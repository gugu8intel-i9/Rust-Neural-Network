//! # Linear (kernelized) attention — an O(N) alternative to FlashAttention.
//!
//! FlashAttention is exact but still does **O(N²·d)** work: it never materialises the full N×N
//! matrix (that is its memory win), but it still *computes* every Q·K score. For long sequences
//! that quadratic compute is the wall.
//!
//! **Linear attention** replaces the softmax kernel `exp(q·k)` with a decomposable feature map
//! `φ(·)` such that `φ(q)·φ(k)` plays the role of the attention logit. Because the kernel is a
//! plain dot product of features, the softmax normaliser and the value mix can be reordered via
//! the **associative trick**:
//!
//! ```text
//!   out_i = ( Σ_j φ(q_i)·φ(k_j) · v_j ) / ( Σ_j φ(q_i)·φ(k_j) )
//!         = φ(q_i)·( Σ_j φ(k_j) ⊗ v_j ) / ( φ(q_i)·( Σ_j φ(k_j) ) )
//!              ^^^^^^^^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^
//!                 "KV state" S ∈ R^{d×d}        normaliser z ∈ R^d
//! ```
//!
//! `S` and `z` are built **once** from all keys/values (one `φ(K)ᵀ·V` GEMM), then every query is
//! answered with one `φ(Q)·S` GEMM. Total work is **O(N·d²)** — *linear* in sequence length — and
//! there is never an N×N matrix, in compute **or** memory.
//!
//! This module ships two positive feature maps (a linear attention kernel must be non-negative so
//! the normaliser behaves):
//! - [`KernelKind::Elu`] — `φ(x) = elu(x)+1` (Katharopoulos et al. 2020), the canonical choice.
//! - [`KernelKind::Relu2`] — `φ(x) = max(0,x)²`, smooth and cheap.
//!
//! Everything runs through the crate's [`blas`](crate::blas) `sgemm`, supports both non-causal
//! (bidirectional) and causal (autoregressive) masks, and is **fully differentiable** via a fused
//! autograd op with an exact O(N·d²) backward.
//!
//! ## When to use it
//! - **Long sequences** (N ≫ d): linear attention is dramatically faster than FlashAttention.
//! - **Bidirectional encoders** (non-causal): the `φ(K)ᵀ·V` state is computed once with a single
//!   parallel GEMM — the fastest path.
//! - Anywhere a small approximation vs. exact softmax is acceptable in exchange for a big speedup.

use crate::blas::{sgemm, Transpose};

/// Feature map used by [`LinearAttention`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// `φ(x) = elu(x) + 1` (x+1 for x>0, eˣ otherwise). The standard linear-attention kernel.
    Elu = 0,
    /// `φ(x) = max(0, x)²`. Smooth, non-negative, very cheap.
    Relu2 = 1,
}

impl KernelKind {
    /// Encode as a `u8` for storage inside the autograd `Op`.
    pub fn to_u8(self) -> u8 {
        self as u8
    }
    /// Decode from a `u8` (defaults to [`KernelKind::Elu`] for any unknown value).
    pub fn from_u8(v: u8) -> Self {
        if v == KernelKind::Relu2 as u8 {
            KernelKind::Relu2
        } else {
            KernelKind::Elu
        }
    }
}

#[inline]
fn phi(kind: KernelKind, x: f32) -> f32 {
    match kind {
        KernelKind::Elu => {
            if x > 0.0 {
                x + 1.0
            } else {
                x.exp()
            }
        }
        KernelKind::Relu2 => {
            let r = x.max(0.0);
            r * r
        }
    }
}

#[inline]
fn phi_prime(kind: KernelKind, x: f32) -> f32 {
    match kind {
        // φ'(x) = 1 for x>0, eˣ (= φ(x)) otherwise.
        KernelKind::Elu => {
            if x > 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        KernelKind::Relu2 => 2.0 * x.max(0.0),
    }
}

fn apply_phi(kind: KernelKind, x: &[f32], scale: f32, out: &mut [f32]) {
    for i in 0..x.len() {
        out[i] = phi(kind, x[i] * scale);
    }
}

// ============================================================================================
// Forward
// ============================================================================================

/// Linear-attention forward for `[batch, seq, d]` inputs (q, k, v must share that shape).
///
/// Returns the attention output `[batch, seq, d]`. `scale` is applied to q and k before the
/// feature map (use `1/√d` to mimic softmax-attention scaling).
pub fn linear_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
    causal: bool,
    kernel: KernelKind,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq * d];
    let sd = seq * d;
    for b in 0..batch {
        let qb = &q[b * sd..(b + 1) * sd];
        let kb = &k[b * sd..(b + 1) * sd];
        let vb = &v[b * sd..(b + 1) * sd];
        let ob = &mut out[b * sd..(b + 1) * sd];
        if causal {
            forward_causal(kernel, qb, kb, vb, seq, d, scale, eps, ob);
        } else {
            forward_ncausal(kernel, qb, kb, vb, seq, d, scale, eps, ob);
        }
    }
    out
}

/// Non-causal (bidirectional) forward: one `φ(K)ᵀ·V` GEMM builds the state, then `φ(Q)·S`.
fn forward_ncausal(
    kind: KernelKind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
    out: &mut [f32],
) {
    let mut phi_q = vec![0.0f32; seq * d];
    let mut phi_k = vec![0.0f32; seq * d];
    apply_phi(kind, q, scale, &mut phi_q);
    apply_phi(kind, k, scale, &mut phi_k);

    // S = φ(K)ᵀ · V   ∈ R^{d×d}    (φ(K) is [seq,d], so transa=Trans, lda=d)
    let mut s = vec![0.0f32; d * d];
    sgemm(
        Transpose::Trans,
        Transpose::NoTrans,
        d,
        d,
        seq,
        1.0,
        &phi_k,
        d,
        v,
        d,
        0.0,
        &mut s,
        d,
    );

    // z = Σ_t φ(k_t)  ∈ R^d
    let mut z = vec![0.0f32; d];
    for t in 0..seq {
        for a in 0..d {
            z[a] += phi_k[t * d + a];
        }
    }

    // num = φ(Q) · S   ∈ R^{seq×d}
    let mut num = vec![0.0f32; seq * d];
    sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        seq,
        d,
        d,
        1.0,
        &phi_q,
        d,
        &s,
        d,
        0.0,
        &mut num,
        d,
    );

    // den_t = φ(q_t)·z ; out_t = num_t / (den_t + eps)
    for t in 0..seq {
        let mut den = 0.0f32;
        for a in 0..d {
            den += phi_q[t * d + a] * z[a];
        }
        let inv = 1.0 / (den + eps);
        for b in 0..d {
            out[t * d + b] = num[t * d + b] * inv;
        }
    }
}

/// Causal (autoregressive) forward via the running recurrence `S_t = S_{t-1} + φ(k_t)⊗v_t`.
/// Position t attends to positions 0..=t (inclusive). Still O(N·d²) — no N×N matrix.
fn forward_causal(
    kind: KernelKind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
    out: &mut [f32],
) {
    let mut s = vec![0.0f32; d * d];
    let mut z = vec![0.0f32; d];
    let mut phi_q_t = vec![0.0f32; d];
    let mut phi_k_t = vec![0.0f32; d];
    let mut num_t = vec![0.0f32; d];
    for t in 0..seq {
        // φ of the current position.
        for a in 0..d {
            phi_q_t[a] = phi(kind, q[t * d + a] * scale);
            phi_k_t[a] = phi(kind, k[t * d + a] * scale);
        }
        // Fold the current key/value into the running state (causal: attend to 0..=t).
        for a in 0..d {
            let pka = phi_k_t[a];
            z[a] += pka;
            let sa = a * d;
            let vt = t * d;
            for b in 0..d {
                s[sa + b] += pka * v[vt + b];
            }
        }
        // num = φ(q_t) · S
        for b in 0..d {
            let mut acc = 0.0f32;
            let mut a = 0;
            while a < d {
                acc += phi_q_t[a] * s[a * d + b];
                a += 1;
            }
            num_t[b] = acc;
        }
        // den = φ(q_t) · z
        let mut den = 0.0f32;
        for a in 0..d {
            den += phi_q_t[a] * z[a];
        }
        let inv = 1.0 / (den + eps);
        for b in 0..d {
            out[t * d + b] = num_t[b] * inv;
        }
    }
}

// ============================================================================================
// Backward (exact, O(N·d²))
// ============================================================================================

/// Exact reverse-mode VJP for linear attention.
///
/// Given the upstream gradient `grad_out` w.r.t. the forward output, returns
/// `(∂L/∂q, ∂L/∂k, ∂L/∂v)`, each `[batch, seq, d]`.
#[allow(clippy::too_many_arguments)]
pub fn linear_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_out: &[f32],
    batch: usize,
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
    causal: bool,
    kernel: KernelKind,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0f32; batch * seq * d];
    let mut dk = vec![0.0f32; batch * seq * d];
    let mut dv = vec![0.0f32; batch * seq * d];
    let sd = seq * d;
    for b in 0..batch {
        let qb = &q[b * sd..(b + 1) * sd];
        let kb = &k[b * sd..(b + 1) * sd];
        let vb = &v[b * sd..(b + 1) * sd];
        let gb = &grad_out[b * sd..(b + 1) * sd];
        let (dqb, dkb, dvb) = if causal {
            backward_causal(kernel, qb, kb, vb, gb, seq, d, scale, eps)
        } else {
            backward_ncausal(kernel, qb, kb, vb, gb, seq, d, scale, eps)
        };
        dq[b * sd..(b + 1) * sd].copy_from_slice(&dqb);
        dk[b * sd..(b + 1) * sd].copy_from_slice(&dkb);
        dv[b * sd..(b + 1) * sd].copy_from_slice(&dvb);
    }
    (dq, dk, dv)
}

#[allow(clippy::too_many_arguments)]
fn backward_ncausal(
    kind: KernelKind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Recompute forward intermediates.
    let mut phi_q = vec![0.0f32; seq * d];
    let mut phi_k = vec![0.0f32; seq * d];
    apply_phi(kind, q, scale, &mut phi_q);
    apply_phi(kind, k, scale, &mut phi_k);

    let mut s = vec![0.0f32; d * d];
    sgemm(
        Transpose::Trans,
        Transpose::NoTrans,
        d,
        d,
        seq,
        1.0,
        &phi_k,
        d,
        v,
        d,
        0.0,
        &mut s,
        d,
    );

    let mut z = vec![0.0f32; d];
    for t in 0..seq {
        for a in 0..d {
            z[a] += phi_k[t * d + a];
        }
    }

    let mut num = vec![0.0f32; seq * d];
    sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        seq,
        d,
        d,
        1.0,
        &phi_q,
        d,
        &s,
        d,
        0.0,
        &mut num,
        d,
    );

    let mut den = vec![0.0f32; seq];
    for t in 0..seq {
        let mut acc = 0.0f32;
        for a in 0..d {
            acc += phi_q[t * d + a] * z[a];
        }
        den[t] = acc;
    }

    // gnum_t = grad_t / (den_t + eps)   ;   gdi_t = -<grad_t, num_t> / (den_t + eps)^2
    let mut gnum = vec![0.0f32; seq * d];
    let mut gdi = vec![0.0f32; seq];
    for t in 0..seq {
        let inv = 1.0 / (den[t] + eps);
        let inv2 = inv * inv;
        let mut dot = 0.0f32;
        for b in 0..d {
            gnum[t * d + b] = grad[t * d + b] * inv;
            dot += grad[t * d + b] * num[t * d + b];
        }
        gdi[t] = -dot * inv2;
    }

    // ∂L/∂S = φ(Q)ᵀ · gnum   ∈ R^{d×d}
    let mut s_grad = vec![0.0f32; d * d];
    sgemm(
        Transpose::Trans,
        Transpose::NoTrans,
        d,
        d,
        seq,
        1.0,
        &phi_q,
        d,
        &gnum,
        d,
        0.0,
        &mut s_grad,
        d,
    );

    // ∂L/∂z = Σ_t φ(q_t) · gdi_t   ∈ R^d
    let mut z_grad = vec![0.0f32; d];
    for t in 0..seq {
        for a in 0..d {
            z_grad[a] += phi_q[t * d + a] * gdi[t];
        }
    }

    // ∂L/∂φ(Q) = gnum · Sᵀ  +  outer(gdi, z)   ∈ R^{seq×d}
    let mut phi_q_grad = vec![0.0f32; seq * d];
    sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        seq,
        d,
        d,
        1.0,
        &gnum,
        d,
        &s,
        d,
        0.0,
        &mut phi_q_grad,
        d,
    );
    for t in 0..seq {
        let g = gdi[t];
        for a in 0..d {
            phi_q_grad[t * d + a] += g * z[a];
        }
    }

    // ∂L/∂V = φ(K) · S_grad    ∈ R^{seq×d}
    let mut dv = vec![0.0f32; seq * d];
    sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        seq,
        d,
        d,
        1.0,
        &phi_k,
        d,
        &s_grad,
        d,
        0.0,
        &mut dv,
        d,
    );

    // ∂L/∂φ(K) = V · S_gradᵀ  +  broadcast(z_grad)   ∈ R^{seq×d}
    let mut phi_k_grad = vec![0.0f32; seq * d];
    sgemm(
        Transpose::NoTrans,
        Transpose::Trans,
        seq,
        d,
        d,
        1.0,
        v,
        d,
        &s_grad,
        d,
        0.0,
        &mut phi_k_grad,
        d,
    );
    for t in 0..seq {
        for a in 0..d {
            phi_k_grad[t * d + a] += z_grad[a];
        }
    }

    // Chain through φ (with the scale factor on q/k).
    let mut dq = vec![0.0f32; seq * d];
    let mut dk = vec![0.0f32; seq * d];
    for t in 0..seq {
        for a in 0..d {
            let idx = t * d + a;
            dq[idx] = phi_q_grad[idx] * phi_prime(kind, q[idx] * scale) * scale;
            dk[idx] = phi_k_grad[idx] * phi_prime(kind, k[idx] * scale) * scale;
        }
    }
    (dq, dk, dv)
}

/// Causal backward: a reverse scan carrying ∂L/∂S and ∂L/∂z through the recurrence.
///
/// Forward states S_t, z_t (cumulative, inclusive) are precomputed once and stored; the reverse
/// loop folds gradients back through `S_t = S_{t-1} + φ(k_t)⊗v_t` and `z_t = z_{t-1} + φ(k_t)`.
#[allow(clippy::too_many_arguments)]
fn backward_causal(
    kind: KernelKind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // ---- precompute forward states per step ----
    let mut phi_q_all = vec![0.0f32; seq * d];
    let mut phi_k_all = vec![0.0f32; seq * d];
    let mut states_s = vec![0.0f32; seq * d * d]; // S_t for each t (inclusive)
    let mut states_z = vec![0.0f32; seq * d];
    let mut num_all = vec![0.0f32; seq * d];
    let mut den_all = vec![0.0f32; seq];
    {
        let mut s = vec![0.0f32; d * d];
        let mut z = vec![0.0f32; d];
        for t in 0..seq {
            for a in 0..d {
                let pqa = phi(kind, q[t * d + a] * scale);
                let pka = phi(kind, k[t * d + a] * scale);
                phi_q_all[t * d + a] = pqa;
                phi_k_all[t * d + a] = pka;
                z[a] += pka;
                let vt = t * d;
                let sa = a * d;
                for b in 0..d {
                    s[sa + b] += pka * v[vt + b];
                }
                states_z[t * d + a] = z[a];
            }
            // copy current S, z into stored state, then compute num/den.
            states_s[t * d * d..(t + 1) * d * d].copy_from_slice(&s);
            let mut den = 0.0f32;
            for a in 0..d {
                den += phi_q_all[t * d + a] * z[a];
            }
            den_all[t] = den;
            for b in 0..d {
                let mut acc = 0.0f32;
                let mut a = 0;
                while a < d {
                    acc += phi_q_all[t * d + a] * s[a * d + b];
                    a += 1;
                }
                num_all[t * d + b] = acc;
            }
        }
    }

    let mut dq = vec![0.0f32; seq * d];
    let mut dk = vec![0.0f32; seq * d];
    let mut dv = vec![0.0f32; seq * d];
    // Running gradients through the recurrence (carry ∂L/∂S_t, ∂L/∂z_t backward in time).
    let mut ds = vec![0.0f32; d * d];
    let mut dz = vec![0.0f32; d];

    for t in (0..seq).rev() {
        let den_t = den_all[t] + eps;
        let inv = 1.0 / den_t;
        let inv2 = inv * inv;
        // gnum_t = grad_t / den_t ; gdi_t = -<grad_t,num_t>/den_t^2
        let mut gnum_t = vec![0.0f32; d];
        let mut dot = 0.0f32;
        for b in 0..d {
            gnum_t[b] = grad[t * d + b] * inv;
            dot += grad[t * d + b] * num_all[t * d + b];
        }
        let gdi_t = -dot * inv2;

        let s_t = &states_s[t * d * d..(t + 1) * d * d];
        let z_t = &states_z[t * d..(t + 1) * d];

        // ∂L/∂φ(q_t) = S_t · gnum_t + z_t · gdi_t
        let mut dphi_q = vec![0.0f32; d];
        for a in 0..d {
            let mut acc = 0.0f32;
            for b in 0..d {
                acc += s_t[a * d + b] * gnum_t[b];
            }
            dphi_q[a] = acc + z_t[a] * gdi_t;
        }

        // Fold local dependencies into the running state gradients:
        // ∂L/∂S_t += φ(q_t) ⊗ gnum_t ; ∂L/∂z_t += φ(q_t) · gdi_t
        for a in 0..d {
            let pqa = phi_q_all[t * d + a];
            dz[a] += pqa * gdi_t;
            let sa = a * d;
            let g = gnum_t[a];
            for b in 0..d {
                ds[sa + b] += pqa * gnum_t[b];
            }
            // (g unused aside from clarity; keep gnum_t[a] usage above)
            let _ = g;
        }

        // Now ds, dz are ∂L/∂S_t, ∂L/∂z_t. Propagate to k_t, v_t:
        // ∂L/∂φ(k_t) = S_grad · v_t + z_grad   ;   ∂L/∂v_t = S_gradᵀ · φ(k_t)
        let vt = t * d;
        let mut dphi_k = vec![0.0f32; d];
        for a in 0..d {
            let mut acc = 0.0f32;
            for b in 0..d {
                acc += ds[a * d + b] * v[vt + b];
            }
            dphi_k[a] = acc + dz[a];
        }
        for b in 0..d {
            let mut acc = 0.0f32;
            for a in 0..d {
                acc += ds[a * d + b] * phi_k_all[t * d + a];
            }
            dv[vt + b] = acc;
        }

        // Chain through φ (scale on q,k).
        for a in 0..d {
            dq[t * d + a] = dphi_q[a] * phi_prime(kind, q[t * d + a] * scale) * scale;
            dk[t * d + a] = dphi_k[a] * phi_prime(kind, k[t * d + a] * scale) * scale;
        }

        // ds, dz carry unchanged to t-1 (S_t = S_{t-1} + ..., so ∂L/∂S_{t-1} = ∂L/∂S_t).
    }

    (dq, dk, dv)
}

// ============================================================================================
// Public struct + convenience
// ============================================================================================

/// Configuration for linear attention.
#[derive(Debug, Clone, Copy)]
pub struct LinearAttention {
    /// Feature-map kernel.
    pub kernel: KernelKind,
    /// Applied to q and k before the feature map (use `1/√d` to mimic softmax scaling).
    pub scale: f32,
    /// Numerical-stability epsilon added to the normaliser.
    pub eps: f32,
    /// If true, position t attends only to positions 0..=t (autoregressive).
    pub causal: bool,
}

impl LinearAttention {
    /// Build with a kernel; scale defaults to `1/√d`, eps to `1e-6`, non-causal.
    pub fn new(kernel: KernelKind, d: usize) -> Self {
        LinearAttention {
            kernel,
            scale: 1.0 / (d as f32).sqrt(),
            eps: 1e-6,
            causal: false,
        }
    }

    /// Set the causal mask.
    pub fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
    /// Set the scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Default ELU linear attention for head dim `d`, non-causal.
    pub fn elu(d: usize) -> Self {
        Self::new(KernelKind::Elu, d)
    }
}

// ============================================================================================
// Naive O(N²) reference (tests only) — explicit kernel computation, no associative trick.
// ============================================================================================

#[cfg(test)]
fn naive_reference(
    kind: KernelKind,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d: usize,
    scale: f32,
    eps: f32,
    causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * d];
    for i in 0..seq {
        let mut acc = vec![0.0f32; d];
        let mut den = 0.0f32;
        for j in 0..seq {
            if causal && j > i {
                continue;
            }
            // kernel score = φ(q_i)·φ(k_j)
            let mut score = 0.0f32;
            for a in 0..d {
                score += phi(kind, q[i * d + a] * scale) * phi(kind, k[j * d + a] * scale);
            }
            for b in 0..d {
                acc[b] += score * v[j * d + b];
            }
            den += score;
        }
        let inv = 1.0 / (den + eps);
        for b in 0..d {
            out[i * d + b] = acc[b] * inv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn lcg(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((*seed >> 8) as f32 / 16777216.0) - 0.5
            })
            .collect()
    }

    #[test]
    fn forward_ncausal_matches_naive_elu() {
        let (batch, seq, d) = (2, 12, 8);
        let mut s = 1u32;
        let q = lcg(&mut s, batch * seq * d);
        let k = lcg(&mut s, batch * seq * d);
        let v = lcg(&mut s, batch * seq * d);
        let scale = 1.0 / (d as f32).sqrt();
        let out = linear_attention_forward(
            &q,
            &k,
            &v,
            batch,
            seq,
            d,
            scale,
            1e-6,
            false,
            KernelKind::Elu,
        );
        // compare each batch against the naive O(N^2) reference
        let sd = seq * d;
        for b in 0..batch {
            let refb = naive_reference(
                KernelKind::Elu,
                &q[b * sd..],
                &k[b * sd..],
                &v[b * sd..],
                seq,
                d,
                scale,
                1e-6,
                false,
            );
            let diff = max_abs_diff(&out[b * sd..(b + 1) * sd], &refb);
            assert!(diff < 1e-4, "batch {b} non-causal ELU diff {diff}");
        }
    }

    #[test]
    fn forward_causal_matches_naive_elu() {
        let (batch, seq, d) = (2, 11, 7);
        let mut s = 2u32;
        let q = lcg(&mut s, batch * seq * d);
        let k = lcg(&mut s, batch * seq * d);
        let v = lcg(&mut s, batch * seq * d);
        let scale = 1.0 / (d as f32).sqrt();
        let out = linear_attention_forward(
            &q,
            &k,
            &v,
            batch,
            seq,
            d,
            scale,
            1e-6,
            true,
            KernelKind::Elu,
        );
        let sd = seq * d;
        for b in 0..batch {
            let refb = naive_reference(
                KernelKind::Elu,
                &q[b * sd..],
                &k[b * sd..],
                &v[b * sd..],
                seq,
                d,
                scale,
                1e-6,
                true,
            );
            let diff = max_abs_diff(&out[b * sd..(b + 1) * sd], &refb);
            assert!(diff < 1e-4, "batch {b} causal ELU diff {diff}");
        }
    }

    #[test]
    fn forward_ncausal_matches_naive_relu2() {
        let (batch, seq, d) = (1, 14, 9);
        let mut s = 3u32;
        let q = lcg(&mut s, batch * seq * d);
        let k = lcg(&mut s, batch * seq * d);
        let v = lcg(&mut s, batch * seq * d);
        let scale = 1.0 / (d as f32).sqrt();
        let out = linear_attention_forward(
            &q,
            &k,
            &v,
            batch,
            seq,
            d,
            scale,
            1e-6,
            false,
            KernelKind::Relu2,
        );
        let sd = seq * d;
        let refb = naive_reference(KernelKind::Relu2, &q, &k, &v, seq, d, scale, 1e-6, false);
        let diff = max_abs_diff(&out, &refb);
        assert!(diff < 1e-4, "non-causal Relu2 diff {diff}");
        let _ = sd;
    }

    /// Numerical gradient check: the fused backward must match finite differences.
    /// Uses central differences (O(h²)) and a combined abs+rel tolerance, so it is robust to the
    /// near-zero-gradient indices that defeat a naive forward-difference check.
    fn grad_check_once(causal: bool, kind: KernelKind) {
        let (batch, seq, d) = (1, 6, 5);
        let mut s = 42u32;
        let q = lcg(&mut s, batch * seq * d);
        let k = lcg(&mut s, batch * seq * d);
        let v = lcg(&mut s, batch * seq * d);
        let w = lcg(&mut s, batch * seq * d); // loss = Σ w_i · out_i  (random weights → O(1) grads)
        let scale = 0.4f32;
        let eps = 1e-6f32;
        let n = batch * seq * d;
        let h = 1e-3f32;

        let (dq, dk, dv) =
            linear_attention_backward(&q, &k, &v, &w, batch, seq, d, scale, eps, causal, kind);

        let fwd = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let o = linear_attention_forward(qq, kk, vv, batch, seq, d, scale, eps, causal, kind);
            w.iter().zip(o.iter()).map(|(a, b)| a * b).sum()
        };

        let check = |which: usize| {
            for idx in 0..n {
                let mut qp = match which {
                    0 => q.clone(),
                    1 => k.clone(),
                    _ => v.clone(),
                };
                let mut qm = qp.clone();
                qp[idx] += h;
                qm[idx] -= h;
                let (fp, fm) = match which {
                    0 => (fwd(&qp, &k, &v), fwd(&qm, &k, &v)),
                    1 => (fwd(&q, &qp, &v), fwd(&q, &qm, &v)),
                    _ => (fwd(&q, &k, &qp), fwd(&q, &k, &qm)),
                };
                let num = (fp - fm) / (2.0 * h);
                let ana = match which {
                    0 => dq[idx],
                    1 => dk[idx],
                    _ => dv[idx],
                };
                let err = (num - ana).abs();
                let tol = 1e-3 + 1e-2 * ana.abs().max(num.abs());
                assert!(
                    err < tol,
                    "grad check failed: which={which} idx={idx} num={num} ana={ana} err={err} tol={tol}"
                );
            }
        };
        check(0);
        check(1);
        check(2);
    }

    #[test]
    fn grad_check_ncausal_elu() {
        grad_check_once(false, KernelKind::Elu);
    }
    #[test]
    fn grad_check_causal_elu() {
        grad_check_once(true, KernelKind::Elu);
    }
    #[test]
    fn grad_check_ncausal_relu2() {
        grad_check_once(false, KernelKind::Relu2);
    }

    /// End-to-end through the autograd `Tensor` API: the fused Op must populate q/k/v grads.
    #[test]
    fn tensor_api_end_to_end_backward() {
        use crate::tensor::Tensor;
        let (batch, seq, d) = (2, 16, 8);
        let q = Tensor::randn(&[batch, seq, d]);
        let k = Tensor::randn(&[batch, seq, d]);
        let v = Tensor::randn(&[batch, seq, d]);
        q.set_requires_grad(true);
        k.set_requires_grad(true);
        v.set_requires_grad(true);
        let scale = 1.0 / (d as f32).sqrt();
        let out = Tensor::linear_attention(&q, &k, &v, scale, 1e-6, false, KernelKind::Elu);
        assert_eq!(out.shape(), vec![batch, seq, d]);
        out.sum().backward();
        assert_eq!(q.grad().unwrap().shape(), vec![batch, seq, d]);
        assert_eq!(k.grad().unwrap().shape(), vec![batch, seq, d]);
        assert_eq!(v.grad().unwrap().shape(), vec![batch, seq, d]);

        // Causal ReLU² path also runs and produces grads.
        let q2 = Tensor::randn(&[1, 12, 6]);
        let k2 = Tensor::randn(&[1, 12, 6]);
        let v2 = Tensor::randn(&[1, 12, 6]);
        q2.set_requires_grad(true);
        k2.set_requires_grad(true);
        v2.set_requires_grad(true);
        let out2 = Tensor::linear_attention(&q2, &k2, &v2, 0.5, 1e-6, true, KernelKind::Relu2);
        out2.sum().backward();
        assert!(q2.grad().is_some() && k2.grad().is_some() && v2.grad().is_some());
    }
}
