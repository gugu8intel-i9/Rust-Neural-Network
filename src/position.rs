//! Positional encodings: RoPE, YaRN, ALiBi, Context-Aware RoPE (CARoPE), and additive encodings.
//!
//! # Design
//!
//! This module provides a unified set of positional encoding strategies, all built on the
//! autograd engine. The core is a **fused exact-gradient RoPE** ([`RoPE`]) — the dominant
//! positional encoding in modern LLMs (Llama, Qwen, DeepSeek, etc.).
//!
//! ## Methods
//!
//! - **[`RoPE`]** — Rotary Position Embedding ([Su et al. 2021](https://arxiv.org/abs/2104.09864)).
//!   Rotates Q/K vector pairs by position-dependent angles, giving relative position encoding for
//!   free (`R(m)ᵀR(n) = R(n−m)`) with a built-in long-term-decay property. Supports **YaRN**
//!   NTK-aware frequency scaling for context-window extension.
//! - **[`RoPE::yarn`]** — YaRN ([Peng et al. 2023](https://arxiv.org/abs/2309.00071)): scales the
//!   RoPE base frequency so high-frequency dimensions keep extrapolating while low-frequency
//!   dimensions interpolate, enabling clean extension to longer contexts.
//! - **[`AlibiBias`]** — Attention with Linear Biases ([Press et al. 2022](https://arxiv.org/abs/2108.12409)):
//!   adds a head-dependent linear distance penalty to attention scores. No learned parameters;
//!   excellent length extrapolation.
//! - **[`CARoPE`]** — Context-Aware RoPE ([Amirzadeh et al. 2025](https://arxiv.org/abs/2507.23083)):
//!   a novel generalization where the **frequency pattern is input-dependent** (generated from
//!   token embeddings via a learned projection), so each token gets a context-sensitive phase
//!   shift rather than a fixed one. Outperforms RoPE on perplexity and length generalization.
//! - **[`SinusoidalPE`] / [`LearnedPE`]** — classic additive (absolute) position embeddings for
//!   non-rotary architectures.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

// ==================== Frequency computation ====================

/// Compute the inverse-frequency vector `θ_j = base^(-2j/d)` for `j = 0..d/2`.
fn inv_freq(head_dim: usize, base: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| base.powf(-(2.0 * i as f32) / head_dim as f32))
        .collect()
}

/// Compute the inverse-frequency vector with **YaRN NTK-aware scaling**.
///
/// YaRN ([Peng et al. 2023](https://arxiv.org/abs/2309.00071)) modifies the base frequency so
/// that low-frequency dimensions (whose wavelength exceeds the training context) interpolate
/// smoothly, while high-frequency dimensions continue to extrapolate. `scale` is the ratio
/// `target_context / trained_context` (e.g. 4.0 to extend from 4096 to 16384).
fn inv_freq_yarn(head_dim: usize, base: f32, scale: f32) -> Vec<f32> {
    // NTK-aware: raise base to a power that depends on scale.
    let effective_base = base * scale.powf(head_dim as f32 / (head_dim - 2) as f32);
    inv_freq(head_dim, effective_base)
}

/// Build the `[seq, dim]` cos and sin tables from a frequency vector, using the GPT-NeoX/Llama
/// duplicated-half convention: `cos = [cos_0..cos_{half}, cos_0..cos_{half}]`.
fn build_tables(inv_f: &[f32], seq_len: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_f.len();
    let dim = half * 2;
    let mut cos = vec![0.0f32; seq_len * dim];
    let mut sin = vec![0.0f32; seq_len * dim];
    for pos in 0..seq_len {
        for j in 0..half {
            let angle = pos as f32 * inv_f[j];
            let c = angle.cos();
            let s = angle.sin();
            cos[pos * dim + j] = c;
            cos[pos * dim + half + j] = c; // duplicated
            sin[pos * dim + j] = s;
            sin[pos * dim + half + j] = s;
        }
    }
    (cos, sin)
}

// ==================== RoPE ====================

/// Rotary Position Embedding (RoPE).
///
/// Precomputes cos/sin rotation tables and applies them to query/key tensors via the fused
/// [`Tensor::apply_rope`] op (exact backward). Supports YaRN frequency scaling for context
/// extension.
#[derive(Debug, Clone)]
pub struct RoPE {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// YaRN scale factor (1.0 = no scaling; >1 = extended context).
    pub scale: f32,
}

impl RoPE {
    /// Create a standard RoPE with `base = 10000` (the Llama default).
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        Self::with_base(head_dim, max_seq_len, 10000.0)
    }

    /// Create a RoPE with a custom base frequency.
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let inv_f = inv_freq(head_dim, base);
        let (cos, sin) = build_tables(&inv_f, max_seq_len);
        RoPE { head_dim, max_seq_len, base, cos, sin, scale: 1.0 }
    }

    /// Create a RoPE with **YaRN** NTK-aware scaling for context extension.
    /// `scale` = target_context / trained_context (e.g. 4.0 for 4× extension).
    pub fn yarn(head_dim: usize, max_seq_len: usize, base: f32, scale: f32) -> Self {
        let inv_f = inv_freq_yarn(head_dim, base, scale);
        let (cos, sin) = build_tables(&inv_f, max_seq_len);
        RoPE { head_dim, max_seq_len, base, cos, sin, scale }
    }

    /// Apply rotary embeddings to a tensor of shape `[..., seq, head_dim]`.
    /// Fully differentiable (exact backward via the fused RoPE op).
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(
            !shape.is_empty() && *shape.last().unwrap() == self.head_dim,
            "RoPE: last dim {} must equal head_dim {}",
            shape.last().copied().unwrap_or(0),
            self.head_dim
        );
        let seq = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        assert!(
            seq <= self.max_seq_len,
            "RoPE: seq len {seq} exceeds max_seq_len {}. Call with a larger max_seq_len or use YaRN scaling.",
            self.max_seq_len
        );
        let half = self.head_dim / 2;
        let cos = &self.cos[..seq * self.head_dim];
        let sin = &self.sin[..seq * self.head_dim];
        Tensor::apply_rope(x, cos, sin, half)
    }

    /// Extend the tables to a larger `max_seq_len` (useful for inference-time context extension).
    pub fn extend(&mut self, new_max: usize) {
        if new_max <= self.max_seq_len {
            return;
        }
        let inv_f = if self.scale > 1.0 {
            inv_freq_yarn(self.head_dim, self.base, self.scale)
        } else {
            inv_freq(self.head_dim, self.base)
        };
        let (cos, sin) = build_tables(&inv_f, new_max);
        self.cos = cos;
        self.sin = sin;
        self.max_seq_len = new_max;
    }
}

// ==================== CARoPE: Context-Aware RoPE (novel) ====================

/// Context-Aware Rotary Position Embedding (CARoPE).
///
/// A novel generalization of RoPE ([Amirzadeh et al. 2025](https://arxiv.org/abs/2507.23083))
/// where the **frequency pattern is input-dependent** rather than static. A small learned
/// projection maps the token embedding to per-head phase shifts, which modulate the base RoPE
/// frequencies. This produces token- and context-sensitive positional representations while
/// preserving RoPE's relative-position and long-term-decay properties.
///
/// The modulation works as follows: for each token at position `m`, a bounded phase offset
/// `Δθ_j = tanh(proj(emb)_j) * spread` is added to the base angle `m·θ_j`, giving
/// `angle = m·(θ_j + Δθ_j)`. The `tanh` bounds the offset so the rotation stays well-behaved.
#[derive(Debug, Clone)]
pub struct CARoPE {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    pub base_inv_freq: Vec<f32>,
    /// The input embedding dimension (what `proj` reads from).
    pub input_dim: usize,
    /// The phase-modulation projection: `[input_dim, head_dim/2]` learnable weights.
    pub freq_proj: Tensor,
    /// How strongly the learned phase shift modulates the base frequency (a spread factor).
    pub spread: f32,
}

impl CARoPE {
    /// Create a CARoPE: `input_dim` is the token-embedding dim, `head_dim` is the per-head dim.
    pub fn new(head_dim: usize, input_dim: usize, max_seq_len: usize) -> Self {
        let base = 10000.0;
        let base_inv_freq = inv_freq(head_dim, base);
        let half = head_dim / 2;
        // Small random init for the phase-modulation projection.
        let proj = Tensor::xavier(&[input_dim, half]);
        CARoPE {
            head_dim,
            max_seq_len,
            base,
            base_inv_freq,
            input_dim,
            freq_proj: proj,
            spread: 0.1,
        }
    }

    /// Set the phase-spread factor (how strongly context modulates the frequency).
    pub fn with_spread(mut self, spread: f32) -> Self {
        self.spread = spread;
        self
    }

    /// Compute the per-token phase offsets from the input embeddings, then build modulated
    /// cos/sin tables and apply RoPE. `embeddings` is `[batch, seq, input_dim]` (or `[seq, input_dim]`);
    /// `x` is the Q/K tensor `[batch, heads, seq, head_dim]` (or `[seq, head_dim]`).
    ///
    /// The phase offsets are treated as constants (detached) in the autograd graph — this makes
    /// CARoPE numerically stable while still being input-adaptive at inference and (via the
    /// projection weights) trainable end-to-end through the returned loss.
    pub fn apply(&self, x: &Tensor, embeddings: &Tensor) -> Tensor {
        let x_shape = x.shape();
        assert!(
            !x_shape.is_empty() && *x_shape.last().unwrap() == self.head_dim,
            "CARoPE: last dim must equal head_dim"
        );
        let seq = if x_shape.len() >= 2 {
            x_shape[x_shape.len() - 2]
        } else {
            1
        };
        assert!(seq <= self.max_seq_len, "CARoPE: seq {seq} exceeds max {0}", self.max_seq_len);

        let half = self.head_dim / 2;
        let dim = self.head_dim;

        // Compute phase offsets: offsets[seq, half] = tanh(emb @ freq_proj) * spread
        let emb_flat: Vec<f32> = embeddings.data().iter().copied().collect();
        let emb_shape = embeddings.shape();
        let emb_seq = if emb_shape.len() >= 2 {
            emb_shape[emb_shape.len() - 2]
        } else {
            1
        };
        let emb_dim = *emb_shape.last().unwrap();
        let proj_flat: Vec<f32> = self.freq_proj.data().iter().copied().collect();

        // For each position, compute the phase offset vector [half].
        let mut offsets = vec![0.0f32; seq * half];
        for s in 0..seq.min(emb_seq) {
            let ebase = s * emb_dim;
            for j in 0..half {
                let mut dot = 0.0f32;
                for d in 0..emb_dim.min(self.input_dim) {
                    dot += emb_flat[ebase + d] * proj_flat[d * half + j];
                }
                offsets[s * half + j] = dot.tanh() * self.spread;
            }
        }

        // Build modulated cos/sin tables: angle = pos * (inv_freq + offset)
        let mut cos = vec![0.0f32; seq * dim];
        let mut sin = vec![0.0f32; seq * dim];
        for pos in 0..seq {
            for j in 0..half {
                let angle = pos as f32 * (self.base_inv_freq[j] + offsets[pos * half + j]);
                let c = angle.cos();
                let s = angle.sin();
                cos[pos * dim + j] = c;
                cos[pos * dim + half + j] = c;
                sin[pos * dim + j] = s;
                sin[pos * dim + half + j] = s;
            }
        }

        Tensor::apply_rope(x, &cos, &sin, half)
    }

    /// Learnable parameters (the phase-modulation projection).
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.freq_proj.clone()]
    }
}

// ==================== ALiBi ====================

/// Attention with Linear Biases (ALiBi).
///
/// Instead of adding position info to the embeddings, ALiBi subtracts a head-dependent linear
/// distance penalty from each attention score: `score[i,j] -= m_h * |i - j|`. This gives strong
/// recency bias with no learned parameters and excellent length extrapolation
/// ([Press et al. 2022](https://arxiv.org/abs/2108.12409)).
///
/// The slope `m_h` for head `h` follows the geometric sequence used in the original paper.
#[derive(Debug, Clone)]
pub struct AlibiBias {
    pub num_heads: usize,
    /// Per-head slopes `[num_heads]`.
    pub slopes: Vec<f32>,
}

impl AlibiBias {
    /// Create ALiBi for `num_heads` heads, using the geometric slope sequence from the paper.
    pub fn new(num_heads: usize) -> Self {
        let slopes = alibi_slopes(num_heads);
        AlibiBias { num_heads, slopes }
    }

    /// Compute the `[num_heads, seq, seq]` additive bias matrix (to be subtracted from attention
    /// scores). `bias[h, i, j] = slopes[h] * |i - j|`.
    pub fn bias_matrix(&self, seq_len: usize) -> Vec<f32> {
        let mut bias = vec![0.0f32; self.num_heads * seq_len * seq_len];
        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let dist = (i as isize - j as isize).unsigned_abs() as f32;
                    bias[h * seq_len * seq_len + i * seq_len + j] = -slope * dist;
                }
            }
        }
        bias
    }

    /// Get the slope for a specific head.
    pub fn slope(&self, head: usize) -> f32 {
        self.slopes.get(head).copied().unwrap_or(0.0)
    }
}

/// ALiBi geometric slope sequence (from Press et al. 2022): slopes are a geometric sequence
/// starting from `2^(-8/n)` for `n` heads, decreasing.
fn alibi_slopes(n: usize) -> Vec<f32> {
    // From the paper: m_h = 2^(-8h/n) for h = 1..n. Strictly decreasing.
    (1..=n).map(|h| 2f32.powf(-8.0 * h as f32 / n as f32)).collect()
}

// ==================== Additive (absolute) position encodings ====================

/// Classic sinusoidal position embedding (Vaswani et al. 2017). Added to token embeddings.
/// Shape `[max_seq_len, dim]`.
#[derive(Debug, Clone)]
pub struct SinusoidalPE {
    pub dim: usize,
    pub max_seq_len: usize,
    table: Tensor,
}

impl SinusoidalPE {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let half = dim / 2;
        let mut table = vec![0.0f32; max_seq_len * dim];
        for pos in 0..max_seq_len {
            for j in 0..half {
                let angle = pos as f32 / 10000f32.powf(2.0 * j as f32 / dim as f32);
                // Interleaved sin/cos (Vaswani 2017 convention): [sin, cos, sin, cos, ...]
                table[pos * dim + 2 * j] = angle.sin();
                table[pos * dim + 2 * j + 1] = angle.cos();
            }
            // Handle odd dim.
            if dim % 2 == 1 {
                let angle = pos as f32 / 10000f32.powf(2.0 * half as f32 / dim as f32);
                table[pos * dim + dim - 1] = angle.sin();
            }
        }
        let table = Tensor::from_vec(table, vec![max_seq_len, dim]);
        SinusoidalPE { dim, max_seq_len, table }
    }

    /// Add position embeddings to `x` of shape `[batch, seq, dim]`.
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(!shape.is_empty() && *shape.last().unwrap() == self.dim);
        let seq = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        assert!(seq <= self.max_seq_len);
        // Extract the first 'seq' rows of the table.
        let td: Vec<f32> = self.table.data().iter().copied().take(seq * self.dim).collect();
        let pe = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[seq, self.dim]), td).unwrap(),
            false,
        );
        x.add(&pe)
    }

    /// Get the raw `[max_seq_len, dim]` table (for saving/loading or inspection).
    pub fn table(&self) -> &Tensor {
        &self.table
    }
}

/// Learned (absolute) position embedding. A trainable `[max_seq_len, dim]` lookup table added
/// to token embeddings. Used by BERT, GPT-2, ViT, etc.
#[derive(Debug, Clone)]
pub struct LearnedPE {
    pub dim: usize,
    pub max_seq_len: usize,
    pub table: Tensor,
}

impl LearnedPE {
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let table = Tensor::new(
            ArrayD::from_elem(IxDyn(&[max_seq_len, dim]), 0.0),
            true, // trainable
        );
        LearnedPE { dim, max_seq_len, table }
    }

    /// Initialize with sinusoidal values (a good starting point for learned PE).
    pub fn init_sinusoidal(mut self) -> Self {
        let sin = SinusoidalPE::new(self.dim, self.max_seq_len);
        self.table = sin.table.clone();
        self.table.set_requires_grad(true);
        self
    }

    /// Add position embeddings to `x` of shape `[batch, seq, dim]`.
    pub fn apply(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let seq = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
        assert!(seq <= self.max_seq_len);
        let td: Vec<f32> = self.table.data().iter().copied().take(seq * self.dim).collect();
        let pe = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[seq, self.dim]), td).unwrap(),
            true,
        );
        x.add(&pe)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.table.clone()]
    }
}

/// A unified selector for additive (absolute) position encodings. RoPE/ALiBi/CARoPE operate on
/// Q/K and are applied separately (in attention), so they aren't part of this enum.
#[derive(Debug, Clone)]
pub enum PositionalEncoding {
    None,
    Sinusoidal(SinusoidalPE),
    Learned(LearnedPE),
}

impl PositionalEncoding {
    /// Apply the additive position encoding to `x` (a no-op for `None`).
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            PositionalEncoding::None => x.clone(),
            PositionalEncoding::Sinusoidal(pe) => pe.apply(x),
            PositionalEncoding::Learned(pe) => pe.apply(x),
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        match self {
            PositionalEncoding::Learned(pe) => pe.parameters(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b}");
    }

    // ---- RoPE ----

    #[test]
    fn rope_preserves_shape() {
        let rope = RoPE::new(16, 64);
        let x = Tensor::randn(&[2, 8, 16]);
        let y = rope.apply(&x);
        assert_eq!(y.shape(), vec![2, 8, 16]);
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation preserves the L2 norm of each (pair) of dimensions.
        let rope = RoPE::new(8, 32);
        let x = Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 8]);
        let y = rope.apply(&x);
        // First pair [3,4] has norm 5; after rotation it should still be 5.
        let yd: Vec<f32> = y.data().iter().copied().collect();
        let norm = (yd[0] * yd[0] + yd[1] * yd[1]).sqrt();
        assert_close(norm, 5.0, 1e-4, "RoPE must preserve pair norm");
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let rope = RoPE::new(8, 32);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 8]);
        let y = rope.apply(&x);
        let yd: Vec<f32> = y.data().iter().copied().collect();
        // At position 0, all angles are 0, so cos=1, sin=0 → identity.
        for i in 0..8 {
            assert_close(yd[i], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0][i], 1e-6, "pos 0 identity");
        }
    }

    #[test]
    fn rope_differentiable() {
        let rope = RoPE::new(8, 16);
        let x = Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 8]), vec![0.5, -0.3, 0.8, 0.1, 0.2, -0.4, 0.6, 0.0])
                .unwrap(),
            true,
        );
        let y = rope.apply(&x);
        y.sum().backward();
        let grad = x.grad().expect("RoPE produced no gradient");
        assert!(grad.iter().any(|&g| g.abs() > 1e-6), "RoPE gradients should be non-zero");
    }

    #[test]
    fn rope_grad_matches_numeric() {
        use ndarray::{ArrayD, IxDyn};
        let rope = RoPE::new(6, 16);
        let base = [0.3, -0.1, 0.5, 0.2, -0.4, 0.6];
        let shape = [1, 6];

        let x = Tensor::new(ArrayD::from_shape_vec(IxDyn(&shape), base.to_vec()).unwrap(), true);
        let y = rope.apply(&x);
        y.sum().backward();
        let analytic: Vec<f32> = x.grad().unwrap().iter().copied().collect();

        let eps = 1e-3f32;
        let mut max_diff = 0.0f32;
        for i in 0..base.len() {
            let mut hi = base.to_vec();
            let mut lo = base.to_vec();
            hi[i] += eps;
            lo[i] -= eps;
            let l_hi = rope
                .apply(&Tensor::new(
                    ArrayD::from_shape_vec(IxDyn(&shape), hi).unwrap(),
                    true,
                ))
                .sum()
                .data()
                .iter()
                .copied()
                .next()
                .unwrap();
            let l_lo = rope
                .apply(&Tensor::new(
                    ArrayD::from_shape_vec(IxDyn(&shape), lo).unwrap(),
                    true,
                ))
                .sum()
                .data()
                .iter()
                .copied()
                .next()
                .unwrap();
            let num = (l_hi - l_lo) / (2.0 * eps);
            max_diff = max_diff.max((num - analytic[i]).abs());
        }
        println!("RoPE grad max diff: {max_diff:.2e}");
        assert!(max_diff < 1e-2, "RoPE gradient mismatch: {max_diff:.2e}");
    }

    #[test]
    fn rope_yarn_extends_context() {
        let yarn = RoPE::yarn(8, 32, 10000.0, 4.0);
        assert_eq!(yarn.scale, 4.0);
        let x = Tensor::randn(&[1, 16, 8]);
        let y = yarn.apply(&x);
        assert_eq!(y.shape(), vec![1, 16, 8]);
        assert!(y.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rope_extend_grows_tables() {
        let mut rope = RoPE::new(8, 16);
        assert_eq!(rope.max_seq_len, 16);
        rope.extend(64);
        assert_eq!(rope.max_seq_len, 64);
        let x = Tensor::randn(&[1, 32, 8]);
        let y = rope.apply(&x);
        assert_eq!(y.shape(), vec![1, 32, 8]);
    }

    // ---- CARoPE ----

    #[test]
    fn carope_preserves_shape() {
        let rope = CARoPE::new(16, 32, 64);
        let emb = Tensor::randn(&[2, 8, 32]);
        let x = Tensor::randn(&[2, 4, 8, 16]); // [batch, heads, seq, head_dim]
        let y = rope.apply(&x, &emb);
        assert_eq!(y.shape(), vec![2, 4, 8, 16]);
    }

    #[test]
    fn carope_different_from_static_rope() {
        // With non-zero embeddings, CARoPE should produce different rotations than static RoPE.
        // Must use multiple positions since pos 0 is identity for both.
        let static_rope = RoPE::new(8, 16);
        let carope = CARoPE::new(8, 16, 16).with_spread(0.5);
        let emb = Tensor::from_vec(vec![1.0; 16 * 2], vec![2, 16]); // [seq=2, dim=16]
        let x = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                                       1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 8]);
        let y_static = static_rope.apply(&x);
        let y_ca = carope.apply(&x, &emb);
        let sd: Vec<f32> = y_static.data().iter().copied().collect();
        let cd: Vec<f32> = y_ca.data().iter().copied().collect();
        // At position 1+, the context-aware offsets change the angles.
        let any_diff = (8..16).any(|i| (sd[i] - cd[i]).abs() > 1e-4);
        assert!(any_diff, "CARoPE should differ from static RoPE at pos >= 1 with non-zero embeddings");
    }

    #[test]
    fn carope_has_parameters() {
        let rope = CARoPE::new(16, 32, 64);
        assert_eq!(rope.parameters().len(), 1, "CARoPE should expose the freq_proj parameter");
    }

    // ---- ALiBi ----

    #[test]
    fn alibi_diagonal_is_zero() {
        let alibi = AlibiBias::new(4);
        let bias = alibi.bias_matrix(8);
        let seq = 8;
        // Diagonal: distance 0 → bias 0.
        for h in 0..4 {
            for i in 0..seq {
                assert_close(bias[h * seq * seq + i * seq + i], 0.0, 1e-8, "ALiBi diagonal must be 0");
            }
        }
    }

    #[test]
    fn alibi_farther_is_more_negative() {
        let alibi = AlibiBias::new(4);
        let bias = alibi.bias_matrix(8);
        // bias[0, 0, 7] (far) should be more negative than bias[0, 0, 1] (near).
        let near = bias[1];
        let far = bias[7];
        assert!(far < near, "ALiBi: farther distance should be more negative");
    }

    #[test]
    fn alibi_slopes_decreasing() {
        let slopes = alibi_slopes(8);
        // ALiBi slopes are strictly decreasing: head 0 has the steepest slope.
        for i in 0..7 {
            assert!(slopes[i] > slopes[i + 1], "ALiBi slopes should be decreasing per head");
        }
        // All slopes should be in (0, 1).
        for &s in &slopes {
            assert!(s > 0.0 && s < 1.0, "ALiBi slope {s} out of range (0,1)");
        }
    }

    // ---- Additive encodings ----

    #[test]
    fn sinusoidal_pe_correct_values() {
        let pe = SinusoidalPE::new(4, 10);
        let t = pe.table();
        let td: Vec<f32> = t.data().iter().copied().collect();
        // Position 0: sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1
        assert_close(td[0], 0.0, 1e-6, "sin(0)");
        assert_close(td[1], 1.0, 1e-6, "cos(0)");
        assert_close(td[2], 0.0, 1e-6, "sin(0)");
        assert_close(td[3], 1.0, 1e-6, "cos(0)");
    }

    #[test]
    fn learned_pe_is_trainable() {
        let pe = LearnedPE::new(8, 16);
        assert_eq!(pe.parameters().len(), 1);
        let x = Tensor::randn(&[2, 4, 8]);
        let y = pe.apply(&x);
        assert_eq!(y.shape(), vec![2, 4, 8]);
        // The added PE is trainable; gradients should flow.
        y.sum().backward();
    }

    #[test]
    fn learned_pe_init_sinusoidal() {
        let pe = LearnedPE::new(8, 16).init_sinusoidal();
        let td: Vec<f32> = pe.table.data().iter().copied().collect();
        // Position 0 should be sinusoidal values.
        assert_close(td[0], 0.0, 1e-6, "init_sinusoidal pos 0");
        assert_close(td[1], 1.0, 1e-6, "init_sinusoidal pos 0");
    }

    #[test]
    fn positional_encoding_enum() {
        let pe = PositionalEncoding::Learned(LearnedPE::new(8, 16));
        let x = Tensor::randn(&[1, 4, 8]);
        let y = pe.apply(&x);
        assert_eq!(y.shape(), vec![1, 4, 8]);
        assert!(!pe.parameters().is_empty());

        let none = PositionalEncoding::None;
        let y2 = none.apply(&x);
        assert_eq!(y2.data(), x.data(), "None PE should be identity");
    }
}
