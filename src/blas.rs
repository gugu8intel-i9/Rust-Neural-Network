//! # BLAS backend — a hand-written, transpose-aware, cache-packed linear-algebra engine.
//!
//! This is a from-scratch [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
//! implementation. It is **not** a binding to OpenBLAS/Accelerate/MKL — every cycle comes from
//! our own kernels. The design follows the production GEMM literature (BLIS / Goto's algorithm)
//! and adds three things that make it both **novel** for this crate and **fast** in practice:
//!
//! 1. **Transpose-aware packing.** The classic BLAS `sgemm` signature
//!    `C = α·op(A)·op(B) + β·C` with `op ∈ {NoTrans, Trans}` is implemented directly. When
//!    `op(B) = Trans` (i.e. we need `Bᵀ`), we do **not** materialise a transposed copy of `B` and
//!    then multiply — instead the *packing* pass reads `B` with a stride and writes a
//!    contiguous panel. This matters enormously for backprop: the gradient rules
//!    `∂L/∂A = ∂L/∂C · Bᵀ` and `∂L/∂B = Aᵀ · ∂L/∂C` are exactly two transposed GEMMs, so a
//!    transposed-multiply that never copies `B` (or `A`) is the difference between a fast
//!    backward and a slow one. See [`sgemm`].
//!
//! 2. **B-panel cache packing (BLIS-style).** The inner micro-kernel works on a small,
//!    *contiguous* panel of `B` of shape `[KC, NC]` that is loaded into a packed buffer once and
//!    reused across a whole `[MC, KC]` block of rows of `A`. A naïve strided kernel jumps by the
//!    full `n` between K-rows of `B` and thrashes the cache for any real model width; the packed
//!    panel has stride `NC` and fits in L2, so every packed element participates in `MC` FMAs
//!    before eviction. This is the single biggest lever on large matmuls.
//!
//! 3. **Backend abstraction + size-aware dispatch.** [`BlasBackend`] is a trait so an external
//!    provider (system BLAS, a future GPU path, …) can be slotted in. The built-in
//!    [`NativeBackend`] auto-dispatches: tiny problems skip blocking and rayon overhead, normal
//!    problems use the packed cache-blocked path, and huge near-square problems can optionally
//!    route through Strassen (see [`gemm_strassen`]).
//!
//! On x86_64 the inner kernel uses AVX2 + FMA via `std::arch` with **runtime** feature detection
//! (`is_x86_feature_detected!`); everything else falls back to an auto-vectorisable scalar loop.
//! No `cc`/no C — it is pure Rust, so it compiles anywhere the rest of the crate does.
//!
//! ## Levels
//!
//! | Level | Routines | What they do |
//! |-------|----------|--------------|
//! | 1     | [`sdot`], [`saxpy`], [`sscal`], [`scopy`], [`snrm2`] | vector–vector |
//! | 2     | [`sgemv`] | matrix–vector |
//! | 3     | [`sgemm`], [`gemm_strassen`] | matrix–matrix |

use rayon::prelude::*;

// ============================================================================
// Blocking parameters — tuned for typical L1/L2 geometry (see Goto/BLIS).
// ============================================================================

/// K-block: how many K-elements live in one packed panel. Sized so a `[MC, KC]` A-panel and a
/// `[KC, NC]` B-panel together stay cache-friendly.
const KC: usize = 256;

/// M-block: rows of A (and C) processed per L2 residence of a B-panel.
const MC: usize = 128;

/// N-block: columns of B (and C) per packed B-panel. Sized so the `[KC, NC]` B-panel fits the
/// per-core L2 (so it stays resident while every M-block of A streams past it).
const NC: usize = 256;

/// Below this flop count we skip blocking + threading entirely (overhead dominates).
const SMALL_FLOPS: usize = 4096;

// ============================================================================
// Public types
// ============================================================================

/// Whether an operand should be used as-is (`NoTrans`) or transposed (`Trans`).
///
/// Mirrors the standard BLAS transpose flag. The matrix storage is always **row-major with a
/// leading dimension** equal to the row stride; `Trans` means the logical operand is the
/// transpose of that storage — we handle it inside packing, never by copying the whole matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// Use the matrix as stored: `op(X) = X`.
    NoTrans,
    /// Use the transpose: `op(X) = Xᵀ`.
    Trans,
}

impl Transpose {
    #[inline]
    fn is_trans(self) -> bool {
        matches!(self, Transpose::Trans)
    }
}

// ============================================================================
// Level 3: SGEMM  —  C = α·op(A)·op(B) + β·C   (row-major, leading-dim strides)
// ============================================================================

/// Single-precision general matrix multiply: `C = α·op(A)·op(B) + β·C`.
///
/// * `transa`, `transb` — how to interpret `A`/`B` (see [`Transpose`]).
/// * `m`, `n`, `k` — the *logical* dimensions: `op(A)` is `[m, k]`, `op(B)` is `[k, n]`, `C` is
///   `[m, n]`.
/// * `a`, `lda` — `A` storage. `NoTrans` ⇒ `A` is `[m, k]` row-major with row stride `lda`;
///   `Trans` ⇒ `A` is `[k, m]` row-major with row stride `lda`.
/// * `b`, `ldb` — analogously for `B`.
/// * `c`, `ldc` — `C` is `[m, n]` row-major with row stride `ldc`. Read (scaled by `β`) and
///   written in place.
///
/// # Panics (debug only)
/// Debug builds assert that the supplied strides make every accessed index in-bounds. Release
/// builds trust the caller (standard BLAS contract).
pub fn sgemm(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert_blas(
        transa,
        transb,
        m,
        n,
        k,
        a.len(),
        lda,
        b.len(),
        ldb,
        c.len(),
        ldc,
    );

    // (1) Scale C by β (and treat β == 0 as "overwrite, do not propagate NaNs").
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 || alpha == 0.0 {
        // No A·B contribution — just `C = β·C`.
        scale_c(c, ldc, m, n, beta);
        return;
    }

    // (2) Small matrices: a single serial, ALLOCATION-FREE pass that folds β into the same loop
    // (no separate scale pass, no thread pool, no packing buffers). For tiny matmuls — the norm
    // in small-model training — this avoids the per-call buffer allocation of the blocked path and
    // uses an i-p-j loop order that auto-vectorizes into FMA on contiguous B rows.
    if m * n * k < SMALL_FLOPS {
        gemm_small(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }

    // (3) Packed, cache-blocked, parallel GEMM.
    // Parallelise across M-blocks: each rayon task uniquely owns its C row-block, so there are
    // no data races even though every task reads the shared `A` and `B`.
    scale_c(c, ldc, m, n, beta);
    // Right-size the per-thread packing buffers to the *actual* matrix dimensions (not the full
    // MC×KC / KC×NC blocks), so small-but-not-tiny matmuls don't over-allocate ~640 KB per call.
    // `buf_a` is rounded up to a multiple of MR=4 rows (zero-padded) so the register-tiled
    // micro-kernels can always read a full 4-row tile without going out of bounds.
    let buf_a_size = ((MC.min(m) + 3) & !3) * KC;
    let buf_b_size = KC * NC.min(n);
    c.par_chunks_mut(MC * ldc)
        .enumerate()
        .for_each(|(blk, cblk)| {
            let ii = blk * MC;
            let mb = MC.min(m - ii);
            // Per-thread reusable packing buffers (allocated once, reused across K/N blocks).
            let mut buf_a = vec![0.0f32; buf_a_size];
            let mut buf_b = vec![0.0f32; buf_b_size];

            for kk in (0..k).step_by(KC) {
                let kb = KC.min(k - kk);
                // Pack the A-panel op(A)[ii..ii+mb, kk..kk+kb] → contiguous [mb, kb].
                pack_a(transa, a, lda, ii, mb, kk, kb, &mut buf_a);

                for jj in (0..n).step_by(NC) {
                    let nb = NC.min(n - jj);
                    // Pack the B-panel op(B)[kk..kk+kb, jj..jj+nb] → contiguous [kb, nb],
                    // pre-multiplying by α so the FMA loop is pure accumulation.
                    pack_b(transb, b, ldb, kk, kb, jj, nb, alpha, &mut buf_b);
                    // C[ii..ii+mb, jj..jj+nb] += buf_a[mb,kb] @ buf_b[kb,nb]
                    kernel_dispatch(mb, kb, nb, &buf_a, &buf_b, cblk, ldc, jj);
                }
            }
        });
}

/// Scale C by β in place (treat β == 0 as "overwrite, do not propagate NaNs"). Serial: this is a
/// single pass over the `[m, n]` region — negligible next to the GEMM — and a plain loop avoids
/// any `&mut`-in-`Fn` capture issue and is trivially obviously correct.
pub(crate) fn scale_c(c: &mut [f32], ldc: usize, m: usize, n: usize, beta: f32) {
    if beta == 1.0 {
        return;
    }
    for i in 0..m {
        let row = i * ldc;
        if beta == 0.0 {
            for j in 0..n {
                c[row + j] = 0.0;
            }
        } else {
            for j in 0..n {
                c[row + j] *= beta;
            }
        }
    }
}

/// Small-matrix GEMM: a single serial, **allocation-free** pass that folds β into the same loop
/// as the multiply (no separate scale pass, no thread pool, no packing buffers). Uses an i-p-j
/// loop order so the inner `j` loop walks a contiguous row of B and a contiguous row of C — which
/// the compiler turns into packed FMA. Handles both transpose flags inline via strides.
#[allow(clippy::too_many_arguments)]
fn gemm_small(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for i in 0..m {
        let crow = i * ldc;
        // Fold β into this C row once (cache-friendly: each row touched once for scale, then accum).
        if beta == 0.0 {
            for j in 0..n {
                c[crow + j] = 0.0;
            }
        } else if beta != 1.0 {
            for j in 0..n {
                c[crow + j] *= beta;
            }
        }
        for p in 0..k {
            let av = if transa.is_trans() {
                a[p * lda + i]
            } else {
                a[i * lda + p]
            };
            if av == 0.0 {
                continue;
            }
            let aav = alpha * av; // invariant in j → hoisted
            if !transb.is_trans() {
                // B row p is contiguous: b[p*ldb + j], j = 0..n. Auto-vectorizes to FMA.
                let brow = p * ldb;
                for j in 0..n {
                    c[crow + j] += aav * b[brow + j];
                }
            } else {
                // op(B)[p][j] = b[j*ldb + p] (strided gather — rare for small; still correct).
                for j in 0..n {
                    c[crow + j] += aav * b[j * ldb + p];
                }
            }
        }
    }
}

// ============================================================================
// Packing — turns a strided, possibly-transposed operand into a contiguous panel.
// ============================================================================

/// Pack `op(A)[ii..ii+mb, kk..kk+kb]` into `out[mb, kb]` (row-major, stride `kb`).
#[allow(clippy::too_many_arguments)]
fn pack_a(
    transa: Transpose,
    a: &[f32],
    lda: usize,
    ii: usize,
    mb: usize,
    kk: usize,
    kb: usize,
    out: &mut [f32],
) {
    if !transa.is_trans() {
        // op(A)[i][p] = A[ii+i][kk+p] = a[(ii+i)*lda + (kk+p)]
        for i in 0..mb {
            let src = (ii + i) * lda + kk;
            let dst = i * kb;
            out[dst..dst + kb].copy_from_slice(&a[src..src + kb]);
        }
    } else {
        // op(A)[i][p] = A[kk+p][ii+i] = a[(kk+p)*lda + (ii+i)]  (strided gather)
        let col = ii;
        for i in 0..mb {
            let dst = i * kb;
            for p in 0..kb {
                out[dst + p] = a[(kk + p) * lda + col + i];
            }
        }
    }
}

/// Pack `op(B)[kk..kk+kb, jj..jj+nb]` into `out[kb, nb]` (row-major, stride `nb`),
/// scaling every element by `alpha`.
#[allow(clippy::too_many_arguments)]
fn pack_b(
    transb: Transpose,
    b: &[f32],
    ldb: usize,
    kk: usize,
    kb: usize,
    jj: usize,
    nb: usize,
    alpha: f32,
    out: &mut [f32],
) {
    if !transb.is_trans() {
        // op(B)[p][t] = B[kk+p][jj+t] = b[(kk+p)*ldb + (jj+t)]  (contiguous within a row)
        for p in 0..kb {
            let src = (kk + p) * ldb + jj;
            let dst = p * nb;
            if alpha == 1.0 {
                out[dst..dst + nb].copy_from_slice(&b[src..src + nb]);
            } else {
                for t in 0..nb {
                    out[dst + t] = alpha * b[src + t];
                }
            }
        }
    } else {
        // op(B)[p][t] = B[jj+t][kk+p] = b[(jj+t)*ldb + (kk+p)]  (strided gather)
        let col = kk;
        for p in 0..kb {
            let dst = p * nb;
            for t in 0..nb {
                out[dst + t] = alpha * b[(jj + t) * ldb + col + p];
            }
        }
    }
}

// ============================================================================
// Micro-kernel: C[mb,·] += A_packed[mb,kb] @ B_packed[kb,nb]
//   A_packed is row-major stride `kb`; B_packed row-major stride `nb`;
//   C is the owning row-block `cblk` with row stride `ldc`, written at column offset `jj`.
// ============================================================================

/// Whether to use the AVX-512 kernel when available. AVX-512 can trigger heavy frequency
/// licensing downclocks on many client CPUs (making it *slower* than AVX2 for real workloads —
/// measured ~2.3× slower on a 512³ GEMM here), so AVX2 is the default. Set `RUSTNN_USE_AVX512=1`
/// to opt in on hardware without that penalty (some server SKX/Ice Lake parts).
fn want_avx512() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("RUSTNN_USE_AVX512")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
fn kernel_dispatch(
    mb: usize,
    kb: usize,
    nb: usize,
    a_packed: &[f32],
    b_packed: &[f32],
    cblk: &mut [f32],
    ldc: usize,
    jj: usize,
) {
    // The packed A buffer is zero-padded to a multiple of MR=4 rows (see `buf_a_size` in `sgemm`),
    // so the register-tiled kernels below may always read a full 4-row tile; out-of-range rows are
    // zero and their (discarded) accumulators are never stored. AVX2 (4×8) is the default; the
    // AVX-512 (4×16) kernel is opt-in via `RUSTNN_USE_AVX512=1`.
    #[cfg(target_arch = "x86_64")]
    {
        if want_avx512() && nb >= 16 && std::is_x86_feature_detected!("avx512f") {
            unsafe { kernel_avx512(mb, kb, nb, a_packed, b_packed, cblk, ldc, jj) };
            return;
        }
        if nb >= 8 && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        {
            unsafe { kernel_avx2(mb, kb, nb, a_packed, b_packed, cblk, ldc, jj) };
            return;
        }
    }
    kernel_scalar(mb, kb, nb, a_packed, b_packed, cblk, ldc, jj);
}

/// Scalar (auto-vectorisable) micro-kernel.
#[allow(clippy::too_many_arguments)]
fn kernel_scalar(
    mb: usize,
    kb: usize,
    nb: usize,
    a_packed: &[f32],
    b_packed: &[f32],
    cblk: &mut [f32],
    ldc: usize,
    jj: usize,
) {
    for i in 0..mb {
        let arow = i * kb; // offset into a_packed
        let crow = i * ldc + jj; // offset into cblk
        for t in 0..nb {
            let mut sum = 0.0f32;
            for p in 0..kb {
                sum += a_packed[arow + p] * b_packed[p * nb + t];
            }
            cblk[crow + t] += sum;
        }
    }
}

/// Register-tiled AVX2 + FMA micro-kernel (MR=4 rows × NR=16 cols). Each K-step loads **two**
/// 8-wide B vectors and reuses them across **4** A-rows via **8** FMA accumulators — and each
/// A-broadcast is amortised over 16 output columns (vs 8 in a 4×8 tile), so A traffic halves.
/// This is the BLIS/Goto register-tiling technique, sized to use ~11 of the 16 YMM registers.
///
/// Requires the packed A buffer to be zero-padded to a multiple of 4 rows (see `kernel_dispatch`).
///
/// # Safety
/// Caller must guarantee `avx2`+`fma` at runtime, A padded to MR=4 rows, and all accessed indices
/// in range.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_avx2(
    mb: usize,
    kb: usize,
    nb: usize,
    a_packed: &[f32],
    b_packed: &[f32],
    cblk: &mut [f32],
    ldc: usize,
    jj: usize,
) {
    use std::arch::x86_64::*;
    let a = a_packed.as_ptr();
    let b = b_packed.as_ptr();
    let c = cblk.as_mut_ptr();
    let z = _mm256_setzero_ps();
    let mut i = 0;
    while i < mb {
        let rb = if i + 4 <= mb { 4 } else { mb - i };
        let abase = i * kb;
        // ---- 16-column tiles: 8 accumulators (4 rows × 2 B-halves) ----
        let mut t = 0;
        while t + 16 <= nb {
            let mut acc = [z, z, z, z, z, z, z, z];
            for p in 0..kb {
                let bv0 = _mm256_loadu_ps(b.add(p * nb + t));
                let bv1 = _mm256_loadu_ps(b.add(p * nb + t + 8));
                let a0 = _mm256_set1_ps(*a.add(abase + p));
                let a1 = _mm256_set1_ps(*a.add(abase + kb + p));
                let a2 = _mm256_set1_ps(*a.add(abase + 2 * kb + p));
                let a3 = _mm256_set1_ps(*a.add(abase + 3 * kb + p));
                acc[0] = _mm256_fmadd_ps(a0, bv0, acc[0]);
                acc[1] = _mm256_fmadd_ps(a0, bv1, acc[1]);
                acc[2] = _mm256_fmadd_ps(a1, bv0, acc[2]);
                acc[3] = _mm256_fmadd_ps(a1, bv1, acc[3]);
                acc[4] = _mm256_fmadd_ps(a2, bv0, acc[4]);
                acc[5] = _mm256_fmadd_ps(a2, bv1, acc[5]);
                acc[6] = _mm256_fmadd_ps(a3, bv0, acc[6]);
                acc[7] = _mm256_fmadd_ps(a3, bv1, acc[7]);
            }
            for r in 0..rb {
                let cp0 = c.add((i + r) * ldc + jj + t);
                let e0 = _mm256_loadu_ps(cp0);
                _mm256_storeu_ps(cp0, _mm256_add_ps(acc[r * 2], e0));
                let cp1 = c.add((i + r) * ldc + jj + t + 8);
                let e1 = _mm256_loadu_ps(cp1);
                _mm256_storeu_ps(cp1, _mm256_add_ps(acc[r * 2 + 1], e1));
            }
            t += 16;
        }
        // ---- 8-column tail (4×8 step) ----
        while t + 8 <= nb {
            let mut acc = [z, z, z, z];
            for p in 0..kb {
                let bv = _mm256_loadu_ps(b.add(p * nb + t));
                acc[0] = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(abase + p)), bv, acc[0]);
                acc[1] = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(abase + kb + p)), bv, acc[1]);
                acc[2] = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(abase + 2 * kb + p)), bv, acc[2]);
                acc[3] = _mm256_fmadd_ps(_mm256_set1_ps(*a.add(abase + 3 * kb + p)), bv, acc[3]);
            }
            for r in 0..rb {
                let cp = c.add((i + r) * ldc + jj + t);
                let e = _mm256_loadu_ps(cp);
                _mm256_storeu_ps(cp, _mm256_add_ps(acc[r], e));
            }
            t += 8;
        }
        // ---- scalar tail ----
        while t < nb {
            for r in 0..rb {
                let mut s = 0.0f32;
                for p in 0..kb {
                    s += *a.add(abase + r * kb + p) * *b.add(p * nb + t);
                }
                *c.add((i + r) * ldc + jj + t) += s;
            }
            t += 1;
        }
        i += 4;
    }
}

/// Register-tiled AVX-512 micro-kernel (MR=4 rows × NR=16 cols). Same 4-row reuse as the AVX2
/// kernel but with 16-wide vectors — double the FMA throughput per K-step. Preferred when
/// `avx512f` is present.
///
/// # Safety
/// Caller must guarantee `avx512f` at runtime, A padded to MR=4 rows, and all accessed indices
/// in range (see [`kernel_avx2`]).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn kernel_avx512(
    mb: usize,
    kb: usize,
    nb: usize,
    a_packed: &[f32],
    b_packed: &[f32],
    cblk: &mut [f32],
    ldc: usize,
    jj: usize,
) {
    use std::arch::x86_64::*;
    let a = a_packed.as_ptr();
    let b = b_packed.as_ptr();
    let c = cblk.as_mut_ptr();
    let z = _mm512_setzero_ps();
    let mut i = 0;
    while i < mb {
        let rb = if i + 4 <= mb { 4 } else { mb - i };
        let abase = i * kb;
        let mut t = 0;
        while t + 16 <= nb {
            let mut acc = [z, z, z, z];
            for p in 0..kb {
                let bv = _mm512_loadu_ps(b.add(p * nb + t));
                acc[0] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(abase + p)), bv, acc[0]);
                acc[1] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(abase + kb + p)), bv, acc[1]);
                acc[2] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(abase + 2 * kb + p)), bv, acc[2]);
                acc[3] = _mm512_fmadd_ps(_mm512_set1_ps(*a.add(abase + 3 * kb + p)), bv, acc[3]);
            }
            for r in 0..rb {
                let cp = c.add((i + r) * ldc + jj + t);
                let e = _mm512_loadu_ps(cp);
                _mm512_storeu_ps(cp, _mm512_add_ps(acc[r], e));
            }
            t += 16;
        }
        // 8-wide tail (AVX2-style step) then scalar tail.
        while t + 8 <= nb {
            for r in 0..rb {
                let mut s = _mm256_setzero_ps();
                let arow = abase + r * kb;
                for p in 0..kb {
                    s = _mm256_fmadd_ps(
                        _mm256_set1_ps(*a.add(arow + p)),
                        _mm256_loadu_ps(b.add(p * nb + t)),
                        s,
                    );
                }
                let cp = c.add((i + r) * ldc + jj + t);
                let e = _mm256_loadu_ps(cp);
                _mm256_storeu_ps(cp, _mm256_add_ps(s, e));
            }
            t += 8;
        }
        while t < nb {
            for r in 0..rb {
                let mut s = 0.0f32;
                for p in 0..kb {
                    s += *a.add(abase + r * kb + p) * *b.add(p * nb + t);
                }
                *c.add((i + r) * ldc + jj + t) += s;
            }
            t += 1;
        }
        i += 4;
    }
}

// ============================================================================
// Level 2: SGEMV — y = α·op(A)·x + β·y   (A is [m,k] logical, row-major)
// ============================================================================

/// Single-precision matrix–vector multiply: `y = α·op(A)·x + β·y`.
///
/// `op(A)` is `[m, k]`; `x` has length `k`; `y` has length `m`. `A` is row-major with row
/// stride `lda` (and is the transpose of that storage when `trans == Trans`, i.e. logical `[k,m]`
/// with `x` length `m` and `y` length `k`).
pub fn sgemv(
    trans: Transpose,
    m: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    // Kept serial: Level-2 (matrix·vector) is O(mk), not the training bottleneck, and a serial
    // loop sidesteps the `Fn`-vs-`FnMut` capture issue that mutating `y` inside a rayon closure
    // would create. The hot path is the Level-3 GEMM in the forward/backward, which is parallel.
    if !trans.is_trans() {
        // y[i] = α·Σ_p A[i][p]·x[p] + β·y[i]
        for i in 0..m {
            let row = i * lda;
            let mut sum = 0.0f32;
            let mut p = 0;
            let mut px = 0;
            while p < k {
                sum += a[row + p] * x[px];
                p += 1;
                px += incx;
            }
            let idx = i * incy;
            y[idx] = if beta == 0.0 {
                alpha * sum
            } else {
                alpha * sum + beta * y[idx]
            };
        }
    } else {
        // y[p] = α·Σ_i A[i][p]·x[i] + β·y[p]  (Aᵀ·x), scale y first then accumulate.
        if beta == 0.0 {
            let mut py = 0;
            for _ in 0..k {
                y[py] = 0.0;
                py += incy;
            }
        } else if beta != 1.0 {
            let mut py = 0;
            for _ in 0..k {
                y[py] *= beta;
                py += incy;
            }
        }
        if alpha != 0.0 {
            for i in 0..m {
                let row = i * lda;
                let xi = alpha * x[i * incx];
                let mut py = 0;
                for p in 0..k {
                    y[py] += xi * a[row + p];
                    py += incy;
                }
            }
        }
    }
}

// ============================================================================
// Level 1: vector–vector BLAS
// ============================================================================

/// DOT product `xᵀ·y` (strided by `incx`/`incy`).
pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut ix = 0;
    let mut iy = 0;
    for _ in 0..n {
        sum += x[ix] * y[iy];
        ix += incx;
        iy += incy;
    }
    sum
}

/// AXPY: `y = α·x + y` (unit strides).
pub fn saxpy(n: usize, alpha: f32, x: &[f32], y: &mut [f32]) {
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

/// SCAL: `x = α·x`.
pub fn sscal(n: usize, alpha: f32, x: &mut [f32]) {
    for i in 0..n {
        x[i] *= alpha;
    }
}

/// COPY: `y = x`.
pub fn scopy(n: usize, x: &[f32], y: &mut [f32]) {
    y[..n].copy_from_slice(&x[..n]);
}

/// NRM2: Euclidean norm `√(Σ xᵢ²)`, summed in f64 for stability.
pub fn snrm2(n: usize, x: &[f32]) -> f32 {
    let mut s = 0.0f64;
    for i in 0..n {
        s += (x[i] as f64) * (x[i] as f64);
    }
    s.sqrt() as f32
}

// ============================================================================
// Backend abstraction
// ============================================================================

/// A pluggable BLAS provider. The built-in implementation is [`NativeBackend`]; downstream code
/// can implement this against a system BLAS (OpenBLAS, Accelerate, Intel MKL) and use it directly.
pub trait BlasBackend: Send + Sync {
    /// Human-readable name of the backend (e.g. `"native-avx2+fma"`).
    fn name(&self) -> &str;

    /// `C = α·op(A)·op(B) + β·C`. See [`sgemm`] for the contract.
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    );
}

/// The built-in backend: packed, cache-blocked, AVX2+FMA (or scalar) kernels with runtime
/// dispatch and size-aware blocking. Zero external dependencies.
#[derive(Default, Debug, Clone, Copy)]
pub struct NativeBackend;

impl BlasBackend for NativeBackend {
    fn name(&self) -> &str {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                if want_avx512() && std::is_x86_feature_detected!("avx512f") {
                    "native-avx512f (4x16 tile)"
                } else {
                    "native-avx2+fma (4x16 tile)"
                }
            } else {
                "native-scalar"
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "native-scalar"
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// Convenience: `C = A·B` for plain contiguous `[m,k]` @ `[k,n]` → `[m,n]` (no transpose, α=1, β=0).
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    sgemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        m,
        n,
        k,
        1.0,
        a,
        k,
        b,
        n,
        0.0,
        c,
        n,
    );
}

// ============================================================================
// Debug assertions
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn debug_assert_blas(
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    a_len: usize,
    lda: usize,
    b_len: usize,
    ldb: usize,
    c_len: usize,
    ldc: usize,
) {
    // op(A) is [m,k]. NoTrans: A stored [m,k], needs (m-1)*lda + k. Trans: A stored [k,m], needs (k-1)*lda + m.
    let a_need = if m == 0 || k == 0 {
        0
    } else if transa.is_trans() {
        (k - 1) * lda + m
    } else {
        (m - 1) * lda + k
    };
    let b_need = if k == 0 || n == 0 {
        0
    } else if transb.is_trans() {
        (n - 1) * ldb + k
    } else {
        (k - 1) * ldb + n
    };
    let c_need = if m == 0 || n == 0 {
        0
    } else {
        (m - 1) * ldc + n
    };
    debug_assert!(
        a_len >= a_need,
        "sgemm: A too short ({}) for need {}",
        a_len,
        a_need
    );
    debug_assert!(
        b_len >= b_need,
        "sgemm: B too short ({}) for need {}",
        b_len,
        b_need
    );
    debug_assert!(
        c_len >= c_need,
        "sgemm: C too short ({}) for need {}",
        c_len,
        c_need
    );
}

// ============================================================================
// Optional Strassen path (opt-in). Larger error than classical GEMM in f32 — keep it
// off the default gradient path; use it for huge, near-square inference matmuls.
// ============================================================================

/// Below this size, Strassen recursion falls back to the classical packed [`sgemm`].
pub const GEMM_STRASSEN_THRESHOLD: usize = 256;

/// Strassen-style recursive GEMM for huge, near-square problems.
///
/// Recurses down to [`GEMM_STRASSEN_THRESHOLD`] using 7 (recursive) sub-multiplies instead of 8,
/// giving `O(n^2.807)` vs `O(n³)`. **Caveat:** Strassen in `f32` has larger accumulated rounding
/// error than classical GEMM, so this is deliberately **opt-in** and never used by the autograd
/// backward pass or by [`sgemm`]. It is appropriate for large inference-time matmuls where a few
/// ULPs of extra error are acceptable.
///
/// Computes `C = A @ B` (overwrite) for contiguous row-major `A` `[m,k]`, `B` `[k,n]`, `C` `[m,n]`.
/// Odd dimensions are peeled off and handled by the classical kernel, and operand quadrants are
/// copied into contiguous temporaries; the asymptotic win still dominates for sufficiently large
/// matrices.
pub fn gemm_strassen(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    strassen_rec(a, k, b, n, c, n, m, k, n, 0);
}

/// Recursive worker: `c = a @ b` (overwrite). `a` stride `lda`, `b` stride `ldb`, `c` stride `ldc`.
#[allow(clippy::too_many_arguments)]
fn strassen_rec(
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    m: usize,
    k: usize,
    n: usize,
    depth: u32,
) {
    // Base case: small enough, or recursion getting deep → classical packed GEMM (uses full k).
    if m.min(k).min(n) <= GEMM_STRASSEN_THRESHOLD || depth > 6 {
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
        return;
    }

    let mf = m & !1; // largest even ≤ m
    let nf = n & !1; // largest even ≤ n
    let kf = k & !1; // largest even ≤ k

    // Need all three even cores to be at least 2 to do a 4-way split; otherwise just go classical.
    if mf < 2 || nf < 2 || kf < 2 {
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            a,
            lda,
            b,
            ldb,
            0.0,
            c,
            ldc,
        );
        return;
    }

    // ---- Even core [mf × kf] @ [kf × nf] → C[0..mf, 0..nf] (Strassen 7-prod, overwrite) ----
    strassen_core(a, lda, b, ldb, c, ldc, mf, kf, nf, depth);

    // ---- Peeled M row (last row), even-K part only ----
    if m & 1 == 1 {
        let i = mf; // == m-1
                    // C[mf, 0..nf] = A[mf, 0..kf] @ B[0..kf, 0..nf]
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1,
            nf,
            kf,
            1.0,
            &a[i * lda..],
            lda,
            b,
            ldb,
            0.0,
            &mut c[i * ldc..],
            ldc,
        );
        // last row's tail column C[mf, nf] handled by the peeled-N step below (uses full m).
    }

    // ---- Peeled N column (last column), even-K part only, all rows [0..m) ----
    if n & 1 == 1 {
        let j = nf; // == n-1
        let mut bcol = vec![0.0f32; kf];
        for p in 0..kf {
            bcol[p] = b[p * ldb + j];
        }
        let mut ccol = vec![0.0f32; m];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            1,
            kf,
            1.0,
            a,
            lda,
            &bcol,
            1,
            0.0,
            &mut ccol,
            1,
        );
        for i in 0..m {
            c[i * ldc + j] = ccol[i];
        }
    }

    // ---- Peeled K tail (odd K): rank-1 add over the WHOLE [m × n] output ----
    if k & 1 == 1 {
        let p = kf; // == k-1
        for i in 0..m {
            let aiv = a[i * lda + p];
            if aiv != 0.0 {
                let crow = i * ldc;
                for j in 0..n {
                    c[crow + j] += aiv * b[p * ldb + j];
                }
            }
        }
    }
}

/// Strassen 7-product core for an all-even `[m,k] @ [k,n] → [m,n]` block, writing into `c`
/// (strided, overwrite). `m`, `k`, `n` must all be even and ≥ 2.
#[allow(clippy::too_many_arguments)]
fn strassen_core(
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
    m: usize,
    k: usize,
    n: usize,
    depth: u32,
) {
    debug_assert!(m & 1 == 0 && n & 1 == 0 && k & 1 == 0 && m >= 2 && n >= 2 && k >= 2);
    let m2 = m / 2;
    let k2 = k / 2;
    let n2 = n / 2;

    // ---- Gather contiguous quadrant copies ----
    let mut a11 = vec![0.0f32; m2 * k2];
    let mut a12 = vec![0.0f32; m2 * k2];
    let mut a21 = vec![0.0f32; m2 * k2];
    let mut a22 = vec![0.0f32; m2 * k2];
    copy_block(a, lda, 0, 0, m2, k2, &mut a11);
    copy_block(a, lda, 0, k2, m2, k2, &mut a12);
    copy_block(a, lda, m2, 0, m2, k2, &mut a21);
    copy_block(a, lda, m2, k2, m2, k2, &mut a22);

    let mut b11 = vec![0.0f32; k2 * n2];
    let mut b12 = vec![0.0f32; k2 * n2];
    let mut b21 = vec![0.0f32; k2 * n2];
    let mut b22 = vec![0.0f32; k2 * n2];
    copy_block(b, ldb, 0, 0, k2, n2, &mut b11);
    copy_block(b, ldb, 0, n2, k2, n2, &mut b12);
    copy_block(b, ldb, k2, 0, k2, n2, &mut b21);
    copy_block(b, ldb, k2, n2, k2, n2, &mut b22);

    // ---- Scratch + 7 product buffers (each [m2, n2]) ----
    let mut sa = vec![0.0f32; m2 * k2];
    let mut sb = vec![0.0f32; k2 * n2];
    let mut p = vec![vec![0.0f32; m2 * n2]; 7];

    // Strassen's 7 multiplies (standard formulation). P[i] is p[i], [m2,n2] contiguous.
    // P1 = (A11+A22)(B11+B22)
    add_into(&a11, &a22, &mut sa);
    add_into(&b11, &b22, &mut sb);
    strassen_rec(&sa, k2, &sb, n2, &mut p[0], n2, m2, k2, n2, depth + 1);
    // P2 = (A21+A22) B11
    add_into(&a21, &a22, &mut sa);
    strassen_rec(&sa, k2, &b11, n2, &mut p[1], n2, m2, k2, n2, depth + 1);
    // P3 = A11 (B12-B22)
    sub_into(&b12, &b22, &mut sb);
    strassen_rec(&a11, k2, &sb, n2, &mut p[2], n2, m2, k2, n2, depth + 1);
    // P4 = A22 (B21-B11)
    sub_into(&b21, &b11, &mut sb);
    strassen_rec(&a22, k2, &sb, n2, &mut p[3], n2, m2, k2, n2, depth + 1);
    // P5 = (A11+A12) B22
    add_into(&a11, &a12, &mut sa);
    strassen_rec(&sa, k2, &b22, n2, &mut p[4], n2, m2, k2, n2, depth + 1);
    // P6 = (A21-A11)(B11+B12)
    sub_into(&a21, &a11, &mut sa);
    add_into(&b11, &b12, &mut sb);
    strassen_rec(&sa, k2, &sb, n2, &mut p[5], n2, m2, k2, n2, depth + 1);
    // P7 = (A12-A22)(B21+B22)
    sub_into(&a12, &a22, &mut sa);
    add_into(&b21, &b22, &mut sb);
    strassen_rec(&sa, k2, &sb, n2, &mut p[6], n2, m2, k2, n2, depth + 1);

    // ---- Combine into C quadrants (strided, overwrite) ----
    // C11 = P1 + P4 - P5 + P7
    // C12 = P3 + P5
    // C21 = P2 + P4
    // C22 = P1 - P2 + P3 + P6
    for i in 0..m2 {
        for j in 0..n2 {
            let idx = i * n2 + j;
            c[i * ldc + j] = p[0][idx] + p[3][idx] - p[4][idx] + p[6][idx]; // C11
            c[i * ldc + j + n2] = p[2][idx] + p[4][idx]; // C12
            c[(i + m2) * ldc + j] = p[1][idx] + p[3][idx]; // C21
            c[(i + m2) * ldc + j + n2] = p[0][idx] - p[1][idx] + p[2][idx] + p[5][idx];
            // C22
        }
    }
}

/// Copy a strided `[r, c]` block starting at `(ro, co)` (with row stride `stride`) into contiguous
/// `dst` of row stride `c`.
#[allow(clippy::needless_range_loop)]
fn copy_block(
    src: &[f32],
    stride: usize,
    ro: usize,
    co: usize,
    r: usize,
    c: usize,
    dst: &mut [f32],
) {
    for i in 0..r {
        let s = (ro + i) * stride + co;
        let d = i * c;
        dst[d..d + c].copy_from_slice(&src[s..s + c]);
    }
}

/// Contiguous elementwise `dst = x + y`.
fn add_into(x: &[f32], y: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), dst.len());
    for i in 0..x.len() {
        dst[i] = x[i] + y[i];
    }
}

/// Contiguous elementwise `dst = x - y`.
fn sub_into(x: &[f32], y: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), dst.len());
    for i in 0..x.len() {
        dst[i] = x[i] - y[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- reference implementations ----------

    fn naive_sgemm(
        ta: Transpose,
        tb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    let av = if ta.is_trans() {
                        a[p * lda + i]
                    } else {
                        a[i * lda + p]
                    };
                    let bv = if tb.is_trans() {
                        b[j * ldb + p]
                    } else {
                        b[p * ldb + j]
                    };
                    s += av * bv;
                }
                c[i * ldc + j] = alpha * s + beta * c[i * ldc + j];
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn rand_mat(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                // Top 24 bits / 2^24 → [0,1); shift to [-0.5, 0.5).
                ((*seed >> 8) as f32 / 16777216.0) - 0.5
            })
            .collect()
    }

    // ---------- sgemm correctness across transpose flags ----------

    #[test]
    fn sgemm_nn_matches_naive() {
        let (m, n, k) = (17, 19, 13);
        let mut s = 1u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, k * n);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1e-3,
            "NN diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_nt_matches_naive() {
        // B is stored [n,k] (transb=Trans). C[m,n] = A[m,k] @ Bᵀ.
        let (m, n, k) = (33, 29, 17);
        let mut s = 2u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, n * k); // [n,k] storage
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            k,
            0.0,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            k,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1e-3,
            "NT diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_tn_matches_naive() {
        // A stored [k,m] (transa=Trans). C[m,n] = Aᵀ @ B.
        let (m, n, k) = (29, 33, 17);
        let mut s = 3u32;
        let a = rand_mat(&mut s, k * m); // [k,m] storage
        let b = rand_mat(&mut s, k * n);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            n,
            0.0,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::Trans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1e-3,
            "TN diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_tt_matches_naive() {
        let (m, n, k) = (40, 35, 24);
        let mut s = 4u32;
        let a = rand_mat(&mut s, k * m);
        let b = rand_mat(&mut s, n * k);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm(
            Transpose::Trans,
            Transpose::Trans,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::Trans,
            Transpose::Trans,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1e-3,
            "TT diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_alpha_beta_accumulation() {
        let (m, n, k) = (20, 20, 20);
        let mut s = 5u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, k * n);
        let c0 = rand_mat(&mut s, m * n);
        let alpha = 2.5f32;
        let beta = -1.5f32;
        let mut c1 = c0.clone();
        let mut c2 = c0.clone();
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            alpha,
            &a,
            k,
            &b,
            n,
            beta,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            alpha,
            &a,
            k,
            &b,
            n,
            beta,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1e-3,
            "alpha/beta diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_large_blocked_path() {
        // Large enough to exercise the full MC/KC/NC blocked + threaded path.
        let (m, n, k) = (200, 180, 160);
        let mut s = 6u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, k * n);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c1,
            n,
        );
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 0.1,
            "large blocked diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn sgemm_zero_and_small_edges() {
        // k=0 → C should be zeroed by beta=0.
        let mut c = vec![7.0f32; 4];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            0,
            1.0,
            &[],
            0,
            &[],
            0,
            0.0,
            &mut c,
            2,
        );
        assert_eq!(c, vec![0.0, 0.0, 0.0, 0.0]);

        // 1x1
        let mut c = vec![0.0];
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1,
            1,
            1,
            1.0,
            &[3.0],
            1,
            &[4.0],
            1,
            0.0,
            &mut c,
            1,
        );
        assert!((c[0] - 12.0).abs() < 1e-6);
    }

    // ---------- convenience + levels 1/2 ----------

    #[test]
    fn matmul_convenience_works() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [2,2]
        let mut c = vec![0.0; 4];
        matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn sgemv_notrans_and_trans() {
        // A [3,2], x[2], y = A x
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 3];
        sgemv(Transpose::NoTrans, 3, 2, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1);
        assert_eq!(y, vec![3.0, 7.0, 11.0]);
        // Aᵀ x : Aᵀ is [2,3], x len 3, y len 2
        let mut y2 = vec![0.0; 2];
        sgemv(
            Transpose::Trans,
            3,
            2,
            1.0,
            &a,
            2,
            &[1.0, 1.0, 1.0],
            1,
            0.0,
            &mut y2,
            1,
        );
        assert_eq!(y2, vec![9.0, 12.0]); // col sums
    }

    #[test]
    fn level1_ops() {
        assert_eq!(sdot(3, &[1.0, 2.0, 3.0], 1, &[4.0, 5.0, 6.0], 1), 32.0);
        let mut y = vec![1.0, 2.0, 3.0];
        saxpy(3, 2.0, &[10.0, 20.0, 30.0], &mut y);
        assert_eq!(y, vec![21.0, 42.0, 63.0]);
        let mut x = vec![1.0, 2.0, 3.0];
        sscal(3, 2.0, &mut x);
        assert_eq!(x, vec![2.0, 4.0, 6.0]);
        let mut z = vec![0.0; 3];
        scopy(3, &[7.0, 8.0, 9.0], &mut z);
        assert_eq!(z, vec![7.0, 8.0, 9.0]);
        assert!((snrm2(2, &[3.0, 4.0]) - 5.0).abs() < 1e-5);
    }

    // ---------- backend abstraction ----------

    #[test]
    fn native_backend_trait_matches_free_fn() {
        let (m, n, k) = (16, 16, 16);
        let mut s = 7u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, k * n);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        NativeBackend.sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c1,
            n,
        );
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(max_abs_diff(&c1, &c2) < 1e-4);
        assert!(!NativeBackend.name().is_empty());
    }

    // ---------- Strassen ----------

    #[test]
    fn strassen_matches_naive_even() {
        let n = 256; // power of two, triggers exactly one recursion level
        let mut s = 8u32;
        let a = rand_mat(&mut s, n * n);
        let b = rand_mat(&mut s, n * n);
        let mut c1 = vec![0.0f32; n * n];
        let mut c2 = vec![0.0f32; n * n];
        gemm_strassen(&a, &b, &mut c1, n, n, n);
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0,
            &a,
            n,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        // Strassen in f32: allow a slightly looser tolerance.
        assert!(
            max_abs_diff(&c1, &c2) < 1.0,
            "strassen even diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn strassen_matches_naive_odd_rect() {
        // Odd + rectangular: exercises all three peeling paths.
        let (m, k, n) = (133, 117, 129);
        let mut s = 9u32;
        let a = rand_mat(&mut s, m * k);
        let b = rand_mat(&mut s, k * n);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        gemm_strassen(&a, &b, &mut c1, m, k, n);
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(
            max_abs_diff(&c1, &c2) < 1.0,
            "strassen odd diff {}",
            max_abs_diff(&c1, &c2)
        );
    }

    #[test]
    fn strassen_small_falls_back_to_classical() {
        let n = 32; // below threshold
        let mut s = 10u32;
        let a = rand_mat(&mut s, n * n);
        let b = rand_mat(&mut s, n * n);
        let mut c1 = vec![0.0f32; n * n];
        let mut c2 = vec![0.0f32; n * n];
        gemm_strassen(&a, &b, &mut c1, n, n, n);
        naive_sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0,
            &a,
            n,
            &b,
            n,
            0.0,
            &mut c2,
            n,
        );
        assert!(max_abs_diff(&c1, &c2) < 1e-3);
    }
}
