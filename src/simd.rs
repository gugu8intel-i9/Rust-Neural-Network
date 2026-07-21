//! SIMD-accelerated CPU kernels: cache-blocked GEMM, vectorized element-wise ops, and reductions.
//!
//! # Design philosophy: "exploit what CPUs are good at"
//!
//! Modern CPUs excel at three things that this module systematically exploits:
//!
//! 1. **SIMD vector units** (AVX2/FMA on x86, NEON on ARM): process 8 (AVX2) or 16 (AVX-512)
//!    single-precision floats per instruction. FMA (`_mm256_fmadd_ps`) does `a*b+c` in one cycle.
//!    This module uses explicit intrinsics via `std::arch` with **runtime feature detection** — it
//!    automatically dispatches to the fastest available kernel and falls back to scalar on
//!    unsupported CPUs.
//!
//! 2. **Cache hierarchy** (L1: ~32 KB, L2: ~256-512 KB): the naive O(n²) working set of a matrix
//!    multiply exceeds cache for any real model. This kernel uses **cache blocking (tiling)**: it
//!    breaks the computation into blocks sized to fit L1/L2, so each cache line is fully consumed
//!    before eviction. The block sizes (`MC`, `KC`, `NC`) are tuned for typical L2 geometry.
//!
//! 3. **Multi-core parallelism**: independent row-blocks are dispatched to rayon worker threads.
//!    Each thread owns its output block, so there are no data races.
//!
//! ## The GEMM kernel
//!
//! The inner kernel broadcasts each element of A to a full 256-bit (8×f32) vector and uses FMA to
//! accumulate 8 columns of C simultaneously. This gives an arithmetic-to-memory ratio of 8:1 —
//! each load from A contributes to 8 FMAs. The outer blocking ensures B's tiles stay in L2 across
//! multiple A rows.
//!
//! References:
//!   - "What Every Programmer Should Know About Memory" (Ulrich Drepper)
//!   - Nadav Rot's "Efficient matrix multiplication" gist (register blocking)
//!   - BLIS packing algorithm for cache-blocked GEMM

use rayon::prelude::*;

/// L1-fitting K-dimension block.
const KC: usize = 256;

/// L2-fitting M-dimension block (rows of A processed per L2 tile).
const MC: usize = 64;

/// L2-fitting N-dimension block (columns of B/C per L2 tile).
const NC: usize = 256;

/// SIMD-accelerated C = A @ B for row-major f32 matrices.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, `c` is `[m, n]` (will be zeroed then filled).
/// Uses cache blocking + FMA + rayon parallelism with runtime dispatch.
pub fn simd_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    // Zero the output.
    c.iter_mut().for_each(|x| *x = 0.0);

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // Small matrices: skip blocking overhead.
    if m * n * k < 4096 {
        inner_kernel(a, b, c, m, k, n);
        return;
    }

    // Cache-blocked, parallelized GEMM.
    // Parallelize across M-blocks: each chunk of C rows is uniquely owned by one thread.
    c.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(block_idx, c_block)| {
            let ii = block_idx * MC;
            let m_block = MC.min(m - ii);

            // Cache-block over K and N dimensions (sequential within the M-block).
            for kk in (0..k).step_by(KC) {
                let k_block = KC.min(k - kk);
                for jj in (0..n).step_by(NC) {
                    let n_block = NC.min(n - jj);

                    // Strided inner kernel: A has stride k, B has stride n, C has stride n.
                    // Within each row, elements are contiguous (good for SIMD loads).
                    inner_kernel_strided(
                        &a[ii * k + kk..], k,
                        &b[kk * n + jj..], n,
                        &mut c_block[jj..], n,
                        m_block, k_block, n_block,
                    );
                }
            }
        });
}

/// Dispatch to the best available inner kernel (AVX2+FMA or scalar).
fn inner_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Contiguous version (lda=k, ldb=n, ldc=n).
    inner_kernel_strided(a, k, b, n, c, n, m, k, n);
}

/// Strided inner kernel dispatcher: processes a [m, k] @ [k, n] → [m, n] sub-block.
/// `lda`/`ldb`/`ldc` are the leading dimensions (row strides) of A, B, C respectively.
#[allow(clippy::too_many_arguments)]
fn inner_kernel_strided(
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
    m: usize, k: usize, n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { inner_kernel_avx2_strided(a, lda, b, ldb, c, ldc, m, k, n) };
            return;
        }
    }
    inner_kernel_scalar_strided(a, lda, b, ldb, c, ldc, m, k, n);
}

/// AVX2 + FMA inner kernel (strided).
///
/// # Safety
///
/// This function uses AVX2 + FMA intrinsics and requires `avx2` and `fma` CPU features.
/// The caller MUST ensure `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
/// before calling. All slice accesses use `get_unchecked` which assumes the indices are in bounds;
/// the strided indexing (`i * lda + p`, `p * ldb + j`) is valid as long as the caller passes
/// correct leading dimensions that do not cause OOB access.
///
/// For each row of A (stride `lda`), broadcasts each element to 8-wide and uses FMA to accumulate
/// 8 columns of C (stride `ldc`). B rows have stride `ldb`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn inner_kernel_avx2_strided(
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
    m: usize, k: usize, n: usize,
) {
    use std::arch::x86_64::*;

    for i in 0..m {
        let c_row_base = i * ldc;
        let mut j = 0;

        // Process 8 columns at a time (256-bit = 8 × f32).
        while j + 8 <= n {
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_set1_ps(*a.get_unchecked(i * lda + p));
                let b_vec = _mm256_loadu_ps(b.get_unchecked(p * ldb + j));
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }
            let c_ptr = c.as_mut_ptr().add(c_row_base + j);
            let existing = _mm256_loadu_ps(c_ptr);
            _mm256_storeu_ps(c_ptr, _mm256_add_ps(acc, existing));
            j += 8;
        }

        // Scalar remainder for n % 8 ≠ 0.
        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a.get_unchecked(i * lda + p) * *b.get_unchecked(p * ldb + j);
            }
            *c.get_unchecked_mut(c_row_base + j) += sum;
            j += 1;
        }
    }
}

/// Scalar fallback inner kernel (strided).
#[allow(clippy::too_many_arguments)]
fn inner_kernel_scalar_strided(
    a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
    m: usize, k: usize, n: usize,
) {
    for i in 0..m {
        let c_row = &mut c[i * ldc..i * ldc + n];
        for p in 0..k {
            let a_val = a[i * lda + p];
            let b_row = &b[p * ldb..p * ldb + n];
            for j in 0..n {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// ==================== SIMD element-wise operations ====================

/// SIMD-accelerated element-wise add: `out = a + b`.
pub fn simd_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe { simd_add_avx2(a, b, out) };
            return;
        }
    }
    // Auto-vectorizable scalar fallback.
    for i in 0..n {
        out[i] = a[i] + b[i];
    }
}

/// SIMD-accelerated element-wise multiply: `out = a * b`.
pub fn simd_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe { simd_mul_avx2(a, b, out) };
            return;
        }
    }
    for i in 0..n {
        out[i] = a[i] * b[i];
    }
}

/// SIMD-accelerated ReLU: `out = max(x, 0)`.
pub fn simd_relu(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe { simd_relu_avx2(x, out) };
            return;
        }
    }
    for i in 0..n {
        out[i] = x[i].max(0.0);
    }
}

/// SIMD-accelerated scalar multiply: `out = x * scale`.
pub fn simd_scale(x: &[f32], scale: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let n = x.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe { simd_scale_avx2(x, scale, out) };
            return;
        }
    }
    for i in 0..n {
        out[i] = x[i] * scale;
    }
}

/// SIMD-accelerated sum reduction.
pub fn simd_sum(x: &[f32]) -> f32 {
    let n = x.len();

    #[cfg(target_arch = "x86_64")]
    {
        if n >= 8 && std::is_x86_feature_detected!("avx2") {
            return unsafe { simd_sum_avx2(x) };
        }
    }
    x.iter().sum()
}

// ==================== AVX2 implementations ====================

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Requires AVX2. All vector loads/stores are on slices the caller has verified
/// are at least 8 elements long, with scalar cleanup for the remainder.
#[target_feature(enable = "avx2")]
unsafe fn simd_add_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) = a.get_unchecked(i) + b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Requires AVX2. All vector loads/stores are on slices the caller has verified
/// are at least 8 elements long, with scalar cleanup for the remainder.
#[target_feature(enable = "avx2")]
unsafe fn simd_mul_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(va, vb));
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) = a.get_unchecked(i) * b.get_unchecked(i);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Requires AVX2. All vector loads/stores are on slices the caller has verified
/// are at least 8 elements long, with scalar cleanup for the remainder.
#[target_feature(enable = "avx2")]
unsafe fn simd_relu_avx2(x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_max_ps(vx, zero));
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) = x.get_unchecked(i).max(0.0);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Requires AVX2. All vector loads/stores are on slices the caller has verified
/// are at least 8 elements long, with scalar cleanup for the remainder.
#[target_feature(enable = "avx2")]
unsafe fn simd_scale_avx2(x: &[f32], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(vx, vscale));
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) = x.get_unchecked(i) * scale;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Requires AVX2. All vector loads/stores are on slices the caller has verified
/// are at least 8 elements long, with scalar cleanup for the remainder.
#[target_feature(enable = "avx2")]
unsafe fn simd_sum_avx2(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.as_ptr().add(i)));
        i += 8;
    }
    // Horizontal sum of the 8 lanes.
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < n {
        sum += *x.get_unchecked(i);
        i += 1;
    }
    sum
}

/// Report which SIMD features are active at runtime.
pub fn simd_features() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            "AVX-512 (falling back to AVX2 kernels)"
        } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            "AVX2 + FMA"
        } else if std::is_x86_feature_detected!("sse4.1") {
            "SSE4.1"
        } else {
            "scalar"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "scalar (non-x86)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[test]
    fn simd_matmul_matches_naive_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3,2]
        let expected = naive_matmul(&a, &b, 2, 3, 2);
        let mut c = vec![0.0; 4];
        simd_matmul(&a, &b, &mut c, 2, 3, 2);
        for i in 0..4 {
            assert!((c[i] - expected[i]).abs() < 1e-4, "matmul mismatch at {i}: {} vs {}", c[i], expected[i]);
        }
    }

    #[test]
    fn simd_matmul_matches_naive_medium() {
        let m = 17;
        let k = 13;
        let n = 19;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos() * 0.3).collect();
        let expected = naive_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0; m * n];
        simd_matmul(&a, &b, &mut c, m, k, n);
        let mut max_diff = 0.0f32;
        for i in 0..m * n {
            max_diff = max_diff.max((c[i] - expected[i]).abs());
        }
        assert!(max_diff < 1e-3, "matmul mismatch (max diff {max_diff:.2e})");
    }

    #[test]
    fn simd_matmul_matches_naive_large_blocked() {
        // Large enough to trigger blocking + threading.
        let m = 100;
        let k = 80;
        let n = 120;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01 - 0.5).collect();
        let expected = naive_matmul(&a, &b, m, k, n);
        let mut c = vec![0.0; m * n];
        simd_matmul(&a, &b, &mut c, m, k, n);
        let mut max_diff = 0.0f32;
        for i in 0..m * n {
            max_diff = max_diff.max((c[i] - expected[i]).abs());
        }
        assert!(max_diff < 0.5, "blocked matmul mismatch (max diff {max_diff:.2e})");
    }

    #[test]
    fn simd_matmul_identity() {
        let n = 8;
        let mut eye = vec![0.0f32; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let x: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut c = vec![0.0; n * n];
        simd_matmul(&x, &eye, &mut c, n, n, n);
        for i in 0..n * n {
            assert!((c[i] - x[i]).abs() < 1e-4, "identity matmul failed at {i}");
        }
    }

    #[test]
    fn simd_matmul_edge_cases() {
        // 1×1
        let mut c = vec![0.0];
        simd_matmul(&[3.0], &[4.0], &mut c, 1, 1, 1);
        assert!((c[0] - 12.0).abs() < 1e-6);

        // Zero dimension
        let mut cz = vec![];
        simd_matmul(&[], &[], &mut cz, 0, 0, 0);
        assert!(cz.is_empty());
    }

    #[test]
    fn simd_add_correct() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut out = vec![0.0; 9];
        simd_add(&a, &b, &mut out);
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0]);
    }

    #[test]
    fn simd_mul_correct() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0; 8];
        let mut out = vec![0.0; 8];
        simd_mul(&a, &b, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn simd_relu_correct() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 3.0, -4.0];
        let mut out = vec![0.0; 9];
        simd_relu(&x, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5, 3.0, 0.0]);
    }

    #[test]
    fn simd_scale_correct() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];
        simd_scale(&x, 0.5, &mut out);
        assert_eq!(out, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn simd_sum_correct() {
        let x: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let s = simd_sum(&x);
        assert!((s - 5050.0).abs() < 1e-2, "sum = {s}, expected 5050");
    }

    #[test]
    fn simd_features_returns_string() {
        let features = simd_features();
        println!("Active SIMD: {features}");
        assert!(!features.is_empty());
    }
}
