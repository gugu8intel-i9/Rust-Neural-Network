//! INT8 quantized inference: symmetric per-channel quantization for 4× memory reduction.
//!
//! Quantizes f32 weights to INT8 with a per-output-channel scale factor, then performs
//! INT8 × INT8 → INT32 accumulate → f32 dequantize matmul. This is the same approach used
//! by llama.cpp, TensorRT, and ONNX Runtime INT8 inference.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// An INT8 quantized weight matrix with per-channel scales.
#[derive(Debug, Clone)]
pub struct Int8Weights {
    /// Quantized weight values (INT8 as i8).
    pub data: Vec<i8>,
    /// Per-output-channel scale: `weight_f32[i, :] = data[i, :] as f32 * scale[i]`.
    pub scales: Vec<f32>,
    pub shape: Vec<usize>, // [out_features, in_features]
}

impl Int8Weights {
    /// Quantize an f32 weight matrix to INT8 with per-channel (per-row) symmetric quantization.
    ///
    /// For each output row `i`: `scale[i] = max(|W[i, :]|) / 127`.
    pub fn quantize(weight: &Tensor) -> Self {
        let data = weight.data();
        let shape = data.shape().to_vec();
        let (out_features, in_features) = (shape[0], shape[1]);
        let flat: Vec<f32> = data.iter().copied().collect();

        let mut q_data = vec![0i8; out_features * in_features];
        let mut scales = vec![0.0f32; out_features];

        for i in 0..out_features {
            // Find max abs in this row.
            let row = &flat[i * in_features..(i + 1) * in_features];
            let max_abs = row.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[i] = scale;

            // Quantize each element.
            for j in 0..in_features {
                let q = (row[j] / scale).round().clamp(-128.0, 127.0) as i8;
                q_data[i * in_features + j] = q;
            }
        }

        Int8Weights {
            data: q_data,
            scales,
            shape,
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Tensor {
        let (out_f, in_f) = (self.shape[0], self.shape[1]);
        let mut result = vec![0.0f32; out_f * in_f];
        for i in 0..out_f {
            let scale = self.scales[i];
            for j in 0..in_f {
                result[i * in_f + j] = self.data[i * in_f + j] as f32 * scale;
            }
        }
        Tensor::from_vec(result, self.shape.clone())
    }

    /// INT8 matmul: `C[m,n] = A_f32[m,k] @ (INT8_weight[k,n] * scale[n])`.
    /// A is the f32 activation, B is the quantized weight (transposed: [in, out]).
    /// Returns f32 result.
    pub fn matmul_f32(&self, activation: &[f32], m: usize) -> Vec<f32> {
        let (out_f, in_f) = (self.shape[0], self.shape[1]);
        assert_eq!(activation.len(), m * in_f, "activation shape mismatch");
        let mut result = vec![0.0f32; m * out_f];

        for row in 0..m {
            for col in 0..out_f {
                let mut acc = 0i32;
                for k in 0..in_f {
                    let a_val = (activation[row * in_f + k] * (1.0 / self.scales[col]))
                        .round()
                        .clamp(-128.0, 127.0) as i8;
                    acc += a_val as i32 * self.data[col * in_f + k] as i32;
                }
                result[row * out_f + col] = acc as f32 * self.scales[col] * self.scales[col];
            }
        }
        result
    }

    /// Memory usage in bytes (INT8 weights + f32 scales).
    pub fn mem_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Compression ratio vs f32 (should be ~4×).
    pub fn compression_ratio(&self) -> f64 {
        let f32_bytes = self.data.len() * 4;
        f32_bytes as f64 / self.mem_bytes() as f64
    }

    /// Quantization error (mean abs difference between original and dequantized).
    pub fn quantization_error(&self, original: &Tensor) -> f32 {
        let dequant = self.dequantize();
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let deq: Vec<f32> = dequant.data().iter().copied().collect();
        let total_diff: f32 = orig
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        total_diff / orig.len() as f32
    }
}

/// A quantized Linear layer for INT8 inference.
#[derive(Debug)]
pub struct Int8Linear {
    pub int8_weight: Int8Weights, // [out, in]
    pub bias: Option<Vec<f32>>,   // [out]
}

impl Int8Linear {
    /// Quantize a Linear layer's weights to INT8 for inference.
    pub fn from_linear(layer: &crate::nn::Linear) -> Self {
        let int8_weight = Int8Weights::quantize(&layer.weight);
        let bias = layer
            .bias
            .as_ref()
            .map(|b| b.data().iter().copied().collect());
        Int8Linear { int8_weight, bias }
    }

    /// Run INT8 inference: `y = x @ W^T + b` using quantized weights.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data();
        let x_shape = x_data.shape();
        let batch = if x_shape.len() >= 2 {
            x_shape[x_shape.len() - 2]
        } else {
            1
        };
        let in_features = *x_shape.last().unwrap();
        let out_features = self.int8_weight.shape[0];

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let mut result = vec![0.0f32; batch * out_features];

        for b in 0..batch {
            for o in 0..out_features {
                let mut acc = 0i32;
                let scale = self.int8_weight.scales[o];
                for k in 0..in_features {
                    // Quantize activation on-the-fly (dynamic quantization).
                    let a_q = (x_flat[b * in_features + k] / scale)
                        .round()
                        .clamp(-128.0, 127.0) as i8;
                    acc += a_q as i32 * self.int8_weight.data[o * in_features + k] as i32;
                }
                result[b * out_features + o] = acc as f32 * scale * scale;
                if let Some(ref bias) = self.bias {
                    result[b * out_features + o] += bias[o];
                }
            }
        }

        let mut out_shape = x_shape.to_vec();
        *out_shape.last_mut().unwrap() = out_features;
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&out_shape), result).unwrap(),
            false,
        )
    }
}

// ============================================================================
// High-performance INT8 GEMM via AVX-512 VNNI (VPDPBUSD)
// ============================================================================
//
// VPDPBUSD (the VNNI instruction) does a per-32-bit-lane dot product of 4
// unsigned×signed bytes:  one ZMM instruction = 16 int32 lanes × 4 byte-MACs =
// 64 INT8 multiply-accumulates per cycle — ~4× the per-cycle throughput of an
// FP32 FMA, and 4× less memory traffic (bytes vs f32). This is exactly the
// hardware feature CPUs add *for ML*, and it is the single biggest lever here.
//
// We compute C[m,n] = Σ_k a_u8[m,k] · b_i8[k,n]  (int32 accumulate; caller
// applies zero-point / scale in the dequant step). B is repacked once into the
// VNNI tile layout so the inner kernel is pure VNNI + a broadcast of 4 A bytes.

use rayon::prelude::*;

/// Repack `b` (`[k,n]` int8, row-major) into the VNNI tile layout: for each
/// `(k4, n16)` tile, 64 bytes ordered as `lane l (0..16) × byte j (0..3)` where
/// byte = `b[k4+j, n16+l]`. `k`/`n` are zero-padded up to `kp`/`np` (multiples
/// of 4/16); out-of-range bytes are 0.
fn pack_b_vnni(b: &[i8], k: usize, n: usize, kp: usize, np: usize) -> Vec<u8> {
    let n_blocks = np / 16;
    let mut out = vec![0u8; (kp / 4) * n_blocks * 64];
    for kb in 0..(kp / 4) {
        let k4 = kb * 4;
        for nb in 0..n_blocks {
            let n16 = nb * 16;
            let base = (kb * n_blocks + nb) * 64;
            for l in 0..16usize {
                let col = n16 + l;
                for j in 0..4usize {
                    let row = k4 + j;
                    let v = if row < k && col < n {
                        b[row * n + col] as u8 // i8 → u8 bit pattern
                    } else {
                        0
                    };
                    out[base + l * 4 + j] = v;
                }
            }
        }
    }
    out
}

/// Scalar INT8 GEMM reference: `c[m,n] = Σ_k a_u8[m,k] · b_i8[k,n]` (i32).
/// `a` bytes are treated as unsigned (0..255), `b` bytes sign-extended.
pub fn igemm_scalar(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0i32;
            for p in 0..k {
                s += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = s;
        }
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn igemm_vnni_block(
    a_pad: &[u8],
    b_pack: &[u8],
    crows: &mut [i32], // rb × np row-major
    i0: usize,
    rb: usize,
    kp: usize,
    np: usize,
) {
    use std::arch::x86_64::*;
    let n_blocks = np / 16;
    let z = _mm512_setzero_si512();
    let a_ptr = a_pad.as_ptr();
    let b_ptr = b_pack.as_ptr();
    // MR=4 register tile: each B-load is reused across 4 A-rows via 4 accumulators.
    for nb in 0..n_blocks {
        let n16 = nb * 16;
        let mut acc = [z, z, z, z];
        for kb in 0..(kp / 4) {
            let k4 = kb * 4;
            let tile_off = (kb * n_blocks + nb) * 64;
            let bvec = _mm512_loadu_epi32(b_ptr.add(tile_off) as *const i32);
            for r in 0..4 {
                // 4 A bytes a[i0+r, k4..k4+4] as a little-endian u32, broadcast to all lanes.
                let au32 = (a_ptr.add((i0 + r) * kp + k4) as *const u32).read_unaligned() as i32;
                let avec = _mm512_set1_epi32(au32);
                acc[r] = _mm512_dpbusd_epi32(acc[r], avec, bvec);
            }
        }
        // Store only the rb real rows.
        for r in 0..rb {
            let mut tmp = [0i32; 16];
            _mm512_storeu_epi32(tmp.as_mut_ptr(), acc[r]);
            let cw = r * np + n16;
            crows[cw..cw + 16].copy_from_slice(&tmp);
        }
    }
}

/// INT8 GEMM: `c[m,n] = Σ_k a_u8[m,k] · b_i8[k,n]` (i32 accumulate).
///
/// `a` is `[m,k]` **unsigned** bytes (0..255), `b` is `[k,n]` **signed** bytes — the
/// VNNI (unsigned × signed) contract. On AVX-512 VNNI hardware this runs the VPDPBUSD
/// micro-kernel (~4× FP32 FMA throughput, 4× less memory); otherwise it falls back to
/// the scalar reference. Caller applies activation zero-point / per-channel scales to
/// dequantize the int32 result.
pub fn igemm(a: &[u8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512vnni") {
        let kp = (k + 3) & !3;
        let np = (n + 15) & !15;
        let mp = (m + 3) & !3;
        // Pad A to mp × kp (zero fill); repack B into the VNNI layout.
        let mut a_pad = vec![0u8; mp * kp];
        for i in 0..m {
            a_pad[i * kp..i * kp + k].copy_from_slice(&a[i * k..i * k + k]);
        }
        let b_pack = pack_b_vnni(b, k, n, kp, np);
        let mut ctmp = vec![0i32; m * np];
        // Parallelise over groups of 4 M-rows (each task owns its output rows).
        ctmp.par_chunks_mut(np * 4)
            .enumerate()
            .for_each(|(blk, rows)| {
                let i0 = blk * 4;
                let rb = rows.len() / np;
                unsafe { igemm_vnni_block(&a_pad, &b_pack, rows, i0, rb, kp, np) };
            });
        // Trim the np-padded columns down to n.
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            c[i * n..i * n + n].copy_from_slice(&ctmp[i * np..i * np + n]);
        }
        return c;
    }
    igemm_scalar(a, b, m, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module};

    #[test]
    fn int8_roundtrip_low_error() {
        let weight = Tensor::from_vec(vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8], vec![2, 4]);
        let q = Int8Weights::quantize(&weight);
        let error = q.quantization_error(&weight);
        assert!(error < 0.01, "quantization error too high: {error}");
    }

    #[test]
    fn int8_compression_ratio() {
        let weight = Tensor::randn(&[64, 128]);
        let q = Int8Weights::quantize(&weight);
        let ratio = q.compression_ratio();
        assert!(ratio > 3.0, "compression should be ~4x, got {ratio}");
    }

    #[test]
    fn int8_memory_smaller() {
        let weight = Tensor::randn(&[32, 64]);
        let f32_bytes = 32 * 64 * 4;
        let q = Int8Weights::quantize(&weight);
        assert!(q.mem_bytes() < f32_bytes);
    }

    #[test]
    fn int8_linear_matches_f32_approximately() {
        let layer = Linear::new(8, 4, true);
        let x = Tensor::randn(&[2, 8]);

        // f32 reference.
        let y_f32 = layer.forward(&x);

        // INT8 inference.
        let q_layer = Int8Linear::from_linear(&layer);
        let y_int8 = q_layer.forward(&x);

        // Should be close (within quantization error).
        let f32_vals: Vec<f32> = y_f32.data().iter().copied().collect();
        let q_vals: Vec<f32> = y_int8.data().iter().copied().collect();
        let max_diff: f32 = f32_vals
            .iter()
            .zip(q_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 5.0,
            "INT8 inference should be close to f32, max diff: {max_diff}"
        );
    }

    #[test]
    fn int8_weights_shape() {
        let w = Tensor::randn(&[16, 32]);
        let q = Int8Weights::quantize(&w);
        assert_eq!(q.shape, vec![16, 32]);
        assert_eq!(q.data.len(), 16 * 32);
        assert_eq!(q.scales.len(), 16);
    }

    // ---------- VNNI INT8 GEMM ----------

    fn lcg_bytes(seed: &mut u32, n: usize, signed: bool) -> Vec<i8> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let b = (*seed >> 16) as u8;
                if signed {
                    b as i8
                } else {
                    (b & 0x7f) as i8
                }
            })
            .collect()
    }

    fn check_igemm(m: usize, k: usize, n: usize) {
        let mut s = 1u32;
        let a_i8 = lcg_bytes(&mut s, m * k, false);
        let b_i8 = lcg_bytes(&mut s, k * n, true);
        let a_u8: Vec<u8> = a_i8.iter().map(|&v| v as u8).collect();
        let expected = igemm_scalar(&a_u8, &b_i8, m, k, n);
        let got = igemm(&a_u8, &b_i8, m, k, n);
        assert_eq!(got.len(), expected.len(), "len mismatch ({m},{k},{n})");
        let mut worst = 0i64;
        for (g, e) in got.iter().zip(expected.iter()) {
            let d = (*g - *e).abs() as i64;
            worst = worst.max(d);
        }
        assert!(
            worst == 0,
            "igemm mismatch ({m},{k},{n}): worst abs diff {worst}"
        );
    }

    #[test]
    fn igemm_known_small() {
        let a: Vec<u8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, -1, 2, 2, -3, 4, 0, -2]; // [4,2]
        let c = igemm(&a, &b, 1, 4, 2);
        // c[0] = 1*1+2*2+3*(-3)+4*0 = -4 ; c[1] = 1*(-1)+2*2+3*4+4*(-2) = 7
        assert_eq!(c, vec![-4, 7]);
    }

    #[test]
    fn igemm_matches_scalar_various() {
        check_igemm(16, 32, 48); // multiples of 4/16
        check_igemm(3, 5, 7); // non-multiples → K(×4) and N(×16) padding
        check_igemm(13, 17, 29);
        check_igemm(1, 1, 1);
        check_igemm(4, 4, 16);
        check_igemm(33, 9, 33);
        check_igemm(64, 64, 64); // bigger — blocked/parallel path
    }
}
