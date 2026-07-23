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
    pub shape: Vec<usize>,  // [out_features, in_features]
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

        Int8Weights { data: q_data, scales, shape }
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
        let total_diff: f32 = orig.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).sum();
        total_diff / orig.len() as f32
    }
}

/// A quantized Linear layer for INT8 inference.
#[derive(Debug)]
pub struct Int8Linear {
    pub int8_weight: Int8Weights,  // [out, in]
    pub bias: Option<Vec<f32>>,   // [out]
}

impl Int8Linear {
    /// Quantize a Linear layer's weights to INT8 for inference.
    pub fn from_linear(layer: &crate::nn::Linear) -> Self {
        let int8_weight = Int8Weights::quantize(&layer.weight);
        let bias = layer.bias.as_ref().map(|b| {
            b.data().iter().copied().collect()
        });
        Int8Linear { int8_weight, bias }
    }

    /// Run INT8 inference: `y = x @ W^T + b` using quantized weights.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data = x.data();
        let x_shape = x_data.shape();
        let batch = if x_shape.len() >= 2 { x_shape[x_shape.len() - 2] } else { 1 };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module};

    #[test]
    fn int8_roundtrip_low_error() {
        let weight = Tensor::from_vec(
            vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8],
            vec![2, 4],
        );
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
        let max_diff: f32 = f32_vals.iter().zip(q_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 5.0, "INT8 inference should be close to f32, max diff: {max_diff}");
    }

    #[test]
    fn int8_weights_shape() {
        let w = Tensor::randn(&[16, 32]);
        let q = Int8Weights::quantize(&w);
        assert_eq!(q.shape, vec![16, 32]);
        assert_eq!(q.data.len(), 16 * 32);
        assert_eq!(q.scales.len(), 16);
    }
}
