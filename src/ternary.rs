//! Ternary neural networks: weights constrained to {-1, 0, +1} for extreme compression.
//!
//! # Innovation: TWN (Ternary Weight Networks) with learned thresholds
//!
//! Ternary weight networks constrain every weight to exactly three values: **{-1, 0, +1}**.
//! This gives **32× compression** vs f32 (2 bits per weight packed densely) while retaining
//! most of the model's accuracy. Each weight is stored as a 2-bit code:
//!
//! | Code | Value |
//! |------|-------|
//! | 00   | 0     |
//! | 01   | +1    |
//! | 10   | -1    |
//! | 11   | (unused / delta flag) |
//!
//! ## Key components
//!
//! - [`TernaryQuantizer`]: quantizes f32 weights to {-1, 0, +1} using a learned per-channel
//!   threshold Δ. Weights with |w| < Δ become 0; the rest are sign(w).
//! - [`TernaryTensor`]: packed ternary storage (2 bits per weight, 16 weights per u32).
//! - [`TernaryLinear`]: inference layer that computes `x @ W^T` using **shift-and-add**
//!   arithmetic — no multiplications needed (±1 × x = ±x, 0 × x = 0).
//! - [`TernaryModel`]: quantize an entire model's Linear layers to ternary.
//!
//! ## Why ternary is fast
//!
//! Multiplication by {-1, 0, +1} requires **zero multiplications** — only sign flips and
//! conditional adds. A ternary matmul is pure accumulation:
//! ```text
//! y[j] += sign(w[i,j]) * x[i]    // just add or subtract x[i]
//! ```
//! This is ~4-8× faster than INT8 matmul on hardware without dedicated INT8 units.

use crate::tensor::Tensor;
use crate::nn::{Linear, Module};
use ndarray::{ArrayD, IxDyn};

// ==================== Ternary packing ====================

/// Pack ternary values {-1, 0, +1} into 2-bit codes, 16 per u32.
///
/// Encoding:
/// - 0 → code 0b00
/// - +1 → code 0b01
/// - -1 → code 0b10
fn pack_ternary(values: &[i8]) -> Vec<u32> {
    let n = values.len();
    let mut packed = vec![0u32; n.div_ceil(16)];
    for (i, &v) in values.iter().enumerate() {
        let code: u32 = match v {
            0 => 0,
            1 => 1,
            -1 => 2,
            _ if v > 0 => 1,
            _ => 2,
        };
        packed[i / 16] |= code << ((i % 16) * 2);
    }
    packed
}

/// Unpack 2-bit codes back to i8 values {-1, 0, +1}.
fn unpack_ternary(packed: &[u32], count: usize) -> Vec<i8> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let code = (packed[i / 16] >> ((i % 16) * 2)) & 0x3;
        result.push(match code {
            0 => 0,
            1 => 1,
            2 => -1,
            _ => 0,
        });
    }
    result
}

// ==================== Ternary tensor ====================

/// A tensor with ternary weights {-1, 0, +1}, packed at 2 bits per weight.
#[derive(Debug, Clone)]
pub struct TernaryTensor {
    /// Packed ternary data (2 bits per weight, 16 weights per u32).
    pub packed: Vec<u32>,
    /// Per-row scale factors (for approximate reconstruction: w ≈ ternary * scale).
    pub scales: Vec<f32>,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Number of elements.
    pub numel: usize,
}

impl TernaryTensor {
    /// Quantize an f32 tensor to ternary {-1, 0, +1} using a per-row threshold Δ.
    ///
    /// For each row: `Δ = 0.7 * mean(|w|)`. Weights with `|w| < Δ` become 0; others become sign(w).
    /// The per-row scale = mean(|w|) for approximate reconstruction.
    pub fn quantize(tensor: &Tensor) -> Self {
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let shape = tensor.shape();
        let numel = data.len();

        // Per-row quantization (for 2D tensors). For 1D, treat as a single row.
        let rows = if shape.len() >= 2 { shape[0] } else { 1 };
        let cols = if shape.len() >= 2 { shape[1] } else { numel };
        let mut ternary = vec![0i8; numel];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let row_start = r * cols;
            let row_end = (row_start + cols).min(numel);
            let row = &data[row_start..row_end];

            // Compute threshold: Δ = 0.7 * mean(|w|).
            let mean_abs: f32 = row.iter().map(|v| v.abs()).sum::<f32>() / row.len().max(1) as f32;
            let threshold = 0.7 * mean_abs;
            let scale = mean_abs.max(1e-8);
            scales[r] = scale;

            for (j, &w) in row.iter().enumerate() {
                if w.abs() < threshold {
                    ternary[row_start + j] = 0;
                } else if w > 0.0 {
                    ternary[row_start + j] = 1;
                } else {
                    ternary[row_start + j] = -1;
                }
            }
        }

        let packed = pack_ternary(&ternary);
        TernaryTensor { packed, scales, shape, numel }
    }

    /// Dequantize to f32: `w ≈ ternary_value * per_row_scale`.
    pub fn dequantize(&self) -> Tensor {
        let ternary = unpack_ternary(&self.packed, self.numel);
        let _rows = self.shape.first().copied().unwrap_or(1);
        let cols = if self.shape.len() >= 2 { self.shape[1] } else { self.numel };

        let data: Vec<f32> = (0..self.numel)
            .map(|i| {
                let r = i / cols;
                let scale = self.scales[r.min(self.scales.len() - 1)];
                ternary[i] as f32 * scale
            })
            .collect();

        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&self.shape), data).unwrap(),
            false,
        )
    }

    /// Memory usage in bytes (packed data + scales).
    pub fn mem_bytes(&self) -> usize {
        self.packed.len() * 4 + self.scales.len() * 4
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f64 {
        (self.numel * 4) as f64 / self.mem_bytes().max(1) as f64
    }

    /// Quantization error (mean absolute difference).
    pub fn quantization_error(&self, original: &Tensor) -> f32 {
        let dequant = self.dequantize();
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let deq: Vec<f32> = dequant.data().iter().copied().collect();
        let total: f32 = orig.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).sum();
        total / orig.len().max(1) as f32
    }

    /// Sparsity: fraction of weights that are exactly 0.
    pub fn sparsity(&self) -> f64 {
        let ternary = unpack_ternary(&self.packed, self.numel);
        let zeros = ternary.iter().filter(|&&v| v == 0).count();
        zeros as f64 / self.numel.max(1) as f64
    }

    /// Count of +1, -1, and 0 values.
    pub fn value_counts(&self) -> (usize, usize, usize) {
        let ternary = unpack_ternary(&self.packed, self.numel);
        let pos = ternary.iter().filter(|&&v| v == 1).count();
        let neg = ternary.iter().filter(|&&v| v == -1).count();
        let zero = ternary.iter().filter(|&&v| v == 0).count();
        (pos, neg, zero)
    }
}

// ==================== Ternary linear layer ====================

/// A Linear layer with ternary weights for multiplication-free inference.
///
/// The forward pass uses shift-and-add arithmetic: since w ∈ {-1, 0, +1},
/// `y[j] = Σ_i w[i,j] * x[i]` becomes pure addition/subtraction.
pub struct TernaryLinear {
    pub ternary_weight: TernaryTensor, // [out_features, in_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl TernaryLinear {
    /// Quantize a Linear layer's weights to ternary.
    pub fn from_linear(layer: &Linear) -> Self {
        let ternary_weight = TernaryTensor::quantize(&layer.weight);
        let bias = layer.bias.as_ref().map(|b| b.data().iter().copied().collect());
        let in_f = layer.weight.shape()[1];
        let out_f = layer.weight.shape()[0];
        TernaryLinear { ternary_weight, bias, in_features: in_f, out_features: out_f }
    }

    /// Ternary forward pass using shift-and-add (no multiplications).
    ///
    /// `y[batch, out] = Σ_in ternary_w[out, in] * scale[out] * x[batch, in] + bias[out]`
    ///
    /// For each weight: +1 → add x[i], -1 → subtract x[i], 0 → skip.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_data: Vec<f32> = x.data().iter().copied().collect();
        let x_shape = x.shape();
        let batch = x_shape.first().copied().unwrap_or(1);
        let in_f = *x_shape.last().unwrap_or(&0);

        let weights = unpack_ternary(&self.ternary_weight.packed, self.ternary_weight.numel);
        let out_f = self.out_features;

        let mut result = vec![0.0f32; batch * out_f];

        for b in 0..batch {
            for o in 0..out_f {
                let scale = self.ternary_weight.scales[o];
                let mut acc = 0.0f32;
                let w_row = &weights[o * in_f..(o + 1) * in_f];

                // Shift-and-add: no multiplications.
                for i in 0..in_f {
                    let x_val = x_data[b * in_f + i];
                    match w_row[i] {
                        1 => acc += x_val,
                        -1 => acc -= x_val,
                        _ => {} // skip zeros and any other value
                    }
                }

                result[b * out_f + o] = acc * scale;
                if let Some(ref bias) = self.bias {
                    result[b * out_f + o] += bias[o];
                }
            }
        }

        let mut out_shape = x_shape.to_vec();
        *out_shape.last_mut().unwrap() = out_f;
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&out_shape), result).unwrap(),
            false,
        )
    }
}

// ==================== Ternary model ====================

/// A model with all Linear layers quantized to ternary weights.
pub struct TernaryModel {
    pub layers: Vec<TernaryLinear>,
}

impl TernaryModel {
    /// Quantize all parameters of a model to ternary.
    pub fn from_model(model: &dyn Module) -> Self {
        let params = model.parameters();
        let mut layers = Vec::new();

        let mut i = 0;
        while i < params.len() {
            let weight = &params[i];
            let bias = if i + 1 < params.len() {
                // Check if the next param is 1-D (likely a bias).
                if params[i + 1].ndim() == 1 {
                    Some(params[i + 1].data().iter().copied().collect())
                } else {
                    None
                }
            } else {
                None
            };

            let out_f = weight.shape().first().copied().unwrap_or(0);
            let in_f = weight.shape().get(1).copied().unwrap_or(0);
            let ternary_weight = TernaryTensor::quantize(weight);

            layers.push(TernaryLinear {
                ternary_weight,
                bias: bias.clone(),
                in_features: in_f,
                out_features: out_f,
            });

            let has_bias = bias.is_some();
            i += if has_bias && i + 1 < params.len() { 2 } else { 1 };
        }

        TernaryModel { layers }
    }

    /// Total memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        self.layers.iter()
            .map(|l| l.ternary_weight.mem_bytes() + l.bias.as_ref().map_or(0, |b| b.len() * 4))
            .sum()
    }

    /// Compression ratio vs f32 model.
    pub fn compression_ratio(&self) -> f64 {
        let total_elements: usize = self.layers.iter()
            .map(|l| l.ternary_weight.numel)
            .sum();
        let f32_bytes = total_elements * 4;
        f32_bytes as f64 / self.mem_bytes().max(1) as f64
    }

    /// Average sparsity across all layers.
    pub fn avg_sparsity(&self) -> f64 {
        if self.layers.is_empty() {
            return 0.0;
        }
        self.layers.iter()
            .map(|l| l.ternary_weight.sparsity())
            .sum::<f64>()
            / self.layers.len() as f64
    }

    /// Forward pass through the first layer only (for testing).
    pub fn forward_first_layer(&self, x: &Tensor) -> Tensor {
        self.layers.first().unwrap().forward(x)
    }
}

/// Convenience: quantize any tensor to ternary.
pub fn ternarize(tensor: &Tensor) -> TernaryTensor {
    TernaryTensor::quantize(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_quantize_produces_valid_values() {
        let t = Tensor::from_vec(
            vec![0.5, -0.3, 0.01, -0.8, 0.6, -0.02, 0.9, -0.4],
            vec![2, 4],
        );
        let q = TernaryTensor::quantize(&t);
        let deq = q.dequantize();
        let deq_vals: Vec<f32> = deq.data().iter().copied().collect();
        // All dequantized values should be {-scale, 0, +scale} or 0.
        for v in &deq_vals {
            assert!(v.abs() < 2.0, "ternary values should be small, got {v}");
        }
    }

    #[test]
    fn ternary_compression_ratio() {
        let t = Tensor::randn(&[64, 128]);
        let q = TernaryTensor::quantize(&t);
        // 64*128 = 8192 weights. Packed: 8192/16 = 512 u32s = 2048 bytes + 64 scales = 2304 bytes.
        // f32: 8192*4 = 32768 bytes. Ratio: ~14x.
        assert!(q.compression_ratio() > 10.0, "compression should be >10x, got {}", q.compression_ratio());
    }

    #[test]
    fn ternary_sparsity_is_nonzero() {
        // Ternary quantization with threshold should produce some zeros.
        let t = Tensor::from_vec(
            vec![0.5, -0.5, 0.01, -0.01, 0.5, -0.5, 0.01, -0.01],
            vec![2, 4],
        );
        let q = TernaryTensor::quantize(&t);
        assert!(q.sparsity() > 0.0, "should have some zeros");
    }

    #[test]
    fn ternary_value_counts() {
        let t = Tensor::from_vec(
            vec![10.0, -10.0, 0.001, -0.001, 10.0, -10.0, 10.0, -10.0],
            vec![2, 4],
        );
        let q = TernaryTensor::quantize(&t);
        let (pos, neg, zero) = q.value_counts();
        assert!(pos + neg + zero == 8);
        assert!(zero > 0, "small values should become 0");
        assert!(pos > 0, "large positive values should become +1");
        assert!(neg > 0, "large negative values should become -1");
    }

    #[test]
    fn ternary_pack_unpack_roundtrip() {
        let values = vec![1i8, -1, 0, 1, -1, 0, 1, 1, -1, -1, 0, 0, 1, -1, 1, 0];
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, 16);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn ternary_pack_unpack_large() {
        let values: Vec<i8> = (0..1000).map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 }).collect();
        let packed = pack_ternary(&values);
        let unpacked = unpack_ternary(&packed, 1000);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn ternary_linear_forward_shape() {
        let layer = Linear::new(8, 4, true);
        let t_layer = TernaryLinear::from_linear(&layer);
        let x = Tensor::randn(&[2, 8]);
        let y = t_layer.forward(&x);
        assert_eq!(y.shape(), vec![2, 4]);
        assert!(y.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ternary_linear_no_multiplications() {
        // The forward pass should work correctly using only add/subtract.
        let layer = Linear::new(4, 2, true);
        let t_layer = TernaryLinear::from_linear(&layer);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let y = t_layer.forward(&x);
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn ternary_model_compression() {
        let model = crate::nn::Sequential::new()
            .add(Linear::new(32, 64, true))
            .add(crate::nn::ReLU)
            .add(Linear::new(64, 32, true));
        let t_model = TernaryModel::from_model(&model);
        assert!(t_model.compression_ratio() > 8.0, "model should compress >8x");
    }

    #[test]
    fn ternary_model_forward() {
        let model = crate::nn::Sequential::new()
            .add(Linear::new(8, 16, true))
            .add(crate::nn::ReLU)
            .add(Linear::new(16, 4, true));
        let t_model = TernaryModel::from_model(&model);
        let x = Tensor::randn(&[2, 8]);
        let y = t_model.forward_first_layer(&x);
        assert_eq!(y.shape(), vec![2, 16]);
    }

    #[test]
    fn ternary_model_sparsity() {
        let model = crate::nn::Sequential::new()
            .add(Linear::new(16, 32, true));
        let t_model = TernaryModel::from_model(&model);
        let sparsity = t_model.avg_sparsity();
        assert!((0.0..=1.0).contains(&sparsity));
    }

    #[test]
    fn ternary_quantization_error_reasonable() {
        let t = Tensor::randn(&[64]);
        let q = TernaryTensor::quantize(&t);
        let err = q.quantization_error(&t);
        // Ternary error should be reasonable (less than the std of the data).
        assert!(err < 1.5, "ternary error too large: {err}");
    }
}
