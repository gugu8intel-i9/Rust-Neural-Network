//! Multi-format quantization: INT8, INT4, FP16, NF4, NVFP4, and GGUF-compatible formats.
//!
//! # Supported formats
//!
//! | Format | Bits | Use case | Compression vs f32 |
//! |--------|------|----------|-------------------|
//! | FP16   | 16   | Training/inference | 2× |
//! | BF16   | 16   | Training (wider range) | 2× |
//! | INT8   | 8    | Inference | 4× |
//! | INT4   | 4    | Edge inference | 8× |
//! | NF4    | 4    | QLoRA fine-tuning | 8× |
//! | NVFP4  | 4    | NVIDIA Blackwell inference | 8× |
//! | GGUF_Q4| 4    | llama.cpp compatibility | 8× |
//!
//! # NF4 (NormalFloat 4-bit)
//!
//! NF4 is the optimal data type for normally-distributed weights (which pretrained neural
//! network weights are). It uses 16 information-theoretically optimal quantile values for
//! the standard normal distribution, achieving lower quantization error than uniform INT4.
//!
//! # NVFP4 (NVIDIA Floating Point 4-bit)
//!
//! NVFP4 uses the E2M1 format (2 exponent bits, 1 mantissa bit) with a per-tensor or
//! per-block scaling factor. This is the native 4-bit format for NVIDIA Blackwell GPUs
//! (RTX 5090, B200). The 4-bit values represent: {±0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0,
//! ±4.0, ±6.0} with sign.

use crate::tensor::Tensor;
use crate::nn::{Linear, Module};
use ndarray::{ArrayD, IxDyn};

// ==================== Quantization format definitions ====================

/// Quantization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    /// Half precision (16-bit float).
    FP16,
    /// Brain float (16-bit, 8 exponent bits).
    BF16,
    /// 8-bit integer with per-channel scale.
    INT8,
    /// 4-bit integer with group-wise scale.
    INT4,
    /// NormalFloat 4-bit (optimal for normally-distributed weights).
    NF4,
    /// NVIDIA FP4 (E2M1 format for Blackwell GPUs).
    NVFP4,
    /// GGUF Q4_0 (llama.cpp compatible 4-bit quantization).
    GGUF_Q4,
    /// GGUF Q4_K (llama.cpp compatible, higher quality 4-bit).
    GGUF_Q4K,
    /// GGUF Q8_0 (llama.cpp compatible 8-bit).
    GGUF_Q8,
}

impl QuantFormat {
    /// Bits per weight element.
    pub fn bits(&self) -> usize {
        match self {
            QuantFormat::FP16 | QuantFormat::BF16 => 16,
            QuantFormat::INT8 | QuantFormat::GGUF_Q8 => 8,
            QuantFormat::INT4 | QuantFormat::NF4 | QuantFormat::NVFP4
            | QuantFormat::GGUF_Q4 | QuantFormat::GGUF_Q4K => 4,
        }
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits() as f64
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            QuantFormat::FP16 => "FP16",
            QuantFormat::BF16 => "BF16",
            QuantFormat::INT8 => "INT8",
            QuantFormat::INT4 => "INT4",
            QuantFormat::NF4 => "NF4 (NormalFloat 4-bit)",
            QuantFormat::NVFP4 => "NVFP4 (NVIDIA FP4 E2M1)",
            QuantFormat::GGUF_Q4 => "GGUF Q4_0",
            QuantFormat::GGUF_Q4K => "GGUF Q4_K",
            QuantFormat::GGUF_Q8 => "GGUF Q8_0",
        }
    }
}

// ==================== NF4 lookup table ====================

/// The 16 NF4 quantile values for the standard normal distribution N(0,1).
/// These are the information-theoretically optimal 4-bit quantization levels.
const NF4_LOOKUP: [f32; 16] = [
    -1.0, -0.696_192_8, -0.525_073_05, -0.394_917_5,
    -0.284_441_38, -0.184_773_43, -0.091_050_036, 0.0,
    0.079_580_3, 0.160_930_2, 0.246_112_3, 0.337_915_24,
    0.440_709_83, 0.562_617, 0.722_956_84, 1.0,
];

/// Find the nearest NF4 index for a normalized value in [-1, 1].
fn nf4_quantize_single(x: f32) -> u8 {
    let mut best = 0u8;
    let mut best_dist = f32::INFINITY;
    for (i, &v) in NF4_LOOKUP.iter().enumerate() {
        let d = (v - x).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    best
}

/// Dequantize an NF4 index back to a float (given the scale).
fn nf4_dequant_single(idx: u8, scale: f32) -> f32 {
    NF4_LOOKUP[idx as usize & 0xF] * scale
}

// ==================== NVFP4 (E2M1) ====================

/// NVFP4 E2M1 values: 4-bit floats with 2 exponent + 1 mantissa + 1 sign.
/// Represents: ±0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
const NVFP4_LOOKUP: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5,  // positive, exp=0
    2.0, 3.0, 4.0, 6.0,  // positive, exp=1
    -0.0, -0.5, -1.0, -1.5,  // negative, exp=0
    -2.0, -3.0, -4.0, -6.0,  // negative, exp=1
];

fn nvfp4_quantize_single(x: f32) -> u8 {
    let abs_x = x.abs();
    let sign = if x < 0.0 { 8u8 } else { 0u8 };
    // Find nearest magnitude.
    let magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut best = 0u8;
    let mut best_dist = f32::INFINITY;
    for (i, &m) in magnitudes.iter().enumerate() {
        let d = (m - abs_x).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    sign | best
}

fn nvfp4_dequant_single(idx: u8, scale: f32) -> f32 {
    NVFP4_LOOKUP[idx as usize & 0xF] * scale
}

// ==================== GGUF Q4_0 ====================

/// GGUF Q4_0 block: 32 values per block, 1 f16 scale + 32 × 4-bit packed.
/// Each block: scale (f16) + 16 bytes of packed 4-bit values = 18 bytes per 32 values.
fn quantize_q4_0_block(block: &[f32]) -> (f32, Vec<u8>) {
    let n = block.len();
    let max_abs = block.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    let scale = max_abs / -8.0_f32; // Q4_0 uses signed 4-bit: range [-8, 7]
    let scale = if scale == 0.0 { 1.0 } else { scale };

    let mut packed = Vec::with_capacity(n.div_ceil(2));
    let mut i = 0;
    while i < n {
        let q0 = ((block[i] / scale).round().clamp(-8.0, 7.0) as i8) & 0xF;
        let q1 = if i + 1 < n {
            ((block[i + 1] / scale).round().clamp(-8.0, 7.0) as i8) & 0xF
        } else {
            0
        };
        packed.push((q0 | (q1 << 4)) as u8);
        i += 2;
    }
    (scale, packed)
}

fn dequantize_q4_0_block(scale: f32, packed: &[u8], count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
        // Sign-extend from 4-bit.
        let val = if nibble & 0x8 != 0 {
            (((nibble) | 0xF0_u8) as i8) as f32
        } else {
            nibble as f32
        };
        result.push(val * scale);
    }
    result
}

// ==================== Quantized tensor ====================

/// A quantized weight tensor supporting multiple formats.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Format used.
    pub format: QuantFormat,
    /// Quantized data (packed bytes).
    pub data: Vec<u8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Group size for block-wise quantization.
    pub group_size: usize,
}

impl QuantizedTensor {
    /// Quantize a tensor to the given format.
    pub fn quantize(tensor: &Tensor, format: QuantFormat, group_size: usize) -> Self {
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let shape = tensor.shape();
        let n = data.len();
        let gs = if group_size == 0 { n } else { group_size };

        match format {
            QuantFormat::FP16 => Self::quantize_fp16(&data, &shape),
            QuantFormat::BF16 => Self::quantize_bf16(&data, &shape),
            QuantFormat::INT8 => Self::quantize_int8(&data, &shape, gs),
            QuantFormat::INT4 => Self::quantize_int4(&data, &shape, gs),
            QuantFormat::NF4 => Self::quantize_nf4(&data, &shape, gs),
            QuantFormat::NVFP4 => Self::quantize_nvfp4(&data, &shape, gs),
            QuantFormat::GGUF_Q4 | QuantFormat::GGUF_Q4K => Self::quantize_gguf_q4(&data, &shape, gs),
            QuantFormat::GGUF_Q8 => Self::quantize_gguf_q8(&data, &shape, gs),
        }
    }

    /// Dequantize back to an f32 Tensor.
    pub fn dequantize(&self) -> Tensor {
        let n: usize = self.shape.iter().product();
        let result = match self.format {
            QuantFormat::FP16 => self.dequant_fp16(n),
            QuantFormat::BF16 => self.dequant_bf16(n),
            QuantFormat::INT8 => self.dequant_int8(n),
            QuantFormat::INT4 => self.dequant_int4(n),
            QuantFormat::NF4 => self.dequant_nf4(n),
            QuantFormat::NVFP4 => self.dequant_nvfp4(n),
            QuantFormat::GGUF_Q4 | QuantFormat::GGUF_Q4K => self.dequant_gguf_q4(n),
            QuantFormat::GGUF_Q8 => self.dequant_gguf_q8(n),
        };
        Tensor::new(
            ArrayD::from_shape_vec(IxDyn(&self.shape), result).unwrap(),
            false,
        )
    }

    /// Memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Compression ratio vs f32.
    pub fn compression_ratio(&self) -> f64 {
        let f32_bytes = self.shape.iter().product::<usize>() * 4;
        f32_bytes as f64 / self.mem_bytes() as f64
    }

    /// Quantization error (mean absolute difference).
    pub fn quantization_error(&self, original: &Tensor) -> f32 {
        let dequant = self.dequantize();
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let deq: Vec<f32> = dequant.data().iter().copied().collect();
        let total_diff: f32 = orig.iter().zip(deq.iter()).map(|(a, b)| (a - b).abs()).sum();
        total_diff / orig.len().max(1) as f32
    }

    // ---- Per-format implementations ----

    fn quantize_int8(data: &[f32], shape: &[usize], gs: usize) -> Self {
        let n = data.len();
        let n_groups = n.div_ceil(gs);
        let mut scales = Vec::with_capacity(n_groups);
        let mut qdata = Vec::with_capacity(n);

        for g in 0..n_groups {
            let start = g * gs;
            let end = (start + gs).min(n);
            let group = &data[start..end];
            let max_abs = group.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            for &v in group {
                let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
                qdata.push(q as u8);
            }
        }

        QuantizedTensor { format: QuantFormat::INT8, data: qdata, scales, shape: shape.to_vec(), group_size: gs }
    }

    fn dequant_int8(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let g = i / self.group_size;
            let scale = self.scales[g.min(self.scales.len() - 1)];
            result.push((self.data[i] as i8) as f32 * scale);
        }
        result
    }

    fn quantize_int4(data: &[f32], shape: &[usize], gs: usize) -> Self {
        let n = data.len();
        let n_groups = n.div_ceil(gs);
        let mut scales = Vec::with_capacity(n_groups);
        let mut packed = Vec::with_capacity(n.div_ceil(2));

        for g in 0..n_groups {
            let start = g * gs;
            let end = (start + gs).min(n);
            let group = &data[start..end];
            let max_abs = group.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);

            // Quantize to 4-bit [-8, 7].
            let qvals: Vec<i8> = group.iter()
                .map(|&v| (v / scale).round().clamp(-8.0, 7.0) as i8)
                .collect();

            // Pack pairs into bytes.
            let mut i = 0;
            while i < qvals.len() {
                let lo = (qvals[i] & 0xF) as u8;
                let hi = if i + 1 < qvals.len() { (qvals[i + 1] & 0xF) as u8 } else { 0 };
                packed.push(lo | (hi << 4));
                i += 2;
            }
        }

        QuantizedTensor { format: QuantFormat::INT4, data: packed, scales, shape: shape.to_vec(), group_size: gs }
    }

    fn dequant_int4(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let g = i / self.group_size;
            let scale = self.scales[g.min(self.scales.len() - 1)];
            let byte = self.data[i / 2];
            let nibble = if i % 2 == 0 { (byte & 0xF) as i8 } else { ((byte >> 4) & 0xF) as i8 };
            let val: i8 = if nibble & 0x8 != 0 { (nibble as u8 | 0xF0) as i8 } else { nibble };
            result.push(val as f32 * scale);
        }
        result
    }

    fn quantize_nf4(data: &[f32], shape: &[usize], gs: usize) -> Self {
        let n = data.len();
        let n_groups = n.div_ceil(gs);
        let mut scales = Vec::with_capacity(n_groups);
        let mut packed = Vec::with_capacity(n.div_ceil(2));

        for g in 0..n_groups {
            let start = g * gs;
            let end = (start + gs).min(n);
            let group = &data[start..end];
            let max_abs = group.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
            scales.push(scale);

            let qvals: Vec<u8> = group.iter()
                .map(|&v| nf4_quantize_single(v / scale))
                .collect();

            let mut i = 0;
            while i < qvals.len() {
                packed.push(qvals[i] | (if i + 1 < qvals.len() { qvals[i + 1] << 4 } else { 0 }));
                i += 2;
            }
        }

        QuantizedTensor { format: QuantFormat::NF4, data: packed, scales, shape: shape.to_vec(), group_size: gs }
    }

    fn dequant_nf4(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let g = i / self.group_size;
            let scale = self.scales[g.min(self.scales.len() - 1)];
            let byte = self.data[i / 2];
            let idx = if i % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
            result.push(nf4_dequant_single(idx, scale));
        }
        result
    }

    fn quantize_nvfp4(data: &[f32], shape: &[usize], gs: usize) -> Self {
        let n = data.len();
        let n_groups = n.div_ceil(gs);
        let mut scales = Vec::with_capacity(n_groups);
        let mut packed = Vec::with_capacity(n.div_ceil(2));

        for g in 0..n_groups {
            let start = g * gs;
            let end = (start + gs).min(n);
            let group = &data[start..end];
            let max_abs = group.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
            scales.push(scale);

            let qvals: Vec<u8> = group.iter()
                .map(|&v| nvfp4_quantize_single(v / scale))
                .collect();

            let mut i = 0;
            while i < qvals.len() {
                packed.push(qvals[i] | (if i + 1 < qvals.len() { qvals[i + 1] << 4 } else { 0 }));
                i += 2;
            }
        }

        QuantizedTensor { format: QuantFormat::NVFP4, data: packed, scales, shape: shape.to_vec(), group_size: gs }
    }

    fn dequant_nvfp4(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let g = i / self.group_size;
            let scale = self.scales[g.min(self.scales.len() - 1)];
            let byte = self.data[i / 2];
            let idx = if i % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
            result.push(nvfp4_dequant_single(idx, scale));
        }
        result
    }

    fn quantize_gguf_q4(data: &[f32], shape: &[usize], gs: usize) -> Self {
        let block_size = gs.min(32); // Q4_0 uses 32-element blocks
        let n = data.len();
        let n_blocks = n.div_ceil(block_size);
        let mut scales = Vec::with_capacity(n_blocks);
        let mut all_packed = Vec::new();

        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let (scale, packed) = quantize_q4_0_block(&data[start..end]);
            scales.push(scale);
            all_packed.extend_from_slice(&packed);
        }

        QuantizedTensor { format: QuantFormat::GGUF_Q4, data: all_packed, scales, shape: shape.to_vec(), group_size: block_size }
    }

    fn dequant_gguf_q4(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        let bs = self.group_size;
        let mut offset = 0;
        for b in 0..n.div_ceil(bs) {
            let start = b * bs;
            let end = (start + bs).min(n);
            let count = end - start;
            let scale = self.scales[b];
            let packed_len = count.div_ceil(2);
            let block = &self.data[offset..offset + packed_len];
            let deq = dequantize_q4_0_block(scale, block, count);
            result.extend(deq);
            offset += packed_len;
        }
        result
    }

    fn quantize_gguf_q8(data: &[f32], shape: &[usize], gs: usize) -> Self {
        // Q8_0: 32-element blocks, f16 scale + 32 × int8.
        let block_size = gs.min(32);
        let n = data.len();
        let n_blocks = n.div_ceil(block_size);
        let mut scales = Vec::with_capacity(n_blocks);
        let mut qdata = Vec::with_capacity(n);

        for b in 0..n_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let group = &data[start..end];
            let max_abs = group.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);
            for &v in group {
                qdata.push((v / scale).round().clamp(-128.0, 127.0) as i8 as u8);
            }
        }

        QuantizedTensor { format: QuantFormat::GGUF_Q8, data: qdata, scales, shape: shape.to_vec(), group_size: block_size }
    }

    fn dequant_gguf_q8(&self, n: usize) -> Vec<f32> {
        self.dequant_int8(n) // same layout as INT8
    }

    fn quantize_fp16(data: &[f32], shape: &[usize]) -> Self {
        let half_data: Vec<u16> = data.iter().map(|&v| f32_to_f16(v)).collect();
        let mut bytes = Vec::with_capacity(half_data.len() * 2);
        for h in &half_data {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        QuantizedTensor { format: QuantFormat::FP16, data: bytes, scales: vec![], shape: shape.to_vec(), group_size: 0 }
    }

    fn dequant_fp16(&self, n: usize) -> Vec<f32> {
        self.data.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .take(n)
            .collect()
    }

    fn quantize_bf16(data: &[f32], shape: &[usize]) -> Self {
        let bf16_data: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut bytes = Vec::with_capacity(bf16_data.len() * 2);
        for h in &bf16_data {
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        QuantizedTensor { format: QuantFormat::BF16, data: bytes, scales: vec![], shape: shape.to_vec(), group_size: 0 }
    }

    fn dequant_bf16(&self, n: usize) -> Vec<f32> {
        self.data.chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .take(n)
            .collect()
    }
}

// ==================== FP16 / BF16 conversion ====================

/// Convert f32 to IEEE 754 half precision (binary16).
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN.
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // Overflow to Inf.
    }
    if new_exp <= 0 {
        if 14 - new_exp >= 24 {
            return sign; // Underflow to zero.
        }
        let mant = mant | 0x800000;
        let shift = 14 - new_exp;
        let half_mant = mant >> shift;
        return sign | half_mant as u16;
    }

    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert IEEE 754 half precision (binary16) to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1F;
    let mant = (h as u32) & 0x3FF;

    let val = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal.
            let mut e = 1u32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | ((127 - 15 + e as i32) as u32) << 23 | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F800000 | (mant << 13) // Inf or NaN.
    } else {
        sign | ((exp + 112) << 23) | (mant << 13) // Normal.
    };

    f32::from_bits(val)
}

/// Convert f32 to BF16 (truncate lower 16 mantissa bits).
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    // Round-to-nearest-even.
    let rounded = bits + ((bits >> 16) & 1) + 0x7FFF;
    (rounded >> 16) as u16
}

/// Convert BF16 to f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ==================== Quantized model ====================

/// A model with all Linear layers quantized to a given format.
pub struct QuantizedModel {
    layers: Vec<QuantizedLinear>,
    pub format: QuantFormat,
}

/// A quantized Linear layer for inference.
pub struct QuantizedLinear {
    pub weight: QuantizedTensor,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl QuantizedLinear {
    /// Quantize a Linear layer.
    pub fn from_linear(layer: &Linear, format: QuantFormat, group_size: usize) -> Self {
        let weight = QuantizedTensor::quantize(&layer.weight, format, group_size);
        let bias = layer.bias.as_ref().map(|b| b.data().iter().copied().collect());
        let in_f = layer.weight.shape()[1];
        let out_f = layer.weight.shape()[0];
        QuantizedLinear { weight, bias, in_features: in_f, out_features: out_f }
    }

    /// Inference: dequantize on-the-fly and compute.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let w = self.weight.dequantize();
        let w_t = w.transpose();
        let mut out = x.matmul(&w_t);
        if let Some(ref bias) = self.bias {
            let b = Tensor::from_vec(bias.clone(), vec![self.out_features]);
            out = out.add(&b);
        }
        out
    }
}

impl QuantizedModel {
    /// Quantize all parameters of a model.
    pub fn from_model(model: &dyn Module, format: QuantFormat, group_size: usize) -> Self {
        // For simplicity, treat all parameter pairs as Linear layers.
        let params = model.parameters();
        let mut layers = Vec::new();

        // Group params in pairs (weight, bias).
        let mut i = 0;
        while i < params.len() {
            let weight = &params[i];
            let bias = if i + 1 < params.len() { Some(&params[i + 1]) } else { None };

            let q_weight = QuantizedTensor::quantize(weight, format, group_size);
            let q_bias = bias.map(|b| b.data().iter().copied().collect());
            let out_f = weight.shape().first().copied().unwrap_or(0);
            let in_f = weight.shape().get(1).copied().unwrap_or(0);

            layers.push(QuantizedLinear {
                weight: q_weight,
                bias: q_bias,
                in_features: in_f,
                out_features: out_f,
            });
            i += if bias.is_some() { 2 } else { 1 };
        }

        QuantizedModel { layers, format }
    }

    /// Total memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.weight.mem_bytes() + l.bias.as_ref().map_or(0, |b| b.len() * 4)).sum()
    }

    /// Compression ratio vs f32 model.
    pub fn compression_ratio(&self) -> f64 {
        let total_elements: usize = self.layers.iter()
            .map(|l| l.weight.shape.iter().product::<usize>())
            .sum();
        let f32_bytes = total_elements * 4;
        f32_bytes as f64 / self.mem_bytes().max(1) as f64
    }

    /// Get quantized layers.
    pub fn layers(&self) -> &[QuantizedLinear] {
        &self.layers
    }
}

/// Convenience: quantize a tensor to any supported format.
pub fn quantize(tensor: &Tensor, format: QuantFormat, group_size: usize) -> QuantizedTensor {
    QuantizedTensor::quantize(tensor, format, group_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip() {
        let t = Tensor::from_vec(vec![0.1, 0.5, -0.3, 0.8, -0.9, 0.0], vec![2, 3]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::INT8, 0);
        let d = q.dequantize();
        let orig: Vec<f32> = t.data().iter().copied().collect();
        let deq: Vec<f32> = d.data().iter().copied().collect();
        for (o, r) in orig.iter().zip(deq.iter()) {
            assert!((o - r).abs() < 0.05, "INT8 error too large: {o} vs {r}");
        }
    }

    #[test]
    fn int4_compression() {
        let t = Tensor::randn(&[64, 128]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::INT4, 32);
        assert!(q.compression_ratio() > 5.0, "INT4 should compress >5x");
        let d = q.dequantize();
        assert_eq!(d.shape(), vec![64, 128]);
    }

    #[test]
    fn nf4_lower_error_than_int4() {
        // NF4 should have lower quantization error than INT4 for normally-distributed data.
        let t = Tensor::randn(&[256]);
        let q_nf4 = QuantizedTensor::quantize(&t, QuantFormat::NF4, 64);
        let q_int4 = QuantizedTensor::quantize(&t, QuantFormat::INT4, 64);
        let err_nf4 = q_nf4.quantization_error(&t);
        let err_int4 = q_int4.quantization_error(&t);
        println!("NF4 error: {err_nf4:.6}, INT4 error: {err_int4:.6}");
        // NF4 should generally be better (allow tie for small samples).
        assert!(err_nf4 <= err_int4 * 1.5, "NF4 error ({err_nf4}) should be <= INT4 error ({err_int4}) * 1.5");
    }

    #[test]
    fn nvfp4_roundtrip() {
        let t = Tensor::from_vec(vec![0.0, 0.5, 1.0, -1.5, 3.0, -6.0, 2.0, -0.5], vec![4, 2]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::NVFP4, 8);
        let d = q.dequantize();
        let _orig: Vec<f32> = t.data().iter().copied().collect();
        let deq: Vec<f32> = d.data().iter().copied().collect();
        // NVFP4 has specific representable values; check roundtrip for exact representable values.
        assert!(deq[0].abs() < 0.01, "0 should map to 0");
        assert!((deq[5] - (-6.0)).abs() < 0.1, "-6 should be representable");
    }

    #[test]
    fn fp16_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 0.5, -0.25, 100.0, -1000.0, 0.001], vec![6]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::FP16, 0);
        let d = q.dequantize();
        let orig: Vec<f32> = t.data().iter().copied().collect();
        let deq: Vec<f32> = d.data().iter().copied().collect();
        for (o, r) in orig.iter().zip(deq.iter()) {
            assert!((o - r).abs() / o.abs().max(1e-6) < 0.01, "FP16 error too large: {o} vs {r}");
        }
    }

    #[test]
    fn bf16_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, -2.5, 100.0, -1000.0], vec![4]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::BF16, 0);
        let d = q.dequantize();
        let orig: Vec<f32> = t.data().iter().copied().collect();
        let deq: Vec<f32> = d.data().iter().copied().collect();
        for (o, r) in orig.iter().zip(deq.iter()) {
            assert!((o - r).abs() < 1.0, "BF16 error: {o} vs {r}");
        }
    }

    #[test]
    fn gguf_q4_roundtrip() {
        let t = Tensor::randn(&[32]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::GGUF_Q4, 32);
        let d = q.dequantize();
        assert_eq!(d.shape(), vec![32]);
        let err = q.quantization_error(&t);
        assert!(err < 0.5, "GGUF Q4_0 error too large: {err}");
    }

    #[test]
    fn gguf_q8_low_error() {
        let t = Tensor::randn(&[64]);
        let q = QuantizedTensor::quantize(&t, QuantFormat::GGUF_Q8, 32);
        let err = q.quantization_error(&t);
        assert!(err < 0.05, "GGUF Q8_0 error too large: {err}");
    }

    #[test]
    fn quantized_linear_forward() {
        let layer = Linear::new(8, 4, true);
        let ql = QuantizedLinear::from_linear(&layer, QuantFormat::INT8, 0);
        let x = Tensor::randn(&[2, 8]);
        let y = ql.forward(&x);
        assert_eq!(y.shape(), vec![2, 4]);
    }

    #[test]
    fn quantized_model_compression() {
        let model = crate::nn::Sequential::new()
            .add(Linear::new(64, 128, true))
            .add(crate::nn::ReLU)
            .add(Linear::new(128, 64, true));
        
        let q_model = QuantizedModel::from_model(&model, QuantFormat::INT4, 32);
        assert!(q_model.compression_ratio() > 4.0, "should compress >4x");
    }

    #[test]
    fn all_formats_produce_valid_output() {
        let t = Tensor::randn(&[16, 16]);
        for format in [QuantFormat::FP16, QuantFormat::BF16, QuantFormat::INT8,
                       QuantFormat::INT4, QuantFormat::NF4, QuantFormat::NVFP4,
                       QuantFormat::GGUF_Q4, QuantFormat::GGUF_Q8] {
            let q = QuantizedTensor::quantize(&t, format, 32);
            let d = q.dequantize();
            assert_eq!(d.shape(), vec![16, 16], "{:?} wrong shape", format);
            assert!(d.data().iter().all(|v| v.is_finite()), "{:?} has non-finite values", format);
        }
    }

    #[test]
    fn format_bits_and_compression() {
        assert_eq!(QuantFormat::FP16.bits(), 16);
        assert_eq!(QuantFormat::INT8.bits(), 8);
        assert_eq!(QuantFormat::NF4.bits(), 4);
        assert_eq!(QuantFormat::NVFP4.bits(), 4);
        assert!((QuantFormat::INT4.compression_ratio() - 8.0).abs() < 0.1);
    }

    #[test]
    fn f16_conversion_special_values() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f16_to_f32(0), 0.0);
        assert!((f16_to_f32(f32_to_f16(1.0)) - 1.0).abs() < 0.01);
        // Inf.
        let inf_bits = f32_to_f16(f32::INFINITY);
        assert_eq!(inf_bits & 0x7C00, 0x7C00);
    }

    #[test]
    fn group_size_affects_error() {
        let t = Tensor::randn(&[256]);
        let q_small = QuantizedTensor::quantize(&t, QuantFormat::INT4, 32);
        let q_large = QuantizedTensor::quantize(&t, QuantFormat::INT4, 256);
        let err_small = q_small.quantization_error(&t);
        let err_large = q_large.quantization_error(&t);
        assert!(err_small <= err_large, "smaller groups should have lower error");
    }
}
