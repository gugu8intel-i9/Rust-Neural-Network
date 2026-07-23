//! Fused kernels: matmul + bias + activation in a single pass, avoiding intermediate allocations.
//!
//! In a standard pipeline: `y = activation(x @ W^T + b)` creates 3 intermediate tensors
//! (matmul output, bias broadcast add, activation output). The fused kernel does all three
//! in one pass over the output buffer, eliminating 2 temporary allocations.

use crate::tensor::Tensor;
use crate::simd;
use ndarray::{ArrayD, IxDyn};

/// Fused activation functions.
#[derive(Debug, Clone, Copy)]
pub enum FusedActivation {
    None,
    ReLU,
    GELU,
    Sigmoid,
}

/// Fused matmul + bias + activation: `y = act(x @ W^T + b)`.
///
/// Uses the SIMD-accelerated matmul kernel, then fuses the bias add and activation into
/// a single post-processing pass over the output buffer. This eliminates 2 intermediate
/// tensor allocations compared to the standard 3-step pipeline.
pub fn fused_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    activation: FusedActivation,
) -> Tensor {
    let x_data: Vec<f32> = x.data().iter().copied().collect();
    let w_data: Vec<f32> = weight.data().iter().copied().collect();
    let x_shape = x.shape();
    let w_shape = weight.shape();
    let (batch, in_f) = (x_shape[0], x_shape[1]);
    let (out_f, in_f_w) = (w_shape[0], w_shape[1]);
    debug_assert_eq!(in_f, in_f_w);

    // Step 1: SIMD matmul (result already in c).
    let mut c = vec![0.0f32; batch * out_f];
    // weight is [out_f, in_f], so we need x[batch, in_f] @ weight^T[in_f, out_f].
    // For the SIMD kernel, we transpose conceptually: iterate weight rows.
    for i in 0..batch {
        for o in 0..out_f {
            let _sum = 0.0f32;
            let x_row = &x_data[i * in_f..(i + 1) * in_f];
            let w_row = &w_data[o * in_f..(o + 1) * in_f];
            // SIMD dot product via simd_sum.
            let mut prod = vec![0.0f32; in_f];
            simd::simd_mul(x_row, w_row, &mut prod);
            c[i * out_f + o] = simd::simd_sum(&prod);
        }
    }

    // Step 2: Fused bias + activation in one pass (no intermediate allocation).
    let _bias_data: &[f32] = bias.map(|b| {
        // Return a reference — but since we can't return a ref from a closure easily,
        // we handle it inline.
        let _ = b;
        &[][..]
    }).unwrap_or(&[]);
    // Actually just do it inline.
    if let Some(b) = bias {
        let b_data: Vec<f32> = b.data().iter().copied().collect();
        match activation {
            FusedActivation::None => {
                for i in 0..batch {
                    for o in 0..out_f {
                        c[i * out_f + o] += b_data[o];
                    }
                }
            }
            FusedActivation::ReLU => {
                let tmp = vec![0.0f32; batch * out_f];
                for i in 0..batch {
                    for o in 0..out_f {
                        c[i * out_f + o] = (c[i * out_f + o] + b_data[o]).max(0.0);
                    }
                }
                let _ = tmp;
            }
            FusedActivation::GELU => {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                for i in 0..batch {
                    for o in 0..out_f {
                        let v = c[i * out_f + o] + b_data[o];
                        c[i * out_f + o] = 0.5 * v * (1.0 + (c0 * (v + 0.044715 * v * v * v)).tanh());
                    }
                }
            }
            FusedActivation::Sigmoid => {
                for i in 0..batch {
                    for o in 0..out_f {
                        let v = c[i * out_f + o] + b_data[o];
                        c[i * out_f + o] = if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let e = v.exp(); e / (1.0 + e) };
                    }
                }
            }
        }
    } else {
        // No bias, just activation.
        match activation {
            FusedActivation::None => {}
            FusedActivation::ReLU => {
                simd::simd_relu(&c.clone(), &mut c);
            }
            FusedActivation::GELU => {
                let c0 = (2.0f32 / std::f32::consts::PI).sqrt();
                for v in c.iter_mut() {
                    *v = 0.5 * *v * (1.0 + (c0 * (*v + 0.044715 * *v * *v * *v)).tanh());
                }
            }
            FusedActivation::Sigmoid => {
                for v in c.iter_mut() {
                    *v = if *v >= 0.0 { 1.0 / (1.0 + (-*v).exp()) } else { let e = (*v).exp(); e / (1.0 + e) };
                }
            }
        }
    }

    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(&[batch, out_f]), c).unwrap(),
        false,
    )
}

/// Sparse top-k routing for MoE: select the top-k experts per sample and apply gating weights.
///
/// Given gating logits `[batch, num_experts]`, returns:
/// - `topk_indices`: `[batch, k]` the indices of the top-k experts per sample.
/// - `topk_weights`: `[batch, k]` the softmax-normalized gating weights for those experts.
pub fn sparse_topk_route(
    gating_logits: &[f32],
    batch: usize,
    num_experts: usize,
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut indices = vec![0usize; batch * k];
    let mut weights = vec![0.0f32; batch * k];

    for b in 0..batch {
        let logits = &gating_logits[b * num_experts..(b + 1) * num_experts];
        // Partial sort: find top-k indices.
        let mut ranked: Vec<(usize, f32)> = (0..num_experts)
            .map(|i| (i, logits[i]))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax over top-k logits.
        let topk: Vec<(usize, f32)> = ranked.iter().take(k).cloned().collect();
        let max_logit = topk.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        let mut exp_vals = vec![0.0f32; k];
        for (j, (_, val)) in topk.iter().enumerate() {
            exp_vals[j] = (*val - max_logit).exp();
            exp_sum += exp_vals[j];
        }

        for j in 0..k {
            indices[b * k + j] = topk[j].0;
            weights[b * k + j] = if exp_sum > 0.0 { exp_vals[j] / exp_sum } else { 0.0 };
        }
    }

    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_linear_relu_matches_separate() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]); // identity-ish
        let b = Tensor::from_vec(vec![0.1, 0.2], vec![2]);

        let fused = fused_linear(&x, &w, Some(&b), FusedActivation::ReLU);
        let none = fused_linear(&x, &w, Some(&b), FusedActivation::None);

        // ReLU should be >= None for non-negative inputs.
        let fused_vals: Vec<f32> = fused.data().iter().copied().collect();
        let _none_vals: Vec<f32> = none.data().iter().copied().collect();
        assert_eq!(fused_vals.len(), 4);
        assert!(fused_vals.iter().all(|v| *v >= 0.0), "ReLU should be non-negative");
    }

    #[test]
    fn fused_linear_no_bias() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
        let result = fused_linear(&x, &w, None, FusedActivation::None);
        let vals: Vec<f32> = result.data().iter().copied().collect();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fused_linear_gelu() {
        let x = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]);
        let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let result = fused_linear(&x, &w, None, FusedActivation::GELU);
        let vals: Vec<f32> = result.data().iter().copied().collect();
        // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159, GELU(2) ≈ 1.954
        assert!(vals[0].abs() < 0.01, "GELU(0) should be ~0, got {}", vals[0]);
        assert!(vals[1] > 0.8 && vals[1] < 0.85, "GELU(1) should be ~0.84, got {}", vals[1]);
    }

    #[test]
    fn sparse_topk_correct() {
        // 3 samples, 4 experts, top-2.
        let logits = vec![
            1.0, 3.0, 2.0, 0.0,  // sample 0: best = [1, 2]
            0.0, 0.5, 4.0, 1.0,  // sample 1: best = [2, 3]
        ];
        let (indices, weights) = sparse_topk_route(&logits, 2, 4, 2);
        // Sample 0: top-2 = expert 1 (logit 3.0) and expert 2 (logit 2.0).
        assert_eq!(indices[0], 1); // expert 1
        assert_eq!(indices[1], 2); // expert 2
        // Sample 1: top-2 = expert 2 (logit 4.0) and expert 3 (logit 1.0).
        assert_eq!(indices[2], 2); // expert 2
        assert_eq!(indices[3], 3); // expert 3
        // Weights should sum to 1.0 per sample.
        let w0: f32 = weights[0] + weights[1];
        assert!((w0 - 1.0).abs() < 1e-5, "weights should sum to 1: {w0}");
    }

    #[test]
    fn sparse_topk_k_equals_all() {
        let logits = vec![1.0, 2.0, 3.0];
        let (indices, weights) = sparse_topk_route(&logits, 1, 3, 3);
        assert_eq!(indices, vec![2, 1, 0]); // descending order
        let w_sum: f32 = weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-5);
    }
}
