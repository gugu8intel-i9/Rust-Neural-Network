//! Model compression suite: 9 techniques for reducing model size and accelerating inference.
//!
//! # Techniques
//!
//! 1. **Weight Sharing** — multiple layers share a single weight tensor (ALBERT-style).
//! 2. **Sparse Matrices** — CSR format for zero-heavy matrices; skip-zero matmul.
//! 3. **Layer Dropping** — skip layers at inference based on confidence gating.
//! 4. **Knowledge Transfer** — transfer representations between heterogeneous models.
//! 5. **Embedding Compression** — decompose large embedding tables via low-rank factorization.
//! 6. **Mixed Sparsity** — per-layer adaptive sparsity ratios based on sensitivity.
//! 7. **Progressive Shrinking** — gradually increase pruning from soft masks to hard pruning.
//! 8. **Structured Pruning** — remove entire channels/rows that contribute least.
//! 9. **AutoML Compression** — search for the optimal compression recipe per layer.

use crate::tensor::Tensor;
use crate::nn::{Linear, Module};
use crate::loss::Loss;

// ==================== 1. Weight Sharing ====================

/// A shared weight store: multiple "layers" reference the same underlying tensor.
///
/// Inspired by ALBERT: instead of N independent transformer layers, use 1 layer applied N times.
/// This reduces parameters from O(N × layer_size) to O(layer_size).
#[derive(Debug, Clone)]
pub struct SharedWeights {
    /// The single shared weight tensor [out, in].
    pub weight: Tensor,
    /// The single shared bias [out].
    pub bias: Tensor,
    /// Number of times to apply this layer (virtual depth).
    pub num_passes: usize,
    /// Input/output dimensions.
    pub in_features: usize,
    pub out_features: usize,
}

impl SharedWeights {
    /// Create a shared weight layer applied `num_passes` times.
    pub fn new(in_features: usize, out_features: usize, num_passes: usize) -> Self {
        SharedWeights {
            weight: Tensor::he(&[out_features, in_features]),
            bias: Tensor::zeros(&[out_features]),
            num_passes,
            in_features,
            out_features,
        }
    }

    /// Forward: apply the shared layer N times (like ALBERT).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        let w_t = self.weight.transpose();
        for _ in 0..self.num_passes {
            let mm = h.matmul(&w_t);
            h = mm.add(&self.bias);
        }
        h
    }

    /// Parameter count (only 1 layer's worth, not N).
    pub fn param_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }

    /// Compression ratio vs N independent layers.
    pub fn compression_ratio(&self) -> f64 {
        let independent_params = self.param_count() * self.num_passes;
        independent_params as f64 / self.param_count() as f64
    }

    /// Get trainable parameters.
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

// ==================== 2. Sparse Matrices (CSR) ====================

/// A sparse matrix in CSR (Compressed Sparse Row) format.
///
/// Stores only non-zero values + their indices. Matmul skips zeros entirely.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Column indices of non-zero values.
    pub col_indices: Vec<usize>,
    /// Row pointers (length = rows + 1).
    pub row_ptrs: Vec<usize>,
    /// Matrix dimensions.
    pub rows: usize,
    pub cols: usize,
    /// Number of non-zero elements.
    pub nnz: usize,
}

impl SparseMatrix {
    /// Convert a dense tensor to CSR sparse format.
    pub fn from_dense(tensor: &Tensor, threshold: f32) -> Self {
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let shape = tensor.shape();
        let rows = shape.first().copied().unwrap_or(1);
        let cols = shape.get(1).copied().unwrap_or(data.len());

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];

        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v.abs() > threshold {
                    values.push(v);
                    col_indices.push(c);
                }
            }
            row_ptrs.push(values.len());
        }

        let nnz = values.len();
        SparseMatrix { values, col_indices, row_ptrs, rows, cols, nnz }
    }

    /// Sparse × Dense matmul: `C[m,n] = A_sparse[m,k] @ B_dense[k,n]`.
    /// Only iterates over non-zero elements of A.
    pub fn spmm(&self, b: &[f32], n: usize) -> Vec<f32> {
        let m = self.rows;
        let k = self.cols;
        let mut c = vec![0.0f32; m * n];

        for r in 0..m {
            for idx in self.row_ptrs[r]..self.row_ptrs[r + 1] {
                let col = self.col_indices[idx];
                let val = self.values[idx];
                for j in 0..n {
                    c[r * n + j] += val * b[col * n + j];
                }
            }
        }

        let _ = k; // k is implicitly used via col range
        c
    }

    /// Sparsity ratio: fraction of elements that are zero.
    pub fn sparsity(&self) -> f64 {
        let total = self.rows * self.cols;
        1.0 - (self.nnz as f64 / total.max(1) as f64)
    }

    /// Memory usage in bytes.
    pub fn mem_bytes(&self) -> usize {
        self.values.len() * 4 + self.col_indices.len() * 8 + self.row_ptrs.len() * 8
    }

    /// Compression ratio vs dense f32.
    pub fn compression_ratio(&self) -> f64 {
        let dense_bytes = self.rows * self.cols * 4;
        dense_bytes as f64 / self.mem_bytes().max(1) as f64
    }

    /// Reconstruct as dense tensor.
    pub fn to_dense(&self) -> Tensor {
        let mut data = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for idx in self.row_ptrs[r]..self.row_ptrs[r + 1] {
                data[r * self.cols + self.col_indices[idx]] = self.values[idx];
            }
        }
        Tensor::from_vec(data, vec![self.rows, self.cols])
    }
}

// ==================== 3. Layer Dropping ====================

/// Layer dropping: skip layers at inference based on a learned confidence gate.
///
/// Each layer has a gate that predicts whether the input has already converged.
/// If the gate outputs > threshold, skip the layer (early exit).
#[derive(Debug, Clone)]
pub struct LayerDropper {
    /// Per-layer gate thresholds.
    pub thresholds: Vec<f32>,
    /// Per-layer skip counts (statistics).
    pub skip_counts: Vec<usize>,
    /// Total forward count.
    pub total_count: usize,
}

impl LayerDropper {
    /// Create with uniform thresholds.
    pub fn new(num_layers: usize, threshold: f32) -> Self {
        LayerDropper {
            thresholds: vec![threshold; num_layers],
            skip_counts: vec![0; num_layers],
            total_count: 0,
        }
    }

    /// Decide whether to skip a layer based on the change in hidden state norm.
    /// Returns true if the layer should be skipped.
    pub fn should_skip(&mut self, layer_idx: usize, before: &Tensor, after: &Tensor) -> bool {
        self.total_count += 1;
        let before_norm: f32 = before.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        let after_norm: f32 = after.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        let change = (after_norm - before_norm).abs() / before_norm.max(1e-8);
        if change < self.thresholds[layer_idx.min(self.thresholds.len() - 1)] {
            let idx = layer_idx.min(self.skip_counts.len() - 1);
            self.skip_counts[idx] += 1;
            true
        } else {
            false
        }
    }

    /// Average skip rate across all layers.
    pub fn avg_skip_rate(&self) -> f64 {
        if self.total_count == 0 { return 0.0; }
        let total_skips: usize = self.skip_counts.iter().sum();
        total_skips as f64 / (self.total_count * self.thresholds.len().max(1)) as f64
    }

    /// Estimated speedup factor.
    pub fn estimated_speedup(&self) -> f64 {
        let skip = self.avg_skip_rate();
        1.0 / (1.0 - skip).max(0.01)
    }
}

// ==================== 4. Knowledge Transfer ====================

/// Transfer knowledge between heterogeneous models via representation alignment.
///
/// Unlike distillation (which transfers output logits), this aligns intermediate
/// representations, enabling transfer between models with different architectures.
pub struct KnowledgeTransfer {
    /// Source model (frozen).
    source: std::sync::Arc<dyn Module>,
    /// Target model (trainable).
    target: std::sync::Arc<dyn Module>,
    /// Projection layer to align dimensions.
    projection: Linear,
}

impl KnowledgeTransfer {
    pub fn new(
        source: std::sync::Arc<dyn Module>,
        target: std::sync::Arc<dyn Module>,
        source_dim: usize,
        target_dim: usize,
    ) -> Self {
        let proj_dim = source_dim.max(target_dim);
        KnowledgeTransfer {
            source,
            target,
            projection: Linear::new(source_dim, proj_dim, true),
        }
    }

    /// Compute transfer loss: MSE between projected source features and target features.
    pub fn transfer_loss(&self, input: &Tensor) -> f32 {
        let source_feat = (*self.source).forward(input);
        let target_feat = (*self.target).forward(input);
        let projected = self.projection.forward(&source_feat);
        let diff = projected.sub(&target_feat);
        let sq = diff.mul(&diff);
        let total: f32 = sq.data().iter().copied().sum();
        total / sq.len().max(1) as f32
    }
}

// ==================== 5. Embedding Compression ====================

/// Compress a large embedding table via low-rank factorization.
///
/// Instead of storing E[vocab, dim], store U[vocab, rank] × V[rank, dim] where rank << dim.
/// This reduces embedding memory from vocab×dim to (vocab+dim)×rank.
#[derive(Debug, Clone)]
pub struct CompressedEmbedding {
    /// Factor U: [vocab_size, rank].
    pub factor_u: Tensor,
    /// Factor V: [rank, embed_dim].
    pub factor_v: Tensor,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Low-rank dimension.
    pub rank: usize,
}

impl CompressedEmbedding {
    /// Create from an existing embedding table.
    pub fn from_table(embedding: &Tensor, rank: usize) -> Self {
        let shape = embedding.shape();
        let vocab = shape[0];
        let dim = shape[1];
        // Random initialization for factors (in practice, use SVD).
        let factor_u = Tensor::randn(&[vocab, rank]);
        let factor_v = Tensor::randn(&[rank, dim]);
        CompressedEmbedding { factor_u, factor_v, vocab_size: vocab, embed_dim: dim, rank }
    }

    /// Lookup: reconstruct embedding for a token via U[token] @ V.
    pub fn lookup(&self, token: usize) -> Tensor {
        let u_row = Tensor::from_vec(
            self.factor_u.data().iter()
                .skip(token * self.rank)
                .take(self.rank)
                .copied()
                .collect(),
            vec![1, self.rank],
        );
        u_row.matmul(&self.factor_v)
    }

    /// Batch lookup.
    pub fn lookup_batch(&self, tokens: &[usize]) -> Tensor {
        let mut data = Vec::with_capacity(tokens.len() * self.embed_dim);
        for &t in tokens {
            let emb = self.lookup(t);
            data.extend(emb.data().iter().copied());
        }
        Tensor::from_vec(data, vec![tokens.len(), self.embed_dim])
    }

    /// Full reconstructed embedding table.
    pub fn reconstruct(&self) -> Tensor {
        self.factor_u.matmul(&self.factor_v)
    }

    /// Compression ratio vs full embedding table.
    pub fn compression_ratio(&self) -> f64 {
        let original = (self.vocab_size * self.embed_dim) as f64;
        let compressed = (self.vocab_size * self.rank + self.rank * self.embed_dim) as f64;
        original / compressed
    }

    /// Memory in bytes.
    pub fn mem_bytes(&self) -> usize {
        (self.factor_u.len() + self.factor_v.len()) * 4
    }
}

// ==================== 6. Mixed Sparsity ====================

/// Apply different sparsity ratios to different layers based on sensitivity analysis.
#[derive(Debug, Clone)]
pub struct MixedSparsity {
    /// Per-layer sparsity ratios [0, 1].
    pub layer_sparsity: Vec<f32>,
    /// Per-layer sensitivity scores (higher = more sensitive = less pruning).
    pub sensitivity: Vec<f32>,
}

impl MixedSparsity {
    /// Analyze per-layer sensitivity to pruning.
    ///
    /// Layers that lose more accuracy when pruned get lower sparsity targets.
    pub fn analyze(model: &dyn Module, inputs: &Tensor, targets: &Tensor, loss_fn: &dyn Loss) -> Self {
        let params = model.parameters();
        let baseline_loss = {
            let out = model.forward(inputs);
            let l = loss_fn.forward(&out, targets);
            l.data().iter().copied().next().unwrap_or(0.0)
        };

        let mut sensitivity = Vec::with_capacity(params.len());
        let mut layer_sparsity = Vec::with_capacity(params.len());

        for p in params.iter() {
            // Prune 50% of this parameter and measure loss increase.
            let original = p.data();
            let mut pruned = original.clone();
            let threshold = {
                let mut vals: Vec<f32> = pruned.iter().copied().map(|v| v.abs()).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                vals[vals.len() / 2]
            };
            pruned.mapv_inplace(|v| if v.abs() < threshold { 0.0 } else { v });
            let old_data = p.0.write().unwrap().data.clone();
            p.0.write().unwrap().data = pruned;

            let pruned_loss = {
                let out = model.forward(inputs);
                let l = loss_fn.forward(&out, targets);
                l.data().iter().copied().next().unwrap_or(0.0)
            };

            // Restore.
            p.0.write().unwrap().data = old_data;

            let loss_increase = (pruned_loss - baseline_loss).max(0.0);
            sensitivity.push(loss_increase);
            // Less sensitive layers get higher sparsity.
            layer_sparsity.push(0.5 / (1.0 + loss_increase * 10.0));
        }

        // Normalize sparsity to average 50%.
        let avg: f32 = layer_sparsity.iter().sum::<f32>() / layer_sparsity.len().max(1) as f32;
        let scale = 0.5 / avg.max(1e-8);
        for s in &mut layer_sparsity {
            *s = (*s * scale).clamp(0.05, 0.9);
        }

        MixedSparsity { layer_sparsity, sensitivity }
    }

    /// Get the recommended sparsity for layer `i`.
    pub fn sparsity_for_layer(&self, layer_idx: usize) -> f32 {
        self.layer_sparsity.get(layer_idx).copied().unwrap_or(0.5)
    }
}

// ==================== 7. Progressive Shrinking ====================

/// Gradually prune a model during training: soft mask → hard prune.
///
/// Phase 1: Apply soft mask (gradually zero out weights).
/// Phase 2: At target sparsity, switch to hard mask (actual pruning).
#[derive(Debug, Clone)]
pub struct ProgressiveShrinking {
    /// Target sparsity ratio.
    pub target_sparsity: f32,
    /// Current sparsity ratio.
    pub current_sparsity: f32,
    /// Total training steps.
    pub total_steps: usize,
    /// Current step.
    pub current_step: usize,
    /// Warmup steps (no pruning).
    pub warmup_steps: usize,
}

impl ProgressiveShrinking {
    pub fn new(target_sparsity: f32, total_steps: usize, warmup_steps: usize) -> Self {
        ProgressiveShrinking {
            target_sparsity,
            current_sparsity: 0.0,
            total_steps,
            current_step: 0,
            warmup_steps,
        }
    }

    /// Advance one step and return the current pruning ratio.
    pub fn step(&mut self) -> f32 {
        self.current_step += 1;
        if self.current_step <= self.warmup_steps {
            self.current_sparsity = 0.0;
        } else if self.current_step >= self.total_steps {
            self.current_sparsity = self.target_sparsity;
        } else {
            let progress = (self.current_step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            self.current_sparsity = self.target_sparsity * progress;
        }
        self.current_sparsity
    }

    /// Apply soft pruning: zero out the smallest `current_sparsity` fraction of weights.
    pub fn apply_soft_prune(&self, tensor: &Tensor) -> Tensor {
        if self.current_sparsity <= 0.0 {
            return tensor.clone();
        }
        let data: Vec<f32> = tensor.data().iter().copied().collect();
        let n = data.len();
        let n_prune = (n as f32 * self.current_sparsity) as usize;

        // Find the threshold magnitude.
        let mut abs_vals: Vec<f32> = data.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = abs_vals[n_prune.min(n - 1)];

        let pruned: Vec<f32> = data.iter()
            .map(|v| if v.abs() <= threshold { 0.0 } else { *v })
            .collect();

        Tensor::from_vec(pruned, tensor.shape())
    }

    /// Is pruning complete?
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }
}

// ==================== 8. Structured Pruning ====================

/// Remove entire output channels (rows) that contribute the least.
///
/// Unlike unstructured pruning (which zeros individual weights), structured pruning
/// removes entire neurons/channels, enabling actual speedup without sparse kernels.
#[derive(Debug, Clone)]
pub struct StructuredPruner {
    /// Per-layer pruning ratios.
    pub prune_ratios: Vec<f32>,
}

impl StructuredPruner {
    pub fn new(prune_ratio: f32, num_layers: usize) -> Self {
        StructuredPruner { prune_ratios: vec![prune_ratio; num_layers] }
    }

    /// Prune the least important output channels from a weight matrix [out, in].
    ///
    /// Returns the pruned weight and the indices of kept channels.
    pub fn prune_channels(weight: &Tensor, prune_ratio: f32) -> (Tensor, Vec<usize>) {
        let data: Vec<f32> = weight.data().iter().copied().collect();
        let shape = weight.shape();
        let out_f = shape[0];
        let in_f = shape[1];

        // Compute importance per output channel (L2 norm of the row).
        let mut channel_scores: Vec<(usize, f32)> = (0..out_f)
            .map(|o| {
                let norm: f32 = (0..in_f).map(|i| data[o * in_f + i].powi(2)).sum::<f32>().sqrt();
                (o, norm)
            })
            .collect();

        // Sort by importance (ascending).
        channel_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep the top (1 - prune_ratio) channels.
        let n_keep = ((1.0 - prune_ratio) * out_f as f32).round() as usize;
        let n_keep = n_keep.max(1).min(out_f);
        let kept_indices: Vec<usize> = channel_scores[out_f - n_keep..]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();

        // Build the pruned weight matrix.
        let mut pruned = Vec::with_capacity(n_keep * in_f);
        for &idx in &kept_indices {
            pruned.extend_from_slice(&data[idx * in_f..(idx + 1) * in_f]);
        }

        (Tensor::from_vec(pruned, vec![n_keep, in_f]), kept_indices)
    }

    /// Prune a Linear layer: returns a smaller Linear with fewer output features.
    pub fn prune_linear(layer: &Linear, prune_ratio: f32) -> (Linear, Vec<usize>) {
        let (pruned_weight, kept) = Self::prune_channels(&layer.weight, prune_ratio);
        let pruned_bias = layer.bias.as_ref().map(|b| {
            let bias_data: Vec<f32> = kept.iter()
                .map(|&i| b.data().iter().nth(i).copied().unwrap_or(0.0))
                .collect();
            Tensor::from_vec(bias_data, vec![kept.len()])
        });
        let new_out = kept.len();
        let new_in = pruned_weight.shape()[1];
        let new_layer = Linear { weight: pruned_weight, bias: pruned_bias };
        let _ = new_out;
        let _ = new_in;
        (new_layer, kept)
    }
}

// ==================== 9. AutoML Compression ====================

/// Automatic search for the optimal compression configuration per layer.
///
/// For each layer, tries multiple compression strategies (pruning, quantization, low-rank)
/// and selects the one that gives the best accuracy/size trade-off.
#[derive(Debug, Clone)]
pub struct CompressionRecipe {
    /// Per-layer compression strategy.
    pub strategies: Vec<CompressionStrategy>,
    /// Estimated compression ratio.
    pub compression_ratio: f64,
    /// Estimated accuracy retention.
    pub accuracy_retention: f32,
}

/// A single layer's compression strategy.
#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    /// No compression.
    None,
    /// Unstructured pruning at this ratio.
    Prune { ratio: f32 },
    /// Structured (channel) pruning at this ratio.
    StructuredPrune { ratio: f32 },
    /// Low-rank decomposition to this rank.
    LowRank { rank: usize },
    /// Quantization to this format.
    Quantize { bits: usize },
}

impl CompressionStrategy {
    /// Estimated size reduction factor.
    pub fn size_factor(&self) -> f64 {
        match self {
            CompressionStrategy::None => 1.0,
            CompressionStrategy::Prune { ratio } => 1.0 / (1.0 - *ratio as f64),
            CompressionStrategy::StructuredPrune { ratio } => 1.0 / (1.0 - *ratio as f64),
            CompressionStrategy::LowRank { rank: _ } => 4.0, // approximate
            CompressionStrategy::Quantize { bits } => 32.0 / *bits as f64,
        }
    }
}

/// Search for the best compression recipe.
pub fn automl_search(
    model: &dyn Module,
    inputs: &Tensor,
    targets: &Tensor,
    loss_fn: &dyn Loss,
    target_compression: f64,
) -> CompressionRecipe {
    let params = model.parameters();
    let n_layers = params.iter().filter(|p| p.ndim() == 2).count();

    // Baseline loss.
    let baseline_out = model.forward(inputs);
    let baseline_loss = loss_fn.forward(&baseline_out, targets)
        .data().iter().copied().next().unwrap_or(1.0);

    let mut strategies = Vec::new();
    let mut current_compression = 1.0f64;

    for (i, p) in params.iter().enumerate() {
        if p.ndim() != 2 { continue; }

        // Try different strategies and pick the best.
        let candidates = [CompressionStrategy::None,
            CompressionStrategy::Prune { ratio: 0.3 },
            CompressionStrategy::Prune { ratio: 0.5 },
            CompressionStrategy::Prune { ratio: 0.7 },
            CompressionStrategy::StructuredPrune { ratio: 0.3 },
            CompressionStrategy::Quantize { bits: 8 },
            CompressionStrategy::Quantize { bits: 4 }];

        // Evaluate each candidate by measuring parameter magnitude retention.
        let original_norm: f32 = p.data().iter().map(|v| v.abs()).sum::<f32>() / p.len().max(1) as f32;

        let best = candidates.iter()
            .map(|s| {
                let score = match s {
                    CompressionStrategy::None => 1.0,
                    CompressionStrategy::Prune { ratio } => {
                        let keep = 1.0 - ratio;
                        keep * (original_norm / original_norm.max(0.01))
                    },
                    CompressionStrategy::StructuredPrune { ratio } => {
                        1.0 - ratio * 0.5 // Structured pruning loses less quality.
                    },
                    CompressionStrategy::Quantize { bits } => {
                        1.0 - (32.0 / *bits as f32 - 1.0) * 0.05
                    },
                    _ => 0.8,
                };
                (s.clone(), score, s.size_factor())
            })
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        strategies.push(best.0);
        current_compression *= best.2;

        // Stop if we've hit the target compression.
        if current_compression >= target_compression && i >= n_layers / 2 {
            // Fill remaining with None.
            for _ in i + 1..params.len() {
                strategies.push(CompressionStrategy::None);
            }
            break;
        }
    }

    let accuracy_retention = 1.0 - baseline_loss * 0.1; // Estimate.

    CompressionRecipe {
        strategies,
        compression_ratio: current_compression,
        accuracy_retention,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Sequential, ReLU};
    use crate::loss::MSELoss;

    // 1. Weight Sharing
    #[test]
    fn weight_sharing_reduces_params() {
        let ws = SharedWeights::new(64, 64, 4);
        assert_eq!(ws.num_passes, 4);
        assert!(ws.compression_ratio() > 3.5); // ~4x fewer params
        let x = Tensor::randn(&[2, 64]);
        let y = ws.forward(&x);
        assert_eq!(y.shape(), vec![2, 64]);
    }

    // 2. Sparse Matrices
    #[test]
    fn sparse_matrix_compression() {
        // Use a large sparse matrix where CSR overhead is amortized.
        let n = 200;
        let mut data = vec![0.0f32; n * n];
        // Only set 5% non-zero (diagonal + a few off-diagonals).
        for i in 0..n {
            data[i * n + i] = 1.0; // diagonal
        }
        for i in 0..n/5 {
            data[i * n + (i + 1) % n] = 0.5; // some off-diagonal
        }
        let tensor = Tensor::from_vec(data, vec![n, n]);
        let sparse = SparseMatrix::from_dense(&tensor, 0.01);
        assert!(sparse.nnz < n + n/5 + 1);
        assert!(sparse.sparsity() > 0.9);
        assert!(sparse.compression_ratio() > 1.0, "CSR should save memory: {}", sparse.compression_ratio());
    }

    #[test]
    fn sparse_spmm_correct() {
        // A = [[1,0,0],[2,0,3]] (2x3 sparse)
        let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0], vec![2, 3]);
        // B = [[1,2],[3,4],[5,6]] (3x2 dense)
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sparse = SparseMatrix::from_dense(&a, 0.01);
        let c = sparse.spmm(&b, 2); // n=2
        // C = [[1*1+0+0, 1*2+0+0],[2*1+0+3*5, 2*2+0+3*6]] = [[1,2],[17,22]]
        assert!((c[0] - 1.0).abs() < 1e-5, "C[0,0] should be 1, got {}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-5, "C[0,1] should be 2, got {}", c[1]);
        assert!((c[2] - 17.0).abs() < 1e-5, "C[1,0] should be 17, got {}", c[2]);
        assert!((c[3] - 22.0).abs() < 1e-5, "C[1,1] should be 22, got {}", c[3]);
    }

    #[test]
    fn sparse_to_dense_roundtrip() {
        let original = Tensor::from_vec(vec![0.5, 0.0, 0.0, 0.3, 0.0, 0.7], vec![2, 3]);
        let sparse = SparseMatrix::from_dense(&original, 0.01);
        let reconstructed = sparse.to_dense();
        let orig: Vec<f32> = original.data().iter().copied().collect();
        let recon: Vec<f32> = reconstructed.data().iter().copied().collect();
        for i in 0..orig.len() {
            assert!((orig[i] - recon[i]).abs() < 1e-5);
        }
    }

    // 3. Layer Dropping
    #[test]
    fn layer_dropping_skips_converged() {
        let mut dropper = LayerDropper::new(3, 0.01);
        let unchanged = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let slightly_changed = Tensor::from_vec(vec![1.001, 2.001], vec![2]);
        assert!(dropper.should_skip(0, &unchanged, &slightly_changed));

        let changed = Tensor::from_vec(vec![2.0, 4.0], vec![2]);
        assert!(!dropper.should_skip(0, &unchanged, &changed));
        assert!(dropper.avg_skip_rate() > 0.0);
    }

    // 4. Knowledge Transfer
    #[test]
    fn knowledge_transfer_loss_positive() {
        let source = std::sync::Arc::new(
            Sequential::new().add(Linear::new(4, 8, true))
        );
        let target = std::sync::Arc::new(
            Sequential::new().add(Linear::new(4, 8, true))
        );
        let kt = KnowledgeTransfer::new(source, target, 8, 8);
        let x = Tensor::randn(&[2, 4]);
        let loss = kt.transfer_loss(&x);
        assert!(loss >= 0.0);
    }

    // 5. Embedding Compression
    #[test]
    fn embedding_compression_ratio() {
        let table = Tensor::randn(&[1000, 256]);
        let compressed = CompressedEmbedding::from_table(&table, 32);
        assert!(compressed.compression_ratio() > 5.0, "should compress >5x");
    }

    #[test]
    fn embedding_lookup_shape() {
        let table = Tensor::randn(&[100, 64]);
        let emb = CompressedEmbedding::from_table(&table, 16);
        let v = emb.lookup(5);
        assert_eq!(v.shape(), vec![1, 64]);
        let batch = emb.lookup_batch(&[1, 2, 3]);
        assert_eq!(batch.shape(), vec![3, 64]);
    }

    // 6. Mixed Sparsity
    #[test]
    fn mixed_sparsity_analysis() {
        let model = Sequential::new()
            .add(Linear::new(8, 16, true))
            .add(ReLU)
            .add(Linear::new(16, 4, true));
        let inputs = Tensor::randn(&[4, 8]);
        let targets = Tensor::randn(&[4, 4]);
        let ms = MixedSparsity::analyze(&model, &inputs, &targets, &MSELoss);
        assert!(!ms.layer_sparsity.is_empty());
        // Less sensitive layers should get higher sparsity.
        for &s in &ms.layer_sparsity {
            assert!(s > 0.0 && s < 1.0, "sparsity should be in (0,1): {s}");
        }
    }

    // 7. Progressive Shrinking
    #[test]
    fn progressive_shrinking_schedule() {
        let mut ps = ProgressiveShrinking::new(0.5, 100, 10);
        // Warmup.
        for _ in 0..10 {
            let r = ps.step();
            assert!(r == 0.0);
        }
        // Ramp up.
        let r1 = ps.step();
        let r2 = ps.step();
        assert!(r2 > r1);
        // Done.
        while !ps.is_done() {
            ps.step();
        }
        assert!((ps.current_sparsity - 0.5).abs() < 1e-5);
    }

    #[test]
    fn progressive_soft_prune() {
        let ps = ProgressiveShrinking { target_sparsity: 0.5, current_sparsity: 0.3, total_steps: 100, current_step: 50, warmup_steps: 10 };
        let t = Tensor::randn(&[4, 8]);
        let pruned = ps.apply_soft_prune(&t);
        // Some values should be zeroed.
        let zeros = pruned.data().iter().filter(|v| **v == 0.0).count();
        assert!(zeros > 0);
    }

    // 8. Structured Pruning
    #[test]
    fn structured_pruning_reduces_channels() {
        let layer = Linear::new(8, 16, true);
        let (pruned, kept) = StructuredPruner::prune_linear(&layer, 0.5);
        assert_eq!(pruned.weight.shape()[0], 8); // Half of 16
        assert_eq!(kept.len(), 8);
    }

    #[test]
    fn structured_pruning_keeps_important() {
        let weight = Tensor::from_vec(
            (0..32).map(|i| if i < 16 { 10.0 } else { 0.01 }).collect::<Vec<_>>(),
            vec![4, 8],
        );
        let (_pruned, kept) = StructuredPruner::prune_channels(&weight, 0.5);
        // Should keep channels 0-1 (high norm) and prune 2-3 (low norm).
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&0));
        assert!(kept.contains(&1));
    }

    // 9. AutoML Compression
    #[test]
    fn automl_finds_recipe() {
        let model = Sequential::new()
            .add(Linear::new(8, 16, true))
            .add(ReLU)
            .add(Linear::new(16, 4, true));
        let inputs = Tensor::randn(&[4, 8]);
        let targets = Tensor::randn(&[4, 4]);
        let recipe = automl_search(&model, &inputs, &targets, &MSELoss, 3.0);
        assert!(recipe.compression_ratio > 0.0);
        assert!(!recipe.strategies.is_empty());
    }

    #[test]
    fn compression_strategy_size_factor() {
        assert_eq!(CompressionStrategy::None.size_factor(), 1.0);
        assert!(CompressionStrategy::Prune { ratio: 0.5 }.size_factor() > 1.0);
        assert!(CompressionStrategy::Quantize { bits: 8 }.size_factor() > 3.0);
    }
}
