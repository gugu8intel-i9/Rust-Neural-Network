//! Looped Transformer: weight-shared iterative computation.
//!
//! A single (shallow) transformer block with **shared parameters** is applied recurrently for
//! `T` loop iterations, decoupling computational depth from parameter count — O(k) parameters
//! but O(k·T) effective depth. This is the core idea behind Universal Transformers (Dehghani
//! et al., 2019), ALBERT, and ByteDance's Ouro-1.4B (LoopLM).
//!
//! The update rule at each loop step `t`:
//! ```text
//! h^(t+1) = h^(t) + block_θ( h^(t) + time_embed(t) )
//! ```
//! where `block_θ` is the same pre-norm attention + FFN block every iteration.
//!
//! Features for high performance:
//!   - **FlashAttention** (memory-efficient, exact, O(seq) memory) for the attention sublayer.
//!   - **Multi-head attention** via fused `permute` + `flash_attention`.
//!   - **Pre-norm LayerNorm** (differentiable) for stability across deep recursion.
//!   - **Timestep conditioning** via sinusoidal embeddings so the shared block can specialize
//!     its behavior per loop iteration.
//!   - **Residual connections** across every loop for gradient flow.
//!   - Optional **adaptive halting** (ACT-style) for input-dependent compute depth.

use crate::nn::{GELU, LayerNorm, Linear, Module, Sequential};
use crate::tensor::Tensor;
use std::sync::Arc;

/// Multi-head self-attention using the fused FlashAttention kernel.
///
/// Projects the input to Q/K/V, splits into `num_heads` heads, applies exact memory-efficient
/// attention per head, then recombines via an output projection. Uses the fused `permute` op
/// for the head-axis rearrangement (fully differentiable).
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        );
        let head_dim = d_model / num_heads;
        MultiHeadAttention {
            q_proj: Linear::new(d_model, d_model, true),
            k_proj: Linear::new(d_model, d_model, true),
            v_proj: Linear::new(d_model, d_model, true),
            out_proj: Linear::new(d_model, d_model, true),
            num_heads,
            head_dim,
            d_model,
        }
    }

    /// Split `[batch, seq, d_model]` into `[batch*num_heads, seq, head_dim]` for per-head attention.
    fn split_heads(&self, t: &Tensor, batch: usize, seq: usize) -> Tensor {
        // [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
        let reshaped = t.reshape(&[batch, seq, self.num_heads, self.head_dim]);
        // -> [batch, num_heads, seq, head_dim]
        let permuted = reshaped.permute(&[0, 2, 1, 3]);
        // -> [batch * num_heads, seq, head_dim]
        permuted.reshape(&[batch * self.num_heads, seq, self.head_dim])
    }

    /// Inverse of [`split_heads`]: `[batch*num_heads, seq, head_dim]` → `[batch, seq, d_model]`.
    fn merge_heads(&self, t: &Tensor, batch: usize, seq: usize) -> Tensor {
        // [batch*num_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        let reshaped = t.reshape(&[batch, self.num_heads, seq, self.head_dim]);
        // -> [batch, seq, num_heads, head_dim]
        let permuted = reshaped.permute(&[0, 2, 1, 3]);
        // -> [batch, seq, d_model]
        permuted.reshape(&[batch, seq, self.d_model])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert!(shape.len() == 3, "MultiHeadAttention expects [batch, seq, d_model]");
        let (batch, seq, _) = (shape[0], shape[1], shape[2]);

        let q = self.split_heads(&self.q_proj.forward(input), batch, seq);
        let k = self.split_heads(&self.k_proj.forward(input), batch, seq);
        let v = self.split_heads(&self.v_proj.forward(input), batch, seq);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = Tensor::flash_attention(&q, &k, &v, scale);

        let merged = self.merge_heads(&attn, batch, seq);
        self.out_proj.forward(&merged)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

/// A pre-norm transformer block: `h + Attention(LN(h))` then `h + FFN(LN(h))`.
///
/// This is the **shared** unit that gets applied repeatedly in a LoopedTransformer. Pre-norm
/// (LayerNorm before the sublayer) is critical for stability in deep recursion.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ff: Sequential,
    pub d_model: usize,
    pub ff_dim: usize,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, ff_dim: usize) -> Self {
        let ff = Sequential::new()
            .add(Linear::new(d_model, ff_dim, true))
            .add(GELU)
            .add(Linear::new(ff_dim, d_model, true));
        TransformerBlock {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, num_heads),
            ln2: LayerNorm::new(d_model),
            ff,
            d_model,
            ff_dim,
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Pre-norm attention sublayer + residual.
        let normed = self.ln1.forward(input);
        let attn_out = self.attn.forward(&normed);
        let h = input.add(&attn_out);

        // Pre-norm FFN sublayer + residual.
        let normed2 = self.ln2.forward(&h);
        let ff_out = self.ff.forward(&normed2);
        h.add(&ff_out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ff.parameters());
        p
    }
}

/// Sinusoidal timestep embedding (reused from the diffusion module).
fn timestep_embedding(step: usize, d_model: usize) -> Tensor {
    crate::diffusion::sinusoidal_embedding(step as f32, d_model)
}

/// **Looped Transformer**: a weight-shared transformer block applied recurrently for `num_loops`
/// iterations, with timestep conditioning and residual connections.
///
/// This achieves deep computational power (O(k·T) effective depth) with very few parameters
/// (O(k)), as in Universal Transformers and ByteDance's Ouro/LoopLM. The shared block is
/// conditioned on the loop step via a sinusoidal embedding so it can specialize per iteration.
///
/// Optionally supports **adaptive halting** (ACT): a learned halting unit predicts a stop
/// probability at each step, enabling input-dependent compute depth at inference time.
#[derive(Debug, Clone)]
pub struct LoopedTransformer {
    /// Input projection: maps input features to `d_model`.
    pub embed: Linear,
    /// The shared transformer block (same weights every loop).
    pub block: TransformerBlock,
    /// Final LayerNorm before the output head.
    pub ln_final: LayerNorm,
    /// Output projection head.
    pub head: Linear,
    /// Projects the timestep embedding to `d_model` for conditioning.
    pub time_proj: Sequential,
    /// Learned halting unit (predicts stop probability per step); only used if `adaptive`.
    pub halt_unit: Linear,
    /// Number of loop iterations (computational depth).
    pub num_loops: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Whether to use adaptive halting (ACT) at inference.
    pub adaptive: bool,
    /// Threshold for the cumulative halting probability (ACT).
    pub halt_threshold: f32,
}

impl LoopedTransformer {
    /// Create a Looped Transformer.
    ///
    /// - `input_dim`: dimensionality of the input features (last axis).
    /// - `d_model`: internal model dimension.
    /// - `num_heads`: number of attention heads (must divide `d_model`).
    /// - `ff_dim`: feed-forward hidden dimension.
    /// - `output_dim`: dimensionality of the output.
    /// - `num_loops`: number of times to apply the shared block (computational depth).
    pub fn new(
        input_dim: usize,
        d_model: usize,
        num_heads: usize,
        ff_dim: usize,
        output_dim: usize,
        num_loops: usize,
    ) -> Self {
        let embed = Linear::new(input_dim, d_model, true);
        let block = TransformerBlock::new(d_model, num_heads, ff_dim);
        let ln_final = LayerNorm::new(d_model);
        let head = Linear::new(d_model, output_dim, true);
        let time_proj = Sequential::new()
            .add(Linear::new(d_model, d_model, true))
            .add(GELU)
            .add(Linear::new(d_model, d_model, true));
        let halt_unit = Linear::new(d_model, 1, true);

        LoopedTransformer {
            embed,
            block,
            ln_final,
            head,
            time_proj,
            halt_unit,
            num_loops,
            d_model,
            adaptive: false,
            halt_threshold: 0.9,
        }
    }

    /// Enable adaptive halting (ACT) for inference. At each loop step, the halting unit's
    /// sigmoid output is accumulated; when it exceeds `threshold`, the loop stops early.
    pub fn with_adaptive_halting(mut self, threshold: f32) -> Self {
        self.adaptive = true;
        self.halt_threshold = threshold;
        self
    }

    /// Run a forward pass, returning the output and the number of loops actually used
    /// (may be < `num_loops` if adaptive halting triggers).
    pub fn forward_with_loops(&self, input: &Tensor) -> (Tensor, usize) {
        let mut h = self.embed.forward(input);

        let mut loops_used = self.num_loops;
        if self.adaptive {
            let mut cumulative = 0.0f32;
            for t in 0..self.num_loops {
                // Timestep conditioning.
                let temb = timestep_embedding(t, self.d_model);
                let tcond = self.time_proj.forward(&temb).reshape(&[self.d_model]);
                let conditioned = h.add(&tcond);

                // Shared block + residual.
                let block_out = self.block.forward(&conditioned);
                h = conditioned.add(&block_out);

                // Halting probability (inference-only; non-differentiable decision).
                let halt_logit = self.halt_unit.forward(&h);
                let p = 1.0 / (1.0 + (-halt_logit.data().iter().copied().next().unwrap_or(0.0)).exp());
                cumulative += p;
                if cumulative >= self.halt_threshold {
                    loops_used = t + 1;
                    break;
                }
            }
        } else {
            for t in 0..self.num_loops {
                // Timestep conditioning: inject the loop-step embedding into the state.
                let temb = timestep_embedding(t, self.d_model);
                let tcond = self.time_proj.forward(&temb).reshape(&[self.d_model]);
                let conditioned = h.add(&tcond);

                // Shared block + residual.
                let block_out = self.block.forward(&conditioned);
                h = conditioned.add(&block_out);
            }
        }

        let h = self.ln_final.forward(&h);
        (self.head.forward(&h), loops_used)
    }
}

impl Module for LoopedTransformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_with_loops(input).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.embed.parameters();
        p.extend(self.block.parameters());
        p.extend(self.ln_final.parameters());
        p.extend(self.head.parameters());
        p.extend(self.time_proj.parameters());
        if self.adaptive {
            p.extend(self.halt_unit.parameters());
        }
        p
    }
}

/// A **standard** (non-looped) transformer: a stack of independently-parameterized blocks.
/// Provided for comparison with [`LoopedTransformer`]. Uses the same block architecture
/// (pre-norm MHA + FFN) but each layer has unique weights.
#[derive(Debug, Clone)]
pub struct Transformer {
    pub embed: Linear,
    pub blocks: Vec<Arc<dyn Module>>,
    pub ln_final: LayerNorm,
    pub head: Linear,
}

impl Transformer {
    pub fn new(
        input_dim: usize,
        d_model: usize,
        num_heads: usize,
        ff_dim: usize,
        output_dim: usize,
        num_layers: usize,
    ) -> Self {
        let embed = Linear::new(input_dim, d_model, true);
        let blocks: Vec<Arc<dyn Module>> = (0..num_layers)
            .map(|_| Arc::new(TransformerBlock::new(d_model, num_heads, ff_dim)) as Arc<dyn Module>)
            .collect();
        Transformer {
            embed,
            blocks,
            ln_final: LayerNorm::new(d_model),
            head: Linear::new(d_model, output_dim, true),
        }
    }
}

impl Module for Transformer {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut h = self.embed.forward(input);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        let h = self.ln_final.forward(&h);
        self.head.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.embed.parameters();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_final.parameters());
        p.extend(self.head.parameters());
        p
    }
}
