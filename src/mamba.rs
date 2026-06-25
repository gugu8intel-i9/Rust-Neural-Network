//! Mamba: Selective State Space Models (S6).
//!
//! Implements the Mamba block (selective SSM + gated SiLU path + causal conv1d), a **full**
//! Mamba stack (pure SSM blocks with residuals), and a **hybrid** Mamba that interleaves Mamba
//! blocks with attention blocks (à la Jamba). All blocks are fully differentiable via the
//! autograd engine — the selective scan, conv1d, softplus, and sigmoid are fused ops with exact
//! backward passes.
//!
//! Core S6 recurrence: `h_t = Ā_t⊙h_{t-1} + B̄_t⊙u_t`, `y_t = C_t·h_t`, where Δ, B, C are
//! input-dependent (the "selection" mechanism) and `Ā = exp(ΔA)`, `B̄ = ΔB`.

use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// A single Mamba (selective-SSM) block.
///
/// Projects the input to an expanded inner dimension, applies a depthwise causal conv1d + SiLU,
/// runs the selective scan with input-dependent Δ/B/C, gates the output, and projects back.
#[derive(Debug, Clone)]
pub struct MambaBlock {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub kernel_size: usize,
    /// Projects the input to the SSM branch (`x`).
    pub in_proj_x: Linear,
    /// Projects the input to the gate branch (`z`).
    pub in_proj_z: Linear,
    /// Depthwise causal conv1d weights, `[d_inner, kernel_size]`.
    pub conv_weight: Tensor,
    /// Diagonal SSM state-transition parameters `[d_inner, d_state]` (negative).
    pub a: Tensor,
    /// Projects to the input-dependent step size Δ.
    pub proj_delta: Linear,
    /// Projects to the input-dependent B vector.
    pub proj_b: Linear,
    /// Projects to the input-dependent C vector.
    pub proj_c: Linear,
    /// Projects the gated output back to `d_model`.
    pub out_proj: Linear,
}

impl MambaBlock {
    /// Create a Mamba block.
    ///
    /// - `d_model`: model dimension (input/output).
    /// - `d_state`: SSM latent state size per channel.
    /// - `expand`: inner dimension = `expand * d_model`.
    /// - `kernel_size`: causal conv1d kernel width.
    pub fn new(d_model: usize, d_state: usize, expand: usize, kernel_size: usize) -> Self {
        let d_inner = expand * d_model;
        // Initialize A as negative values (HiPPO-like decay); a in [d_inner, d_state].
        let a_data: Vec<f32> = (0..(d_inner * d_state))
            .map(|i| -(((i % d_state) + 1) as f32) * 0.5)
            .collect();
        let a = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[d_inner, d_state]), a_data).unwrap(), true);
        // Conv1d weights initialized to ~1 (so the conv is close to identity initially).
        let mut conv_data = vec![0.0f32; d_inner * kernel_size];
        for c in 0..d_inner {
            conv_data[c * kernel_size + kernel_size - 1] = 1.0; // identity (last tap)
        }
        let conv_weight =
            Tensor::new(ArrayD::from_shape_vec(IxDyn(&[d_inner, kernel_size]), conv_data).unwrap(), true);

        MambaBlock {
            d_model,
            d_inner,
            d_state,
            kernel_size,
            in_proj_x: Linear::new(d_model, d_inner, true),
            in_proj_z: Linear::new(d_model, d_inner, true),
            conv_weight,
            a,
            proj_delta: Linear::new(d_inner, d_inner, true),
            proj_b: Linear::new(d_inner, d_state, true),
            proj_c: Linear::new(d_inner, d_state, true),
            out_proj: Linear::new(d_inner, d_model, true),
        }
    }
}

impl Module for MambaBlock {
    fn forward(&self, input: &Tensor) -> Tensor {
        // x, z branches
        let x = self.in_proj_x.forward(input); // [b, s, d_inner]
        let z = self.in_proj_z.forward(input); // [b, s, d_inner]

        // Causal conv1d + SiLU on the SSM branch.
        let x_conv = Tensor::conv1d_causal(&x, &self.conv_weight);
        let x_act = x_conv.silu();

        // Input-dependent SSM parameters.
        let delta = self.proj_delta.forward(&x_act).softplus(); // Δ > 0
        let b_vec = self.proj_b.forward(&x_act); // [b, s, d_state]
        let c_vec = self.proj_c.forward(&x_act); // [b, s, d_state]

        // Selective scan.
        let y = Tensor::selective_scan(&delta, &b_vec, &c_vec, &x_act, &self.a); // [b, s, d_inner]

        // Gate: y * silu(z)
        let gate = z.silu();
        let y_gated = Tensor::mul(&y, &gate);

        // Output projection.
        self.out_proj.forward(&y_gated)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.in_proj_x.parameters();
        p.extend(self.in_proj_z.parameters());
        p.push(self.conv_weight.clone());
        p.push(self.a.clone());
        p.extend(self.proj_delta.parameters());
        p.extend(self.proj_b.parameters());
        p.extend(self.proj_c.parameters());
        p.extend(self.out_proj.parameters());
        p
    }
}

/// **Full Mamba**: a stack of Mamba blocks with residual connections. Pure selective-SSM
/// sequence mixing (no attention), giving linear O(seq) complexity.
#[derive(Debug, Clone)]
pub struct Mamba {
    pub blocks: Vec<Arc<dyn Module>>,
    pub d_model: usize,
}

impl Mamba {
    /// `num_layers` Mamba blocks, each `d_state`-wide with the given expansion and conv kernel.
    pub fn new(d_model: usize, d_state: usize, expand: usize, kernel_size: usize, num_layers: usize) -> Self {
        let blocks: Vec<Arc<dyn Module>> = (0..num_layers)
            .map(|_| Arc::new(MambaBlock::new(d_model, d_state, expand, kernel_size)) as Arc<dyn Module>)
            .collect();
        Mamba { blocks, d_model }
    }
}

impl Module for Mamba {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for block in &self.blocks {
            let out = block.forward(&x);
            x = x.add(&out); // residual connection
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p
    }
}

/// **Hybrid Mamba**: interleaves Mamba (selective-SSM) blocks with attention blocks, combining
/// Mamba's efficient linear-time long-context modeling with attention's precise retrieval
/// (as in the Jamba architecture).
///
/// Constructed from a user-provided list of blocks in their desired order, e.g. M, M, M, A, ...
#[derive(Debug, Clone)]
pub struct HybridMamba {
    pub blocks: Vec<Arc<dyn Module>>,
    pub d_model: usize,
}

impl HybridMamba {
    pub fn new(d_model: usize) -> Self {
        HybridMamba { blocks: Vec::new(), d_model }
    }

    /// Append a block (typically `MambaBlock` or a model-wrapped attention layer) and return self.
    pub fn with_block<M: Module + 'static>(mut self, block: M) -> Self {
        self.blocks.push(Arc::new(block));
        self
    }

    /// Convenience: append a Mamba block.
    pub fn with_mamba(mut self, d_state: usize, expand: usize, kernel_size: usize) -> Self {
        self.blocks.push(Arc::new(MambaBlock::new(self.d_model, d_state, expand, kernel_size)));
        self
    }

    /// Convenience: append an arbitrary `Module` (e.g. a `Sequential` wrapping attention + MLP).
    pub fn with_layer<M: Module + 'static>(mut self, layer: M) -> Self {
        self.blocks.push(Arc::new(layer));
        self
    }
}

impl Module for HybridMamba {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for block in &self.blocks {
            let out = block.forward(&x);
            x = x.add(&out); // residual
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p
    }
}
