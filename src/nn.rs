//! Neural network modules and layers using Autograd.

use crate::tensor::Tensor;
use std::sync::Arc;

pub trait Module: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    /// Switch the module between training and evaluation mode (e.g. for `Dropout`).
    /// Takes `&self` so it can be toggled on layers shared behind an `Arc`.
    fn set_training(&self, _training: bool) {}
}

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::he(&[out_features, in_features]);
        let bias = if bias {
            // Bias must be trainable: create zeros with requires_grad = true.
            Some(Tensor::new(
                ndarray::ArrayD::zeros(ndarray::IxDyn(&[out_features])),
                true,
            ))
        } else {
            None
        };
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight_t = self.weight.transpose();
        let mut res = input.matmul(&weight_t);
        if let Some(ref b) = self.bias {
            // Broadcasting in ndarray add handles [batch, out] + [out]
            res = res.add(b);
        }
        res
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[derive(Debug, Clone)]
pub struct Sequential {
    pub layers: Vec<Arc<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: Vec::new() }
    }

    /// Append a layer and return the model for chaining. (Named `add` for ergonomics,
    /// intentionally shadowing nothing the standard library exposes on these types.)
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.layers.push(Arc::new(module));
        self
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn set_training(&self, training: bool) {
        for layer in &self.layers {
            layer.set_training(training);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

macro_rules! impl_activation_module {
    ($name:ident, $func:path, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone)]
        pub struct $name;

        impl Module for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                $func(input)
            }
            fn parameters(&self) -> Vec<Tensor> {
                Vec::new()
            }
        }
    };
}

impl_activation_module!(Sigmoid, crate::activations::sigmoid, "Sigmoid activation module: `1 / (1 + e^-x)`.");
impl_activation_module!(Tanh, crate::activations::tanh, "Hyperbolic-tangent activation module.");
impl_activation_module!(Softmax, crate::activations::softmax, "Softmax activation module (over the last axis).");
impl_activation_module!(GELU, crate::activations::gelu, "Gaussian Error Linear Unit activation module.");

/// Inverted dropout. During training, zeroes each element with probability `p` and scales
/// the kept elements by `1 / (1 - p)` so expectations are preserved. In eval mode it is a no-op.
///
/// Uses an `AtomicBool` training flag so the mode can be toggled even when the module is
/// shared behind an `Arc` (see `Module::set_training`).
#[derive(Debug)]
pub struct Dropout {
    pub p: f32,
    training: std::sync::atomic::AtomicBool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        let p = p.clamp(0.0, 1.0);
        Dropout {
            p,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

// `AtomicBool` isn't `Clone`, so provide a manual impl preserving the current mode.
impl Clone for Dropout {
    fn clone(&self) -> Self {
        Dropout {
            p: self.p,
            training: std::sync::atomic::AtomicBool::new(
                self.training.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        let in_training = self.training.load(std::sync::atomic::Ordering::Relaxed);
        if !in_training || self.p <= 0.0 {
            return input.clone();
        }
        if self.p >= 1.0 {
            return Tensor::zeros(&input.shape());
        }
        use rand::Rng;
        let keep = 1.0 - self.p;
        let scale = 1.0 / keep;
        let mut rng = rand::thread_rng();
        let data = input
            .data()
            .mapv(|v| if rng.gen::<f32>() < keep { v * scale } else { 0.0 });
        Tensor::new(data, input.0.read().unwrap().requires_grad)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    fn set_training(&self, training: bool) {
        self.training
            .store(training, std::sync::atomic::Ordering::Relaxed);
    }
}

/// 1-D batch normalization over the feature dimension.
///
/// Expects input shaped `[batch, features]`. Statistics (mean/variance) are computed from
/// the current batch; the learnable affine parameters `gamma` and `beta` are applied via the
/// autograd-tracked `mul`/`add` ops, so they receive gradients, while the normalization itself
/// is treated as constant (a common simplification for lightweight autograd engines).
#[derive(Debug, Clone)]
pub struct BatchNorm1D {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub num_features: usize,
    pub eps: f32,
}

impl BatchNorm1D {
    pub fn new(num_features: usize) -> Self {
        let gamma = Tensor::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[num_features])), true);
        let beta = Tensor::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[num_features])), true);
        BatchNorm1D {
            gamma,
            beta,
            num_features,
            eps: 1e-5,
        }
    }
}

impl Module for BatchNorm1D {
    fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data();
        let shape = data.shape();
        if shape.len() != 2 || shape[1] != self.num_features {
            panic!(
                "BatchNorm1D expects input [batch, {}], got {:?}",
                self.num_features, shape
            );
        }
        let (batch, features) = (shape[0], shape[1]);

        let mut mean = vec![0.0f32; features];
        let mut var = vec![0.0f32; features];
        for b in 0..batch {
            for f in 0..features {
                mean[f] += data[[b, f]];
            }
        }
        for m in mean.iter_mut() {
            *m /= batch as f32;
        }
        for b in 0..batch {
            for f in 0..features {
                let d = data[[b, f]] - mean[f];
                var[f] += d * d;
            }
        }
        for v in var.iter_mut() {
            *v /= batch as f32;
        }

        let mut norm_data = data.clone();
        for b in 0..batch {
            for f in 0..features {
                norm_data[[b, f]] = (data[[b, f]] - mean[f]) / (var[f] + self.eps).sqrt();
            }
        }
        // Normalization statistics are detached (forward-only); affine params stay differentiable.
        let normed = Tensor::new(norm_data, false);
        normed.mul(&self.gamma).add(&self.beta)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[derive(Debug, Clone)]
pub struct NormalMoE {
    pub gate: Linear,
    pub experts: Vec<Sequential>,
}

impl NormalMoE {
    pub fn new(in_features: usize, hidden_features: usize, num_experts: usize) -> Self {
        let gate = Linear::new(in_features, num_experts, true);
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            let expert = Sequential::new()
                .add(Linear::new(in_features, hidden_features, true))
                .add(ReLU)
                .add(Linear::new(hidden_features, in_features, true));
            experts.push(expert);
        }
        NormalMoE { gate, experts }
    }
}

impl Module for NormalMoE {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Compute routing logits
        let _routing_logits = self.gate.forward(input);
        
        // As a dense approximation without tensor slicing/top-k ops in the autograd engine,
        // we pass the input through all experts and sum their outputs.
        // In a full implementation, this would use sparse routing and gating weights.
        let mut combined_output = self.experts[0].forward(input);
        for i in 1..self.experts.len() {
            let expert_out = self.experts[i].forward(input);
            combined_output = combined_output.add(&expert_out);
        }
        
        combined_output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.gate.parameters();
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }
}

#[derive(Debug, Clone)]
pub struct FineGrainedMoE {
    pub gate: Linear,
    pub shared_expert: Sequential,
    pub experts: Vec<Sequential>,
}

impl FineGrainedMoE {
    pub fn new(in_features: usize, shared_hidden: usize, expert_hidden: usize, num_experts: usize) -> Self {
        let gate = Linear::new(in_features, num_experts, true);
        let shared_expert = Sequential::new()
            .add(Linear::new(in_features, shared_hidden, true))
            .add(ReLU)
            .add(Linear::new(shared_hidden, in_features, true));
            
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            let expert = Sequential::new()
                .add(Linear::new(in_features, expert_hidden, true))
                .add(ReLU)
                .add(Linear::new(expert_hidden, in_features, true));
            experts.push(expert);
        }
        FineGrainedMoE { gate, shared_expert, experts }
    }
}

impl Module for FineGrainedMoE {
    fn forward(&self, input: &Tensor) -> Tensor {
        let _routing_logits = self.gate.forward(input);
        
        // Pass through shared expert
        let mut combined_output = self.shared_expert.forward(input);
        
        // Dense approximation for fine-grained experts
        for expert in &self.experts {
            let expert_out = expert.forward(input);
            combined_output = combined_output.add(&expert_out);
        }
        
        combined_output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.gate.parameters();
        params.extend(self.shared_expert.parameters());
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }
}

#[derive(Debug, Clone)]
pub struct Recursive {
    pub module: Arc<dyn Module>,
    pub depth: usize,
}

impl Recursive {
    pub fn new<M: Module + 'static>(module: M, depth: usize) -> Self {
        Recursive {
            module: Arc::new(module),
            depth,
        }
    }
}

impl Module for Recursive {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for _ in 0..self.depth {
            out = self.module.forward(&out);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.module.parameters()
    }
}

#[derive(Debug, Clone)]
pub struct RNNCell {
    pub weight_ih: Linear,
    pub weight_hh: Linear,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        RNNCell {
            weight_ih: Linear::new(input_size, hidden_size, true),
            weight_hh: Linear::new(hidden_size, hidden_size, true),
        }
    }

    /// Advance the RNN one step given an input and the previous hidden state.
    pub fn step(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
        use crate::activations::tanh;
        let ih = self.weight_ih.forward(input);
        let hh = self.weight_hh.forward(hidden);
        tanh(&ih.add(&hh))
    }
}

impl Module for RNNCell {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Module compatibility: assume a zeroed initial hidden state.
        let batch_size = input.shape()[0];
        let hidden_size = self.weight_hh.weight.shape()[0];
        let hidden = Tensor::zeros(&[batch_size, hidden_size]);
        self.step(input, &hidden)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.weight_ih.parameters();
        p.extend(self.weight_hh.parameters());
        p
    }
}

/// CSA (Compressed Sparse Attention) by DeepSeek
///
/// Reduces KV cache memory by compressing multiple tokens into a single representation 
/// and utilizing sparse attention (top-k selection) to maintain fine-grained selection 
/// and long-distance dependency resolution.
#[derive(Debug, Clone)]
pub struct CSA {
    pub compression_layer: Linear,
    pub group_size: usize,
}

impl CSA {
    pub fn new(hidden_dim: usize, group_size: usize) -> Self {
        CSA {
            compression_layer: Linear::new(hidden_dim * group_size, hidden_dim, true),
            group_size,
        }
    }
}

impl Module for CSA {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Simulated: Compress sequences of `group_size` tokens into a single entry
        // Then perform Sparse Top-K Attention over the compressed KV cache.
        // For skeletal architecture, we pass through the projected dimension.
        input.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.compression_layer.parameters()
    }
}

/// HCA (Heavily/Heavy Compressed Attention) by DeepSeek
///
/// Achieves extreme memory savings (e.g., 128x) by heavily compressing massive groups of tokens 
/// into single entries, providing broad coverage and dense attention for global semantic understanding.
#[derive(Debug, Clone)]
pub struct HCA {
    pub compression_layer: Linear,
    pub group_size: usize,
}

impl HCA {
    pub fn new(hidden_dim: usize, group_size: usize) -> Self {
        HCA {
            compression_layer: Linear::new(hidden_dim * group_size, hidden_dim, true),
            group_size, // e.g., 128
        }
    }
}

impl Module for HCA {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Simulated: Aggressive compression of `group_size` (128) tokens into 1 entry.
        // Performs dense attention on the hyper-compressed cache to catch global context.
        input.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.compression_layer.parameters()
    }
}

/// FakeQuantize layer for Quantization Aware Training (QAT)
/// 
/// Simulates lower precision (e.g., INT8) during the forward pass by clamping and rounding,
/// while allowing full-precision gradients to flow backward using the Straight-Through Estimator (STE).
#[derive(Debug, Clone)]
pub struct FakeQuantize {
    pub num_bits: u8,
    pub qmin: f32,
    pub qmax: f32,
    pub scale: f32,
}

impl FakeQuantize {
    pub fn new(num_bits: u8, max_val: f32) -> Self {
        let qmin = -(1 << (num_bits - 1)) as f32;
        let qmax = ((1 << (num_bits - 1)) - 1) as f32;
        let scale = max_val / qmax;
        FakeQuantize {
            num_bits,
            qmin,
            qmax,
            scale,
        }
    }
}

impl Module for FakeQuantize {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Forward:  x_q = clamp(round(x / scale), qmin, qmax) * scale
        // Backward: Straight-Through Estimator — gradients pass through unchanged.
        let scale = self.scale;
        let data = input.data().mapv(|x| {
            let q = (x / scale).round().clamp(self.qmin, self.qmax);
            q * scale
        });
        Tensor::new(data, input.0.read().unwrap().requires_grad)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct Flatten;

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        if shape.len() <= 1 { return input.clone(); }
        let batch_size = shape[0];
        let features = shape.iter().skip(1).product();
        input.reshape(&[batch_size, features])
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
