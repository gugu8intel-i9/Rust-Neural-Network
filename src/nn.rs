//! Neural network modules and layers using Autograd.

use crate::tensor::Tensor;
use std::sync::Arc;

pub trait Module: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn set_training(&mut self, _training: bool) {}
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
            Some(Tensor::zeros(&[out_features]))
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

    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.layers.push(Arc::new(module));
        self
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

    fn set_training(&mut self, _training: bool) {
        // This is a bit tricky with Arc, we need to iterate and set if possible
        // but since we want to be high performance and simple, we'll skip for now
        // or just accept that sequential doesn't propagate set_training easily without RefCell
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
