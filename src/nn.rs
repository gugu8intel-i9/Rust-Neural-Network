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
