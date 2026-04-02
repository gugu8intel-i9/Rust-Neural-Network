//! Activation functions for neural networks.

use crate::tensor::Tensor;

/// Rectified Linear Unit activation: `max(0, x)`
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// Sigmoid activation: `1 / (1 + exp(-x))`
pub fn sigmoid(x: &Tensor) -> Tensor {
    // Sigmoid is not yet implemented in Tensor ops, adding it
    let data = x.data().mapv(|v| 1.0 / (1.0 + (-v).exp()));
    Tensor::new(data, x.0.read().unwrap().requires_grad)
}

/// Hyperbolic tangent activation: `tanh(x)`
pub fn tanh(x: &Tensor) -> Tensor {
    let data = x.data().mapv(|v| v.tanh());
    Tensor::new(data, x.0.read().unwrap().requires_grad)
}
