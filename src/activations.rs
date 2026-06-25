//! Activation functions for neural networks.

use crate::tensor::Tensor;

/// Rectified Linear Unit activation: `max(0, x)`
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// Sigmoid activation: `1 / (1 + exp(-x))`
pub fn sigmoid(x: &Tensor) -> Tensor {
    let data = x.data().mapv(|v| 1.0 / (1.0 + (-v).exp()));
    Tensor::new(data, x.0.read().unwrap().requires_grad)
}

/// Hyperbolic tangent activation: `tanh(x)`
pub fn tanh(x: &Tensor) -> Tensor {
    let data = x.data().mapv(|v| v.tanh());
    Tensor::new(data, x.0.read().unwrap().requires_grad)
}

/// Softmax over the last axis, computed with the numerically stable
/// "subtract the max" trick so large logits can't overflow `exp`.
pub fn softmax(x: &Tensor) -> Tensor {
    let data = x.data();
    let ndim = data.ndim();
    if ndim == 0 {
        return Tensor::from_vec(vec![1.0], vec![1]);
    }

    let last = data.shape()[ndim - 1];
    if last == 0 {
        return Tensor::new(data.clone(), x.0.read().unwrap().requires_grad);
    }

    // Collapse the leading axes into one so we can process each "row" of the last axis.
    let leading = data.len() / last;
    let mut out = data
        .clone()
        .into_shape(ndarray::IxDyn(&[leading, last]))
        .expect("softmax reshape");

    for i in 0..leading {
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..last {
            row_max = row_max.max(out[[i, j]]);
        }
        let mut s = 0.0;
        for j in 0..last {
            out[[i, j]] = (out[[i, j]] - row_max).exp();
            s += out[[i, j]];
        }
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for j in 0..last {
            out[[i, j]] *= inv;
        }
    }

    let out = out.into_shape(ndarray::IxDyn(data.shape())).expect("softmax restore shape");
    Tensor::new(out, x.0.read().unwrap().requires_grad)
}

/// Gaussian Error Linear Unit approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
pub fn gelu(x: &Tensor) -> Tensor {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let data = x.data().mapv(|v| {
        let inner = c * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    });
    Tensor::new(data, x.0.read().unwrap().requires_grad)
}
