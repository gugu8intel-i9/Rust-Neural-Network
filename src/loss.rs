//! Loss functions for training neural networks.
//!
//! Every loss below returns a scalar `Tensor` whose gradient is tracked by the autograd
//! engine, so `loss.backward()` flows all the way back to the model parameters. The more
//! complex losses (cross-entropy, BCE, L1, Huber) are implemented as fused ops with exact
//! analytic gradients and numerically stable forward passes.

use crate::tensor::Tensor;

/// Trait implemented by all loss functions.
pub trait Loss {
    /// Compute the (scalar) loss tensor. Calling `.backward()` on the result propagates
    /// gradients to the prediction/inputs.
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}

/// Mean Squared Error: `mean((prediction - target)^2)`.
pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target);
        let squared = diff.mul(&diff);
        squared.sum()
    }
}

/// Softmax + cross-entropy loss. Expects logits shaped `[batch, classes]` and a one-hot
/// (or soft) `target` of the same shape. Numerically stable and fully differentiable.
pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        prediction.cross_entropy_logits(target)
    }
}

/// Binary cross-entropy computed from logits (numerically stable). `prediction` holds raw
/// logits and `target` holds {0, 1} labels of the same shape.
pub struct BCEWithLogitsLoss;

impl Loss for BCEWithLogitsLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        prediction.bce_with_logits(target)
    }
}

/// Binary cross-entropy computed from probabilities in `(0, 1)`.
pub struct BCELoss;

impl Loss for BCELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        prediction.bce(target)
    }
}

/// Mean Absolute Error: `mean(|prediction - target|)`.
pub struct L1Loss;

impl Loss for L1Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        prediction.l1_loss(target)
    }
}

/// Huber (smooth-L1) loss, robust to outliers. `delta` is the transition point between the
/// quadratic and linear regions.
pub struct HuberLoss {
    pub delta: f32,
}

impl Default for HuberLoss {
    fn default() -> Self {
        HuberLoss { delta: 1.0 }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        prediction.huber_loss(target, self.delta)
    }
}
