//! Loss functions for training neural networks.

use crate::tensor::Tensor;

pub trait Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}

pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        let diff = prediction.sub(target);
        let squared = diff.mul(&diff);
        squared.sum()
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor {
        // Simple implementation of CE loss with logits
        // This is a placeholder for a more stable implementation
        let exp = prediction.data().mapv(|x| x.exp());
        let sum = exp.sum();
        let probs = exp / sum;
        let log_probs = probs.mapv(|x| x.ln());
        
        let target_data = target.data();
        let loss = -(target_data * log_probs).sum();
        
        Tensor::new(ndarray::arr0(loss).into_dyn(), true)
    }
}
