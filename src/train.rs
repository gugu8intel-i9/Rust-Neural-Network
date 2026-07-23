//! Training utilities and abstractions.

use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optim::Optimizer;
use crate::loss::Loss;

pub struct SimpleDataLoader {
    inputs: Vec<Tensor>,
    targets: Vec<Tensor>,
    batch_size: usize,
    current_index: usize,
    /// Pre-allocated batch buffers to avoid per-sample allocation.
    input_batch_buf: Vec<f32>,
    target_batch_buf: Vec<f32>,
    input_feature_dim: usize,
    target_feature_dim: usize,
}

impl SimpleDataLoader {
    pub fn new(inputs: Tensor, targets: Tensor, batch_size: usize) -> Self {
        let n_samples = inputs.shape()[0];
        let mut input_list = Vec::new();
        let mut target_list = Vec::new();
        
        for i in 0..n_samples {
            // Slice the input and target tensors
            let input_slice = inputs.data().index_axis(ndarray::Axis(0), i).to_owned().into_dyn();
            let target_slice = targets.data().index_axis(ndarray::Axis(0), i).to_owned().into_dyn();
            
            input_list.push(Tensor::new(input_slice, false));
            target_list.push(Tensor::new(target_slice, false));
        }

        let input_feature_dim = if n_samples > 0 { input_list[0].len() } else { 0 };
        let target_feature_dim = if n_samples > 0 { target_list[0].len() } else { 0 };
        SimpleDataLoader {
            inputs: input_list,
            targets: target_list,
            batch_size,
            current_index: 0,
            input_batch_buf: Vec::with_capacity(batch_size * input_feature_dim),
            target_batch_buf: Vec::with_capacity(batch_size * target_feature_dim),
            input_feature_dim,
            target_feature_dim,
        }
    }
}

impl Iterator for SimpleDataLoader {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.inputs.len() {
            return None;
        }

        let end = (self.current_index + self.batch_size).min(self.inputs.len());
        let batch_len = end - self.current_index;

        // Reuse pre-allocated buffers — avoid per-sample Vec allocation.
        self.input_batch_buf.clear();
        self.target_batch_buf.clear();
        self.input_batch_buf.reserve(batch_len * self.input_feature_dim);
        self.target_batch_buf.reserve(batch_len * self.target_feature_dim);

        for i in self.current_index..end {
            let idata = self.inputs[i].data();
            self.input_batch_buf.extend(idata.iter().copied());
            let tdata = self.targets[i].data();
            self.target_batch_buf.extend(tdata.iter().copied());
        }

        self.current_index = end;

        let inputs = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[batch_len, self.input_feature_dim]),
            std::mem::take(&mut self.input_batch_buf),
        ).unwrap();
        let targets = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[batch_len, self.target_feature_dim]),
            std::mem::take(&mut self.target_batch_buf),
        ).unwrap();

        Some((
            Tensor::new(inputs, false),
            Tensor::new(targets, false),
        ))
    }
}

pub struct Trainer<O: Optimizer, L: Loss> {
    model: Arc<dyn Module>,
    optimizer: O,
    loss_fn: L,
}

use std::sync::Arc;

impl<O: Optimizer, L: Loss> Trainer<O, L> {
    pub fn new(model: Arc<dyn Module>, optimizer: O, loss_fn: L) -> Self {
        Trainer {
            model,
            optimizer,
            loss_fn,
        }
    }

    pub fn train_epoch(&mut self, mut loader: SimpleDataLoader) -> f32 {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for (inputs, targets) in &mut loader {
            self.optimizer.zero_grad();
            
            let outputs = self.model.forward(&inputs);
            let loss = self.loss_fn.forward(&outputs, &targets);
            
            loss.backward();
            self.optimizer.step();

            total_loss += loss.data().first().cloned().unwrap_or(0.0);
            n_batches += 1;
        }

        total_loss / n_batches as f32
    }
}
