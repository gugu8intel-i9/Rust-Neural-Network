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

        SimpleDataLoader {
            inputs: input_list,
            targets: target_list,
            batch_size,
            current_index: 0,
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
        
        // Collate batch
        let mut input_batch = Vec::new();
        let mut target_batch = Vec::new();
        
        for i in self.current_index..end {
            input_batch.push(self.inputs[i].data().clone());
            target_batch.push(self.targets[i].data().clone());
        }

        // Use ndarray stack
        let inputs = ndarray::stack(ndarray::Axis(0), &input_batch.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let targets = ndarray::stack(ndarray::Axis(0), &target_batch.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();

        self.current_index = end;

        Some((
            Tensor::new(inputs.into_dyn(), false),
            Tensor::new(targets.into_dyn(), false),
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
