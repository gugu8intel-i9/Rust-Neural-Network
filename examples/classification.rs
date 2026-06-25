//! Synthetic classification example.
//!
//! Trains a small MLP with cross-entropy on a linearly-separable-ish synthetic dataset
//! and reports accuracy. Demonstrates the Trainer + optimizer + loss pipeline.

use rust_nn::loss::CrossEntropyLoss;
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::Adam;
use rust_nn::tensor::Tensor;
use rust_nn::train::{SimpleDataLoader, Trainer};
use std::sync::Arc;

fn main() {
    println!("=== Rust Neural Network - Classification Example ===\n");

    // ----- synthetic data: class-dependent feature means -----
    let n_samples = 600;
    let n_features = 12;
    let n_classes = 4;

    println!(
        "Generating synthetic data: {} samples, {} features, {} classes",
        n_samples, n_features, n_classes
    );

    let mut inputs = Vec::with_capacity(n_samples * n_features);
    let mut onehot = vec![0.0f32; n_samples * n_classes];
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        labels.push(class);
        onehot[i * n_classes + class] = 1.0;
        for j in 0..n_features {
            let mean = class as f32 * 0.8 + (j as f32) * 0.05;
            let noise = (rand::random::<f32>() - 0.5) * 0.6;
            inputs.push(mean + noise);
        }
    }

    let inputs = Tensor::from_vec(inputs, vec![n_samples, n_features]);
    let targets = Tensor::from_vec(onehot, vec![n_samples, n_classes]);

    // ----- model -----
    let model = Arc::new(
        Sequential::new()
            .add(Linear::new(n_features, 64, true))
            .add(ReLU)
            .add(Linear::new(64, 32, true))
            .add(ReLU)
            .add(Linear::new(32, n_classes, true)),
    );
    println!("Model: Linear({n_features}->64) -> ReLU -> Linear(64->32) -> ReLU -> Linear(32->{n_classes})");

    // ----- optimizer + loss + trainer -----
    let params = model.parameters();
    let optimizer = Adam::new(params, 0.05);
    let loss_fn = CrossEntropyLoss;
    let mut trainer = Trainer::new(model.clone(), optimizer, loss_fn);

    // ----- training loop -----
    println!("\nTraining...\n{}", "-".repeat(50));
    let epochs = 40;
    let batch_size = 32;
    for epoch in 0..epochs {
        let loader = SimpleDataLoader::new(inputs.clone(), targets.clone(), batch_size);
        let loss = trainer.train_epoch(loader);
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!("Epoch {:>2}: avg loss = {:.4}", epoch + 1, loss);
        }
    }
    println!("{}", "-".repeat(50));

    // ----- accuracy on the training set -----
    let logits = model.forward(&inputs);
    let data = logits.data();
    let shape = data.shape();
    let (n, c) = (shape[0], shape[1]);
    let mut correct = 0usize;
    for i in 0..n {
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..c {
            let v = data[[i, j]];
            if v > best_val {
                best_val = v;
                best = j;
            }
        }
        if best == labels[i] {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / n as f32;
    println!("\nTraining accuracy: {:.2}% ({}/{})", accuracy * 100.0, correct, n);
    println!("\n=== Example Complete ===");
}
