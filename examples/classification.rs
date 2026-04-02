//! MNIST-like digit classification example.
//!
//! This example demonstrates training a simple neural network
//! on a synthetic digit classification task.

use rust_nn::tensor::Tensor;
use rust_nn::nn::{Module, Sequential, Linear, ReLU, Dropout};
use rust_nn::optim::{Optimizer, Adam};
use rust_nn::loss::CrossEntropyLoss;
use rust_nn::train::{SimpleDataLoader, Trainer, TrainConfig};

fn main() {
    println!("=== Rust Neural Network Example ===\n");

    // Generate synthetic classification data
    println!("Generating synthetic data...");
    let n_samples = 1000;
    let n_features = 20;
    let n_classes = 5;

    let mut inputs_data = Vec::with_capacity(n_samples * n_features);
    let mut targets_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        targets_data.push(class as f64);

        // Generate features with class-dependent means
        for j in 0..n_features {
            let mean = class as f64 * 0.5 + (j as f64) * 0.1;
            let noise = (rand::random::<f64>() - 0.5) * 0.5;
            inputs_data.push(mean + noise);
        }
    }

    let inputs = Tensor::from_vec(inputs_data, vec![n_samples, n_features]);
    let targets = Tensor::from_vec(targets_data, vec![n_samples]);

    println!("  Inputs shape: {:?}", inputs.shape());
    println!("  Targets shape: {:?}", targets.shape());
    println!("  Classes: {}", n_classes);

    // Define the model
    println!("\nBuilding model...");
    let model = Sequential::new()
        .add(Linear::new(n_features, 64))
        .add(ReLU)
        .add(Dropout::new(0.2))
        .add(Linear::new(64, 32))
        .add(ReLU)
        .add(Linear::new(32, n_classes));

    println!("  Model architecture:");
    println!("    Linear({} -> 64) -> ReLU -> Dropout(0.2)", n_features);
    println!("    Linear(64 -> 32) -> ReLU");
    println!("    Linear(32 -> {})", n_classes);

    // Create optimizer and loss
    let optimizer = Adam::new(0.01)
        .with_weight_decay(1e-4);
    let loss_fn = CrossEntropyLoss::new();

    // Create trainer
    let config = TrainConfig {
        epochs: 50,
        learning_rate: 0.01,
        batch_size: 32,
        verbose: true,
        eval_every: 1,
    };

    let mut trainer = Trainer::new(model, optimizer, loss_fn)
        .with_config(config);

    // Create data loader
    let mut train_loader = SimpleDataLoader::new(
        inputs.clone(),
        targets.clone(),
        32,
    );
    train_loader.set_shuffle(true);

    // Train the model
    println!("\nTraining...");
    println!("{}", "-".repeat(60));
    let history = trainer.fit(train_loader);

    println!("{}", "-".repeat(60));
    println!("\nTraining complete!");
    println!("  Final training loss: {:.6}", history.train_loss.last().unwrap());
    if !history.val_loss.is_empty() {
        println!("  Final validation accuracy: {:.4}", history.val_accuracy.last().unwrap());
    }

    // Make predictions on a few samples
    println!("\nMaking predictions on first 5 samples...");
    let test_inputs = inputs.reshape(&[n_samples as isize, -1]);
    let predictions = trainer.predict(&test_inputs.row(0).reshape(&[1, -1]));

    println!("  Sample predictions (logits):");
    for i in 0..n_classes {
        println!("    Class {}: {:.4}", i, predictions.get(&[0, i]));
    }

    println!("\n=== Example Complete ===");
}
