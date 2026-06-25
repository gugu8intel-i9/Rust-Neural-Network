//! Example showing how to use rust-nn as a library.
//!
//! Run with: cargo run --example library_usage

use rust_nn::loss::{Loss, MSELoss};
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;

fn main() {
    println!("=== Using rust-nn as a Library ===\n");

    // 1. Tensor operations
    println!("1. Tensor Operations");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("Tensor A:\n{}", a);
    println!("Tensor B:\n{}", b);
    println!("A + B:\n{}", a.add(&b));
    println!("A @ B (matmul):\n{}\n", a.matmul(&b));

    // 2. Build a neural network
    println!("2. Neural Network");
    let model = Sequential::new()
        .add(Linear::new(10, 32, true))
        .add(ReLU)
        .add(Linear::new(32, 16, true))
        .add(ReLU)
        .add(Linear::new(16, 5, true));

    println!("Model: Linear(10->32) -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->5)");

    // 3. Forward pass
    let input = Tensor::randn(&[4, 10]); // batch of 4
    let output = model.forward(&input);

    println!("Input shape:  {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());

    // 4. Parameters
    let params = model.parameters();
    println!("Total parameter tensors: {}\n", params.len());

    // 5. Real training loop (autograd backward + optimizer step)
    println!("3. Training Example");
    let mut optimizer = Adam::new(model.parameters(), 0.01);
    let loss_fn = MSELoss;

    // Dummy regression target
    let inputs = Tensor::randn(&[32, 10]);
    let targets = Tensor::randn(&[32, 5]);

    let mut loader = rust_nn::train::SimpleDataLoader::new(inputs, targets, 8);

    let mut running = 0.0f32;
    let mut batch_count = 0;
    for (batch_x, batch_y) in &mut loader {
        optimizer.zero_grad();

        let pred = model.forward(&batch_x);
        let loss = loss_fn.forward(&pred, &batch_y);

        // Backpropagate and update parameters.
        loss.backward();
        optimizer.step();

        let loss_val = loss.data().iter().copied().next().unwrap_or(0.0);
        running += loss_val;
        batch_count += 1;
        println!("Batch {batch_count} - Loss: {:.6}", running / batch_count as f32);
    }

    println!("\n=== Library Usage Complete ===");
}
