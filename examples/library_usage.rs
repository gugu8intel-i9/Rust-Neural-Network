//! Example showing how to use rust-nn as a library.
//!
//! Run with: cargo run --example library_usage

use rust_nn::tensor::Tensor;
use rust_nn::nn::{Module, Sequential, Linear, ReLU};
use rust_nn::optim::{Optimizer, Adam};
use rust_nn::loss::{Loss, MSELoss};

fn main() {
    println!("=== Using rust-nn as a Library ===\n");

    // 1. Create tensor operations
    println!("1. Tensor Operations");
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    println!("Tensor A:\n{:?}", a);
    println!("Tensor B:\n{:?}", b);
    println!("A + B:\n{:?}", a.add(&b));
    println!("A @ B (matmul):\n{:?}", a.matmul(&b));

    // 2. Build a neural network
    println!("\n2. Neural Network");
    let model = Sequential::new()
        .add(Linear::new(10, 32, true))
        .add(ReLU)
        .add(Linear::new(32, 16, true))
        .add(ReLU)
        .add(Linear::new(16, 5, true));

    println!("Model: Linear(10→32) → ReLU → Linear(32→16) → ReLU → Linear(16→5)");

    // 3. Forward pass
    let input = Tensor::randn(&[4, 10]);  // batch of 4
    let output = model.forward(&input);
    
    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());

    // 4. Get parameters
    let params = model.parameters();
    println!("Total parameters: {}", params.len());

    // 5. Training loop example
    println!("\n3. Training Example");
    let mut optimizer = Adam::new(params, 0.001);
    let loss_fn = MSELoss;

    // Create some dummy data
    let inputs = Tensor::randn(&[32, 10]);
    let targets = Tensor::randn(&[32, 5]);
    
    let mut loader = rust_nn::train::SimpleDataLoader::new(inputs, targets, 8);
    
    // Single training epoch
    let mut total_loss = 0.0;
    let mut batch_count = 0;
    
    for (batch_x, batch_y) in &mut loader {
        optimizer.zero_grad();
        
        let pred = model.forward(&batch_x);
        let loss = loss_fn.forward(&pred, &batch_y);
        
        // Note: backward() requires autograd support
        // For now, we just show the forward pass
        
        total_loss += loss.data().first().copied().unwrap_or(0.0);
        batch_count += 1;
        
        println!("Batch {} - Loss: {:.6}", batch_count, total_loss / batch_count as f32);
    }

    println!("\n=== Library Usage Complete ===");
}
