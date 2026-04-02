//! Basic tensor operations and neural network example.
//!
//! This example demonstrates core tensor operations and
//! building a simple neural network.

use rust_nn::tensor::Tensor;
use rust_nn::nn::{Module, Sequential, Linear, ReLU, Sigmoid};
use rust_nn::activations::{relu, sigmoid, softmax};

fn main() {
    println!("=== Rust Neural Network - Basic Example ===\n");

    // ==================== Tensor Operations ====================
    println!("1. Tensor Operations");
    println!("{}", "-".repeat(40));

    // Creating tensors
    let zeros = Tensor::zeros(&[2, 3]);
    println!("Zeros tensor (2x3):\n{}\n", zeros);

    let ones = Tensor::ones(&[2, 3]);
    println!("Ones tensor (2x3):\n{}\n", ones);

    let random = Tensor::randn(&[2, 3]);
    println!("Random tensor (2x3):\n{}\n", random);

    // From vector
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("Tensor A:\n{}\n", a);
    println!("Tensor B:\n{}\n", b);

    // Element-wise operations
    let sum = a.add(&b);
    println!("A + B:\n{}\n", sum);

    let product = a.mul(&b);
    println!("A * B (element-wise):\n{}\n", product);

    // Matrix multiplication
    let matmul = a.matmul(&b);
    println!("A @ B (matrix multiply):\n{}\n", matmul);

    // Reshape
    let flat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let reshaped = flat.reshape(&[3, 2]);
    println!("Reshaped from [2,3] to [3,2]:\n{}\n", reshaped);

    // Transpose
    let transposed = a.transpose();
    println!("A transposed:\n{}\n", transposed);

    // Reduction operations
    println!("A sum: {:.4}", a.sum());
    println!("A mean: {:.4}", a.mean());
    println!("A max: {:.4}", a.max());

    // ==================== Activation Functions ====================
    println!("\n2. Activation Functions");
    println!("{}", "-".repeat(40));

    let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    println!("Input: {:?}", x.data());

    let relu_out = relu(&x);
    println!("ReLU:  {:?}", relu_out.data());

    let sigmoid_out = sigmoid(&x);
    println!("Sigmoid: {:?}", sigmoid_out.data());

    let softmax_input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let softmax_out = softmax(&softmax_input);
    println!("Softmax([1,2,3]): {:?}", softmax_out.data());
    println!("  (sum = {:.6})", softmax_out.sum());

    // ==================== Neural Network ====================
    println!("\n3. Neural Network");
    println!("{}", "-".repeat(40));

    // Build a simple network
    let model = Sequential::new()
        .add(Linear::new(10, 32))
        .add(ReLU)
        .add(Linear::new(32, 16))
        .add(ReLU)
        .add(Linear::new(16, 5));

    println!("Model: Linear(10->32) -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->5)");

    // Forward pass
    let batch_size = 4;
    let input = Tensor::randn(&[batch_size, 10]);
    let output = model.forward(&input);

    println!("Input shape: {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("\nOutput (first sample):");
    for i in 0..5 {
        println!("  Class {}: {:.4}", i, output.get(&[0, i]));
    }

    // Get model parameters
    let params = model.parameters();
    println!("\nModel parameters:");
    for (name, param) in &params {
        println!("  {}: {:?}", name, param.shape());
    }

    // ==================== Broadcasting ====================
    println!("\n4. Broadcasting");
    println!("{}", "-".repeat(40));

    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::from_vec(vec![0.1, 0.2, 0.3], vec![3]);

    println!("Matrix (2x3):\n{}", matrix);
    println!("Bias (3,):\n{}", bias);
    println!("Matrix + Bias (broadcasted):\n{}", matrix.add(&bias));

    println!("\n=== Example Complete ===");
}
