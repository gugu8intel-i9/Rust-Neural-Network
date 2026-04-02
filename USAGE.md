# Using rust-nn as a Library

This document shows how to use the `rust-nn` library in your own projects.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-nn = { path = "../rust-nn" }  # Adjust path as needed
```

Or from a git repository:

```toml
[dependencies]
rust-nn = { git = "https://github.com/yourusername/rust-nn" }
```

## Basic Usage

### Tensor Operations

```rust
use rust_nn::Tensor;

fn main() {
    // Create tensors
    let zeros = Tensor::zeros(&[2, 3]);
    let ones = Tensor::ones(&[2, 3]);
    let random = Tensor::randn(&[2, 3]);
    
    // From vector data
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Operations
    let sum = a.add(&b);
    let product = a.mul(&b);  // element-wise
    let matmul = a.matmul(&b);  // matrix multiplication
    
    // Reshape and transpose
    let reshaped = a.reshape(&[4]);
    let transposed = a.transpose();
    
    // Reductions
    let total = a.sum();
    let average = a.mean();
    let maximum = a.max();
}
```

### Activation Functions

```rust
use rust_nn::{Tensor, relu, sigmoid, tanh};

fn main() {
    let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    
    let relu_out = relu(&x);
    let sigmoid_out = sigmoid(&x);
    let tanh_out = tanh(&x);
}
```

### Building Neural Networks

```rust
use rust_nn::{Tensor, Module, Sequential, Linear, ReLU};

fn main() {
    // Build a multi-layer perceptron
    let model = Sequential::new()
        .add(Linear::new(784, 256, true))  // input: 784, output: 256, with bias
        .add(ReLU)
        .add(Linear::new(256, 128, true))
        .add(ReLU)
        .add(Linear::new(128, 10, true));  // output: 10 classes
    
    // Forward pass
    let input = Tensor::randn(&[32, 784]);  // batch of 32 samples
    let output = model.forward(&input);
    
    println!("Output shape: {:?}", output.shape());  // [32, 10]
    
    // Access model parameters
    let params = model.parameters();
    println!("Total parameters: {}", params.len());
}
```

### Training a Model

```rust
use rust_nn::{Tensor, Module, Sequential, Linear, ReLU};
use rust_nn::{Adam, MSELoss, SimpleDataLoader, Trainer};
use std::sync::Arc;

fn main() {
    // Create model
    let model = Arc::new(
        Sequential::new()
            .add(Linear::new(784, 256, true))
            .add(ReLU)
            .add(Linear::new(256, 10, true))
    );
    
    // Create optimizer with model parameters
    let params = model.parameters();
    let mut optimizer = Adam::new(params, 0.001);
    
    // Loss function
    let loss_fn = MSELoss;
    
    // Create trainer
    let mut trainer = Trainer::new(model, optimizer, loss_fn);
    
    // Prepare data
    let inputs = Tensor::randn(&[1000, 784]);
    let targets = Tensor::randn(&[1000, 10]);
    let train_loader = SimpleDataLoader::new(inputs, targets, 32);
    
    // Training loop (manual example)
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        for (batch_x, batch_y) in &mut train_loader {
            // Forward pass, compute loss, backward, step
            // (Implementation depends on autograd support)
        }
        
        println!("Epoch {} complete", epoch);
    }
}
```

## API Reference

### Core Types

- `Tensor` - N-dimensional array with GPU-like operations
- `Module` - Trait for neural network components
- `Optimizer` - Trait for optimization algorithms
- `Loss` - Trait for loss functions

### Available Modules

- `Sequential` - Container for stacking layers
- `Linear` - Fully connected layer
- `ReLU` - ReLU activation
- `Flatten` - Flatten multi-dimensional input

### Available Optimizers

- `SGD` - Stochastic Gradient Descent with momentum
- `Adam` - Adaptive Moment Estimation

### Available Loss Functions

- `MSELoss` - Mean Squared Error
- `CrossEntropyLoss` - Cross-Entropy Loss

## Examples

Run the included examples:

```bash
# Library usage example
cargo run --example library_usage

# Basic tensor operations
cargo run --example basic
```

## Notes

- Tensors use `RwLock` for interior mutability
- The library uses `ndarray` for efficient N-dimensional arrays
- All operations are CPU-based (no GPU support yet)
- Automatic differentiation is a work in progress
