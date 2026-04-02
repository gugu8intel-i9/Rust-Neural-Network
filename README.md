# rust-nn

A high-performance, ergonomic neural network library in Rust.

## Features

- **Efficient Tensor Operations**: Multi-dimensional tensors with contiguous memory layout, broadcasting, and optimized matrix multiplication
- **Neural Network Layers**: Linear, Dropout, BatchNorm, and various activation layers
- **Optimizers**: SGD (with momentum), Adam, and RMSprop
- **Loss Functions**: MSE, Cross-Entropy, BCE, Huber, and more
- **Training Utilities**: Data loaders, learning rate schedulers, early stopping
- **Pure Rust**: No external BLAS dependencies, compiles on stable Rust

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-nn = { path = "." }  # or use a git repository
```

## Quick Start

### Basic Tensor Operations

```rust
use rust_nn::tensor::Tensor;

// Create tensors
let zeros = Tensor::zeros(&[2, 3]);
let ones = Tensor::ones(&[2, 3]);
let random = Tensor::randn(&[2, 3]);

// From data
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

// Operations
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let sum = a.add(&b);
let product = a.matmul(&b);  // Matrix multiplication
let reshaped = a.reshape(&[4]);

// Activations
use rust_nn::activations::{relu, sigmoid, softmax};
let activated = relu(&a);
```

### Building a Neural Network

```rust
use rust_nn::tensor::Tensor;
use rust_nn::nn::{Module, Sequential, Linear, ReLU, Dropout};

// Define a model using the Sequential API
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Dropout::new(0.5))
    .add(Linear::new(256, 128))
    .add(ReLU)
    .add(Linear::new(128, 10));

// Forward pass
let input = Tensor::randn(&[32, 784]);  // batch of 32
let output = model.forward(&input);
println!("Output shape: {:?}", output.shape());  // [32, 10]
```

### Training a Model

```rust
use rust_nn::tensor::Tensor;
use rust_nn::nn::{Module, Sequential, Linear, ReLU};
use rust_nn::optim::{Optimizer, Adam};
use rust_nn::loss::CrossEntropyLoss;
use rust_nn::train::{SimpleDataLoader, Trainer, TrainConfig};

// Create model
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

// Create optimizer and loss
let optimizer = Adam::new(0.001);
let loss_fn = CrossEntropyLoss::new();

// Create trainer
let config = TrainConfig {
    epochs: 10,
    batch_size: 32,
    verbose: true,
    ..Default::default()
};

let mut trainer = Trainer::new(model, optimizer, loss_fn)
    .with_config(config);

// Prepare data
let inputs = Tensor::randn(&[1000, 784]);
let targets = Tensor::from_vec((0..1000).map(|i| (i % 10) as f64).collect(), vec![1000]);
let train_loader = SimpleDataLoader::new(inputs, targets, 32);

// Train
trainer.fit(train_loader);
```

## Architecture

### Tensor

The core `Tensor` type provides:
- Contiguous row-major memory layout
- Efficient indexing and slicing
- Broadcasting support for element-wise operations
- Matrix multiplication with cache-friendly loop ordering
- Reduction operations (sum, mean, max, argmax)

### Modules (`nn`)

The `Module` trait is the foundation for all neural network components:

```rust
pub trait Module: std::fmt::Debug {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<(String, Tensor)> { ... }
    fn gradients(&self) -> Vec<(String, Tensor)> { ... }
    fn zero_grad(&mut self) { ... }
}
```

Built-in modules include:
- `Linear`: Fully connected layer
- `Dropout`: Dropout regularization
- `BatchNorm1D`: Batch normalization
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`: Activation functions
- `Sequential`: Container for stacking layers

### Optimizers (`optim`)

All optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn add_param(&mut self, name: String, param: Tensor, grad: Tensor);
}
```

Available optimizers:
- `SGD`: Stochastic Gradient Descent with optional momentum
- `Adam`: Adaptive Moment Estimation
- `RMSprop`: Root Mean Square Propagation

### Loss Functions (`loss`)

```rust
pub trait Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> f64;
    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}
```

Available losses:
- `MSELoss`: Mean Squared Error (regression)
- `CrossEntropyLoss`: Cross-Entropy (classification)
- `BCELoss`: Binary Cross-Entropy
- `BCEWithLogitsLoss`: BCE with numerically stable sigmoid
- `L1Loss`: Mean Absolute Error
- `HuberLoss`: Robust loss for outliers

## Examples

Run the included examples:

```bash
# Basic tensor operations and simple network
cargo run --example basic

# Full training example
cargo run --example classification
```

## Performance Considerations

- Tensors use contiguous row-major memory layout for cache efficiency
- Matrix multiplication uses i-k-n loop ordering for better cache utilization
- SIMD auto-vectorization is enabled in release mode
- For production use, consider integrating with BLAS libraries (e.g., `blas-src`)

## Roadmap

- [ ] Automatic differentiation (autograd)
- [ ] GPU acceleration via CUDA/Metal
- [ ] Conv2D and pooling layers
- [ ] LSTM/GRU recurrent layers
- [ ] Model serialization
- [ ] More optimizers (AdaGrad, AdamW)
- [ ] Learning rate schedulers integration

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
