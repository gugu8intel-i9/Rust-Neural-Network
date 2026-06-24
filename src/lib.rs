//! # rust-nn
//!
//! A high-performance, ergonomic neural network library in Rust.
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! rust-nn = { path = "../rust-nn" }
//! ```
//!
//! ## Example
//!
//! ```rust
//! use rust_nn::tensor::Tensor;
//! use rust_nn::nn::{Module, Sequential, Linear, ReLU};
//!
//! // Create a simple neural network
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256, true))
//!     .add(ReLU)
//!     .add(Linear::new(256, 10, true));
//!
//! // Forward pass
//! let input = Tensor::randn(&[32, 784]);
//! let output = model.forward(&input);
//! ```
//!
//! ## Features
//!
//! - **Tensor Operations**: N-dimensional arrays with broadcasting
//! - **Neural Network Layers**: Linear, Flatten, activations
//! - **Optimizers**: SGD, Adam
//! - **Loss Functions**: MSE, Cross-Entropy
//! - **Training Utilities**: Data loaders and trainers

pub mod tensor;
pub mod activations;
pub mod nn;
pub mod optim;
pub mod loss;
pub mod train;
pub mod reasoning;

// Re-export main types for convenient access
pub use tensor::Tensor;
pub use activations::{relu, sigmoid, tanh};
pub use nn::{Module, Sequential, Linear, ReLU, Flatten, FakeQuantize, CSA, HCA};
pub use reasoning::{SwiReasoning, MarkovianRSA};
pub use optim::{Optimizer, SGD, Adam, RMSprop, Muon};
pub use loss::{Loss, MSELoss, CrossEntropyLoss};
pub use train::{SimpleDataLoader, Trainer};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
