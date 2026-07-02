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
pub mod quant;
pub mod mamba;
pub mod diffusion;
pub mod rl;
pub mod self_improve;
pub mod looped_transformer;
pub mod tokenizer;
pub mod position;
pub mod serialize;
pub mod gpu;
pub mod simd;

// Re-export main types for convenient access
pub use tensor::Tensor;
pub use activations::{relu, sigmoid, tanh, softmax, gelu};
pub use nn::{
    Module, Sequential, Linear, ReLU, Sigmoid, Tanh, Softmax, GELU, Flatten, Dropout,
    BatchNorm1D, LayerNorm, NormalMoE, FineGrainedMoE, Recursive, RNNCell, FakeQuantize, CSA, HCA,
    attention, flash_attention,
};
pub use reasoning::{SwiReasoning, MarkovianRSA, ChainOfThought, TreeOfThoughts};
pub use optim::{Optimizer, SGD, Adam, RMSprop, Muon};
pub use loss::{Loss, MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, L1Loss, HuberLoss};
pub use train::{SimpleDataLoader, Trainer};
pub use quant::{Rotor, RotorQuant};
pub use mamba::{MambaBlock, Mamba, HybridMamba};
pub use diffusion::{NoiseSchedule, DenoiseNet, DDPM, ScheduleType, sinusoidal_embedding};
pub use looped_transformer::{LoopedTransformer, Transformer, TransformerBlock, MultiHeadAttention};
pub use tokenizer::{BpeTokenizer, MergeScoring};
pub use position::{RoPE, CARoPE, AlibiBias, SinusoidalPE, LearnedPE, PositionalEncoding};
pub use serialize::{serialize, deserialize, save_model, load_model, save_model_named, safetensors_export, safetensors_import};
pub use gpu::{gpu_matmul, gpu_add, gpu_mul, has_gpu, GpuBackend};
pub use simd::{simd_matmul, simd_add, simd_mul, simd_relu, simd_scale, simd_sum, simd_features};
pub use rl::{
    Environment, Reinforce, ActorCritic, Dqn, Ppo, ReplayBuffer, Transition,
    BanditEnv, ChainEnv, sample_categorical, discounted_returns,
};
pub use self_improve::{Critic, SelfImprover};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
