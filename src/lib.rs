// Production lint configuration.
#![allow(missing_docs)] // TODO: add docs in a dedicated PR
#![allow(clippy::unwrap_in_result)] // TODO: eliminate all unwraps in Result-returning functions
#![allow(clippy::panic)] // TODO: replace panics with Result in production paths
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

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

pub mod error;
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
pub mod data;
pub mod interactive;
pub mod gpu_kernels;
pub mod distributed;
pub mod int8;
pub mod fused;
pub mod offload;
pub mod finetune;
pub mod quantize;
pub mod ternary;
pub mod distill;
pub mod grpo;
pub mod compression;
pub mod gui;

// Re-export main types for convenient access
pub use error::{RustNnError, Result};
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
pub use data::{
    Dataset, Column, Credentials, DatasetBuilder, DatasetListing,
    load_csv, load_tsv, load_jsonl, load_huggingface, load_kaggle,
    load_huggingface_auth, load_kaggle_auth,
    search_huggingface, search_kaggle, format_listings,
    make_classification, make_regression,
};
pub use interactive::{run_repl, Session};
pub use gpu_kernels::{
    GpuBackendKind, TileConfig, kernel_matmul, kernel_matmul_with_backend,
    detect_backend, active_backend, set_backend, kernel_source, extract_kernels,
    backend_report, NVIDIA_PTX_KERNEL, APPLE_MSL_KERNEL, AMD_HIP_KERNEL,
};
pub use int8::{Int8Weights, Int8Linear};
pub use fused::{fused_linear, FusedActivation, sparse_topk_route};
pub use compression::{
    SharedWeights, SparseMatrix, LayerDropper, KnowledgeTransfer,
    CompressedEmbedding, MixedSparsity, ProgressiveShrinking,
    StructuredPruner, CompressionRecipe, CompressionStrategy, automl_search,
};
pub use grpo::{
    GrpoConfig, GrpoGroup, GrpoTrainer, GrpoStats,
    RewardModel, RewardScore, RewardWeights, RewardDimension,
    CoEvolutionTrainer, CoEvolutionStats, AdversarialEpisode,
    RepoGraph, FileNode, StructureEdge, StructureEdgeType, parse_rust_file,
};
pub use distill::{Distiller, DistillConfig, DistillResult, ProgressiveDistiller};
pub use ternary::{TernaryTensor, TernaryLinear, TernaryModel, ternarize};
pub use quantize::{QuantFormat, QuantizedTensor, QuantizedModel, QuantizedLinear, quantize};
pub use finetune::{LoraAdapter, LrSchedule, FastTrainer, FastTrainConfig, TrainPoint};
pub use offload::{
    MemoryTier, OffloadConfig, TieredStore, TieredTensor, SsdTensor, OffloadModel,
};
pub use distributed::{
    DistributedConfig, DistributedWorker, Message, MessageType,
    ring_all_reduce_simulated, average_gradients, flatten_gradients,
    unflatten_gradients, sync_gradients, clip_gradients,
    send_message, recv_message,
};
pub use gui::{ModelDashboard, TrainingDashboard, tensor_heatmap_html, launch, full_dashboard};
pub use rl::{
    Environment, Reinforce, ActorCritic, Dqn, Ppo, ReplayBuffer, Transition,
    BanditEnv, ChainEnv, sample_categorical, discounted_returns,
};
pub use self_improve::{Critic, SelfImprover};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
