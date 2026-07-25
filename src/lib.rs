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

pub mod activations;
pub mod blas;
pub mod compression;
pub mod data;
pub mod diffusion;
pub mod distill;
pub mod distributed;
pub mod error;
pub mod finetune;
pub mod fused;
pub mod gpu;
pub mod gpu_kernels;
pub mod grpo;
pub mod gui;
pub mod int8;
pub mod interactive;
pub mod linear_attention;
pub mod looped_transformer;
pub mod loss;
pub mod mamba;
pub mod nn;
pub mod offload;
pub mod optim;
pub mod position;
pub mod quant;
pub mod quantize;
pub mod reasoning;
pub mod rl;
pub mod self_improve;
pub mod serialize;
pub mod simd;
pub mod tensor;
pub mod ternary;
pub mod tokenizer;
pub mod train;

// Re-export main types for convenient access
pub use activations::{gelu, relu, sigmoid, softmax, tanh};
pub use blas::{
    gemm_strassen, matmul as blas_matmul, saxpy, scopy, sdot, sgemm, sgemv, snrm2, sscal,
    BlasBackend, NativeBackend, Transpose, GEMM_STRASSEN_THRESHOLD,
};
pub use compression::{
    automl_search, CompressedEmbedding, CompressionRecipe, CompressionStrategy, KnowledgeTransfer,
    LayerDropper, MixedSparsity, ProgressiveShrinking, SharedWeights, SparseMatrix,
    StructuredPruner,
};
pub use data::{
    format_listings, load_csv, load_huggingface, load_huggingface_auth, load_jsonl, load_kaggle,
    load_kaggle_auth, load_tsv, make_classification, make_regression, search_huggingface,
    search_kaggle, Column, Credentials, Dataset, DatasetBuilder, DatasetListing,
};
pub use diffusion::{sinusoidal_embedding, DenoiseNet, NoiseSchedule, ScheduleType, DDPM};
pub use distill::{DistillConfig, DistillResult, Distiller, ProgressiveDistiller};
pub use distributed::{
    average_gradients, clip_gradients, flatten_gradients, recv_message, ring_all_reduce_simulated,
    send_message, sync_gradients, unflatten_gradients, DistributedConfig, DistributedWorker,
    Message, MessageType,
};
pub use error::{Result, RustNnError};
pub use finetune::{FastTrainConfig, FastTrainer, LoraAdapter, LrSchedule, TrainPoint};
pub use fused::{fused_linear, sparse_topk_route, FusedActivation};
pub use gpu::{gpu_add, gpu_matmul, gpu_mul, has_gpu, GpuBackend};
pub use gpu_kernels::{
    active_backend, backend_report, detect_backend, extract_kernels, kernel_matmul,
    kernel_matmul_with_backend, kernel_source, set_backend, GpuBackendKind, TileConfig,
    AMD_HIP_KERNEL, APPLE_MSL_KERNEL, NVIDIA_PTX_KERNEL,
};
pub use grpo::{
    parse_rust_file, AdversarialEpisode, CoEvolutionStats, CoEvolutionTrainer, FileNode,
    GrpoConfig, GrpoGroup, GrpoStats, GrpoTrainer, RepoGraph, RewardDimension, RewardModel,
    RewardScore, RewardWeights, StructureEdge, StructureEdgeType,
};
pub use gui::{full_dashboard, launch, tensor_heatmap_html, ModelDashboard, TrainingDashboard};
pub use int8::{Int8Linear, Int8Weights};
pub use interactive::{run_repl, Session};
pub use linear_attention::{KernelKind, LinearAttention as LinearAttentionLayer};
pub use looped_transformer::{
    AttentionKind, LoopedTransformer, MultiHeadAttention, Transformer, TransformerBlock,
};
pub use loss::{BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, HuberLoss, L1Loss, Loss, MSELoss};
pub use mamba::{HybridMamba, Mamba, MambaBlock};
pub use nn::{
    attention, flash_attention, BatchNorm1D, Dropout, FakeQuantize, FineGrainedMoE, Flatten,
    LayerNorm, Linear, Module, NormalMoE, RNNCell, ReLU, Recursive, Sequential, Sigmoid, Softmax,
    Tanh, CSA, GELU, HCA,
};
pub use offload::{MemoryTier, OffloadConfig, OffloadModel, SsdTensor, TieredStore, TieredTensor};
pub use optim::{Adam, Muon, Optimizer, RMSprop, SGD};
pub use position::{AlibiBias, CARoPE, LearnedPE, PositionalEncoding, RoPE, SinusoidalPE};
pub use quant::{Rotor, RotorQuant};
pub use quantize::{quantize, QuantFormat, QuantizedLinear, QuantizedModel, QuantizedTensor};
pub use reasoning::{ChainOfThought, MarkovianRSA, SwiReasoning, TreeOfThoughts};
pub use rl::{
    discounted_returns, sample_categorical, ActorCritic, BanditEnv, ChainEnv, Dqn, Environment,
    Ppo, Reinforce, ReplayBuffer, Transition,
};
pub use self_improve::{Critic, SelfImprover};
pub use serialize::{
    deserialize, load_model, safetensors_export, safetensors_import, save_model, save_model_named,
    serialize,
};
pub use simd::{simd_add, simd_features, simd_matmul, simd_mul, simd_relu, simd_scale, simd_sum};
pub use tensor::Tensor;
pub use ternary::{ternarize, TernaryLinear, TernaryModel, TernaryTensor};
pub use tokenizer::{BpeTokenizer, MergeScoring};
pub use train::{SimpleDataLoader, Trainer};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
