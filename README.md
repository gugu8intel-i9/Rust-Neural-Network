# rust-nn

A high-performance, ergonomic neural network library in **pure Rust**, with a small but
correct reverse-mode autograd engine built on top of [`ndarray`](https://crates.io/crates/ndarray)
and [`rayon`](https://crates.io/crates/rayon).

## Features

- **Tensors with autograd**: N-dimensional tensors (f32) that track a computation graph for
  reverse-mode automatic differentiation — `add`, `sub`, `mul`, `matmul`, `relu`, `reshape`,
  `transpose`, `sum`, plus fused loss ops. Broadcasting is correctly handled in the backward pass.
- **Neural network layers**: `Linear`, `Flatten`, `Dropout`, `BatchNorm1D`,
  `NormalMoE`, `FineGrainedMoE`, `Recursive`, `RNNCell`, `FakeQuantize` (QAT), DeepSeek-style
  `CSA` / `HCA`, and activation layers `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`.
- **Optimizers**: `SGD` (with momentum), `Adam`, `RMSprop`, `Muon`.
- **Loss functions** (all fully differentiable): `MSELoss`, `CrossEntropyLoss`, `BCELoss`,
  `BCEWithLogitsLoss`, `L1Loss`, `HuberLoss`.
- **Attention**: an exact, memory-efficient **FlashAttention** implementation (online-softmax
  tiling, O(seq) memory, no materialized N×N matrix, fully differentiable).
- **Mamba**: full (`Mamba`) and hybrid (`HybridMamba`) Selective State Space Models (S6) with
  input-dependent selective scan, causal conv1d, and SiLU gating — linear O(seq) complexity.
- **Diffusion**: a full **DDPM** pipeline (forward/reverse process, sinusoidal timestep
  conditioning, linear & cosine schedules, ancestral sampling).
- **Reinforcement Learning**: REINFORCE, Actor-Critic (A2C), DQN (replay + target net), and
  PPO, with a Gym-style `Environment` trait and example bandit/chain MDPs.
- **Looped Transformer**: weight-shared iterative transformer (Universal Transformer / LoopLM
  style) — O(k) params but O(k·T) effective depth, with multi-head FlashAttention, pre-norm
  LayerNorm, timestep conditioning, and optional adaptive halting (ACT).
- **Tokenizer**: a high-performance byte-level BPE tokenizer with **information-theoretic merge
  scoring** (frequency, PMI, or hybrid), flat contiguous vocab storage (memcpy decode), parallel
  training & batch encoding, regex-free pre-tokenization, and guaranteed lossless round-trip.
- **Quantization**: `FakeQuantize` for QAT, plus **RotorQuant** — block-diagonal Cl(3,0)
  Clifford-rotor decorrelation + scalar quantization for inference-time KV-cache/activation
  compression.
- **Training utilities**: `SimpleDataLoader` and a generic `Trainer`.
- **Reasoning strategies**: `ChainOfThought` (CoT), `TreeOfThoughts` (ToT), `SwiReasoning`,
  and `MarkovianRSA`.
- **Self-Improvement (RLAIF)**: `SelfImprover` loop with heuristic or reward-model `Critic` evaluations allowing models to self-train on unlabelled data.
- **Pure Rust**: no external BLAS required to build (optional BLAS backends available as Cargo features).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rust-nn = { git = "https://github.com/gugu8intel-i9/Rust-Neural-Network" }
```

## Quick Start

### Basic tensor operations

```rust
use rust_nn::tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

let sum = a.add(&b);
let product = a.mul(&b);          // element-wise
let matmul = a.matmul(&b);        // matrix multiplication
let reshaped = a.reshape(&[4]);
let transposed = a.transpose();

println!("sum = {:.4}", a.sum());
println!("mean = {:.4}", a.mean());
println!("max = {:.4}", a.max());
println!("element [0,1] = {:.4}", a.get(&[0, 1]));

// Tensors implement Display:
println!("{a}");
```

### Activations

```rust
use rust_nn::tensor::Tensor;
use rust_nn::activations::{relu, sigmoid, tanh, softmax, gelu};

let x = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
let _ = relu(&x);
let _ = sigmoid(&x);
let _ = tanh(&x);
let _ = softmax(&x);   // numerically stable softmax over the last axis
let _ = gelu(&x);
```

### Building a model

Layers take an explicit `bias: bool` flag:

```rust
use rust_nn::nn::{Sequential, Linear, ReLU, Module};
use rust_nn::tensor::Tensor;

let model = Sequential::new()
    .add(Linear::new(784, 256, true))
    .add(ReLU)
    .add(Linear::new(256, 128, true))
    .add(ReLU)
    .add(Linear::new(128, 10, true));

let input = Tensor::randn(&[32, 784]);   // batch of 32
let output = model.forward(&input);
println!("Output shape: {:?}", output.shape()); // [32, 10]
```

### Training a model

Optimizers are constructed with the model's parameters (the parameter tensors are shared by
reference with the model, so optimizer updates flow straight back into it):

```rust
use rust_nn::nn::{Sequential, Linear, ReLU, Module};
use rust_nn::optim::Adam;
use rust_nn::loss::{MSELoss, Loss};
use rust_nn::train::{SimpleDataLoader, Trainer};
use rust_nn::tensor::Tensor;
use std::sync::Arc;

let model = Arc::new(
    Sequential::new()
        .add(Linear::new(784, 256, true))
        .add(ReLU)
        .add(Linear::new(256, 10, true)),
);

let params = model.parameters();          // shared with `model`
let optimizer = Adam::new(params, 0.001);
let loss_fn = MSELoss;

let mut trainer = Trainer::new(model.clone(), optimizer, loss_fn);

let inputs = Tensor::randn(&[1000, 784]);
let targets = Tensor::randn(&[1000, 10]);

for epoch in 0..10 {
    let loader = SimpleDataLoader::new(inputs.clone(), targets.clone(), 32);
    let loss = trainer.train_epoch(loader);
    println!("epoch {epoch}: loss = {loss:.4}");
}
```

### Attention (FlashAttention)

`rust_nn::nn::flash_attention` computes **exact** scaled dot-product attention,
`softmax(Q·Kᵀ / √d) · V`, for `[batch, seq, d]` inputs. It uses the FlashAttention algorithm:
rather than building the full `seq × seq` score matrix, it streams over keys per query while
maintaining running max/sum statistics (the "online softmax"). The result is numerically
identical to standard attention, but peak memory is **O(seq)** instead of O(seq²), and it is
fully differentiable (backward recomputes attention on-chip, as in the paper).

```rust
use rust_nn::tensor::Tensor;
use rust_nn::nn::flash_attention;

let (batch, seq, d) = (2, 128, 64);
let q = Tensor::randn(&[batch, seq, d]);
let k = Tensor::randn(&[batch, seq, d]);
let v = Tensor::randn(&[batch, seq, d]);

let out = flash_attention(&q, &k, &v);      // [batch, seq, d], scaled by 1/sqrt(d)
// out.backward(); works — gradients flow to q, k, v.
```

`nn::attention(q, k, v)` is the same computation under the standard name, and
`Tensor::flash_attention(q, k, v, scale)` lets you pass a custom scale.

### Quantization (RotorQuant)

**RotorQuant** decorrelates vectors with sparse block-diagonal rotations drawn from the
Clifford algebra Cl(3,0) — the "rotor sandwich" `RxR̃` — before independently scalar-quantizing
each coordinate, then recovering the vector with the inverse rotation. The rotation
homogenizes the per-coordinate distribution so a simple uniform quantizer reaches near-optimal
distortion using only ~4 rotor parameters per 3-D block (versus a dense d×d transform). It is a
compression quantizer (round-trips with quantization error), suited to inference-time
KV-cache / activation quantization.

```rust
use rust_nn::tensor::Tensor;
use rust_nn::quant::RotorQuant;
use rust_nn::nn::Module;

let rq = RotorQuant::new(128, 4);           // 128-dim vectors, 4-bit
let kv_cache = Tensor::randn(&[8, 128]);    // 8 vectors of dim 128

let compressed = rq.quantize(&kv_cache);    // [8, 128], quantized+dequantized
// or use it as a Module:
let compressed = rq.forward(&kv_cache);
```

For training-time quantization with straight-through gradients, use `nn::FakeQuantize` instead.

### Mamba (Full & Hybrid)

**Mamba** is a Selective State Space Model (S6): it replaces attention's quadratic
sequence-mixing with a linear-time, **input-dependent** selective scan. `MambaBlock` projects
to an expanded inner dim, applies a depthwise causal conv1d + SiLU, runs the selective scan
(`h_t = Ā_t⊙h_{t-1} + B̄_t⊙u_t`, with Δ/B/C predicted from the input and diagonal `A`), gates
the output, and projects back. The scan, conv1d, softplus, and sigmoid are all fused autograd
ops with exact backward passes.

```rust
use rust_nn::mamba::Mamba;
use rust_nn::nn::Module;
use rust_nn::tensor::Tensor;

// Full Mamba: d_model=64, d_state=16, expand=2, conv kernel=4, 4 stacked blocks.
let model = Mamba::new(64, 16, 2, 4, 4);
let x = Tensor::randn(&[2, 128, 64]);   // [batch, seq, d_model]
let y = model.forward(&x);              // [2, 128, 64]
```

**Hybrid Mamba** interleaves Mamba blocks with attention blocks (à la Jamba), combining
linear-time long-context modeling with attention's precise retrieval:

```rust
use rust_nn::mamba::HybridMamba;
use rust_nn::nn::{Sequential, Linear, SiLU, Module};

let model = HybridMamba::new(64)
    .with_mamba(16, 2, 4)
    .with_mamba(16, 2, 4)
    .with_layer(Sequential::new().add(Linear::new(64, 64, true)).add(SiLU)) // attention stand-in
    .with_mamba(16, 2, 4);
```

### Diffusion Models (DDPM)

`DDPM` provides the full denoising-diffusion pipeline: a forward (noising) process
`x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`, a noise-predicting denoising network conditioned on the timestep
via a sinusoidal embedding, MSE noise-prediction training, and ancestral sampling from pure
noise. `Linear` and `Cosine` ("Improved DDPM") schedules are supported.

```rust
use rust_nn::diffusion::{DDPM, ScheduleType};
use rust_nn::optim::Adam;

let ddpm = DDPM::new(8, 64, 100, ScheduleType::Linear);   // 8-dim data, 100 steps
let mut opt = Adam::new(ddpm.parameters(), 0.01);

let x0 = rust_nn::Tensor::randn(&[32, 8]);
for step in 0..200 {
    let loss = ddpm.train_batch(&mut opt, &x0);           // one denoising training step
    if step % 50 == 0 { println!("step {step}: loss {loss:.4}"); }
}

let samples = ddpm.sample(16);                            // generate 16 new samples [16, 8]
```

### Reinforcement Learning

`rl` provides a Gym-style [`Environment`](rust_nn::rl::Environment) trait plus four classic and
modern agents, all built on the autograd engine:

- **REINFORCE** — vanilla policy gradient with a moving-average baseline.
- **Actor-Critic (A2C)** — policy gradient with a learned value-function baseline.
- **DQN** — Deep Q-Network with experience replay (`ReplayBuffer`) and a periodic target network.
- **PPO** — Proximal Policy Optimization with a clipped surrogate objective.

Agents use discrete action spaces and `Vec<f32>` observations. Action sampling is
non-differentiable; the policy-gradient agents apply the REINFORCE score-function trick
(backprop through the action's log-probability, weighted by the return/advantage).

```rust
use rust_nn::rl::{Reinforce, BanditEnv, Environment};
use rust_nn::optim::Adam;
use rust_nn::nn::Module;

let mut agent = Reinforce::new(env.observation_dim(), env.num_actions(), 32);
let mut opt = Adam::new(agent.policy.parameters(), 0.02);
let mut env = BanditEnv::new(4);                 // implement Environment for your own MDP

for episode in 0..200 {
    let reward = agent.train_episode(&mut opt, &mut env);
    if episode % 50 == 0 { println!("ep {episode}: reward {reward:.2}"); }
}
let action = agent.act(&env.reset());            // pick an action greedily/stochastically
```

The included `BanditEnv` (k-armed, with Bernoulli / deterministic / sparse reward variants) and
`ChainEnv` (a 1-D goal-reaching corridor) are ready to use and exercised by the test suite.

### Looped Transformer

A **Looped Transformer** reuses a single weight-shared transformer block across `T` loop
iterations, achieving deep computation (O(k·T) effective depth) with very few parameters
(O(k)) — as in Universal Transformers and ByteDance's Ouro-1.4B. Each iteration applies the
shared pre-norm attention + FFN block with a sinusoidal **timestep conditioning** signal and a
residual connection:

```rust
use rust_nn::looped_transformer::LoopedTransformer;
use rust_nn::nn::Module;
use rust_nn::tensor::Tensor;

// 1 shared block applied 8 times: deep computation, minimal params.
let model = LoopedTransformer::new(64, 128, 4, 256, 10, 8);
//               input  d_model heads  ff   out  loops

let x = Tensor::randn(&[2, 16, 64]);   // [batch, seq, input_dim]
let y = model.forward(&x);             // [2, 16, 10]

// With adaptive halting (ACT): stop early when confident.
let model = LoopedTransformer::new(64, 128, 4, 256, 10, 16)
    .with_adaptive_halting(0.9);
let (y, loops_used) = model.forward_with_loops(&x); // loops_used <= 16
```

A standard (non-looped) `Transformer` with independently-parameterized layers is also available
for comparison.

### Positional Encodings

A full suite of position-encoding strategies, centered on a **fused exact-gradient RoPE** (the
dominant method in modern LLMs like Llama, Qwen, and DeepSeek):

```rust
use rust_nn::position::{RoPE, CARoPE, AlibiBias};

// Standard RoPE (Llama default): rotate Q/K by position-dependent angles.
let rope = RoPE::new(64, 2048);              // head_dim=64, max_seq=2048
let q_rotated = rope.apply(&q);              // exact backward, norm-preserving

// YaRN: extend a 2048-trained model to 8192 context.
let yarn = RoPE::yarn(64, 8192, 10000.0, 4.0);

// CARoPE (novel): context-aware frequencies from token embeddings.
let carope = CARoPE::new(64, 512, 2048);      // head_dim, input_dim, max_seq
let q_rotated = carope.apply(&q, &embeddings); // input-dependent phase shifts

// ALiBi: additive distance bias on attention scores (no learned params).
let alibi = AlibiBias::new(8);                 // 8 heads
let bias = alibi.bias_matrix(seq_len);         // [heads, seq, seq]
```

### Tokenizer (BPE)

A high-performance byte-level BPE tokenizer with **information-theoretic merge scoring**. Unlike
standard BPE (frequency-only), it can merge using **PMI** (Pointwise Mutual Information) — pairs
that co-occur far more than chance predicts — yielding more semantically coherent subwords.

```rust
use rust_nn::tokenizer::{BpeTokenizer, MergeScoring};

// Train on a corpus. PMI scoring finds more meaningful merges than raw frequency.
let corpus = "the quick brown fox jumps over the lazy dog ...";
let tok = BpeTokenizer::train(corpus, 1000, MergeScoring::PMI);

let ids = tok.encode("the quick fox");        // Vec<u32>
let text = tok.decode(&ids);                  // "the quick fox" (lossless round-trip)

// Batch encode/decode in parallel (rayon):
let batch = tok.encode_batch(&["the fox".into(), "the dog".into()]);

// Analytics + offsets:
let ratio = tok.compression_ratio("the quick brown fox"); // tokens/char
let spans = tok.encode_with_offsets("hello");             // (id, byte-range) pairs
```

### Reasoning: Chain of Thought (CoT) & Tree of Thoughts (ToT)

**Chain of Thought** refines a hidden state by applying a shared "thought" transformation
repeatedly for a fixed number of steps, with residual connections for stability. The whole
chain is differentiable, so the thought layer trains end-to-end:

```rust
use rust_nn::reasoning::ChainOfThought;
use rust_nn::nn::Module;
use rust_nn::tensor::Tensor;

let cot = ChainOfThought::new(64, 5);        // 64-dim state, 5 reasoning steps
let state = Tensor::randn(&[4, 64]);
let refined = cot.forward(&state);           // [4, 64]
```

**Tree of Thoughts** explores multiple reasoning paths via beam search: at each step every beam
spawns `branching_factor` candidate thoughts, an evaluator scores them, and only the top
`beam_width` survive. This is a test-time-compute technique (the selection is non-differentiable;
gradients flow only through the returned best path):

```rust
use rust_nn::reasoning::TreeOfThoughts;
use rust_nn::nn::Module;
use rust_nn::tensor::Tensor;

let tot = TreeOfThoughts::new(64, 4, 3, 2)   // 64-dim, depth 4, branch 3, keep 2
    .with_exploration_noise(0.15);
let state = Tensor::randn(&[4, 64]);
let best = tot.forward(&state);              // [4, 64] — best reasoning path found
```

## Architecture

### `Tensor`

The core `Tensor` type (`Arc<RwLock<TensorData>>`) provides:

- Contiguous row-major memory layout and N-dimensional shapes.
- Element-wise ops (`add`/`sub`/`mul`), matrix multiplication, reductions (`sum`, `mean`, `max`,
  `min`, `argmax`), indexing (`get`), reshaping and transposition.
- Reverse-mode autograd: call `.backward()` on any scalar output and read `.grad()` on leaf tensors.
- A `Display` impl for pretty-printing.

### `Module`

```rust
pub trait Module: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn set_training(&self, _training: bool) {}
}
```

Built-in modules: `Linear`, `Flatten`, `Dropout`, `BatchNorm1D`, `NormalMoE`, `FineGrainedMoE`,
`Recursive`, `RNNCell`, `FakeQuantize`, `CSA`, `HCA`, `RotorQuant`, `Sequential`, and the
activation modules `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`. Free-function attention is
available via `nn::attention` / `nn::flash_attention`.

`Sequential::set_training` propagates training/eval mode to its layers (used by `Dropout`).

### `Optimizer`

```rust
pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}
```

| Optimizer | Constructor |
|-----------|-------------|
| `SGD`     | `SGD::new(params, lr, momentum)` |
| `Adam`    | `Adam::new(params, lr)` |
| `RMSprop` | `RMSprop::new(params, lr)` |
| `Muon`    | `Muon::new(params, lr, momentum)` |

### `Loss`

```rust
pub trait Loss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}
```

Every loss returns a scalar `Tensor`; call `.backward()` on it to populate parameter gradients.

- `MSELoss`, `CrossEntropyLoss` (logits + one-hot, numerically stable),
  `BCELoss`, `BCEWithLogitsLoss`, `L1Loss`, `HuberLoss`.

## Examples

```bash
cargo run --example basic           # tensor ops, activations, a small network
cargo run --example xor             # train an MLP to solve XOR
cargo run --example classification  # train an MLP with cross-entropy (100% on synthetic data)
cargo run --example library_usage   # end-to-end mini training loop
```

## Tests

Gradient correctness is verified against finite differences:

```bash
cargo test
```

## Performance & cross-platform notes

- `.cargo/config.toml` enables `target-cpu=native` for autovectorization (AVX2/AVX-512 on x86,
  NEON on Apple Silicon).
- Optional BLAS backends: `apple-accelerate`, `intel-mkl`, `openblas` Cargo features.
- `rayon` parallelizes batch computation across cores.
- A cross-platform GPU backend is stubbed via `to_device(Device)` (CPU/Gpu/Metal/Cuda); the
  WebGPU/WGPU execution path remains on the roadmap.

## Changelog

### 0.12.0 — Platform-specific GPU kernels (NVIDIA / AMD / Apple)

- **NVIDIA CUDA (PTX)**: a shared-memory-tiled, register-blocked FP32 GEMM in PTX assembly.
  Each block cooperatively loads tiles via `st.shared.f32`, computes with `fma.rn.f32` FMA
  instructions, and synchronizes with `bar.sync`. Register blocking: each thread accumulates 8
  output elements in registers (8:1 arithmetic-to-memory ratio).
- **Apple Silicon (Metal MSL)**: a GEMM using `simdgroup_float8x8` hardware matrix units and
  `simdgroup_multiply_accumulate` — Apple`s equivalent of Tensor Cores. Features `simdgroup_async_copy`
  for double-buffered device->threadgroup transfer that overlaps memory with computation.
- **AMD ROCm (HIP-C)**: a GEMM using LDS (Local Data Share) tiling with bank-conflict-free padded
  layouts, targeting MFMA (Matrix FMA) wave-level instructions on CDNA GPUs.
- **Unified auto-dispatch** (`detect_backend`): probes for CUDA/ROCm/Metal/WebGPU at runtime and
  selects the optimal backend. Falls back to the SIMD CPU kernel when no GPU is available.
- **Tile auto-tuning** (`TileConfig`): optimal BM/BN/BK tile sizes per architecture, with
  arithmetic-intensity and shared-memory estimates.
- **Kernel extraction** (`extract_kernels`): writes `.ptx`, `.metal`, `.hip` files for offline
  compilation via nvrtc / Metal compiler / hipRTC.
- 15 new tests validating kernel source correctness, dispatch logic, tile configs, and matmul.

### 0.11.0 — Interactive REPL + multi-source dataset loading

- **Interactive REPL** (`src/interactive.rs`): a readline-style session for building models
  layer-by-layer, loading datasets, and training with live ASCII sparkline loss curves.
  Commands: `model new/add/summary`, `data csv/hf/synthetic`, `train`, `predict`, `info`.
- **Dataset module** (`src/data.rs`): unified `Dataset` abstraction with:
  - **HuggingFace Hub loading** via the Datasets Server REST API (`/rows` endpoint) — streams
    rows in batches of 100 without downloading the whole dataset. Local caching.
  - **CSV/TSV/JSONL** loading with dependency-free parsers and automatic type inference.
  - **Kaggle** loading (requires API credentials).
  - **Synthetic generators**: `make_classification` and `make_regression`.
  - **Tensor conversion**: `to_tensor()` for numeric features, `to_token_tensor()` for text.
  - **Train/test splitting**, `head()`, and `summary()`.
- 12 new tests covering CSV parsing (quoted fields, type inference), JSONL loading, tensor
  conversion, dataset splitting, synthetic generation, sparkline rendering, and REPL commands.

### 0.10.0 — SIMD-accelerated CPU kernels

- **Cache-blocked SIMD GEMM** (`src/simd.rs`): matrix multiplication using AVX2+FMA intrinsics
  (8 × f32 per instruction, fused multiply-add), cache blocking (L1/L2-sized tiles), and
  multi-threaded parallelism across row-blocks (rayon). Runtime feature detection with scalar
  fallback on unsupported CPUs. Wired into `Tensor::matmul` — every matmul in the library now
  goes through this kernel.
- **Vectorized element-wise ops**: SIMD `add`, `mul`, `relu`, `scale`, and `sum` reduction,
  all with AVX2 dispatch and scalar fallback.
- **Innovations**: exploits all three CPU strengths simultaneously — SIMD vectorization
  (amortize per-element compute across 8-wide registers), cache blocking (keep working set in
  L1/L2 to avoid memory bandwidth stalls), and multi-core parallelism (independent row-blocks).

### 0.9.0 — Iterative autograd, GPU acceleration, model serialization

- **Iterative topological-sort backward**: replaced the recursive `backward()` with a
  non-recursive iterative engine that topo-sorts the computation graph once, then processes
  nodes with a gradient accumulation map. Eliminates stack-overflow failures on deep graphs
  (looped transformers, long training loops) and is measurably faster. Same exact gradients.
- **GPU acceleration** (`src/gpu.rs`): WGSL compute shaders for GEMM (matrix multiply) and
  element-wise add/mul via WebGPU (wgpu). Cross-platform (Vulkan/Metal/DX12). Automatic CPU
  fallback when no GPU adapter is available. Lazy initialization via `OnceLock`.
- **Model serialization** (`src/serialize.rs`): compact `rnnb` binary format for tensors and
  full models, plus **safetensors** export/import for HuggingFace/PyTorch ecosystem interop.
  Includes a dependency-free JSON parser for the safetensors header.
- All 115 tests pass (including GPU matmul correctness, serialization round-trips, safetensors
  interop, and the full existing gradient-check suite with the new iterative engine).

### 0.8.0 — Positional Encodings (RoPE / YaRN / ALiBi / CARoPE)

- **RoPE**: fused exact-gradient Rotary Position Embedding (the dominant positional encoding in
  modern LLMs). Precomputes cos/sin tables; applies via a fused autograd op with exact backward
  (transpose of the rotation). Norm-preserving, relative-position, with long-term decay.
- **YaRN**: NTK-aware frequency scaling for context-window extension (`RoPE::yarn`).
- **ALiBi**: head-dependent linear distance bias on attention scores (`AlibiBias`), with the
  geometric slope sequence from the paper. No learned parameters; excellent extrapolation.
- **CARoPE** (novel): Context-Aware RoPE where frequencies are input-dependent (generated from
  token embeddings via a learned projection), producing context-sensitive phase shifts.
- **SinusoidalPE / LearnedPE**: classic additive (absolute) position embeddings, plus a
  unified `PositionalEncoding` enum.
- 17 tests covering shape preservation, norm preservation, position-0 identity, gradient
  correctness (finite-difference), YaRN extension, ALiBi properties, CARoPE novelty, and more.

### 0.7.0 — High-performance BPE Tokenizer

- **BpeTokenizer**: a byte-level BPE tokenizer with three **pluggable merge-scoring strategies**
  — `Frequency` (classic), `PMI` (Pointwise Mutual Information, the WordPiece insight), and
  `Hybrid` (a normalized blend). PMI finds statistically meaningful merges rather than just
  frequent ones.
- **Flat contiguous vocab storage**: all token bytes live in one `Vec<u8>` with an offsets array,
  so decode is a pair of memcpy operations per token (no per-token allocation).
- **Parallel training** (rayon merge counting) and **parallel batch encode/decode**.
- **Regex-free pre-tokenizer**: a hand-written byte scanner (no `regex` dependency) with proper
  multibyte-UTF-8 grouping and lossless partition guarantees.
- APIs: `encode`, `encode_with_offsets`, `decode`, `encode_batch`, `decode_batch`,
  `count_tokens`, `compression_ratio`, `save`/`load`, special tokens.
- 17 unit tests covering round-trip (ASCII, unicode, emoji, raw bytes), all scoring modes,
  pre-tokenization partition, offsets, batch, save/load, determinism.

### 0.6.0 — Looped Transformer

- **LoopedTransformer**: a weight-shared transformer block applied recurrently for `T` loops,
  decoupling computational depth from parameter count (Universal Transformer / LoopLM style).
  Includes multi-head attention (via fused `permute` + FlashAttention), pre-norm LayerNorm,
  timestep conditioning (sinusoidal embedding), residual connections, and optional adaptive
  halting (ACT-style input-dependent compute depth). Also includes a standard `Transformer`
  (non-looped) for comparison.
- **Core autograd ops**: added `permute` (axis permutation, fully differentiable), `layer_norm`
  (fused, exact backward for input/gamma/beta), and `LayerNorm` module.
- Added `tests/looped_transformer.rs` (16 tests: permute correctness + grad check, LayerNorm
  correctness + grad check, MHA shape/differentiability, looped transformer shape/
  differentiability/learning/adaptive-halting/param-efficiency).

### 0.5.0 — Reinforcement Learning

- **RL module** (`src/rl.rs`): a Gym-style `Environment` trait plus four agents — `Reinforce`
  (policy gradient + moving-average baseline), `ActorCritic` (A2C with a learned value head),
  `Dqn` (experience replay + target network, ε-greedy), and `Ppo` (clipped surrogate objective).
  Includes `ReplayBuffer`, categorical sampling, and discounted-return computation.
- Example environments: `BanditEnv` (k-armed bandit, Bernoulli/deterministic/sparse reward
  variants) and `ChainEnv` (1-D goal-reaching corridor with a configurable step limit).
- Agents backprop per-step through the autograd engine (gradients accumulate into shared
  parameters), avoiding deep computation graphs. `ReplayBuffer` and `Dqn` are seedable for
  reproducible evaluation.
- Added `tests/rl.rs` (8 tests: all agents learn the optimal bandit arm, DQN reduces TD error,
  target-network sync, replay buffer, sampling & returns).

### 0.4.0 — Mamba (full & hybrid) & Diffusion (DDPM)

- **Mamba**: added `MambaBlock` (selective-SSM block with input-dependent Δ/B/C, diagonal `A`,
  causal conv1d, SiLU gating, in/out projections), `Mamba` (full stack of pure SSM blocks with
  residuals — linear O(seq) complexity), and `HybridMamba` (interleaves Mamba + attention blocks,
  à la Jamba). All fully differentiable.
- Core Mamba math is a fused **selective-scan** autograd op (`Tensor::selective_scan`) with an
  exact reverse-scan backward; verified against finite differences for all five inputs. Added
  supporting fused ops: `conv1d_causal`, `softplus`, and a differentiable `sigmoid`/`silu`
  (the previous `sigmoid` in activations was forward-only — it now flows gradients).
- **Linear now supports N-D inputs** (applies to the last dimension), so layers work on
  `[batch, seq, dim]` tensors, enabling sequence models like Mamba.
- **Diffusion (DDPM)**: added `NoiseSchedule` (linear & cosine), a `DenoiseNet` MLP with
  sinusoidal timestep conditioning, `DDPM` with `q_sample`, `train_batch` (MSE noise prediction),
  and `sample` (ancestral sampling from `x_T ~ N(0,I)`).
- Added `tests/selective_scan.rs` (gradient checks) and `tests/mamba_diffusion.rs` (shape,
  differentiability, loss-decrease, sampling, schedule validity).

### 0.3.1 — Chain of Thought & Tree of Thoughts

- **Chain of Thought (CoT)**: added a differentiable sequential reasoning module that refines a
  hidden state through a chain of intermediate thought steps, with optional residual connections.
  Gradients flow through the entire chain. Exposed as `reasoning::ChainOfThought`.
- **Tree of Thoughts (ToT)**: added a beam-search reasoning module that branches multiple candidate
  next-states per step, scores them with a learned evaluator, and prunes to the top-K beams.
  A test-time-compute technique (selection is non-differentiable; gradients flow through the
  returned best path). Exposed as `reasoning::TreeOfThoughts`.
- Added `tests/reasoning.rs` covering shape preservation, differentiability (CoT), and beam-search
  behavior (ToT).

### 0.3.0 — FlashAttention & RotorQuant, roadmap restored

- **FlashAttention**: added an exact, memory-efficient scaled dot-product attention built on the
  FlashAttention algorithm (online-softmax tiling → O(seq) memory, no materialized N×N matrix,
  exact gradients via on-chip recomputation). Exposed as `nn::flash_attention`,
  `nn::attention`, and `Tensor::flash_attention`. Verified against a naive attention reference
  (max diff < 1e-5) and finite-difference gradient checks.
- **RotorQuant**: added a rotation-assisted quantizer using block-diagonal Cl(3,0) Clifford
  rotors (the sparse rotor-sandwich `RxR̃`) + per-group uniform scalar quantization. Exposed as
  `quant::RotorQuant` (also a `Module`). Verified norm-preserving, exact-inverse, bounded-error.
- Restored the **Next-Generation Roadmap** section that was lost when the README was rewritten.

### 0.2.0 — bug fixes & missing features

This release makes the crate **actually compile and train**. Notable fixes:

- **Library did not compile**: removed a duplicate `#[derive(Debug)]` on `Device` and added the
  missing `Debug` impl for `TensorData`.
- **Broken broadcasting in autograd** (`Add`/`Sub`/`Mul` backward): gradients were not reduced to
  operand shape, so a `Linear` layer's **bias gradient had the wrong shape and panicked the
  optimizer** on any batched input. Broadcasting is now undone correctly in the backward pass.
- **Biases never trained**: `Linear` created its bias with `requires_grad = false`, so bias
  parameters received no gradients and were never updated. Bias is now trainable.
- **`CrossEntropyLoss` did not train**: it computed a loss value but returned a leaf tensor with no
  creator, so `backward()` never reached the weights. Replaced with a fused, numerically-stable
  softmax + cross-entropy autograd op with an exact gradient. Added `BCELoss`,
  `BCEWithLogitsLoss`, `L1Loss`, `HuberLoss` (all differentiable) the same way.
- **`FakeQuantize` did not quantize**: it computed `scale` but never applied it. It now performs
  `clamp(round(x / scale), qmin, qmax) * scale`.
- **`RNNCell` method shadowing**: the `Module` impl had confusing duplicate `forward`/`parameters`
  methods; the step function is now `RNNCell::step` and the recursion-prone `parameters` was cleaned up.
- **`Sequential::set_training` was a no-op**; it now propagates to its layers so `Dropout`
  respects training/eval mode.
- Added missing pieces referenced by docs/examples: `Tensor` `mean`/`max`/`min`/`argmax`/`get` and
  `Display`; `softmax` and `gelu` activations; `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `Dropout`,
  `BatchNorm1D` modules; full loss re-exports.
- Rewrote all four examples so they compile and run against the real API.
- Fixed `Cargo.toml` metadata: corrected the license (`AGPL-3.0-only`) and repository URL.
- Added a gradient-correctness test suite (`tests/grad_check.rs`).

## 🚀 Next-Generation Roadmap

What's done and what's planned:

- [x] **Autograd engine**: reverse-mode automatic differentiation with correct broadcasting.
- [x] **SIMD-accelerated CPU**: cache-blocked AVX2/FMA GEMM + vectorized ops.
- [x] **Native GPU kernels**: NVIDIA PTX, AMD HIP, Apple Metal with auto-dispatch.
- [x] **Iterative autograd**: non-recursive topological-sort backward (no stack overflow).
- [x] **GPU acceleration**: WebGPU compute shaders (wgpu) with CPU fallback.
- [x] **Model serialization**: binary format + safetensors interop.
- [x] **Interactive REPL**: exploratory model building + live training visualization.
- [x] **Dataset loading**: HuggingFace Hub, CSV/TSV/JSONL, Kaggle, synthetic.
- [x] **Core layers & optimizers**: Linear, Dropout, BatchNorm, MoE, RNN; SGD/Adam/RMSprop/Muon.
- [x] **Attention**: exact, memory-efficient FlashAttention + multi-head, with RoPE.
- [x] **Positional Encodings**: RoPE, YaRN, ALiBi, CARoPE, sinusoidal, learned.
- [x] **Looped Transformer**: weight-shared iterative computation (Universal Transformer style).
- [x] **State Space Models**: full & hybrid Mamba (selective SSM, linear-time).
- [x] **Diffusion**: DDPM noise schedule + denoising + sampling.
- [x] **Reinforcement Learning**: REINFORCE, Actor-Critic, DQN, PPO + environments.
- [x] **Quantization**: `FakeQuantize` (QAT) + `RotorQuant` (rotation-assisted compression).
- [x] **Reasoning strategies**: CoT, ToT, Swi-Reasoning, Markovian RSA.
- [x] **Tokenizer**: high-performance byte-level BPE with PMI scoring.
- [ ] **GPU compute backend**: execute `to_device(Device)` kernels via WebGPU/WGPU/Vulkan/Metal.
      The CPU autograd engine currently handles all real work; the GPU path is stubbed.
- [ ] **Fused FlashAttention GPU kernels**: SRAM-tiled, IO-aware CUDA/Metal implementations.
- [ ] **Full attention layers**: multi-head attention with causal masks, KV-cache, and rotary
      position embeddings (RoPE) — building on the FlashAttention primitive.
- [ ] **INT8 / FP16 inference**: end-to-end low-precision execution paths (RotorQuant is a first step).
- [ ] **Graph compilation**: operator fusion / XLA-style JIT over the autograd graph.
- [ ] **Distributed training**: multi-node training via MPI/NCCL.
- [ ] **ONNX ecosystem**: import and export models to/from ONNX.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a Pull Request.
