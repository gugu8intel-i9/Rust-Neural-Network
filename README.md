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
- **Quantization**: `FakeQuantize` for QAT, plus **RotorQuant** — block-diagonal Cl(3,0)
  Clifford-rotor decorrelation + scalar quantization for inference-time KV-cache/activation
  compression.
- **Training utilities**: `SimpleDataLoader` and a generic `Trainer`.
- **Reasoning strategies**: `SwiReasoning` and `MarkovianRSA`.
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
- [x] **Core layers & optimizers**: Linear, Dropout, BatchNorm, MoE, RNN; SGD/Adam/RMSprop/Muon.
- [x] **Attention**: exact, memory-efficient FlashAttention (this release).
- [x] **Quantization**: `FakeQuantize` (QAT) + `RotorQuant` (rotation-assisted compression).
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
