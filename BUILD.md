# Build Options & Hardware Features

## Quick start

```bash
cargo build --release          # Optimized CPU build (AVX2/FMA auto-detected at runtime)
cargo test                     # Run all 226 tests
cargo bench --bench core_ops   # Run performance benchmarks
```

## SIMD / CPU features

The SIMD kernel uses **runtime feature detection** (`is_x86_feature_detected!`):
- AVX2 + FMA: 8× f32 per FMA instruction (Intel Haswell+, AMD Zen+)
- Scalar fallback: for older CPUs / non-x86

For maximum SIMD performance, build with native CPU targeting:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## GPU acceleration

### WebGPU (default, cross-platform)
Always compiled in. Automatically detects and uses Vulkan (Linux), Metal (macOS), or DX12 (Windows).
Falls back to the SIMD CPU kernel when no GPU is available.

### NVIDIA CUDA
The PTX kernel source is in `src/gpu_kernels.rs` (`NVIDIA_PTX_KERNEL`).
Extract and compile offline:
```bash
# From Rust code:
rust_nn::gpu_kernels::extract_kernels("kernels/")?;
# Then compile with nvrtc:
ptxas --gpu-name sm_80 kernels/sgemm.ptx -o kernels/sgemm.cubin
```

### AMD ROCm
The HIP kernel source is in `src/gpu_kernels.rs` (`AMD_HIP_KERNEL`).
Compile with hipRTC:
```bash
hiprtc --genco kernels/sgemm.hip -o kernels/sgemm.co
```

### Apple Metal
The MSL kernel source is in `src/gpu_kernels.rs` (`APPLE_MSL_KERNEL`).
Compile with the Metal compiler:
```bash
xcrun -sdk macosx metal -std=metal3.2 -c kernels/sgemm.metal -o kernels/sgemm.ir
```

## INT8 quantized inference

```rust
use rust_nn::int8::Int8Linear;

// Quantize a trained Linear layer for 4× memory reduction.
let q = Int8Linear::from_linear(&trained_linear);
let y = q.forward(&x);  // INT8 inference
```

## Fused kernels

```rust
use rust_nn::fused::{fused_linear, FusedActivation};

// Matmul + bias + ReLU in one pass (no intermediate allocation).
let y = fused_linear(&x, &weight, Some(&bias), FusedActivation::ReLU);
```

## Distributed training

```bash
# Worker 0 (rank 0):
RUST_NN_RANK=0 RUST_NN_WORLD_SIZE=4 RUST_NN_MASTER=0.0.0.0:29500 cargo run --release

# Worker 1-3 (on other machines):
RUST_NN_RANK=1 RUST_NN_WORLD_SIZE=4 RUST_NN_MASTER=<master_ip>:29500 cargo run --release
```

## Feature flags

```toml
[dependencies]
rust-nn = {
    git = "https://github.com/gugu8intel-i9/Rust-Neural-Network",
    features = ["serde"]  # Enable serialization (for save/load)
}
```

| Feature | Description |
|---------|-------------|
| `serde` | Enable serde serialization for tensors and models |
| `apple-accelerate` | Use Apple Accelerate framework for BLAS (macOS only) |
| `intel-mkl` | Use Intel MKL for BLAS |
| `openblas` | Use OpenBLAS for BLAS |

## Deterministic builds

For reproducible research, seed the RNG:
```rust
use rand::SeedableRng;
let rng = rand::rngs::StdRng::seed_from_u64(42);
```
