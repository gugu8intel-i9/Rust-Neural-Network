use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_nn::blas;
use rust_nn::fused::{fused_linear, FusedActivation};
use rust_nn::linear_attention::{linear_attention_forward, KernelKind};
use rust_nn::nn::{Dropout, Linear, Module, ReLU, Sequential};
use rust_nn::simd;
use rust_nn::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for size in [64, 128, 256, 512].iter() {
        let a = Tensor::randn(&[*size, *size]);
        let b = Tensor::randn(&[*size, *size]);
        group.bench_function(format!("{size}x{size}"), |bencher| {
            bencher.iter(|| black_box(a.matmul(black_box(&b))));
        });
    }
    group.finish();
}

fn bench_simd_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_matmul");
    for size in [64, 128, 256].iter() {
        let a: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).cos()).collect();
        let mut c = vec![0.0f32; size * size];
        group.bench_function(format!("{size}x{size}"), |bencher| {
            bencher.iter(|| {
                simd::simd_matmul(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c),
                    *size,
                    *size,
                    *size,
                )
            });
        });
    }
    group.finish();
}

fn bench_linear_vs_flash_attention(c: &mut Criterion) {
    // The headline "super fast" comparison: linear (O(N·d²)) attention vs FlashAttention (O(N²·d))
    // at increasing sequence lengths. Linear attention's cost grows linearly with N.
    let mut group = c.benchmark_group("attention");
    let (batch, d) = (2, 64);
    for seq in [128, 512, 1024].iter() {
        let q = Tensor::randn(&[batch, *seq, d]);
        let k = Tensor::randn(&[batch, *seq, d]);
        let v = Tensor::randn(&[batch, *seq, d]);
        let scale = 1.0 / (d as f32).sqrt();

        group.bench_function(format!("flash_seq{seq}"), |b| {
            b.iter(|| {
                black_box(Tensor::flash_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    scale,
                ))
            });
        });
        group.bench_function(format!("linear_seq{seq}"), |b| {
            let qf: Vec<f32> = q.data().iter().copied().collect();
            let kf: Vec<f32> = k.data().iter().copied().collect();
            let vf: Vec<f32> = v.data().iter().copied().collect();
            b.iter(|| {
                black_box(linear_attention_forward(
                    black_box(&qf),
                    black_box(&kf),
                    black_box(&vf),
                    batch,
                    *seq,
                    d,
                    scale,
                    1e-6,
                    false,
                    KernelKind::Elu,
                ))
            });
        });
    }
    group.finish();
}

fn bench_blas_sgemm(c: &mut Criterion) {
    // The BLAS engine: transpose-aware, B-panel cache-packed GEMM. The transposed-B variant is
    // the exact shape the matmul BACKWARD now uses (∂L/∂A = ∂L/∂C · Bᵀ).
    // The BLAS engine: transpose-aware, B-panel cache-packed GEMM. The transposed-B variant is
    // the exact shape the matmul BACKWARD now uses (∂L/∂A = ∂L/∂C · Bᵀ).
    let mut group = c.benchmark_group("blas_sgemm");
    for size in [64, 128, 256, 512].iter() {
        let a: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size * size).map(|i| (i as f32 * 0.001).cos()).collect();
        let mut cbuf = vec![0.0f32; size * size];
        group.bench_function(format!("blas_{size}x{size}"), |bencher| {
            bencher.iter(|| {
                blas::sgemm(
                    blas::Transpose::NoTrans,
                    blas::Transpose::NoTrans,
                    *size,
                    *size,
                    *size,
                    1.0,
                    black_box(&a),
                    *size,
                    black_box(&b),
                    *size,
                    0.0,
                    black_box(&mut cbuf),
                    *size,
                )
            });
        });
        // Transposed-B multiply (the matmul backward shape): exercises the packing path that a
        // strided kernel would choke on.
        let bt: Vec<f32> = (0..size * size)
            .map(|i| (i as f32 * 0.001).tan() * 0.1)
            .collect();
        group.bench_function(format!("blas_transB_{size}x{size}"), |bencher| {
            bencher.iter(|| {
                blas::sgemm(
                    blas::Transpose::NoTrans,
                    blas::Transpose::Trans,
                    *size,
                    *size,
                    *size,
                    1.0,
                    black_box(&a),
                    *size,
                    black_box(&bt),
                    *size,
                    0.0,
                    black_box(&mut cbuf),
                    *size,
                )
            });
        });
    }
    group.finish();
}

fn bench_dropout(c: &mut Criterion) {
    let mut group = c.benchmark_group("dropout");
    for size in [256, 1024, 4096].iter() {
        let dropout = Dropout::new(0.5);
        let x = Tensor::randn(&[32, *size]);
        group.bench_function(format!("32x{size}"), |bencher| {
            bencher.iter(|| black_box(dropout.forward(black_box(&x))));
        });
    }
    group.finish();
}

fn bench_fused_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_linear");
    let x = Tensor::randn(&[32, 256]);
    let w = Tensor::randn(&[128, 256]);
    let b = Tensor::randn(&[128]);

    group.bench_function("matmul+bias+relu", |bencher| {
        bencher.iter(|| {
            black_box(fused_linear(
                black_box(&x),
                black_box(&w),
                Some(black_box(&b)),
                FusedActivation::ReLU,
            ))
        });
    });

    // Compare against separate ops.
    group.bench_function("separate_matmul+bias+relu", |bencher| {
        bencher.iter(|| {
            let layer = Linear::new(256, 128, true);
            let out = layer.forward(black_box(&x));
            black_box(ReLU.forward(&out));
        });
    });
    group.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");
    let n = 65536;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut out = vec![0.0f32; n];

    group.bench_function("simd_add_64k", |bencher| {
        bencher.iter(|| simd::simd_add(black_box(&a), black_box(&b), black_box(&mut out)));
    });
    group.bench_function("simd_mul_64k", |bencher| {
        bencher.iter(|| simd::simd_mul(black_box(&a), black_box(&b), black_box(&mut out)));
    });
    group.bench_function("simd_relu_64k", |bencher| {
        let x: Vec<f32> = a.iter().map(|v| v - 0.5).collect();
        bencher.iter(|| simd::simd_relu(black_box(&x), black_box(&mut out)));
    });
    group.bench_function("simd_sum_64k", |bencher| {
        bencher.iter(|| black_box(simd::simd_sum(black_box(&a))));
    });
    group.finish();
}

fn bench_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward");
    let model = Sequential::new()
        .add(Linear::new(64, 128, true))
        .add(ReLU)
        .add(Linear::new(128, 64, true))
        .add(ReLU)
        .add(Linear::new(64, 10, true));

    let x = Tensor::randn(&[8, 64]);
    group.bench_function("forward+backward", |bencher| {
        bencher.iter(|| {
            let out = model.forward(black_box(&x));
            out.sum().backward();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_simd_matmul,
    bench_blas_sgemm,
    bench_linear_vs_flash_attention,
    bench_dropout,
    bench_fused_linear,
    bench_elementwise,
    bench_backward
);
criterion_main!(benches);
