use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_nn::tensor::Tensor;
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::nn::{Dropout};
use rust_nn::simd;
use rust_nn::fused::{fused_linear, FusedActivation};

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
            bencher.iter(|| simd::simd_matmul(black_box(&a), black_box(&b), black_box(&mut c), *size, *size, *size));
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
        bencher.iter(|| black_box(fused_linear(black_box(&x), black_box(&w), Some(black_box(&b)), FusedActivation::ReLU)));
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

criterion_group!(benches, bench_matmul, bench_simd_matmul, bench_dropout, bench_fused_linear, bench_elementwise, bench_backward);
criterion_main!(benches);
