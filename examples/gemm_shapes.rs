use rust_nn::blas::{sgemm, Transpose};
use std::hint::black_box;
use std::time::Instant;

fn bench(m: usize, n: usize, k: usize) {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c = vec![0.0f32; m * n];
    for _ in 0..3 {
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            black_box(&a),
            k,
            black_box(&b),
            n,
            0.0,
            black_box(&mut c),
            n,
        );
    }
    let iters = if m <= 128 { 500 } else { 100 };
    let mut acc = 0.0f32;
    let t0 = Instant::now();
    for _ in 0..iters {
        sgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            m,
            n,
            k,
            1.0,
            black_box(&a),
            k,
            black_box(&b),
            n,
            0.0,
            black_box(&mut c),
            n,
        );
        acc += black_box(c.iter().sum::<f32>());
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = 2.0 * (m * n * k) as f64 / dt / 1e9;
    println!("sgemm m={m:4} n={n} k={k}: {dt:.4}ms  {gflops:.1} GFLOP/s  (cksum {acc:.0})");
}

fn main() {
    for &(m, n, k) in &[
        (64, 512, 512),
        (512, 512, 512),
        (32, 512, 512),
        (128, 512, 512),
    ] {
        bench(m, n, k);
    }
}
