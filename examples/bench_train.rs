//! Honest head-to-head vs PyTorch (eager, CPU, single-thread): train the same small MLPs and
//! time per step. Run with `RAYON_NUM_THREADS=1` to match a single-threaded PyTorch run.
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;
use std::time::Instant;

fn bench(d: usize, h: usize, out: usize, batch: usize, steps: usize, label: &str) {
    let model = Sequential::new()
        .add(Linear::new(d, h, true))
        .add(ReLU)
        .add(Linear::new(h, h, true))
        .add(ReLU)
        .add(Linear::new(h, out, true));
    let mut opt = Adam::new(model.parameters(), 1e-2);
    let x = Tensor::randn(&[batch, d]);
    let y = Tensor::randn(&[batch, out]);

    let mut step = || {
        opt.zero_grad();
        let o = model.forward(&x);
        let diff = o.sub(&y);
        let sq = diff.mul(&diff);
        let loss = sq.sum();
        loss.backward();
        opt.step();
    };
    for _ in 0..5 {
        step();
    }
    let t0 = Instant::now();
    for _ in 0..steps {
        step();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{label}: d={d} h={h} out={out} batch={batch} steps={steps}  total={:.2}ms  per_step={:.3}ms",
        dt * 1e3,
        dt / steps as f64 * 1e3
    );
}

fn main() {
    bench(16, 32, 4, 8, 2000, "rust_tiny");
    bench(64, 128, 10, 32, 1000, "rust_small");
    bench(128, 256, 10, 32, 500, "rust_medium");
}
