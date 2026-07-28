//! Profile a large training step: forward-only vs forward+backward vs full (with Adam).
//! Reveals whether large-model time is matmul-bound or overhead-bound.
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;
use std::time::Instant;

fn main() {
    let model = Sequential::new()
        .add(Linear::new(256, 512, true))
        .add(ReLU)
        .add(Linear::new(512, 512, true))
        .add(ReLU)
        .add(Linear::new(512, 256, true));
    let mut opt = Adam::new(model.parameters(), 1e-2);
    let x = Tensor::randn(&[64, 256]);
    let y = Tensor::randn(&[64, 256]);
    let steps = 200;

    // warmup
    for _ in 0..5 {
        opt.zero_grad();
        let o = model.forward(&x);
        let diff = o.sub(&y);
        let sq = diff.mul(&diff);
        sq.sum().backward();
        opt.step();
    }

    // (a) forward only
    let t0 = Instant::now();
    for _ in 0..steps {
        let _o = model.forward(&x);
    }
    let fwd = t0.elapsed().as_secs_f64() / steps as f64 * 1e3;

    // (b) forward + backward (build graph + backward)
    let t0 = Instant::now();
    for _ in 0..steps {
        opt.zero_grad();
        let o = model.forward(&x);
        let diff = o.sub(&y);
        let sq = diff.mul(&diff);
        sq.sum().backward();
    }
    let fwbwd = t0.elapsed().as_secs_f64() / steps as f64 * 1e3;

    // (c) full step (+ adam)
    let t0 = Instant::now();
    for _ in 0..steps {
        opt.zero_grad();
        let o = model.forward(&x);
        let diff = o.sub(&y);
        let sq = diff.mul(&diff);
        sq.sum().backward();
        opt.step();
    }
    let full = t0.elapsed().as_secs_f64() / steps as f64 * 1e3;

    println!("large (256-512-512-256, b64) breakdown:");
    println!("  forward only      : {fwd:.3} ms");
    println!(
        "  forward+backward  : {fwbwd:.3} ms  (backward ~{:.3} ms)",
        fwbwd - fwd
    );
    println!(
        "  full step (+Adam) : {full:.3} ms  (adam ~{:.3} ms)",
        full - fwbwd
    );
}
