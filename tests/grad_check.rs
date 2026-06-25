//! Gradient-correctness and behavior tests for the autograd engine.
//!
//! Each loss is checked against a finite-difference approximation, and the broadcasting
//! fix (bias gradients) plus a tiny XOR training run are exercised.

use ndarray::{ArrayD, IxDyn};
use rust_nn::loss::{BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, HuberLoss, L1Loss, Loss, MSELoss};
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;
use rust_nn::train::{SimpleDataLoader, Trainer};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(),
        true,
    )
}

/// Compare the autograd gradient of `loss_fn(leaf)` against a central finite difference.
fn check(name: &str, base: &[f32], shape: &[usize], loss_fn: impl Fn(&Tensor) -> Tensor) {
    let eps = 1e-3f32;
    let n = base.len();

    // Analytic gradient via autograd.
    let t = leaf(base, shape);
    let loss = loss_fn(&t);
    loss.backward();
    let analytic: Vec<f32> = t.grad().expect("no grad").iter().copied().collect();
    assert_eq!(analytic.len(), n, "{name}: grad length mismatch");

    // Numeric gradient via central finite differences.
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let mut hi = base.to_vec();
        let mut lo = base.to_vec();
        hi[i] += eps;
        lo[i] -= eps;
        let l_hi = loss_fn(&leaf(&hi, shape)).data().iter().copied().next().unwrap();
        let l_lo = loss_fn(&leaf(&lo, shape)).data().iter().copied().next().unwrap();
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("{name:<18}: max |analytic - numeric| = {max_diff:.2e}");
    assert!(max_diff < 1e-2, "{name}: gradient mismatch (max diff {max_diff:.2e})");
}

#[test]
fn mse_gradient_matches_numeric() {
    let pred = [0.3, -0.7, 1.2, 0.0];
    let target = leaf(&[0.0, 1.0, -1.0, 0.5], &[4]);
    check("MSELoss", &pred, &[4], |p| MSELoss.forward(p, &target));
}

#[test]
fn cross_entropy_gradient_matches_numeric() {
    let logits = vec![0.8, -0.5, 0.1, -0.2, 1.3, 0.4];
    let target = leaf(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    check("CrossEntropy", &logits, &[2, 3], |p| {
        CrossEntropyLoss.forward(p, &target)
    });
}

#[test]
fn bce_with_logits_gradient_matches_numeric() {
    let logits = vec![0.9, -0.4, 0.2, 1.1];
    let target = leaf(&[1.0, 0.0, 1.0, 0.0], &[4]);
    check("BCEWithLogits", &logits, &[4], |p| BCEWithLogitsLoss.forward(p, &target));
}

#[test]
fn bce_gradient_matches_numeric() {
    let probs = vec![0.7, 0.2, 0.6, 0.9];
    let target = leaf(&[1.0, 0.0, 1.0, 0.0], &[4]);
    check("BCE", &probs, &[4], |p| BCELoss.forward(p, &target));
}

#[test]
fn l1_gradient_matches_numeric() {
    let pred = [0.3, -0.7, 1.2, -0.1];
    let target = leaf(&[0.0, 1.0, -1.0, 0.5], &[4]);
    check("L1Loss", &pred, &[4], |p| L1Loss.forward(p, &target));
}

#[test]
fn huber_gradient_matches_numeric() {
    let pred = [0.3, -0.7, 1.2, -0.1];
    let target = leaf(&[0.0, 1.0, -1.0, 0.5], &[4]);
    check("HuberLoss", &pred, &[4], |p| HuberLoss::default().forward(p, &target));
}

#[test]
fn matmul_backward_matches_numeric() {
    // A [1x2] @ B [2x1] summed -> scalar; check grad of B.
    let b = [1.5, -0.5];
    let a = leaf(&[0.4, -0.3], &[1, 2]);
    check("MatMul", &b, &[2, 1], |b| a.matmul(b).sum());
}

#[test]
fn bias_gradient_uses_unbroadcast() {
    // Linear with bias over a batch must produce a bias gradient shaped [out] (not [batch, out]),
    // and an optimizer step must not panic.
    let layer = Linear::new(3, 4, true);
    let x = leaf(&[0.1, -0.2, 0.3, 0.9, 0.0, -0.5, -0.4, 0.2, 0.7], &[3, 3]);
    let target = leaf(&[0.5f32; 12], &[3, 4]);

    let out = layer.forward(&x);
    let loss = MSELoss.forward(&out, &target);
    loss.backward();

    let bias = layer.bias.as_ref().expect("bias should exist");
    let grad = bias.grad().expect("bias grad missing");
    assert_eq!(grad.shape(), &[4], "bias grad must be [out], got {:?}", grad.shape());

    let params = layer.parameters();
    let mut opt = Adam::new(params, 0.01);
    opt.step(); // must not panic
}

#[test]
fn xor_learns() {
    let inputs = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let targets = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    let model = std::sync::Arc::new(
        Sequential::new()
            .add(Linear::new(2, 16, true))
            .add(ReLU)
            .add(Linear::new(16, 1, true)),
    );

    let params = model.parameters();
    let optimizer = Adam::new(params, 0.05);
    let loss_fn = MSELoss;
    let mut trainer = Trainer::new(model.clone(), optimizer, loss_fn);

    let mut first = f32::INFINITY;
    let mut last = 0.0;
    for epoch in 0..400 {
        let loader = SimpleDataLoader::new(inputs.clone(), targets.clone(), 4);
        let loss = trainer.train_epoch(loader);
        if epoch == 0 {
            first = first.min(loss);
        }
        last = loss;
    }
    println!("XOR loss: first={first:.4} last={last:.4}");
    assert!(last < first, "loss should decrease ({last:.4} >= {first:.4})");
    assert!(last < 0.1, "XOR should converge below 0.1, got {last:.4}");
}
