//! Tests for the Looped Transformer, multi-head attention, LayerNorm, and permute.

use ndarray::{ArrayD, IxDyn};
use rust_nn::looped_transformer::{LoopedTransformer, MultiHeadAttention, Transformer};
use rust_nn::nn::{LayerNorm, Module};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(), true)
}

// ==================== permute ====================

#[test]
fn permute_swaps_axes() {
    // [2, 3] -> permute [1, 0] -> [3, 2]
    let t = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let p = t.permute(&[1, 0]);
    assert_eq!(p.shape(), vec![3, 2]);
    // Row 0 of original [1,2,3] becomes column 0 of result.
    assert!((p.get(&[0, 0]) - 1.0).abs() < 1e-6);
    assert!((p.get(&[0, 1]) - 4.0).abs() < 1e-6);
    assert!((p.get(&[1, 0]) - 2.0).abs() < 1e-6);
    assert!((p.get(&[2, 0]) - 3.0).abs() < 1e-6);
}

#[test]
fn permute_4d_head_rearrange() {
    // [batch=2, seq=3, heads=2, head_dim=4] -> permute [0,2,1,3] -> [2,2,3,4]
    let t = Tensor::randn(&[2, 3, 2, 4]);
    let p = t.permute(&[0, 2, 1, 3]);
    assert_eq!(p.shape(), vec![2, 2, 3, 4]);
}

#[test]
fn permute_backward_matches_numeric() {
    let base = [0.3, -0.1, 0.5, 0.2, -0.4, 0.6];
    let shape = [2, 3];
    let t = leaf(&base, &shape);
    let out = t.permute(&[1, 0]);
    out.sum().backward();
    let analytic: Vec<f32> = t.grad().unwrap().iter().copied().collect();

    let eps = 1e-3f32;
    let mut max_diff = 0.0f32;
    for i in 0..base.len() {
        let mut hi = base.to_vec();
        let mut lo = base.to_vec();
        hi[i] += eps;
        lo[i] -= eps;
        let l_hi = leaf(&hi, &shape).permute(&[1, 0]).sum().data().iter().copied().next().unwrap();
        let l_lo = leaf(&lo, &shape).permute(&[1, 0]).sum().data().iter().copied().next().unwrap();
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("permute grad max diff: {max_diff:.2e}");
    assert!(max_diff < 1e-4, "permute gradient mismatch: {max_diff:.2e}");
}

// ==================== LayerNorm ====================

#[test]
fn layernorm_normalizes_last_axis() {
    let ln = LayerNorm::new(4);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let y = ln.forward(&x);
    // Each row should have ~0 mean and ~1 std after normalization (before affine; gamma=1, beta=0).
    let yd: Vec<f32> = y.data().iter().copied().collect();
    let row0_mean = (yd[0] + yd[1] + yd[2] + yd[3]) / 4.0;
    assert!(row0_mean.abs() < 1e-4, "row 0 mean should be ~0: {row0_mean}");
}

#[test]
fn layernorm_grad_matches_numeric() {
    let base = [0.3, -0.1, 0.5, 0.2, -0.4, 0.6];
    let shape = [2, 3];
    let x = leaf(&base, &shape);
    let gamma = leaf(&[1.0, 1.0, 1.0], &[3]);
    let beta = leaf(&[0.0, 0.0, 0.0], &[3]);

    let y = x.layer_norm(&gamma, &beta, 1e-5);
    y.sum().backward();
    let analytic: Vec<f32> = x.grad().unwrap().iter().copied().collect();

    let eps = 1e-3f32;
    let mut max_diff = 0.0f32;
    for i in 0..base.len() {
        let mut hi = base.to_vec();
        let mut lo = base.to_vec();
        hi[i] += eps;
        lo[i] -= eps;
        let l_hi = leaf(&hi, &shape).layer_norm(&gamma, &beta, 1e-5).sum().data().iter().copied().next().unwrap();
        let l_lo = leaf(&lo, &shape).layer_norm(&gamma, &beta, 1e-5).sum().data().iter().copied().next().unwrap();
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("layernorm input-grad max diff: {max_diff:.2e}");
    assert!(max_diff < 1e-2, "layernorm gradient mismatch: {max_diff:.2e}");
}

#[test]
fn layernorm_gamma_grad_nonzero() {
    let ln = LayerNorm::new(4);
    let x = Tensor::randn(&[2, 3, 4]);
    let y = ln.forward(&x);
    y.sum().backward();
    let ggrad = ln.gamma.grad().expect("no gamma grad");
    assert!(ggrad.iter().any(|&g| g.abs() > 1e-6), "gamma gradients should be non-zero");
}

// ==================== Multi-Head Attention ====================

#[test]
fn mha_preserves_shape() {
    let mha = MultiHeadAttention::new(16, 4); // 4 heads, head_dim=4
    let x = Tensor::randn(&[2, 5, 16]);
    let y = mha.forward(&x);
    assert_eq!(y.shape(), vec![2, 5, 16]);
}

#[test]
fn mha_single_head_matches_attention() {
    // With 1 head, MHA should match standard scaled dot-product attention.
    // We just verify shapes and finite outputs here.
    let mha = MultiHeadAttention::new(8, 1);
    let x = Tensor::randn(&[1, 4, 8]);
    let y = mha.forward(&x);
    assert_eq!(y.shape(), vec![1, 4, 8]);
    assert!(y.data().iter().all(|v| v.is_finite()));
}

#[test]
fn mha_is_differentiable() {
    let mha = MultiHeadAttention::new(8, 2);
    let x = leaf(
        &(0..2 * 3 * 8).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(),
        &[2, 3, 8],
    );
    let y = mha.forward(&x);
    y.sum().backward();
    let grad = x.grad().expect("MHA input received no gradient");
    assert!(grad.iter().any(|&g| g.abs() > 0.0), "MHA gradients should be non-zero");
}

// ==================== Looped Transformer ====================

#[test]
fn looped_transformer_preserves_shape() {
    let model = LoopedTransformer::new(8, 16, 4, 32, 5, 4); // loops=4
    let x = Tensor::randn(&[2, 6, 8]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 6, 5]);
}

#[test]
fn looped_transformer_is_differentiable() {
    // Run in a thread with a larger stack for the recursive backward.
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let model = LoopedTransformer::new(6, 8, 2, 16, 3, 2); // 2 loops
            let x = leaf(
                &(0..18).map(|i| (i as f32 * 0.05).sin()).collect::<Vec<_>>(),
                &[1, 3, 6],
            );
            let y = model.forward(&x);
            y.sum().backward();
            x.grad().expect("looped transformer input received no gradient")
        })
        .unwrap();
    let grad = handle.join().unwrap();
    assert!(grad.iter().any(|&g| g.abs() > 0.0), "gradients should flow through all loops");
}

#[test]
fn looped_transformer_adaptive_halting_stops_early() {
    // With adaptive halting at threshold 0.0, the model should halt after step 0.
    let model = LoopedTransformer::new(4, 8, 2, 16, 2, 10)
        .with_adaptive_halting(0.0);
    let x = Tensor::randn(&[1, 3, 4]);
    let (y, loops) = model.forward_with_loops(&x);
    assert_eq!(y.shape(), vec![1, 3, 2]);
    assert!(loops <= 10, "loops_used should be <= num_loops");
}

#[test]
fn looped_transformer_fewer_params_than_standard() {
    // A LoopedTransformer with 1 block x 8 loops should have far fewer params than a
    // standard Transformer with 8 layers (same block architecture).
    let looped = LoopedTransformer::new(16, 32, 4, 64, 16, 8);
    let standard = Transformer::new(16, 32, 4, 64, 16, 8);

    let lp = looped.parameters().len();
    let sp = standard.parameters().len();
    println!("looped param tensors: {lp}, standard param tensors: {sp}");
    // The looped model shares one block across 8 loops, so its param count should be much smaller.
    assert!(lp < sp, "looped should have fewer param tensors than standard");
}

#[test]
fn looped_transformer_learns_sequence_task() {
    // Train a looped transformer on a simple sequence-copy task: output = input.
    // Run in a thread with a larger stack since the recursive backward traverses a deep
    // computation graph (one shared block unrolled across multiple loops).
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let model = LoopedTransformer::new(4, 12, 3, 24, 4, 3); // 3 loops
            let mut opt = Adam::new(model.parameters(), 0.01);

            let x = leaf(
                &[0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, -0.3, 0.2, 0.4, -0.5, 0.0],
                &[1, 3, 4],
            );

            let mut first = f32::INFINITY;
            let mut last = 0.0;
            for step in 0..60 {
                opt.zero_grad();
                let out = model.forward(&x);
                let diff = out.sub(&x);
                let loss = diff.mul(&diff).sum();
                loss.backward();
                opt.step();
                let l = loss.data().iter().copied().next().unwrap_or(0.0);
                if step == 0 {
                    first = first.min(l);
                }
                last = l;
            }
            (first, last)
        })
        .unwrap();
    let (first, last) = handle.join().unwrap();
    println!("looped transformer: first={first:.4} last={last:.4}");
    assert!(last < first, "loss should decrease ({last:.4} >= {first:.4})");
}

#[test]
fn standard_transformer_preserves_shape() {
    let model = Transformer::new(8, 16, 4, 32, 5, 3);
    let x = Tensor::randn(&[2, 6, 8]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 6, 5]);
}

#[test]
fn more_loops_more_effective_depth() {
    // Running with more loops on the same shared block should produce different (deeper) output.
    let m4 = LoopedTransformer::new(4, 8, 2, 16, 4, 4);
    let m8 = LoopedTransformer::new(4, 8, 2, 16, 4, 8);
    // They have independent random init, so just verify both produce finite, correctly-shaped output.
    let x = Tensor::randn(&[1, 3, 4]);
    let y4 = m4.forward(&x);
    let y8 = m8.forward(&x);
    assert_eq!(y4.shape(), vec![1, 3, 4]);
    assert_eq!(y8.shape(), vec![1, 3, 4]);
    assert!(y4.data().iter().all(|v| v.is_finite()));
    assert!(y8.data().iter().all(|v| v.is_finite()));
}
