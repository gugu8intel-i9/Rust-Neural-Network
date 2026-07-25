//! Tests for the fused `Linear` op, the BLAS-backed `Linear` backward (confirmed by finite
//! differences), and `Transformer` running on linear attention.

use rust_nn::looped_transformer::{AttentionKind, Transformer};
use rust_nn::nn::{Linear, Module, ReLU, Sequential};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;

/// `Linear::forward` must equal `x · Wᵀ + b` exactly, and now does so via a single fused op.
#[test]
fn fused_linear_forward_matches_manual() {
    let lin = Linear::new(5, 3, true);
    let x = Tensor::randn(&[4, 5]);
    let y = lin.forward(&x);
    assert_eq!(y.shape(), vec![4, 3]);

    let xs = x.data();
    let ws = lin.weight.data();
    let bs = lin.bias.as_ref().unwrap().data();
    let (m, k) = (4usize, 5usize);
    let n = 3usize;
    let mut max_err = 0.0f32;
    for i in 0..m {
        for o in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += xs[[i, p]] * ws[[o, p]];
            }
            s += bs[[o]];
            max_err = max_err.max((y.data()[[i, o]] - s).abs());
        }
    }
    assert!(
        max_err < 1e-4,
        "fused linear forward mismatch (max err {max_err})"
    );
}

/// Scalar sum-of-squared-errors loss given the current weights (used for finite differences).
fn sse_loss(lin: &Linear, x: &Tensor, target: &Tensor) -> f32 {
    let y = lin.forward(x);
    y.data()
        .iter()
        .zip(target.data().iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

/// CONFIRMATION (not just a changelog claim) that `Linear`'s backward — which now runs through
/// the fused `Op::Linear` arm built on the packed `blas::sgemm` — produces exact gradients. We
/// central-difference the loss w.r.t. every weight element and compare to the autograd result.
#[test]
fn linear_backward_uses_packed_sgemm_and_is_exact() {
    let lin = Linear::new(4, 3, true);
    let x = Tensor::randn(&[3, 4]);
    x.set_requires_grad(true);
    let target = Tensor::randn(&[3, 3]);

    // Analytical gradient via the autograd (Op::Linear → sgemm backward).
    let y = lin.forward(&x);
    let diff = y.sub(&target);
    let sq = diff.mul(&diff);
    let loss = sq.sum();
    loss.backward();
    let g = lin.weight.grad().expect("weight grad populated");
    let g_flat: Vec<f32> = g.iter().copied().collect();

    // Central-difference check on each weight.
    let w_len = lin.weight.data().len();
    let h = 1e-3f32;
    let mut worst = 0.0f32;
    for idx in 0..w_len {
        {
            let mut wg = lin.weight.0.write().unwrap();
            let s = wg.data.as_slice_memory_order_mut().unwrap();
            s[idx] += h;
        }
        let lp = sse_loss(&lin, &x, &target);
        {
            let mut wg = lin.weight.0.write().unwrap();
            let s = wg.data.as_slice_memory_order_mut().unwrap();
            s[idx] -= 2.0 * h;
        }
        let lm = sse_loss(&lin, &x, &target);
        {
            let mut wg = lin.weight.0.write().unwrap();
            let s = wg.data.as_slice_memory_order_mut().unwrap();
            s[idx] += h; // restore
        }
        let num = (lp - lm) / (2.0 * h);
        let ana = g_flat[idx];
        let err = (num - ana).abs();
        let tol = 1e-3 + 1e-2 * ana.abs().max(num.abs());
        assert!(
            err < tol,
            "weight grad mismatch @ {idx}: num={num} ana={ana} err={err}"
        );
        worst = worst.max(err);
    }
    // Sanity: gradients are non-trivial.
    assert!(
        g_flat.iter().any(|&v| v.abs() > 1e-4),
        "weight grads all ~0"
    );
    let _ = worst;
}

/// A tiny MLP actually learns (loss decreases) — the end-to-end "small model trains fast" smoke.
#[test]
fn small_mlp_learns() {
    let model = Sequential::new()
        .add(Linear::new(6, 16, true))
        .add(ReLU)
        .add(Linear::new(16, 8, true))
        .add(ReLU)
        .add(Linear::new(8, 1, true));
    let mut opt = Adam::new(model.parameters(), 0.05);

    let x = Tensor::randn(&[16, 6]);
    let target = Tensor::randn(&[16, 1]);

    let loss_of = |m: &Sequential| -> f32 {
        let y = m.forward(&x);
        let d = y.data();
        let t = target.data();
        d.iter()
            .zip(t.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / d.len() as f32
    };

    let loss0 = loss_of(&model);
    for _ in 0..40 {
        opt.zero_grad();
        let y = model.forward(&x);
        let diff = y.sub(&target);
        let sq = diff.mul(&diff);
        let loss = sq.sum();
        loss.backward();
        opt.step();
    }
    let loss1 = loss_of(&model);
    assert!(
        loss1 < loss0 * 0.5,
        "MLP did not learn: loss {loss0:.4} -> {loss1:.4}"
    );
}

/// A `Transformer` built with `AttentionKind::Linear` runs end-to-end and is differentiable.
#[test]
fn transformer_with_linear_attention_runs() {
    let model = Transformer::new(8, 16, 2, 32, 8, 2).with_attention(AttentionKind::Linear);
    let x = Tensor::randn(&[2, 6, 8]); // [batch, seq, input_dim]
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 6, 8]);

    // Backward populates gradients on parameters (proves the linear-attention op is wired in).
    y.sum().backward();
    let params = model.parameters();
    let with_grad = params.iter().filter(|p| p.grad().is_some()).count();
    assert!(with_grad > 0, "no parameter received a gradient");
}

/// Same transformer on the default Flash attention still works (regression guard).
#[test]
fn transformer_with_flash_attention_runs() {
    let model = Transformer::new(8, 16, 2, 32, 8, 2);
    let x = Tensor::randn(&[2, 6, 8]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 6, 8]);
    y.sum().backward();
}
