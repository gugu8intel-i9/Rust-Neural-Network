//! Correctness tests for FlashAttention: forward exactness vs a naive reference and
//! gradient correctness vs finite differences.

use ndarray::{ArrayD, IxDyn};
use rust_nn::tensor::Tensor;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(),
        true,
    )
}

/// Naive standard attention (full softmax matrix) as an independent reference.
fn naive_attention(q: &ArrayD<f32>, k: &ArrayD<f32>, v: &ArrayD<f32>, scale: f32) -> ArrayD<f32> {
    let sh = q.shape();
    let (b, n, d) = (sh[0], sh[1], sh[2]);
    let mut out = ArrayD::zeros(IxDyn(&[b, n, d]));
    for bi in 0..b {
        for i in 0..n {
            let mut s = vec![0.0f32; n];
            let mut m = f32::NEG_INFINITY;
            for j in 0..n {
                let mut dot = 0.0;
                for t in 0..d {
                    dot += q[[bi, i, t]] * k[[bi, j, t]];
                }
                s[j] = dot * scale;
                m = m.max(s[j]);
            }
            let mut z = 0.0;
            let mut p = vec![0.0f32; n];
            for j in 0..n {
                p[j] = (s[j] - m).exp();
                z += p[j];
            }
            for t in 0..d {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += p[j] * v[[bi, j, t]];
                }
                out[[bi, i, t]] = acc / z;
            }
        }
    }
    out
}

#[test]
fn flash_forward_matches_naive() {
    let (b, n, d) = (2, 5, 4);
    let qv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.13 - 1.0).sin()).collect();
    let kv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.27 + 0.5).cos()).collect();
    let vv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.09).tan() * 0.1).collect();

    let q = leaf(&qv, &[b, n, d]);
    let k = leaf(&kv, &[b, n, d]);
    let v = leaf(&vv, &[b, n, d]);
    let scale = 1.0 / (d as f32).sqrt();

    let out = Tensor::flash_attention(&q, &k, &v, scale);
    let ref_out = naive_attention(&q.data(), &k.data(), &v.data(), scale);

    let od = out.data();
    let mut max_diff = 0.0f32;
    for i in 0..od.len() {
        let r = ref_out.iter().nth(i).copied().unwrap_or(0.0);
        max_diff = max_diff.max((od.iter().nth(i).copied().unwrap_or(0.0) - r).abs());
    }
    println!("flash vs naive max diff: {max_diff:.2e}");
    assert!(max_diff < 1e-5, "flash attention forward differs from naive: {max_diff:.2e}");
}

#[test]
fn flash_grad_matches_numeric_q() {
    // Check gradients flow to Q via a scalar reduction of the output.
    let (b, n, d) = (1, 4, 3);
    let base: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.31).sin() * 0.5).collect();
    let kv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.17).cos() * 0.4).collect();
    let vv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.23).tan() * 0.1).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let shape = [b, n, d];

    let k = leaf(&kv, &shape);
    let v = leaf(&vv, &shape);

    let eps = 1e-3f32;
    let n = base.len();

    let qt = leaf(&base, &shape);
    let out = Tensor::flash_attention(&qt, &k, &v, scale);
    out.sum().backward();
    let analytic: Vec<f32> = qt.grad().expect("no q grad").iter().copied().collect();

    let mut max_diff = 0.0f32;
    for i in 0..n {
        let mut hi = base.clone();
        let mut lo = base.clone();
        hi[i] += eps;
        lo[i] -= eps;
        let o_hi = Tensor::flash_attention(&leaf(&hi, &shape), &k, &v, scale).sum();
        let o_lo = Tensor::flash_attention(&leaf(&lo, &shape), &k, &v, scale).sum();
        let l_hi = o_hi.data().iter().copied().next().unwrap();
        let l_lo = o_lo.data().iter().copied().next().unwrap();
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("flash q-grad max |analytic - numeric| = {max_diff:.2e}");
    assert!(max_diff < 1e-2, "flash attention q-gradient mismatch: {max_diff:.2e}");
}

#[test]
fn flash_grad_matches_numeric_v() {
    let (b, n, d) = (1, 3, 4);
    let base: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.2).sin() * 0.3).collect();
    let qv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.15).cos() * 0.5).collect();
    let kv: Vec<f32> = (0..b * n * d).map(|i| (i as f32 * 0.19).sin() * 0.5).collect();
    let scale = 1.0 / (d as f32).sqrt();
    let shape = [b, n, d];

    let q = leaf(&qv, &shape);
    let k = leaf(&kv, &shape);
    let eps = 1e-3f32;
    let n = base.len();

    let vt = leaf(&base, &shape);
    let out = Tensor::flash_attention(&q, &k, &vt, scale);
    out.sum().backward();
    let analytic: Vec<f32> = vt.grad().expect("no v grad").iter().copied().collect();

    let mut max_diff = 0.0f32;
    for i in 0..n {
        let mut hi = base.clone();
        let mut lo = base.clone();
        hi[i] += eps;
        lo[i] -= eps;
        let o_hi = Tensor::flash_attention(&q, &k, &leaf(&hi, &shape), scale).sum();
        let o_lo = Tensor::flash_attention(&q, &k, &leaf(&lo, &shape), scale).sum();
        let num = (o_hi.data().iter().copied().next().unwrap()
            - o_lo.data().iter().copied().next().unwrap())
            / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("flash v-grad max |analytic - numeric| = {max_diff:.2e}");
    assert!(max_diff < 1e-2, "flash attention v-gradient mismatch: {max_diff:.2e}");
}
