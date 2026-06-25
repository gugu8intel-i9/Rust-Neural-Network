//! Gradient-correctness test for the Mamba selective scan (fused autograd op).

use ndarray::{ArrayD, IxDyn};
use rust_nn::tensor::Tensor;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(), true)
}

fn run_sum(delta: &Tensor, b_vec: &Tensor, c_vec: &Tensor, u: &Tensor, a: &Tensor) -> f32 {
    Tensor::selective_scan(delta, b_vec, c_vec, u, a)
        .sum()
        .data()
        .iter()
        .copied()
        .next()
        .unwrap()
}

#[test]
fn selective_scan_forward_shape() {
    let (batch, seq, dim, n) = (2, 5, 4, 3);
    let delta = leaf(&vec![0.1; batch * seq * dim], &[batch, seq, dim]);
    let bv = leaf(&vec![0.5; batch * seq * n], &[batch, seq, n]);
    let cv = leaf(&vec![0.3; batch * seq * n], &[batch, seq, n]);
    let u = leaf(&vec![0.2; batch * seq * dim], &[batch, seq, dim]);
    let a = leaf(&vec![-1.0; dim * n], &[dim, n]);
    let y = Tensor::selective_scan(&delta, &bv, &cv, &u, &a);
    assert_eq!(y.shape(), vec![batch, seq, dim]);
}

#[test]
fn selective_scan_grad_a_matches_numeric() {
    let (batch, seq, dim, n) = (1, 4, 3, 2);
    let base: Vec<f32> = (0..dim * n).map(|i| -0.5 - i as f32 * 0.1).collect();
    let shape = [dim, n];

    let delta = leaf(&vec![0.2; batch * seq * dim], &[batch, seq, dim]);
    let bv = leaf(&vec![0.4; batch * seq * n], &[batch, seq, n]);
    let cv = leaf(&vec![0.3; batch * seq * n], &[batch, seq, n]);
    let u = leaf(&vec![0.6; batch * seq * dim], &[batch, seq, dim]);

    let at = leaf(&base, &shape);
    let y = Tensor::selective_scan(&delta, &bv, &cv, &u, &at);
    y.sum().backward();
    let analytic: Vec<f32> = at.grad().expect("no a grad").iter().copied().collect();

    let eps = 1e-3f32;
    let mut max_diff = 0.0f32;
    for i in 0..base.len() {
        let mut hi = base.clone();
        let mut lo = base.clone();
        hi[i] += eps;
        lo[i] -= eps;
        let l_hi = run_sum(&delta, &bv, &cv, &u, &leaf(&hi, &shape));
        let l_lo = run_sum(&delta, &bv, &cv, &u, &leaf(&lo, &shape));
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("selective_scan a-grad max |analytic-numeric| = {max_diff:.2e}");
    assert!(max_diff < 1e-2, "a-gradient mismatch: {max_diff:.2e}");
}

#[test]
fn selective_scan_grad_delta_matches_numeric() {
    let (batch, seq, dim, n) = (1, 4, 3, 2);
    let base: Vec<f32> = (0..batch * seq * dim).map(|i| 0.1 + i as f32 * 0.05).collect();
    let shape = [batch, seq, dim];

    let bv = leaf(&vec![0.4; batch * seq * n], &[batch, seq, n]);
    let cv = leaf(&vec![0.3; batch * seq * n], &[batch, seq, n]);
    let u = leaf(&vec![0.6; batch * seq * dim], &[batch, seq, dim]);
    let a = leaf(&[-1.0, -0.8, -0.6, -0.5, -0.4, -0.3], &[dim, n]);

    let dt = leaf(&base, &shape);
    let y = Tensor::selective_scan(&dt, &bv, &cv, &u, &a);
    y.sum().backward();
    let analytic: Vec<f32> = dt.grad().expect("no delta grad").iter().copied().collect();

    let eps = 1e-3f32;
    let mut max_diff = 0.0f32;
    for i in 0..base.len() {
        let mut hi = base.clone();
        let mut lo = base.clone();
        hi[i] += eps;
        lo[i] -= eps;
        let l_hi = run_sum(&leaf(&hi, &shape), &bv, &cv, &u, &a);
        let l_lo = run_sum(&leaf(&lo, &shape), &bv, &cv, &u, &a);
        let num = (l_hi - l_lo) / (2.0 * eps);
        max_diff = max_diff.max((num - analytic[i]).abs());
    }
    println!("selective_scan delta-grad max |analytic-numeric| = {max_diff:.2e}");
    assert!(max_diff < 1e-2, "delta-gradient mismatch: {max_diff:.2e}");
}

#[test]
fn selective_scan_grad_bcu_match_numeric() {
    let (batch, seq, dim, n) = (1, 3, 2, 2);
    let bv_base: Vec<f32> = (0..batch * seq * n).map(|i| 0.3 + i as f32 * 0.1).collect();
    let cv_base: Vec<f32> = (0..batch * seq * n).map(|i| 0.2 + i as f32 * 0.1).collect();
    let u_base: Vec<f32> = (0..batch * seq * dim).map(|i| 0.5 + i as f32 * 0.1).collect();
    let bn = [batch, seq, n];
    let un = [batch, seq, dim];

    let delta = leaf(&vec![0.2; batch * seq * dim], &[batch, seq, dim]);
    let a = leaf(&[-1.0, -0.7, -0.5, -0.4], &[dim, n]);

    // B grad
    let bv = leaf(&bv_base, &bn);
    let cv = leaf(&cv_base, &bn);
    let u = leaf(&u_base, &un);
    Tensor::selective_scan(&delta, &bv, &cv, &u, &a).sum().backward();
    let eps = 1e-3f32;
    let check = |base: &[f32], shape: &[usize], analytic: &[f32], mk: &dyn Fn(&Tensor, &Tensor, &Tensor) -> Tensor| {
        let mut md = 0.0f32;
        for i in 0..base.len() {
            let mut hi = base.to_vec();
            let mut lo = base.to_vec();
            hi[i] += eps;
            lo[i] -= eps;
            let l_hi = mk(&leaf(&hi, shape), &leaf(&cv_base, &bn), &leaf(&u_base, &un)).sum().data().iter().copied().next().unwrap();
            let l_lo = mk(&leaf(&lo, shape), &leaf(&cv_base, &bn), &leaf(&u_base, &un)).sum().data().iter().copied().next().unwrap();
            let num = (l_hi - l_lo) / (2.0 * eps);
            md = md.max((num - analytic[i]).abs());
        }
        md
    };

    let bv_analytic: Vec<f32> = bv.grad().expect("no b grad").iter().copied().collect();
    let md_b = check(&bv_base, &bn, &bv_analytic, &|x, cv, u| Tensor::selective_scan(&delta, x, cv, u, &a));
    println!("b-grad max diff: {md_b:.2e}");
    assert!(md_b < 1e-2, "b-gradient mismatch: {md_b:.2e}");

    let cv_analytic: Vec<f32> = cv.grad().expect("no c grad").iter().copied().collect();
    let md_c = check(&cv_base, &bn, &cv_analytic, &|x, _cv, u| Tensor::selective_scan(&delta, &leaf(&bv_base, &bn), x, u, &a));
    println!("c-grad max diff: {md_c:.2e}");
    assert!(md_c < 1e-2, "c-gradient mismatch: {md_c:.2e}");

    let u_analytic: Vec<f32> = u.grad().expect("no u grad").iter().copied().collect();
    let md_u = check(&u_base, &un, &u_analytic, &|x, _cv, _u| Tensor::selective_scan(&delta, &leaf(&bv_base, &bn), &leaf(&cv_base, &bn), x, &a));
    println!("u-grad max diff: {md_u:.2e}");
    assert!(md_u < 1e-2, "u-gradient mismatch: {md_u:.2e}");
}
