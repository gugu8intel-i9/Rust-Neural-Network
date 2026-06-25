//! Tests for Mamba (full + hybrid) and Diffusion (DDPM).

use ndarray::{ArrayD, IxDyn};
use rust_nn::diffusion::{DDPM, NoiseSchedule, ScheduleType};
use rust_nn::mamba::{HybridMamba, Mamba, MambaBlock};
use rust_nn::nn::{Linear, Module, Sequential, SiLU};
use rust_nn::optim::{Adam, Optimizer};
use rust_nn::tensor::Tensor;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(), true)
}

// ==================== Mamba ====================

#[test]
fn mamba_block_preserves_shape() {
    let block = MambaBlock::new(8, 4, 2, 3);
    let x = Tensor::randn(&[2, 5, 8]);
    let y = block.forward(&x);
    assert_eq!(y.shape(), vec![2, 5, 8], "MambaBlock must preserve [batch, seq, d_model]");
}

#[test]
fn mamba_block_is_differentiable() {
    let block = MambaBlock::new(6, 3, 2, 3);
    let x = leaf(&(0..2 * 4 * 6).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>(), &[2, 4, 6]);
    let y = block.forward(&x);
    y.sum().backward();
    let grad = x.grad().expect("MambaBlock input received no gradient");
    assert!(grad.iter().any(|&g| g.abs() > 0.0), "MambaBlock gradients should be non-zero");
}

#[test]
fn mamba_full_preserves_shape() {
    let model = Mamba::new(8, 4, 2, 3, 3);
    let x = Tensor::randn(&[2, 6, 8]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 6, 8]);
}

#[test]
fn mamba_full_has_parameters() {
    let model = Mamba::new(8, 4, 2, 3, 2);
    let params = model.parameters();
    assert!(params.len() > 10, "Mamba should expose many parameter tensors, got {}", params.len());
}

#[test]
fn hybrid_mamba_preserves_shape() {
    // 3 Mamba blocks + 1 attention-via-Sequential (a Linear stand-in for a learned mixer).
    let model = HybridMamba::new(8)
        .with_mamba(4, 2, 3)
        .with_mamba(4, 2, 3)
        .with_layer(Sequential::new().add(Linear::new(8, 8, true)).add(SiLU))
        .with_mamba(4, 2, 3);
    let x = Tensor::randn(&[2, 5, 8]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), vec![2, 5, 8]);
}

#[test]
fn mamba_block_train_step_reduces_loss() {
    // A tiny sequence-copy task: target = input, so the model learns (near-)identity.
    let model = Mamba::new(4, 4, 2, 3, 2);
    let params = model.parameters();
    let mut opt = Adam::new(params, 0.01);
    let x = leaf(&[0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, -0.3, 0.2, 0.4, -0.5, 0.0], &[1, 3, 4]);

    let mut first = f32::INFINITY;
    let mut last = 0.0;
    for step in 0..30 {
        opt.zero_grad();
        let out = model.forward(&x);
        let loss = out.sub(&x).mul(&out.sub(&x)).sum();
        loss.backward();
        opt.step();
        let l = loss.data().iter().copied().next().unwrap_or(0.0);
        if step == 0 {
            first = first.min(l);
        }
        last = l;
    }
    println!("Mamba train: first={first:.4} last={last:.4}");
    assert!(last < first, "Mamba loss should decrease ({last:.4} >= {first:.4})");
}

// ==================== Diffusion ====================

#[test]
fn noise_schedule_linear_is_valid() {
    let s = NoiseSchedule::new(ScheduleType::Linear, 100);
    assert_eq!(s.betas.len(), 100);
    for &b in &s.betas {
        assert!((0.0..1.0).contains(&b), "beta out of range: {b}");
    }
    // alpha_cumprod must be monotonically non-increasing in (0,1].
    for w in s.alpha_cumprod.windows(2) {
        assert!(w[1] <= w[0] + 1e-6, "alpha_cumprod not decreasing");
        assert!(w[0] > 0.0 && w[0] <= 1.0);
    }
    assert!((s.alpha_cumprod[0] - 1.0).abs() < 1e-3, "alpha_cumprod[0] should be ~1");
}

#[test]
fn noise_schedule_cosine_is_valid() {
    let s = NoiseSchedule::new(ScheduleType::Cosine, 50);
    for &b in &s.betas {
        assert!((0.0..1.0).contains(&b), "cosine beta out of range: {b}");
    }
    assert!(s.alpha_cumprod[0] > 0.99);
    assert!(s.alpha_cumprod[49] < s.alpha_cumprod[0]);
}

#[test]
fn q_sample_endpoints() {
    let ddpm = DDPM::new(3, 16, 100, ScheduleType::Linear);
    let x0 = leaf(&[1.0, 2.0, 3.0], &[1, 3]);
    let noise = leaf(&[0.0, 0.0, 0.0], &[1, 3]);
    // At t=0, alpha_bar ~ 1, so x_t ~ x_0.
    let xt0 = ddpm.q_sample(&x0, 0, &noise);
    let d0: Vec<f32> = xt0.data().iter().copied().collect();
    assert!((d0[0] - 1.0).abs() < 0.05, "q_sample at t=0 should be ~x_0, got {}", d0[0]);
    // With zero noise, x_t = sqrt(alpha_bar) * x_0 everywhere, so it's a scaled x_0 (same sign).
    let xt99 = ddpm.q_sample(&x0, 99, &noise);
    let d99: Vec<f32> = xt99.data().iter().copied().collect();
    assert!(d99[0].abs() < 1.0, "q_sample at high t with zero noise should be small, got {}", d99[0]);
}

#[test]
fn ddpm_train_batch_runs_and_reduces_loss() {
    let ddpm = DDPM::new(4, 32, 50, ScheduleType::Linear);
    let params = ddpm.parameters();
    let mut opt = Adam::new(params, 0.01);
    let x0 = Tensor::randn(&[8, 4]);

    let mut first = f32::INFINITY;
    let mut last = 0.0;
    for step in 0..40 {
        let l = ddpm.train_batch(&mut opt, &x0);
        if step == 0 {
            first = first.min(l);
        }
        last = l;
    }
    println!("DDPM train: first={first:.4} last={last:.4}");
    assert!(last < first, "DDPM loss should decrease ({last:.4} >= {first:.4})");
}

#[test]
fn ddpm_sample_shape() {
    let ddpm = DDPM::new(4, 16, 30, ScheduleType::Linear);
    let samples = ddpm.sample(5);
    assert_eq!(samples.shape(), vec![5, 4]);
    assert!(samples.data().iter().all(|v| v.is_finite()), "samples must be finite");
}

#[test]
fn denoise_predict_preserves_shape() {
    let ddpm = DDPM::new(6, 16, 20, ScheduleType::Linear);
    let xt = Tensor::randn(&[3, 6]);
    let pred = ddpm.predict_noise(&xt, 5);
    assert_eq!(pred.shape(), vec![3, 6]);
}
