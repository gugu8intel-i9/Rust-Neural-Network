//! Optimizers for training neural networks.

use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    momentum_buffers: Vec<Option<ndarray::ArrayD<f32>>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let n_params = params.len();
        SGD {
            params,
            lr,
            momentum,
            momentum_buffers: vec![None; n_params],
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let mut inner = param.0.write().unwrap();
            if let Some(grad) = inner.grad.take() {
                if self.momentum > 0.0 {
                    if let Some(ref mut buffer) = self.momentum_buffers[i] {
                        *buffer = buffer.clone() * self.momentum + &grad;
                        inner.data -= &(buffer.clone() * self.lr);
                    } else {
                        self.momentum_buffers[i] = Some(grad.clone());
                        inner.data -= &(grad * self.lr);
                    }
                } else {
                    inner.data -= &(grad * self.lr);
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    m: Vec<ndarray::ArrayD<f32>>,
    v: Vec<ndarray::ArrayD<f32>>,
    t: u32,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &params {
            let shape = p.shape();
            m.push(ndarray::ArrayD::zeros(ndarray::IxDyn(&shape)));
            v.push(ndarray::ArrayD::zeros(ndarray::IxDyn(&shape)));
        }
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m,
            v,
            t: 0,
        }
    }
}

/// Scalar fused Adam update (one pass, no temporaries).
#[allow(clippy::too_many_arguments)]
fn adam_update_scalar(
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    d: &mut [f32],
    b1: f32,
    b2: f32,
    one_b1: f32,
    one_b2: f32,
    eps: f32,
    lr_t: f32,
) {
    for k in 0..g.len() {
        let gk = g[k];
        let mk = m[k] * b1 + gk * one_b1;
        let vk = v[k] * b2 + gk * gk * one_b2;
        m[k] = mk;
        v[k] = vk;
        d[k] -= lr_t * mk / (vk.sqrt() + eps);
    }
}

/// AVX2 + FMA fused Adam update: 8 elements/iteration with vectorised mul/fmadd/sqrt/div.
/// Far faster than a scalar loop for large parameters (where `sqrt`/`div` dominate).
///
/// # Safety
/// Caller must ensure `avx2`+`fma` are available and `g`,`m`,`v`,`d` have equal length.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn adam_update_avx2(
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    d: &mut [f32],
    b1: f32,
    b2: f32,
    one_b1: f32,
    one_b2: f32,
    eps: f32,
    lr_t: f32,
) {
    use std::arch::x86_64::*;
    let vb1 = _mm256_set1_ps(b1);
    let vb2 = _mm256_set1_ps(b2);
    let vob1 = _mm256_set1_ps(one_b1);
    let vob2 = _mm256_set1_ps(one_b2);
    let veps = _mm256_set1_ps(eps);
    let vlr = _mm256_set1_ps(lr_t);
    let mut k = 0;
    while k + 8 <= g.len() {
        let gg = _mm256_loadu_ps(g.as_ptr().add(k));
        let mm = _mm256_loadu_ps(m.as_ptr().add(k));
        let vv = _mm256_loadu_ps(v.as_ptr().add(k));
        let m_new = _mm256_fmadd_ps(mm, vb1, _mm256_mul_ps(gg, vob1));
        let gsq = _mm256_mul_ps(gg, gg);
        let v_new = _mm256_fmadd_ps(vv, vb2, _mm256_mul_ps(gsq, vob2));
        let denom = _mm256_add_ps(_mm256_sqrt_ps(v_new), veps);
        let upd = _mm256_div_ps(m_new, denom);
        let dd = _mm256_loadu_ps(d.as_ptr().add(k));
        _mm256_storeu_ps(m.as_mut_ptr().add(k), m_new);
        _mm256_storeu_ps(v.as_mut_ptr().add(k), v_new);
        _mm256_storeu_ps(
            d.as_mut_ptr().add(k),
            _mm256_sub_ps(dd, _mm256_mul_ps(upd, vlr)),
        );
        k += 8;
    }
    while k < g.len() {
        let gk = g[k];
        let mk = m[k] * b1 + gk * one_b1;
        let vk = v[k] * b2 + gk * gk * one_b2;
        m[k] = mk;
        v[k] = vk;
        d[k] -= lr_t * mk / (vk.sqrt() + eps);
        k += 1;
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.lr * bc2.sqrt() / bc1;
        let (b1, b2, eps, one_b1, one_b2) = (
            self.beta1,
            self.beta2,
            self.eps,
            1.0 - self.beta1,
            1.0 - self.beta2,
        );

        for (i, param) in self.params.iter_mut().enumerate() {
            let mut inner = param.0.write().unwrap();
            let Some(grad) = inner.grad.take() else {
                continue;
            };
            // In-place fused update — one pass, zero temporary arrays, vectorised on AVX2.
            // (The previous version allocated ~9 full-parameter-sized temporaries per step and
            // ran the sqrt/div scalarly, dominating step time for non-tiny layers.)
            let g = grad.as_slice_memory_order().expect("grad is contiguous");
            let m = self.m[i]
                .as_slice_memory_order_mut()
                .expect("m is contiguous");
            let v = self.v[i]
                .as_slice_memory_order_mut()
                .expect("v is contiguous");
            let d = inner
                .data
                .as_slice_memory_order_mut()
                .expect("data is contiguous");
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    unsafe { adam_update_avx2(g, m, v, d, b1, b2, one_b1, one_b2, eps, lr_t) };
                    continue;
                }
            }
            adam_update_scalar(g, m, v, d, b1, b2, one_b1, one_b2, eps, lr_t);
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

pub struct RMSprop {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    v: Vec<ndarray::ArrayD<f32>>,
}

impl RMSprop {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let mut v = Vec::new();
        for p in &params {
            let shape = p.shape();
            v.push(ndarray::ArrayD::zeros(ndarray::IxDyn(&shape)));
        }
        RMSprop {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            v,
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let mut inner = param.0.write().unwrap();
            if let Some(grad) = inner.grad.take() {
                self.v[i] = &self.v[i] * self.alpha + (&grad * &grad) * (1.0 - self.alpha);
                let update = &grad / (self.v[i].mapv(|x| x.sqrt()) + self.eps);
                inner.data -= &(update * self.lr);
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

pub struct Muon {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    momentum_buffers: Vec<Option<ndarray::ArrayD<f32>>>,
}

impl Muon {
    pub fn new(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        let n_params = params.len();
        Muon {
            params,
            lr,
            momentum,
            momentum_buffers: vec![None; n_params],
        }
    }
}

impl Optimizer for Muon {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let mut inner = param.0.write().unwrap();
            if let Some(grad) = inner.grad.take() {
                // RMS normalization of the gradient per tensor (a core part of Muon strategy for ND arrays)
                let sq_sum: f32 = grad.iter().map(|&x| x * x).sum();
                let rms = (sq_sum / grad.len() as f32).sqrt().max(1e-8);
                let orth_grad = grad.mapv(|x| x / rms);

                if self.momentum > 0.0 {
                    if let Some(ref mut buffer) = self.momentum_buffers[i] {
                        *buffer = buffer.clone() * self.momentum + &orth_grad;
                        inner.data -= &(buffer.clone() * self.lr);
                    } else {
                        self.momentum_buffers[i] = Some(orth_grad.clone());
                        inner.data -= &(orth_grad * self.lr);
                    }
                } else {
                    inner.data -= &(orth_grad * self.lr);
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
