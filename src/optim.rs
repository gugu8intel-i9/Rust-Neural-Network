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

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let lr_t = self.lr * (1.0 - self.beta2.powi(self.t as i32)).sqrt() / (1.0 - self.beta1.powi(self.t as i32));

        for (i, param) in self.params.iter_mut().enumerate() {
            let mut inner = param.0.write().unwrap();
            if let Some(grad) = inner.grad.take() {
                self.m[i] = &self.m[i] * self.beta1 + &grad * (1.0 - self.beta1);
                self.v[i] = &self.v[i] * self.beta2 + (&grad * &grad) * (1.0 - self.beta2);
                
                let update = &self.m[i] / (self.v[i].mapv(|x| x.sqrt()) + self.eps);
                inner.data -= &(update * lr_t);
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
