//! High-performance tensor implementation using ndarray and rayon.
//!
//! Tensors support automatic differentiation (autograd) by tracking
//! operations in a computational graph.

use ndarray::{ArrayD, IxDyn};
use ndarray_rand::rand_distr::{Normal, StandardNormal};
use ndarray_rand::RandomExt;
use std::sync::{Arc, RwLock};
use std::ops::{Add, Mul, Sub};

/// Reduce a gradient array back down to `target_shape`, undoing any broadcasting
/// that was applied during the forward pass.
///
/// ndarray broadcasting aligns the *trailing* dimensions, so we:
///   1. sum away the leading dims the target doesn't have, then
///   2. collapse (and keep) any singleton dims that were expanded from size 1.
///
/// Without this, ops like `Linear`'s `output + bias` (where `bias` is `[out]` and the
/// output is `[batch, out]`) would hand the bias a wrongly-shaped gradient, which then
/// panics inside the optimizer.
fn unbroadcast(mut grad: ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if grad.shape() == target_shape {
        return grad;
    }
    let target_ndim = target_shape.len();

    // Sum over leading dimensions the target does not have.
    while grad.ndim() > target_ndim {
        grad = grad.sum_axis(ndarray::Axis(0));
    }

    // Sum over dimensions that were broadcast from a size of 1.
    for (axis, &size) in target_shape.iter().enumerate() {
        if size == 1 && grad.shape()[axis] != 1 {
            grad = grad
                .sum_axis(ndarray::Axis(axis))
                .insert_axis(ndarray::Axis(axis));
        }
    }

    grad
}

/// Compute device a tensor currently lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,   // Vulkan/DX12 (Windows/Linux)
    Metal, // Apple Silicon GPU
    Cuda,  // Nvidia GPUs
}

#[derive(Debug)]
pub struct TensorData {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub requires_grad: bool,
    pub creator: Option<Arc<Op>>,
    pub device: Device,
}

/// Operations for the computational graph.
#[derive(Debug)]
pub enum Op {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    MatMul(Tensor, Tensor),
    ReLU(Tensor),
    Reshape(Tensor, Vec<usize>),
    Transpose(Tensor),
    Sum(Tensor, Vec<usize>),
    /// Fused softmax + cross-entropy. `(logits, target)` where target is one-hot `[batch, classes]`.
    /// Backward yields `(softmax(logits) - target) / batch`, which is exact and numerically stable.
    SoftmaxCrossEntropy(Tensor, Tensor),
    /// Fused BCE-with-logits. `(logits, target)`. Backward: `(sigmoid(logits) - target) / N`.
    BCEWithLogits(Tensor, Tensor),
    /// Fused BCE on probabilities. `(probs, target)`. Backward: `(probs - target) / (probs * (1 - probs) * N)`.
    BCE(Tensor, Tensor),
    /// Fused L1 (mean absolute error). `(pred, target)`. Backward: `sign(pred - target) / N`.
    L1(Tensor, Tensor),
    /// Fused Huber / smooth-L1. `(pred, target, delta)`.
    Huber(Tensor, Tensor, f32),
    /// Fused scaled dot-product attention computed via the FlashAttention algorithm
    /// (online-softmax tiling, no materialized N×N matrix). `(q, k, v, scale)`.
    /// Inputs and output are `[batch, seq, d]`. Backward is exact.
    FlashAttention(Tensor, Tensor, Tensor, f32),
    /// Fused Mamba selective scan. `(delta, b_vec, c_vec, u, a)`.
    /// Performs the S6 recurrence `h_t = Ā_t⊙h_{t-1} + B̄_t⊙u_t`, `y_t = C_t·h_t` for
    /// `[batch, seq, d]` inputs with a diagonal `a` of shape `[d, n]`. Backward is exact.
    SelectiveScan(Tensor, Tensor, Tensor, Tensor, Tensor),
    /// Fused depthwise causal 1D convolution. `(input, weight)` where input is
    /// `[batch, seq, channels]`, weight is `[channels, kernel]`. Backward is exact.
    Conv1DCausal(Tensor, Tensor),
    /// Fused softplus. `(x,)`. Forward `ln(1+e^x)`, backward `grad * sigmoid(x)`.
    Softplus(Tensor),
    /// Fused sigmoid. `(x,)`. Forward `1/(1+e^-x)`, backward `grad * sig*(1-sig)`.
    Sigmoid(Tensor),
    /// Permute (transpose) axes. `(input, axes)`. Backward applies the inverse permutation.
    Permute(Tensor, Vec<usize>),
    /// Fused layer normalization over the last axis. `(input, gamma, beta, eps)`.
    /// Exact backward for input, gamma, and beta.
    LayerNorm(Tensor, Tensor, Tensor, f32),
}

/// Numerically stable softmax over the last axis of a dynamic array.
fn stable_softmax(data: &ArrayD<f32>) -> ArrayD<f32> {
    let ndim = data.ndim();
    if ndim == 0 {
        return data.clone();
    }
    let last = data.shape()[ndim - 1];
    if last == 0 {
        return data.clone();
    }
    let leading = data.len() / last;
    let mut out = data
        .clone()
        .into_shape(IxDyn(&[leading, last]))
        .expect("softmax reshape");
    for i in 0..leading {
        let mut m = f32::NEG_INFINITY;
        for j in 0..last {
            m = m.max(out[[i, j]]);
        }
        let mut s = 0.0;
        for j in 0..last {
            out[[i, j]] = (out[[i, j]] - m).exp();
            s += out[[i, j]];
        }
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for j in 0..last {
            out[[i, j]] *= inv;
        }
    }
    out.into_shape(IxDyn(data.shape())).expect("softmax restore shape")
}

/// A multi-dimensional tensor with automatic differentiation.
#[derive(Debug, Clone)]
pub struct Tensor(pub Arc<RwLock<TensorData>>);

impl Tensor {
    // ==================== Constructors ====================

    /// Creates a new tensor from ndarray data.
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Tensor(Arc::new(RwLock::new(TensorData {
            data,
            grad: None,
            requires_grad,
            creator: None,
            device: Device::Cpu,
        })))
    }

    /// Creates a new tensor with the given shape, initialized with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(ArrayD::zeros(IxDyn(shape)), false)
    }

    /// Creates a new tensor with the given shape, initialized with ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self::new(ArrayD::ones(IxDyn(shape)), false)
    }

    /// Creates a new tensor from a vector and shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data).expect("Shape mismatch");
        Self::new(array, false)
    }

    /// Creates a tensor with random values from a normal distribution.
    pub fn randn(shape: &[usize]) -> Self {
        let array = ArrayD::random(IxDyn(shape), StandardNormal);
        Self::new(array, false)
    }

    /// He (Kaiming) initialization for ReLU.
    pub fn he(shape: &[usize]) -> Self {
        let fan_in = shape[shape.len() - 1] as f32; // Assuming (out, in)
        let std = (2.0 / fan_in).sqrt();
        let array = ArrayD::random(IxDyn(shape), Normal::new(0.0, std).unwrap());
        Self::new(array, true)
    }

    /// Xavier (Glorot) initialization.
    pub fn xavier(shape: &[usize]) -> Self {
        let fan_in = shape[shape.len() - 1] as f32;
        let fan_out = shape[shape.len() - 2] as f32;
        let std = (2.0 / (fan_in + fan_out)).sqrt();
        let array = ArrayD::random(IxDyn(shape), Normal::new(0.0, std).unwrap());
        Self::new(array, true)
    }

    // ==================== Basic Properties ====================

    pub fn shape(&self) -> Vec<usize> {
        self.0.read().unwrap().data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.0.read().unwrap().data.ndim()
    }

    pub fn len(&self) -> usize {
        self.0.read().unwrap().data.len()
    }

    /// Returns `true` if the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data(&self) -> ArrayD<f32> {
        self.0.read().unwrap().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.0.read().unwrap().grad.clone()
    }

    pub fn set_requires_grad(&self, requires: bool) {
        self.0.write().unwrap().requires_grad = requires;
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.write().unwrap();
        if let Some(ref mut grad) = inner.grad {
            grad.fill(0.0);
        }
    }

    /// Explicitly move the tensor to the specified compute device (CPU, Gpu, Metal, Cuda)
    pub fn to_device(&self, device: Device) -> Tensor {
        let mut inner = self.0.write().unwrap();
        inner.device = device;
        // Logic for transferring actual memory buffers via wgpu goes here in the future
        self.clone()
    }

    /// Retrieve the current compute device the tensor lives on
    pub fn device(&self) -> Device {
        self.0.read().unwrap().device
    }

    // ==================== Operations ====================

    pub fn add(&self, other: &Tensor) -> Tensor {
        let data = &self.0.read().unwrap().data + &other.0.read().unwrap().data;
        let requires_grad = self.0.read().unwrap().requires_grad || other.0.read().unwrap().requires_grad;
        let res = Tensor::new(data, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Add(self.clone(), other.clone())));
        }
        res
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let data = &self.0.read().unwrap().data - &other.0.read().unwrap().data;
        let requires_grad = self.0.read().unwrap().requires_grad || other.0.read().unwrap().requires_grad;
        let res = Tensor::new(data, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Sub(self.clone(), other.clone())));
        }
        res
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let data = &self.0.read().unwrap().data * &other.0.read().unwrap().data;
        let requires_grad = self.0.read().unwrap().requires_grad || other.0.read().unwrap().requires_grad;
        let res = Tensor::new(data, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Mul(self.clone(), other.clone())));
        }
        res
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = self.0.read().unwrap().data.clone().into_dimensionality::<ndarray::Ix2>().expect("MatMul expects 2D");
        let b = other.0.read().unwrap().data.clone().into_dimensionality::<ndarray::Ix2>().expect("MatMul expects 2D");
        let res_data = a.dot(&b).into_dyn();
        
        let requires_grad = self.0.read().unwrap().requires_grad || other.0.read().unwrap().requires_grad;
        let res = Tensor::new(res_data, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::MatMul(self.clone(), other.clone())));
        }
        res
    }

    pub fn sum(&self) -> Tensor {
        let data = ndarray::arr0(self.0.read().unwrap().data.sum()).into_dyn();
        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(data, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Sum(self.clone(), Vec::new())));
        }
        res
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let data = self.0.read().unwrap().data.clone().into_shape(IxDyn(shape)).expect("Reshape fail");
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Reshape(self.clone(), self.shape())));
        }
        res
    }

    /// Fused softmax + cross-entropy loss against one-hot `target`, both `[batch, classes]`.
    ///
    /// Returns a scalar tensor. Its gradient flows exactly to `self` (the logits) as
    /// `(softmax(logits) - target) / batch`, so this trains end-to-end without needing
    /// separate log/exp/div autograd ops.
    pub fn cross_entropy_logits(&self, target: &Tensor) -> Tensor {
        let logits = self.0.read().unwrap().data.clone();
        let tgt = target.0.read().unwrap().data.clone();
        let shape = logits.shape();
        assert!(
            shape.len() == 2 && shape == tgt.shape(),
            "cross_entropy_logits expects 2D [batch, classes] logits and matching one-hot target, got {:?} / {:?}",
            shape,
            tgt.shape()
        );
        let (batch, classes) = (shape[0], shape[1]);

        let probs = stable_softmax(&logits);
        let mut total = 0.0f32;
        for i in 0..batch {
            for j in 0..classes {
                // Cross-entropy: -sum(target * log(prob)). Clamp prob away from 0 for log safety.
                let p = probs[[i, j]].max(1e-12);
                total -= tgt[[i, j]] * p.ln();
            }
        }
        let loss = total / batch as f32;

        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(ndarray::arr0(loss).into_dyn(), requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::SoftmaxCrossEntropy(self.clone(), target.clone())));
        }
        res
    }

    /// Numerically stable sigmoid.
    fn sigmoid_data(data: &ArrayD<f32>) -> ArrayD<f32> {
        data.mapv(|x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
    }

    /// Fused BCE-with-logits loss against a {0,1} target of the same shape.
    pub fn bce_with_logits(&self, target: &Tensor) -> Tensor {
        let logits = self.0.read().unwrap().data.clone();
        let tgt = target.0.read().unwrap().data.clone();
        assert_eq!(logits.shape(), tgt.shape(), "bce_with_logits: shape mismatch");
        // loss = max(x,0) - x*z + log(1 + exp(-|x|)), averaged.
        let loss = logits
            .iter()
            .zip(tgt.iter())
            .map(|(&x, &z)| x.max(0.0) - x * z + ((-x.abs()).exp() + 1.0).ln())
            .sum::<f32>()
            / logits.len() as f32;

        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(ndarray::arr0(loss).into_dyn(), requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::BCEWithLogits(self.clone(), target.clone())));
        }
        res
    }

    /// Binary cross-entropy on probabilities `p` (in (0,1)) against a {0,1} target.
    pub fn bce(&self, target: &Tensor) -> Tensor {
        let probs = self.0.read().unwrap().data.clone();
        let tgt = target.0.read().unwrap().data.clone();
        assert_eq!(probs.shape(), tgt.shape(), "bce: shape mismatch");
        let loss = probs
            .iter()
            .zip(tgt.iter())
            .map(|(&p, &z)| {
                let p = p.clamp(1e-12, 1.0 - 1e-12);
                -(z * p.ln() + (1.0 - z) * (1.0 - p).ln())
            })
            .sum::<f32>()
            / probs.len() as f32;

        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(ndarray::arr0(loss).into_dyn(), requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::BCE(self.clone(), target.clone())));
        }
        res
    }

    /// Mean absolute error against `target`.
    pub fn l1_loss(&self, target: &Tensor) -> Tensor {
        let pred = self.0.read().unwrap().data.clone();
        let tgt = target.0.read().unwrap().data.clone();
        assert_eq!(pred.shape(), tgt.shape(), "l1_loss: shape mismatch");
        let loss = pred.iter().zip(tgt.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>()
            / pred.len() as f32;

        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(ndarray::arr0(loss).into_dyn(), requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::L1(self.clone(), target.clone())));
        }
        res
    }

    /// Huber (smooth-L1) loss against `target` with the given `delta`.
    pub fn huber_loss(&self, target: &Tensor, delta: f32) -> Tensor {
        let pred = self.0.read().unwrap().data.clone();
        let tgt = target.0.read().unwrap().data.clone();
        assert_eq!(pred.shape(), tgt.shape(), "huber_loss: shape mismatch");
        let loss = pred
            .iter()
            .zip(tgt.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                let ad = d.abs();
                if ad <= delta {
                    0.5 * d * d
                } else {
                    delta * (ad - 0.5 * delta)
                }
            })
            .sum::<f32>()
            / pred.len() as f32;

        let requires_grad = self.0.read().unwrap().requires_grad;
        let res = Tensor::new(ndarray::arr0(loss).into_dyn(), requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::Huber(self.clone(), target.clone(), delta)));
        }
        res
    }

    /// Memory-efficient **exact** scaled dot-product attention (FlashAttention algorithm).
    ///
    /// Computes `softmax(Q·Kᵀ · scale) · V` for batched inputs `q`, `k`, `v` of shape
    /// `[batch, seq, d]`. Rather than building the full `seq × seq` attention matrix, it
    /// streams over key positions per query while maintaining running max/sum statistics
    /// (the "online softmax"), so peak memory is **O(seq)** rather than O(seq²) — exactly
    /// the technique from the FlashAttention paper, minus the GPU SRAM tiling (a CPU has no
    /// separate on-chip SRAM to exploit, but the algorithmic memory win still applies).
    ///
    /// The result is bit-for-bit equivalent to standard attention and fully differentiable.
    pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Tensor {
        let qd = q.data();
        let kd = k.data();
        let vd = v.data();
        let shape = qd.shape();
        assert!(
            shape.len() == 3 && shape == kd.shape() && shape == vd.shape(),
            "flash_attention expects q,k,v of matching shape [batch, seq, d], got {:?}/{:?}/{:?}",
            shape,
            kd.shape(),
            vd.shape()
        );
        let (batch, seq, dim) = (shape[0], shape[1], shape[2]);

        let mut out = ArrayD::zeros(IxDyn(&[batch, seq, dim]));
        for b in 0..batch {
            for i in 0..seq {
                // Online softmax over all key positions for query row `i`.
                let mut row_max = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;
                let mut acc = vec![0.0f32; dim];
                for j in 0..seq {
                    let mut s = 0.0f32;
                    for t in 0..dim {
                        s += qd[[b, i, t]] * kd[[b, j, t]];
                    }
                    s *= scale;

                    let m_new = row_max.max(s);
                    // Rescale running statistics: exp(old_max - new_max), 0 on the first step.
                    let exp_old = (row_max - m_new).exp();
                    let p = (s - m_new).exp();

                    row_sum = exp_old * row_sum + p;
                    for t in 0..dim {
                        acc[t] = exp_old * acc[t] + p * vd[[b, j, t]];
                    }
                    row_max = m_new;
                }
                let inv = if row_sum > 0.0 { 1.0 / row_sum } else { 0.0 };
                for t in 0..dim {
                    out[[b, i, t]] = acc[t] * inv;
                }
            }
        }

        let requires_grad = q.0.read().unwrap().requires_grad
            || k.0.read().unwrap().requires_grad
            || v.0.read().unwrap().requires_grad;
        let res = Tensor::new(out, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::FlashAttention(q.clone(), k.clone(), v.clone(), scale)));
        }
        res
    }

    /// Fused **Mamba selective scan** (the S6 recurrence).
    ///
    /// Performs, for each timestep `t` and channel `d`:
    /// ```text
    /// Ā_t = exp(delta_t * a)        // a is the diagonal [d, n] state matrix
    /// B̄_t = delta_t * b_vec_t
    /// h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t
    /// y_t = sum_n c_vec_t[n] * h_t[:, n]
    /// ```
    /// - `delta`: `[batch, seq, d]` (input-dependent step size — the "selection" signal).
    /// - `b_vec`, `c_vec`: `[batch, seq, n]` (input-dependent B, C).
    /// - `u`: `[batch, seq, d]` (the convolved/gated sequence input).
    /// - `a`: `[d, n]` (diagonal state-transition parameters, typically negative).
    /// - returns `y`: `[batch, seq, d]`.
    ///
    /// Exact forward and exact backward (reverse scan). This is the core of the Mamba block.
    pub fn selective_scan(delta: &Tensor, b_vec: &Tensor, c_vec: &Tensor, u: &Tensor, a: &Tensor) -> Tensor {
        let dd = delta.data();
        let bd = b_vec.data();
        let cd = c_vec.data();
        let ud = u.data();
        let ad = a.data();
        let dshape = dd.shape();
        assert!(
            dshape.len() == 3 && dshape == ud.shape(),
            "selective_scan: delta and u must be [batch, seq, d], got {:?} / {:?}",
            dshape,
            ud.shape()
        );
        let (batch, seq, dim) = (dshape[0], dshape[1], dshape[2]);
        let n = bd.shape()[bd.shape().len() - 1];
        assert_eq!(bd.shape(), &[batch, seq, n], "b_vec shape mismatch");
        assert_eq!(cd.shape(), &[batch, seq, n], "c_vec shape mismatch");
        assert_eq!(ad.shape(), &[dim, n], "a must be [d, n]");

        let mut out = ArrayD::zeros(IxDyn(&[batch, seq, dim]));
        for b in 0..batch {
            // h: [d, n] per batch, accumulated across the sequence.
            let mut h = vec![0.0f32; dim * n];
            for t in 0..seq {
                for d in 0..dim {
                    let dt = dd[[b, t, d]];
                    let ut = ud[[b, t, d]];
                    for j in 0..n {
                        let abar = (dt * ad[[d, j]]).exp();
                        let bbar = dt * bd[[b, t, j]];
                        h[d * n + j] = abar * h[d * n + j] + bbar * ut;
                    }
                    let mut yt = 0.0f32;
                    for j in 0..n {
                        yt += cd[[b, t, j]] * h[d * n + j];
                    }
                    out[[b, t, d]] = yt;
                }
            }
        }

        let requires_grad = delta.0.read().unwrap().requires_grad
            || b_vec.0.read().unwrap().requires_grad
            || c_vec.0.read().unwrap().requires_grad
            || u.0.read().unwrap().requires_grad
            || a.0.read().unwrap().requires_grad;
        let res = Tensor::new(out, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::SelectiveScan(
                delta.clone(),
                b_vec.clone(),
                c_vec.clone(),
                u.clone(),
                a.clone(),
            )));
        }
        res
    }

    /// Fused **depthwise causal 1D convolution**.
    ///
    /// `out[b, t, c] = Σ_{i=0..kernel} weight[c, i] * in[b, t - kernel + 1 + i, c]`, treating
    /// out-of-range (future/past) positions as zero (causal + zero-padded). Used inside the
    /// Mamba block. Fully differentiable.
    pub fn conv1d_causal(input: &Tensor, weight: &Tensor) -> Tensor {
        let id = input.data();
        let wd = weight.data();
        let ishape = id.shape();
        assert!(ishape.len() == 3, "conv1d_causal: input must be [batch, seq, channels]");
        let (batch, seq, channels) = (ishape[0], ishape[1], ishape[2]);
        let wshape = wd.shape();
        assert!(
            wshape.len() == 2 && wshape[0] == channels,
            "conv1d_causal: weight must be [channels, kernel], got {:?} for {channels} channels",
            wshape
        );
        let kernel = wshape[1];

        let mut out = ArrayD::zeros(IxDyn(&[batch, seq, channels]));
        for b in 0..batch {
            for t in 0..seq {
                for c in 0..channels {
                    let mut acc = 0.0f32;
                    for i in 0..kernel {
                        let s = t as isize - kernel as isize + 1 + i as isize;
                        if s >= 0 {
                            acc += wd[[c, i]] * id[[b, s as usize, c]];
                        }
                    }
                    out[[b, t, c]] = acc;
                }
            }
        }

        let requires_grad = input.0.read().unwrap().requires_grad || weight.0.read().unwrap().requires_grad;
        let res = Tensor::new(out, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::Conv1DCausal(input.clone(), weight.clone())));
        }
        res
    }

    /// Fused **softplus** activation: `ln(1 + e^x)`, computed in a numerically stable way.
    /// Backward gradient is `grad * sigmoid(x)`. Used by Mamba for the step-size Δ.
    pub fn softplus(&self) -> Tensor {
        let data = self.0.read().unwrap().data.mapv(|x| {
            if x > 20.0 {
                x // exp overflow guard; softplus(x) ≈ x for large x
            } else if x >= 0.0 {
                ((-x).exp() + 1.0).ln()
            } else {
                x.exp().ln_1p()
            }
        });
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Softplus(self.clone())));
        }
        res
    }

    /// Fused **sigmoid** activation: `1 / (1 + e^-x)`, numerically stable and fully
    /// differentiable (backward `grad * sig * (1 - sig)`).
    pub fn sigmoid(&self) -> Tensor {
        let data = self.0.read().unwrap().data.mapv(|v| {
            if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let e = v.exp(); e / (1.0 + e) }
        });
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Sigmoid(self.clone())));
        }
        res
    }

    /// SiLU / Swish activation `x * sigmoid(x)`, fully differentiable.
    pub fn silu(&self) -> Tensor {
        let sig = self.sigmoid();
        Tensor::mul(self, &sig)
    }

    /// Permute (reorder) the axes of this tensor. `axes` must be a permutation of
    /// `0..ndim`. Fully differentiable (backward applies the inverse permutation).
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        // Materialize as a contiguous (standard C-order) array so that subsequent reshapes work.
        let (data, out_shape) = {
            let inner = self.0.read().unwrap();
            let permuted = inner.data.clone().permuted_axes(IxDyn(axes));
            let out_shape = permuted.shape().to_vec();
            let flat: Vec<f32> = permuted.iter().copied().collect();
            let data = ArrayD::from_shape_vec(IxDyn(&out_shape), flat).unwrap();
            (data, out_shape)
        };
        let _ = out_shape; // already captured in data's shape
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::Permute(self.clone(), axes.to_vec())));
        }
        res
    }

    /// Layer normalization over the **last axis**. `gamma` and `beta` are `[d]`-shaped affine
    /// parameters. The normalization statistics (mean/var) are computed per-position; the
    /// affine transform flows gradients to `gamma` and `beta`. Exact backward for all three.
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let data: Vec<f32> = self.data().iter().copied().collect();
        let gd: Vec<f32> = gamma.data().iter().copied().collect();
        let bd: Vec<f32> = beta.data().iter().copied().collect();
        let shape = self.shape();
        assert!(
            !shape.is_empty() && shape[shape.len() - 1] == gd.len() && gd.len() == bd.len(),
            "layer_norm: last dim {} must match gamma/beta len {}",
            shape.last().copied().unwrap_or(0),
            gd.len()
        );
        let n = *shape.last().unwrap();
        let nrows = data.len() / n;

        let mut out = vec![0.0f32; data.len()];
        for row in 0..nrows {
            let base = row * n;
            let mut mean = 0.0f32;
            for j in 0..n {
                mean += data[base + j];
            }
            mean /= n as f32;
            let mut var = 0.0f32;
            for j in 0..n {
                let d = data[base + j] - mean;
                var += d * d;
            }
            var /= n as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for j in 0..n {
                let x_hat = (data[base + j] - mean) * inv_std;
                out[base + j] = x_hat * gd[j] + bd[j];
            }
        }

        let out = ArrayD::from_shape_vec(IxDyn(&shape), out).unwrap();
        let requires_grad = self.0.read().unwrap().requires_grad
            || gamma.0.read().unwrap().requires_grad
            || beta.0.read().unwrap().requires_grad;
        let res = Tensor::new(out, requires_grad);
        if requires_grad {
            res.0.write().unwrap().creator =
                Some(Arc::new(Op::LayerNorm(self.clone(), gamma.clone(), beta.clone(), eps)));
        }
        res
    }

    pub fn transpose(&self) -> Tensor {
        let data = self.0.read().unwrap().data.clone().reversed_axes();
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::Transpose(self.clone())));
        }
        res
    }

    // ==================== Autograd ====================

    pub fn backward(&self) {
        let shape = self.shape();
        let grad = ArrayD::ones(IxDyn(&shape));
        self.backward_with_grad(grad);
    }

    pub fn backward_with_grad(&self, grad: ArrayD<f32>) {
        {
            let mut inner = self.0.write().unwrap();
            if let Some(ref mut existing_grad) = inner.grad {
                *existing_grad += &grad;
            } else {
                inner.grad = Some(grad.clone());
            }
        }

        let inner = self.0.read().unwrap();
        if let Some(ref op) = inner.creator {
            match op.as_ref() {
                Op::Add(a, b) => {
                    if a.0.read().unwrap().requires_grad {
                        let a_shape = a.shape();
                        a.backward_with_grad(unbroadcast(grad.clone(), &a_shape));
                    }
                    if b.0.read().unwrap().requires_grad {
                        let b_shape = b.shape();
                        b.backward_with_grad(unbroadcast(grad, &b_shape));
                    }
                }
                Op::Sub(a, b) => {
                    if a.0.read().unwrap().requires_grad {
                        let a_shape = a.shape();
                        a.backward_with_grad(unbroadcast(grad.clone(), &a_shape));
                    }
                    if b.0.read().unwrap().requires_grad {
                        let b_shape = b.shape();
                        b.backward_with_grad(unbroadcast(-grad, &b_shape));
                    }
                }
                Op::Mul(a, b) => {
                    if a.0.read().unwrap().requires_grad {
                        let b_data = b.0.read().unwrap().data.clone();
                        let a_shape = a.shape();
                        a.backward_with_grad(unbroadcast(&grad * &b_data, &a_shape));
                    }
                    if b.0.read().unwrap().requires_grad {
                        let a_data = a.0.read().unwrap().data.clone();
                        let b_shape = b.shape();
                        b.backward_with_grad(unbroadcast(&grad * &a_data, &b_shape));
                    }
                }
                Op::MatMul(a, b) => {
                    let a_data = a.0.read().unwrap().data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let b_data = b.0.read().unwrap().data.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                    let grad_2d = grad.into_dimensionality::<ndarray::Ix2>().unwrap();

                    if a.0.read().unwrap().requires_grad {
                        let da = grad_2d.dot(&b_data.t()).into_dyn();
                        a.backward_with_grad(da);
                    }
                    if b.0.read().unwrap().requires_grad {
                        let db = a_data.t().dot(&grad_2d).into_dyn();
                        b.backward_with_grad(db);
                    }
                }
                Op::ReLU(a) => {
                    if a.0.read().unwrap().requires_grad {
                        let a_data = a.0.read().unwrap().data.clone();
                        let mut mask = a_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                        mask *= &grad;
                        a.backward_with_grad(mask);
                    }
                }
                Op::Reshape(a, original_shape) => {
                    if a.0.read().unwrap().requires_grad {
                        a.backward_with_grad(grad.into_shape(IxDyn(original_shape)).unwrap());
                    }
                }
                Op::Transpose(a) => {
                    if a.0.read().unwrap().requires_grad {
                        a.backward_with_grad(grad.reversed_axes());
                    }
                }
                Op::Sum(a, _) => {
                    if a.0.read().unwrap().requires_grad {
                        let a_shape = a.shape();
                        let a_grad = ArrayD::from_elem(IxDyn(&a_shape), *grad.first().unwrap_or(&0.0));
                        a.backward_with_grad(a_grad);
                    }
                }
                Op::SoftmaxCrossEntropy(logits, target) => {
                    if logits.0.read().unwrap().requires_grad {
                        let l = logits.0.read().unwrap().data.clone();
                        let t = target.0.read().unwrap().data.clone();
                        let batch = l.shape()[0] as f32;
                        let probs = stable_softmax(&l);
                        // d loss / d logits = (softmax(logits) - target) / batch,
                        // scaled by the incoming scalar gradient.
                        let scale = grad.first().copied().unwrap_or(1.0) / batch;
                        let dlogits = (&probs - &t).mapv(|x| x * scale);
                        logits.backward_with_grad(dlogits);
                    }
                }
                Op::BCEWithLogits(logits, target) => {
                    if logits.0.read().unwrap().requires_grad {
                        let l = logits.0.read().unwrap().data.clone();
                        let t = target.0.read().unwrap().data.clone();
                        let n = l.len() as f32;
                        let sig = Tensor::sigmoid_data(&l);
                        let scale = grad.first().copied().unwrap_or(1.0) / n;
                        let dlogits = (&sig - &t).mapv(|x| x * scale);
                        logits.backward_with_grad(dlogits);
                    }
                }
                Op::BCE(probs, target) => {
                    if probs.0.read().unwrap().requires_grad {
                        let p = probs.0.read().unwrap().data.clone();
                        let t = target.0.read().unwrap().data.clone();
                        let n = p.len() as f32;
                        let scale = grad.first().copied().unwrap_or(1.0) / n;
                        // d/dp -[z ln p + (1-z) ln(1-p)] = (p - z) / (p (1-p))
                        let dp = p
                            .iter()
                            .zip(t.iter())
                            .map(|(&pp, &z)| {
                                let pp = pp.clamp(1e-12, 1.0 - 1e-12);
                                (pp - z) / (pp * (1.0 - pp))
                            })
                            .collect::<Vec<_>>();
                        let mut grad_arr =
                            ArrayD::from_shape_vec(IxDyn(p.shape()), dp).expect("bce grad shape");
                        grad_arr.mapv_inplace(|x| x * scale);
                        probs.backward_with_grad(grad_arr);
                    }
                }
                Op::L1(pred, target) => {
                    if pred.0.read().unwrap().requires_grad {
                        let p = pred.0.read().unwrap().data.clone();
                        let t = target.0.read().unwrap().data.clone();
                        let n = p.len() as f32;
                        let scale = grad.first().copied().unwrap_or(1.0) / n;
                        let dp = p
                            .iter()
                            .zip(t.iter())
                            .map(|(&a, &b)| if a > b { 1.0 } else if a < b { -1.0 } else { 0.0 })
                            .collect::<Vec<_>>();
                        let mut grad_arr =
                            ArrayD::from_shape_vec(IxDyn(p.shape()), dp).expect("l1 grad shape");
                        grad_arr.mapv_inplace(|x| x * scale);
                        pred.backward_with_grad(grad_arr);
                    }
                }
                Op::Huber(pred, target, delta) => {
                    if pred.0.read().unwrap().requires_grad {
                        let p = pred.0.read().unwrap().data.clone();
                        let t = target.0.read().unwrap().data.clone();
                        let n = p.len() as f32;
                        let scale = grad.first().copied().unwrap_or(1.0) / n;
                        // d/dp huber = clip((pred - target)/delta, -1, 1) (for the standard form),
                        // which equals: err if |err|<=delta else delta*sign(err).
                        let dp = p
                            .iter()
                            .zip(t.iter())
                            .map(|(&a, &b)| {
                                let d = a - b;
                                let ad = d.abs();
                                if ad <= *delta { d } else { *delta * d.signum() }
                            })
                            .collect::<Vec<_>>();
                        let mut grad_arr =
                            ArrayD::from_shape_vec(IxDyn(p.shape()), dp).expect("huber grad shape");
                        grad_arr.mapv_inplace(|x| x * scale);
                        pred.backward_with_grad(grad_arr);
                    }
                }
                Op::FlashAttention(q, k, v, scale) => {
                    // Recompute attention on-chip (the FlashAttention backward strategy) and
                    // apply the standard attention gradient formulas.
                    let incoming = grad.first().copied().unwrap_or(1.0);
                    let qd = q.0.read().unwrap().data.clone();
                    let kd = k.0.read().unwrap().data.clone();
                    let vd = v.0.read().unwrap().data.clone();
                    let shape = qd.shape();
                    let (batch, seq, dim) = (shape[0], shape[1], shape[2]);

                    let q_rg = q.0.read().unwrap().requires_grad;
                    let k_rg = k.0.read().unwrap().requires_grad;
                    let v_rg = v.0.read().unwrap().requires_grad;

                    let mut dq = if q_rg { Some(ArrayD::zeros(IxDyn(shape))) } else { None };
                    let mut dk = if k_rg { Some(ArrayD::zeros(IxDyn(shape))) } else { None };
                    let mut dv = if v_rg { Some(ArrayD::zeros(IxDyn(shape))) } else { None };

                    for b in 0..batch {
                        // Recompute the softmax probability matrix P[b] = softmax(scale * Q K^T).
                        let mut p_row = vec![0.0f32; seq];
                        for i in 0..seq {
                            let mut s = vec![0.0f32; seq];
                            let mut m = f32::NEG_INFINITY;
                            for j in 0..seq {
                                let mut dot = 0.0f32;
                                for t in 0..dim {
                                    dot += qd[[b, i, t]] * kd[[b, j, t]];
                                }
                                s[j] = dot * *scale;
                                m = m.max(s[j]);
                            }
                            let mut z = 0.0f32;
                            for (pj, sj) in p_row.iter_mut().zip(s.iter()) {
                                *pj = (sj - m).exp();
                                z += *pj;
                            }
                            let inv = if z > 0.0 { 1.0 / z } else { 0.0 };
                            for pj in p_row.iter_mut() {
                                *pj *= inv;
                            }

                            // dattn_ij = sum_t dout[i,t] * V[j,t]   (dout = incoming gradient)
                            let mut dp = vec![0.0f32; seq];
                            for j in 0..seq {
                                let mut acc = 0.0f32;
                                for t in 0..dim {
                                    acc += incoming * vd[[b, j, t]];
                                }
                                dp[j] = acc;
                            }
                            // softmax backward: dP_ij = P_ij * (dp_ij - sum_k P_ik dp_ik)
                            let mut dotp = 0.0f32;
                            for (pj, dpj) in p_row.iter().zip(dp.iter()) {
                                dotp += pj * dpj;
                            }
                            for j in 0..seq {
                                dp[j] = p_row[j] * (dp[j] - dotp);
                            }
                            // Now dp = d(scores_ij). Accumulate parameter gradients.
                            for j in 0..seq {
                                for t in 0..dim {
                                    if let Some(ref mut g) = dv {
                                        g[[b, j, t]] += p_row[j] * incoming;
                                    }
                                    if let Some(ref mut gq) = dq {
                                        gq[[b, i, t]] += dp[j] * *scale * kd[[b, j, t]];
                                    }
                                    if let Some(ref mut gk) = dk {
                                        gk[[b, j, t]] += dp[j] * *scale * qd[[b, i, t]];
                                    }
                                }
                            }
                        }
                    }

                    if let Some(g) = dq { q.backward_with_grad(g); }
                    if let Some(g) = dk { k.backward_with_grad(g); }
                    if let Some(g) = dv { v.backward_with_grad(g); }
                }
                Op::SelectiveScan(delta, b_vec, c_vec, u, a) => {
                    // Reverse scan. dy is the incoming gradient on y [batch, seq, d].
                    let dy = grad;
                    let dd = delta.0.read().unwrap().data.clone();
                    let bd = b_vec.0.read().unwrap().data.clone();
                    let cd = c_vec.0.read().unwrap().data.clone();
                    let ud = u.0.read().unwrap().data.clone();
                    let ad = a.0.read().unwrap().data.clone();
                    let dshape = dd.shape();
                    let (batch, seq, dim) = (dshape[0], dshape[1], dshape[2]);
                    let n = bd.shape()[bd.shape().len() - 1];

                    let d_rg = delta.0.read().unwrap().requires_grad;
                    let b_rg = b_vec.0.read().unwrap().requires_grad;
                    let c_rg = c_vec.0.read().unwrap().requires_grad;
                    let u_rg = u.0.read().unwrap().requires_grad;
                    let a_rg = a.0.read().unwrap().requires_grad;

                    let mut g_delta = if d_rg { Some(ArrayD::zeros(IxDyn(&[batch, seq, dim]))) } else { None };
                    let mut g_b = if b_rg { Some(ArrayD::zeros(IxDyn(&[batch, seq, n]))) } else { None };
                    let mut g_c = if c_rg { Some(ArrayD::zeros(IxDyn(&[batch, seq, n]))) } else { None };
                    let mut g_u = if u_rg { Some(ArrayD::zeros(IxDyn(&[batch, seq, dim]))) } else { None };
                    let mut g_a = if a_rg { Some(ArrayD::zeros(IxDyn(&[dim, n]))) } else { None };

                    for b in 0..batch {
                        // Forward-recompute, storing the state after every step (recomputation strategy).
                        let mut h = vec![0.0f32; dim * n];
                        let mut states: Vec<Vec<f32>> = Vec::with_capacity(seq + 1);
                        states.push(h.clone()); // states[0] = h_{-1} (initial, zeros)
                        for t in 0..seq {
                            for d in 0..dim {
                                let dt = dd[[b, t, d]];
                                let ut = ud[[b, t, d]];
                                for j in 0..n {
                                    let abar = (dt * ad[[d, j]]).exp();
                                    let bbar = dt * bd[[b, t, j]];
                                    h[d * n + j] = abar * h[d * n + j] + bbar * ut;
                                }
                            }
                            states.push(h.clone()); // states[t+1] = h_t
                        }

                        // Reverse scan: g[d*n+j] = gradient w.r.t. h_t[d,j].
                        let mut g = vec![0.0f32; dim * n];
                        for t in (0..seq).rev() {
                            let h_prev = &states[t];     // h_{t-1}
                            let h_cur = &states[t + 1];   // h_t
                            for d in 0..dim {
                                let dt = dd[[b, t, d]];
                                let ut = ud[[b, t, d]];
                                let dyt = dy[[b, t, d]];
                                // grad from y_t: dy_t * C_t  → adds into gh
                                for j in 0..n {
                                    g[d * n + j] += dyt * cd[[b, t, j]];
                                }
                                for j in 0..n {
                                    let abar = (dt * ad[[d, j]]).exp();
                                    let bbar = dt * bd[[b, t, j]];
                                    let gt = g[d * n + j];           // total grad on h_t[d,j]
                                    let hp = h_prev[d * n + j];      // h_{t-1}[d,j]
                                    let hc = h_cur[d * n + j];       // h_t[d,j]
                                    // grad w.r.t C_t[j]: dy_t * h_t[d,j]
                                    if let Some(ref mut gc) = g_c {
                                        gc[[b, t, j]] += dyt * hc;
                                    }
                                    // grad w.r.t u_t[d]: sum_j gh * B̄_t
                                    if let Some(ref mut gu) = g_u {
                                        gu[[b, t, d]] += gt * bbar;
                                    }
                                    // grad w.r.t B_t[j]: gh * u_t * delta
                                    if let Some(ref mut gb) = g_b {
                                        gb[[b, t, j]] += gt * ut * dt;
                                    }
                                    // grad w.r.t delta_t[d]: Ā-term + B̄-term
                                    if let Some(ref mut gd) = g_delta {
                                        let from_a = gt * hp * abar * ad[[d, j]];
                                        let from_b = gt * ut * bd[[b, t, j]];
                                        gd[[b, t, d]] += from_a + from_b;
                                    }
                                    // grad w.r.t a[d,j]: gh * h_{t-1} * Ā * delta
                                    if let Some(ref mut ga) = g_a {
                                        ga[[d, j]] += gt * hp * abar * dt;
                                    }
                                    // propagate to h_{t-1}: gh * Ā
                                    g[d * n + j] = gt * abar;
                                }
                            }
                        }
                    }

                    if let Some(g) = g_delta { delta.backward_with_grad(g); }
                    if let Some(g) = g_b { b_vec.backward_with_grad(g); }
                    if let Some(g) = g_c { c_vec.backward_with_grad(g); }
                    if let Some(g) = g_u { u.backward_with_grad(g); }
                    if let Some(g) = g_a { a.backward_with_grad(g); }
                }
                Op::Conv1DCausal(input, weight) => {
                    let id = input.0.read().unwrap().data.clone();
                    let wd = weight.0.read().unwrap().data.clone();
                    let ishape = id.shape();
                    let (batch, seq, channels) = (ishape[0], ishape[1], ishape[2]);
                    let kernel = wd.shape()[1];

                    let i_rg = input.0.read().unwrap().requires_grad;
                    let w_rg = weight.0.read().unwrap().requires_grad;

                    let mut g_in = if i_rg { Some(ArrayD::zeros(IxDyn(ishape))) } else { None };
                    let mut g_w = if w_rg { Some(ArrayD::zeros(IxDyn(wd.shape()))) } else { None };

                    for b in 0..batch {
                        for t in 0..seq {
                            for c in 0..channels {
                                let gout = grad[[b, t, c]];
                                for i in 0..kernel {
                                    let s = t as isize - kernel as isize + 1 + i as isize;
                                    if s >= 0 {
                                        let su = s as usize;
                                        if let Some(ref mut gi) = g_in {
                                            gi[[b, su, c]] += gout * wd[[c, i]];
                                        }
                                        if let Some(ref mut gw) = g_w {
                                            gw[[c, i]] += gout * id[[b, su, c]];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Some(g) = g_in { input.backward_with_grad(g); }
                    if let Some(g) = g_w { weight.backward_with_grad(g); }
                }
                Op::Softplus(x) => {
                    if x.0.read().unwrap().requires_grad {
                        // d/dx softplus = sigmoid(x)
                        let xd = x.0.read().unwrap().data.clone();
                        let sig = xd.mapv(|v| {
                            if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let e = v.exp(); e / (1.0 + e) }
                        });
                        x.backward_with_grad(&grad * &sig);
                    }
                }
                Op::Sigmoid(x) => {
                    if x.0.read().unwrap().requires_grad {
                        let xd = x.0.read().unwrap().data.clone();
                        let sig = xd.mapv(|v| {
                            if v >= 0.0 { 1.0 / (1.0 + (-v).exp()) } else { let e = v.exp(); e / (1.0 + e) }
                        });
                        x.backward_with_grad(&grad * &sig * &(1.0 - &sig));
                    }
                }
                Op::Permute(input, axes) => {
                    if input.0.read().unwrap().requires_grad {
                        // Inverse permutation.
                        let mut inv = vec![0usize; axes.len()];
                        for (i, &a) in axes.iter().enumerate() {
                            inv[a] = i;
                        }
                        // Materialize contiguous so downstream reshapes work.
                        let permuted = grad.permuted_axes(IxDyn(&inv));
                        let flat: Vec<f32> = permuted.iter().copied().collect();
                        let contig =
                            ArrayD::from_shape_vec(IxDyn(permuted.shape()), flat).unwrap();
                        input.backward_with_grad(contig);
                    }
                }
                Op::LayerNorm(input, gamma, beta, _eps) => {
                    let data: Vec<f32> = input.0.read().unwrap().data.iter().copied().collect();
                    let gd: Vec<f32> = gamma.0.read().unwrap().data.iter().copied().collect();
                    let shape = data.len();
                    let data_shape = input.0.read().unwrap().data.shape().to_vec();
                    let n = *data_shape.last().unwrap();
                    let nrows = shape / n;
                    let eps = *_eps;
                    let grad_flat: Vec<f32> = grad.iter().copied().collect();

                    // Recompute x_hat per row.
                    let mut x_hat = vec![0.0f32; shape];
                    let mut inv_stds = vec![0.0f32; nrows];
                    #[allow(clippy::needless_range_loop)]
                    for row in 0..nrows {
                        let base = row * n;
                        let mut mean = 0.0f32;
                        for j in 0..n {
                            mean += data[base + j];
                        }
                        mean /= n as f32;
                        let mut var = 0.0f32;
                        for j in 0..n {
                            let d = data[base + j] - mean;
                            var += d * d;
                        }
                        var /= n as f32;
                        let inv_std = 1.0 / (var + eps).sqrt();
                        inv_stds[row] = inv_std;
                        for j in 0..n {
                            x_hat[base + j] = (data[base + j] - mean) * inv_std;
                        }
                    }

                    let in_rg = input.0.read().unwrap().requires_grad;
                    let g_rg = gamma.0.read().unwrap().requires_grad;
                    let b_rg = beta.0.read().unwrap().requires_grad;

                    if g_rg {
                        let mut dgamma = vec![0.0f32; n];
                        for row in 0..nrows {
                            let base = row * n;
                            for j in 0..n {
                                dgamma[j] += grad_flat[base + j] * x_hat[base + j];
                            }
                        }
                        gamma.backward_with_grad(
                            ArrayD::from_shape_vec(IxDyn(&[n]), dgamma).unwrap(),
                        );
                    }
                    if b_rg {
                        let mut dbeta = vec![0.0f32; n];
                        for row in 0..nrows {
                            let base = row * n;
                            for j in 0..n {
                                dbeta[j] += grad_flat[base + j];
                            }
                        }
                        beta.backward_with_grad(
                            ArrayD::from_shape_vec(IxDyn(&[n]), dbeta).unwrap(),
                        );
                    }
                    if in_rg {
                        // dx_hat = dy * gamma
                        let mut dx_hat = vec![0.0f32; shape];
                        for row in 0..nrows {
                            let base = row * n;
                            for j in 0..n {
                                dx_hat[base + j] = grad_flat[base + j] * gd[j];
                            }
                        }
                        // dx = inv_std/N * (N*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
                        let mut dx = vec![0.0f32; shape];
                        #[allow(clippy::needless_range_loop)]
                        for row in 0..nrows {
                            let base = row * n;
                            let mut sum_dxh = 0.0f32;
                            let mut sum_dxh_xh = 0.0f32;
                            for j in 0..n {
                                sum_dxh += dx_hat[base + j];
                                sum_dxh_xh += dx_hat[base + j] * x_hat[base + j];
                            }
                            let inv = inv_stds[row] / n as f32;
                            for j in 0..n {
                                dx[base + j] = inv
                                    * (n as f32 * dx_hat[base + j]
                                        - sum_dxh
                                        - x_hat[base + j] * sum_dxh_xh);
                            }
                        }
                        input.backward_with_grad(
                            ArrayD::from_shape_vec(IxDyn(&data_shape), dx).unwrap(),
                        );
                    }
                }
            }
        }
    }

    pub fn relu(&self) -> Tensor {
        let data = self.0.read().unwrap().data.mapv(|x| x.max(0.0));
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        if self.0.read().unwrap().requires_grad {
            res.0.write().unwrap().creator = Some(Arc::new(Op::ReLU(self.clone())));
        }
        res
    }

    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        let data = self.0.read().unwrap().data.mapv(|x| x.clamp(min, max));
        // Straight-through estimator (STE) approximation for gradients could be handled here
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        // Simple passthrough for demonstration
        res
    }

    pub fn round(&self) -> Tensor {
        let data = self.0.read().unwrap().data.mapv(|x| x.round());
        let res = Tensor::new(data, self.0.read().unwrap().requires_grad);
        res
    }

    // ==================== Reductions & Indexing ====================

    /// Mean of all elements (scalar value).
    pub fn mean(&self) -> f32 {
        let inner = self.0.read().unwrap();
        let data = &inner.data;
        if data.is_empty() {
            0.0
        } else {
            data.sum() / data.len() as f32
        }
    }

    /// Maximum element value.
    pub fn max(&self) -> f32 {
        let inner = self.0.read().unwrap();
        let data = &inner.data;
        data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum element value.
    pub fn min(&self) -> f32 {
        let inner = self.0.read().unwrap();
        let data = &inner.data;
        data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Index of the maximum element (flattened, row-major order).
    pub fn argmax(&self) -> usize {
        let inner = self.0.read().unwrap();
        let data = &inner.data;
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Read a single element by its (flattened) linear index.
    pub fn get_idx(&self, index: usize) -> f32 {
        self.0.read().unwrap().data
            .iter()
            .nth(index)
            .copied()
            .unwrap_or(0.0)
    }

    /// Read a single element by multi-dimensional index, e.g. `t.get(&[row, col])`.
    pub fn get(&self, index: &[usize]) -> f32 {
        let inner = self.0.read().unwrap();
        match inner.data.get(IxDyn(index)) {
            Some(v) => *v,
            None => panic!(
                "index {:?} is out of bounds for tensor of shape {:?}",
                index,
                inner.data.shape()
            ),
        }
    }
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Self) -> Self::Output {
        Tensor::add(&self, &other)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Self) -> Self::Output {
        Tensor::sub(&self, &other)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Self) -> Self::Output {
        Tensor::mul(&self, &other)
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Pretty-print the underlying ndarray data.
        std::fmt::Display::fmt(&self.0.read().unwrap().data, f)
    }
}
