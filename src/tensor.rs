//! High-performance tensor implementation using ndarray and rayon.
//!
//! Tensors support automatic differentiation (autograd) by tracking
//! operations in a computational graph.

use ndarray::{ArrayD, IxDyn};
use ndarray_rand::rand_distr::{Normal, StandardNormal};
use ndarray_rand::RandomExt;
use std::sync::{Arc, RwLock};
use std::ops::{Add, Mul, Sub};

/// Internal data for a Tensor, including data, gradients, and graph info.
#[derive(Debug)]
pub struct TensorData {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub requires_grad: bool,
    pub creator: Option<Arc<Op>>,
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
                    if a.0.read().unwrap().requires_grad { a.backward_with_grad(grad.clone()); }
                    if b.0.read().unwrap().requires_grad { b.backward_with_grad(grad); }
                }
                Op::Sub(a, b) => {
                    if a.0.read().unwrap().requires_grad { a.backward_with_grad(grad.clone()); }
                    if b.0.read().unwrap().requires_grad { b.backward_with_grad(-grad); }
                }
                Op::Mul(a, b) => {
                    if a.0.read().unwrap().requires_grad {
                        let b_data = b.0.read().unwrap().data.clone();
                        a.backward_with_grad(&grad * &b_data);
                    }
                    if b.0.read().unwrap().requires_grad {
                        let a_data = a.0.read().unwrap().data.clone();
                        b.backward_with_grad(&grad * &a_data);
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
}

impl Add for Tensor {
    type Output = Tensor;
    fn add(self, other: Self) -> Self::Output { (&self).add(&other) }
}

impl Sub for Tensor {
    type Output = Tensor;
    fn sub(self, other: Self) -> Self::Output { (&self).sub(&other) }
}

impl Mul for Tensor {
    type Output = Tensor;
    fn mul(self, other: Self) -> Self::Output { (&self).mul(&other) }
}
