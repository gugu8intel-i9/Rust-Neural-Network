//! # Compiled training step — zero-autograd-overhead MLP training.
//!
//! Traces the MLP architecture once into pre-allocated flat buffers. Every step is just
//! raw `sgemm` + elementwise loops — zero allocation, zero locking, zero graph traversal.

use crate::blas::{sgemm, Transpose};
use std::slice;

#[derive(Clone, Copy, PartialEq)]
enum Act {
    ReLU,
    Identity,
}

pub struct CompiledMLP {
    dims: Vec<(usize, usize)>,
    acts: Vec<Act>,
    batch: usize,
    lr: f32,
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    mw: Vec<Vec<f32>>,
    vw: Vec<Vec<f32>>,
    mb: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
    t: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    z: Vec<Vec<f32>>,
    a: Vec<Vec<f32>>,
    grad_a: Vec<f32>,
    grad_b: Vec<f32>,
    dweight: Vec<Vec<f32>>,
    dbias: Vec<Vec<f32>>,
}

impl CompiledMLP {
    pub fn new(dims: &[(usize, usize)], batch: usize, lr: f32) -> Self {
        let n = dims.len();
        let max_dim = dims.iter().flat_map(|&(i, o)| [i, o]).max().unwrap_or(1);
        let mut out = CompiledMLP {
            dims: dims.to_vec(),
            acts: (0..n)
                .map(|i| if i + 1 < n { Act::ReLU } else { Act::Identity })
                .collect(),
            batch,
            lr,
            weights: Vec::new(),
            biases: Vec::new(),
            mw: Vec::new(),
            vw: Vec::new(),
            mb: Vec::new(),
            vb: Vec::new(),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            z: Vec::new(),
            a: Vec::new(),
            grad_a: vec![0.0; batch * max_dim],
            grad_b: vec![0.0; batch * max_dim],
            dweight: Vec::new(),
            dbias: Vec::new(),
        };
        for &(lin, lout) in dims {
            let std = (2.0 / lin as f32).sqrt();
            let wlen = lout * lin;
            let mut w = vec![0.0f32; wlen];
            let mut seed = lin.wrapping_mul(31).wrapping_add(lout) as u32;
            for v in &mut w {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (((seed >> 8) as f32 / 16777216.0) - 0.5) * 2.0 * std;
            }
            out.weights.push(w);
            out.biases.push(vec![0.0; lout]);
            out.mw.push(vec![0.0; wlen]);
            out.vw.push(vec![0.0; wlen]);
            out.mb.push(vec![0.0; lout]);
            out.vb.push(vec![0.0; lout]);
            out.z.push(vec![0.0; batch * lout]);
            out.a.push(vec![0.0; batch * lout]);
            out.dweight.push(vec![0.0; wlen]);
            out.dbias.push(vec![0.0; lout]);
        }
        out
    }

    /// One training step: forward + MSE + backward + Adam. Returns the loss. Zero allocation.
    pub fn step(&mut self, x: &[f32], target: &[f32]) -> f32 {
        let b = self.batch;
        let n = self.dims.len();

        // ===== Forward =====
        for l in 0..n {
            let (lin, lout) = self.dims[l];
            let (iptr, ilen) = if l == 0 {
                (x.as_ptr(), x.len())
            } else {
                (self.a[l - 1].as_ptr(), self.a[l - 1].len())
            };
            let input = unsafe { slice::from_raw_parts(iptr, ilen) };
            let w = self.weights[l].as_ptr();
            let bias = self.biases[l].as_ptr();
            let z = self.z[l].as_mut_ptr();
            let a = self.a[l].as_mut_ptr();
            let z_s = unsafe { slice::from_raw_parts_mut(z, b * lout) };
            let a_s = unsafe { slice::from_raw_parts_mut(a, b * lout) };
            let w_s = unsafe { slice::from_raw_parts(w, lout * lin) };
            let b_s = unsafe { slice::from_raw_parts(bias, lout) };
            sgemm(
                Transpose::NoTrans,
                Transpose::Trans,
                b,
                lout,
                lin,
                1.0,
                input,
                lin,
                w_s,
                lin,
                0.0,
                z_s,
                lout,
            );
            if self.acts[l] == Act::ReLU {
                for i in 0..b {
                    let r = i * lout;
                    for o in 0..lout {
                        let v = z_s[r + o] + b_s[o];
                        z_s[r + o] = v;
                        a_s[r + o] = v.max(0.0);
                    }
                }
            } else {
                for i in 0..b {
                    let r = i * lout;
                    for o in 0..lout {
                        let v = z_s[r + o] + b_s[o];
                        z_s[r + o] = v;
                        a_s[r + o] = v;
                    }
                }
            }
        }

        // ===== Loss + seed grad =====
        let out_len = b * self.dims[n - 1].1;
        let scale = 2.0 / out_len as f32;
        let mut loss = 0.0f32;
        for i in 0..out_len {
            let d = self.a[n - 1][i] - target[i];
            loss += d * d;
            self.grad_a[i] = d * scale;
        }
        loss /= out_len as f32;

        // ===== Backward =====
        let mut cur_a = true;
        for l in (0..n).rev() {
            let (lin, lout) = self.dims[l];
            let glen = b * lout;

            // ReLU mask in-place
            if self.acts[l] == Act::ReLU {
                let zl = self.z[l].as_ptr();
                let gptr = if cur_a {
                    self.grad_a.as_mut_ptr()
                } else {
                    self.grad_b.as_mut_ptr()
                };
                unsafe {
                    let zl_s = slice::from_raw_parts(zl, glen);
                    let g_s = slice::from_raw_parts_mut(gptr, glen);
                    for i in 0..glen {
                        if zl_s[i] <= 0.0 {
                            g_s[i] = 0.0;
                        }
                    }
                }
            }

            // Raw pointers for zero-conflict reads
            let gptr = if cur_a {
                self.grad_a.as_ptr()
            } else {
                self.grad_b.as_ptr()
            };
            let (iptr, ilen) = if l == 0 {
                (x.as_ptr(), x.len())
            } else {
                (self.a[l - 1].as_ptr(), self.a[l - 1].len())
            };
            let wptr = self.weights[l].as_ptr();
            let dwptr = self.dweight[l].as_mut_ptr();
            let dbptr = self.dbias[l].as_mut_ptr();

            unsafe {
                let g = slice::from_raw_parts(gptr, glen);
                let input = slice::from_raw_parts(iptr, ilen);
                let w = slice::from_raw_parts(wptr, lout * lin);
                let dw = slice::from_raw_parts_mut(dwptr, lout * lin);
                let db = slice::from_raw_parts_mut(dbptr, lout);

                // dW = g^T @ input
                sgemm(
                    Transpose::Trans,
                    Transpose::NoTrans,
                    lout,
                    lin,
                    b,
                    1.0,
                    g,
                    lout,
                    input,
                    lin,
                    0.0,
                    dw,
                    lin,
                );
                // db = Σ g
                for o in 0..lout {
                    db[o] = 0.0;
                }
                for i in 0..b {
                    let r = i * lout;
                    for o in 0..lout {
                        db[o] += g[r + o];
                    }
                }
                // dx = g @ W
                if l > 0 {
                    let gnptr = if cur_a {
                        self.grad_b.as_mut_ptr()
                    } else {
                        self.grad_a.as_mut_ptr()
                    };
                    let gn = slice::from_raw_parts_mut(gnptr, b * lin);
                    sgemm(
                        Transpose::NoTrans,
                        Transpose::NoTrans,
                        b,
                        lin,
                        lout,
                        1.0,
                        g,
                        lout,
                        w,
                        lin,
                        0.0,
                        gn,
                        lin,
                    );
                }
            }
            cur_a = !cur_a;
        }

        // ===== Adam (AVX2-accelerated, zero allocation) =====
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr_t = self.lr * bc2.sqrt() / bc1;
        let (b1, b2, eps, ob1, ob2) = (
            self.beta1,
            self.beta2,
            self.eps,
            1.0 - self.beta1,
            1.0 - self.beta2,
        );

        for l in 0..n {
            let (lin, lout) = self.dims[l];
            let wlen = lout * lin;
            // Weights
            let g = &self.dweight[l];
            let m = &mut self.mw[l];
            let v = &mut self.vw[l];
            let w = &mut self.weights[l];
            debug_assert_eq!(g.len(), m.len());
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    unsafe {
                        crate::optim::adam_update_avx2(g, m, v, w, b1, b2, ob1, ob2, eps, lr_t)
                    };
                    // Biases
                    let gb = &self.dbias[l];
                    let mb = &mut self.mb[l];
                    let vb = &mut self.vb[l];
                    let bb = &mut self.biases[l];
                    unsafe {
                        crate::optim::adam_update_avx2(gb, mb, vb, bb, b1, b2, ob1, ob2, eps, lr_t)
                    };
                    continue;
                }
            }
            crate::optim::adam_update_scalar(g, m, v, w, b1, b2, ob1, ob2, eps, lr_t);
            let gb = &self.dbias[l];
            let mb = &mut self.mb[l];
            let vb = &mut self.vb[l];
            let bb = &mut self.biases[l];
            crate::optim::adam_update_scalar(gb, mb, vb, bb, b1, b2, ob1, ob2, eps, lr_t);
            let _ = (lin, lout, wlen);
        }
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiled_mlp_learns() {
        let mut m = CompiledMLP::new(&[(8, 16), (16, 16), (16, 4)], 8, 0.01);
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let t: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let l0 = m.step(&x, &t);
        for _ in 0..100 {
            m.step(&x, &t);
        }
        let l1 = m.step(&x, &t);
        assert!(
            l1 < l0 * 0.5,
            "compiled MLP did not learn: {l0:.4} -> {l1:.4}"
        );
    }
}
