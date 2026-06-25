//! Rotation-assisted quantization: **RotorQuant**.
//!
//! RotorQuant decorrelates vectors with sparse block-diagonal rotations drawn from the
//! Clifford algebra Cl(3,0) (the "rotor sandwich" `RxR̃`) before independently scalar-
//! quantizing each coordinate, then recovers the vector with the inverse rotation.
//!
//! The rotation homogenizes the per-coordinate distribution so a simple uniform scalar
//! quantizer reaches near-optimal distortion with only ~4 rotor parameters per 3-D block
//! instead of a dense d×d transform. This mirrors the RotorQuant / TurboQuant lineage of
//! KV-cache compression methods, adapted here as a pure-Rust tensor quantizer.
//!
//! References:
//!   - RotorQuant (Cl(3,0) rotors): <http://scrya.com/rotorquant/>
//!   - TurboQuant (dense random rotation + scalar quant): ICLR 2026.
//!   - Rotation-assisted PTQ overview.

use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Multivector component layout for Cl(3,0): `[1, e1, e2, e3, e12, e13, e23, e123]`.
type MV = [f32; 8];

/// A unit rotor in Cl(3,0): `R = s + p12·e12 + p13·e13 + p23·e23`.
/// Its "sandwich" `RxR̃` acts as an orthogonal rotation on grade-1 vectors.
#[derive(Debug, Clone, Copy)]
pub struct Rotor {
    pub s: f32,
    pub p12: f32,
    pub p13: f32,
    pub p23: f32,
}

impl Rotor {
    /// Build a random *unit* rotor from three small random bivector angles. A unit rotor
    /// guarantees the sandwich is a proper rotation (norm-preserving).
    pub fn random(rng: &mut impl Rng) -> Self {
        // Bivector generator B = a·e12 + b·e13 + c·e23, then R = exp(B/2).
        // For a unit bivector direction this reduces to R = cos(θ) + sin(θ)·B̂, with B̂ unit.
        let a: f32 = rng.gen_range(-1.0..1.0);
        let b: f32 = rng.gen_range(-1.0..1.0);
        let c: f32 = rng.gen_range(-1.0..1.0);
        let norm = (a * a + b * b + c * c).sqrt();
        let theta: f32 = rng.gen_range(0.0..std::f32::consts::PI);
        if norm < 1e-8 {
            // Degenerate: identity rotor.
            return Rotor { s: 1.0, p12: 0.0, p13: 0.0, p23: 0.0 };
        }
        let (ua, ub, uc) = (a / norm, b / norm, c / norm);
        let s = (theta / 2.0).cos();
        let mag = (theta / 2.0).sin();
        Rotor { s, p12: mag * ua, p13: mag * ub, p23: mag * uc }
    }

    /// Sparse geometric product `r = rotor * x` (28 FMAs vs 64 for the full table), from the
    /// RotorQuant reference implementation. `rotor` is `(s, p12, p13, p23)`, all other
    /// multivector components zero.
    fn apply(x: MV, s: f32, p12: f32, p13: f32, p23: f32) -> MV {
        [
            s * x[0] - p12 * x[4] - p13 * x[5] - p23 * x[6],
            s * x[1] + p12 * x[2] + p13 * x[3] + p23 * x[7],
            s * x[2] - p12 * x[1] + p23 * x[3] - p13 * x[7],
            s * x[3] - p13 * x[1] - p23 * x[2] + p12 * x[7],
            s * x[4] + p12 * x[0],
            s * x[5] + p13 * x[0],
            s * x[6] + p23 * x[0],
            s * x[7] - p23 * x[1] + p13 * x[2] - p12 * x[3],
        ]
    }

    /// Forward sandwich `R x R̃` (R̃ = reverse: negate bivector parts).
    pub fn sandwich(&self, x: MV) -> MV {
        let Rotor { s, p12, p13, p23 } = *self;
        let tmp = Self::apply(x, s, p12, p13, p23);
        // R̃ = (s, -p12, -p13, -p23)
        Self::apply(tmp, s, -p12, -p13, -p23)
    }

    /// Inverse sandwich `R̃ x R` — the exact inverse of [`sandwich`].
    pub fn inv_sandwich(&self, x: MV) -> MV {
        let Rotor { s, p12, p13, p23 } = *self;
        let tmp = Self::apply(x, s, -p12, -p13, -p23);
        Self::apply(tmp, s, p12, p13, p23)
    }

    /// The 3×3 orthogonal rotation induced by the sandwich on grade-1 vectors. Applied to a
    /// plain 3-vector this is equivalent to the full multivector sandwich (and far cheaper to
    /// compose into a block-diagonal rotation).
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let e1: MV = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2: MV = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e3: MV = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let r1 = self.sandwich(e1);
        let r2 = self.sandwich(e2);
        let r3 = self.sandwich(e3);
        [[r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]], [r1[3], r2[3], r3[3]]]
    }
}

/// RotorQuant: block-diagonal Cl(3,0)-rotor decorrelation + uniform scalar quantization.
///
/// Operates on the last tensor dimension (treated as a feature/dim vector); each group of 3
/// coordinates is rotated by its own random unit rotor, scalar-quantized, then inverse-rotated.
/// This is a *compression* quantizer (round-trips with quantization error), primarily intended
/// for inference-time KV-cache / activation quantization rather than training.
#[derive(Debug, Clone)]
pub struct RotorQuant {
    pub dim: usize,
    pub bits: u8,
    /// One rotor per 3-D group (groups operate on coordinates `[3g, 3g+1, 3g+2]`).
    pub rotors: Vec<Rotor>,
    /// Padded dim (next multiple of 3 >= `dim`).
    pub padded_dim: usize,
}

impl RotorQuant {
    /// Create a RotorQuant for vectors of length `dim`, quantizing to `bits` per coordinate
    /// (a deterministic rotor set is seeded for reproducibility).
    pub fn new(dim: usize, bits: u8) -> Self {
        assert!(bits >= 2, "RotorQuant needs at least 2 bits");
        let padded_dim = dim.div_ceil(3) * 3;
        let n_groups = padded_dim / 3;
        let mut rng = StdRng::seed_from_u64(0xC11F_C0DE_u64);
        let rotors = (0..n_groups).map(|_| Rotor::random(&mut rng)).collect();
        RotorQuant { dim, bits, rotors, padded_dim }
    }

    /// Create with a custom RNG seed.
    pub fn with_seed(dim: usize, bits: u8, seed: u64) -> Self {
        assert!(bits >= 2, "RotorQuant needs at least 2 bits");
        let padded_dim = dim.div_ceil(3) * 3;
        let n_groups = padded_dim / 3;
        let mut rng = StdRng::seed_from_u64(seed);
        let rotors = (0..n_groups).map(|_| Rotor::random(&mut rng)).collect();
        RotorQuant { dim, bits, rotors, padded_dim }
    }

    fn qmax(&self) -> f32 {
        ((1u32 << (self.bits - 1)) - 1) as f32
    }

    /// Quantize a single feature vector (length `padded_dim`), returning the dequantized vector.
    /// Each 3-group is rotated, uniform-quantized per group scale, then inverse-rotated.
    fn quantize_vec(&self, mut row: Vec<f32>) -> Vec<f32> {
        debug_assert_eq!(row.len(), self.padded_dim);
        let qmax = self.qmax();
        let n_groups = self.padded_dim / 3;
        for g in 0..n_groups {
            let rotor = self.rotors[g];
            let rot = rotor.rotation_matrix();
            // Rotate group.
            let v = [row[3 * g], row[3 * g + 1], row[3 * g + 2]];
            let mut rv = [
                rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
                rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
                rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
            ];
            // Per-group symmetric uniform scalar quantization.
            let mut max_abs = 0.0f32;
            for &x in &rv {
                max_abs = max_abs.max(x.abs());
            }
            if max_abs < 1e-12 {
                continue; // all-zero group: nothing to quantize.
            }
            let scale = max_abs / qmax;
            for x in rv.iter_mut() {
                let q = (*x / scale).round().clamp(-qmax, qmax);
                *x = q * scale;
            }
            // Inverse-rotate (rotation matrix is orthogonal => inverse = transpose).
            let out = [
                rot[0][0] * rv[0] + rot[1][0] * rv[1] + rot[2][0] * rv[2],
                rot[0][1] * rv[0] + rot[1][1] * rv[1] + rot[2][1] * rv[2],
                rot[0][2] * rv[0] + rot[1][2] * rv[1] + rot[2][2] * rv[2],
            ];
            row[3 * g] = out[0];
            row[3 * g + 1] = out[1];
            row[3 * g + 2] = out[2];
        }
        row
    }

    /// Quantize (and dequantize) a tensor along its last dimension. Each feature vector is
    /// padded to a multiple of 3 with zeros, quantized, then trimmed back to `dim`.
    pub fn quantize(&self, input: &Tensor) -> Tensor {
        let data = input.data();
        let shape = data.shape();
        assert!(
            !shape.is_empty() && shape[shape.len() - 1] == self.dim,
            "RotorQuant configured for dim {} but got last-dim {}",
            self.dim,
            shape.last().copied().unwrap_or(0)
        );

        let rows = data.len() / self.dim;
        let flat: Vec<f32> = data.iter().copied().collect();
        let mut out = Vec::with_capacity(data.len());

        for r in 0..rows {
            let base = r * self.dim;
            let mut row = vec![0.0f32; self.padded_dim];
            for (t, slot) in row.iter_mut().enumerate().take(self.dim) {
                *slot = flat[base + t];
            }
            let qrow = self.quantize_vec(row);
            out.extend(qrow.iter().take(self.dim));
        }

        let out = ArrayD::from_shape_vec(IxDyn(shape), out).expect("rotorquant output shape");
        // Quantization is a compression op: gradients do not flow through it.
        Tensor::new(out, false)
    }
}

impl crate::nn::Module for RotorQuant {
    /// Round-trip quantization (quantize then dequantize). As a hard quantizer it does not
    /// propagate gradients; use [`crate::nn::FakeQuantize`] for straight-through QAT gradients.
    fn forward(&self, input: &Tensor) -> Tensor {
        self.quantize(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotor_is_norm_preserving() {
        // The sandwich of a unit rotor must preserve vector norms (orthogonal rotation).
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let r = Rotor::random(&mut rng);
            let rot = r.rotation_matrix();
            // Orthogonality: R R^T == I.
            for i in 0..3 {
                for j in 0..3 {
                    let dot = (0..3).map(|k| rot[i][k] * rot[j][k]).sum::<f32>();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-4, "rotor not orthogonal: {dot}");
                }
            }
        }
    }

    #[test]
    fn sandwich_inverse_round_trips() {
        let mut rng = StdRng::seed_from_u64(7);
        let r = Rotor::random(&mut rng);
        let x: MV = [0.0, 0.3, -1.7, 2.1, 0.0, 0.0, 0.0, 0.0];
        let y = r.sandwich(x);
        let x_rec = r.inv_sandwich(y);
        for i in 0..8 {
            assert!((x[i] - x_rec[i]).abs() < 1e-5, "sandwich inverse failed at {i}");
        }
    }

    #[test]
    fn rotorquant_round_trips_with_bounded_error() {
        let dim = 12;
        let rq = RotorQuant::with_seed(dim, 4, 123);
        let data: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.37 - 1.5).collect();
        let t = Tensor::from_vec(data.clone(), vec![dim]);
        let q = rq.quantize(&t);
        let qd: Vec<f32> = q.data().iter().copied().collect();
        let mut max_err = 0.0f32;
        for i in 0..dim {
            max_err = max_err.max((qd[i] - data[i]).abs());
        }
        // 4-bit per-group quant: error should be a small fraction of the value range.
        let range = data.iter().copied().fold(-f32::INFINITY, f32::max)
            - data.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_err < 0.15 * range, "quantization error too large: {max_err} (range {range})");
    }

    #[test]
    fn rotorquant_preserves_shape_and_handles_remainder() {
        // dim not divisible by 3.
        let dim = 10;
        let rq = RotorQuant::new(dim, 3);
        let rows = 2;
        let t = Tensor::from_vec((0..rows * dim).map(|i| i as f32).collect(), vec![rows, dim]);
        let q = rq.quantize(&t);
        assert_eq!(q.shape(), vec![rows, dim]);
    }

    #[test]
    fn rotorquant_batched_2d() {
        let dim = 6;
        let rq = RotorQuant::new(dim, 4);
        let t = Tensor::randn(&[5, dim]);
        let q = rq.quantize(&t);
        assert_eq!(q.shape(), vec![5, dim]);
    }
}
