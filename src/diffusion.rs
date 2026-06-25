//! Diffusion models: Denoising Diffusion Probabilistic Models (DDPM).
//!
//! Implements the full DDPM pipeline:
//!   - **Forward process** `q(x_t | x_0)`: progressively corrupt data with Gaussian noise,
//!     `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε` (closed form).
//!   - **Reverse process**: a denoising network `ε_θ(x_t, t)` predicts the added noise,
//!     conditioned on the timestep via a sinusoidal embedding.
//!   - **Training**: simple MSE between predicted and true noise, `L = E[‖ε − ε_θ(x_t,t)‖²]`.
//!   - **Sampling**: iterative denoising from pure noise `x_T ~ N(0, I)`.
//!
//! Supports linear and cosine ("Improved DDPM") noise schedules.

use crate::nn::{Linear, Module, Sequential, SiLU};
use crate::optim::Optimizer;
use crate::tensor::Tensor;

/// Noise schedule type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleType {
    /// Linear β schedule (Ho et al., 2020).
    Linear,
    /// Cosine schedule (Nichol & Dhariwal, "Improved DDPM", 2021).
    Cosine,
}

/// Precomputed DDPM noise schedule: β_t, α_t, ᾱ_t and their square roots.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    pub timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alpha_cumprod: Vec<f32>,
    pub sqrt_alpha_cumprod: Vec<f32>,
    pub sqrt_one_minus_alpha_cumprod: Vec<f32>,
}

impl NoiseSchedule {
    /// Build a schedule of `timesteps` steps.
    pub fn new(schedule: ScheduleType, timesteps: usize) -> Self {
        let betas = match schedule {
            ScheduleType::Linear => {
                let beta_start = 0.0001;
                let beta_end = 0.02;
                (0..timesteps)
                    .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / ((timesteps - 1) as f32))
                    .collect()
            }
            ScheduleType::Cosine => {
                let max_beta = 0.999;
                let mut betas = Vec::with_capacity(timesteps);
                for t in 0..timesteps {
                    let t1 = (t as f32) / (timesteps as f32);
                    let t2 = ((t + 1) as f32) / (timesteps as f32);
                    let f = |x: f32| ((x * std::f32::consts::PI / 2.0).cos()).powi(2);
                    betas.push((1.0 - f(t2) / f(t1)).min(max_beta));
                }
                if betas[0] < 1e-6 {
                    betas[0] = 1e-6;
                }
                betas
            }
        };

        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
        let mut alpha_cumprod = Vec::with_capacity(timesteps);
        let mut acc = 1.0f32;
        for &a in &alphas {
            acc *= a;
            alpha_cumprod.push(acc.max(1e-12));
        }
        let sqrt_alpha_cumprod: Vec<f32> = alpha_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alpha_cumprod: Vec<f32> =
            alpha_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();

        NoiseSchedule {
            timesteps,
            betas,
            alphas,
            alpha_cumprod,
            sqrt_alpha_cumprod,
            sqrt_one_minus_alpha_cumprod,
        }
    }
}

/// Sinusoidal timestep embedding (fixed, non-learned), as used in transformers/diffusion.
pub fn sinusoidal_embedding(t: f32, dim: usize) -> Tensor {
    let half = dim / 2;
    let max_period: f32 = 10000.0;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(max_period.ln()) * (i as f32) / (half.max(1) as f32)).exp())
        .collect();
    let mut emb = Vec::with_capacity(dim);
    for &f in &freqs {
        let arg = t * f;
        emb.push(arg.sin());
        emb.push(arg.cos());
    }
    while emb.len() < dim {
        emb.push(0.0);
    }
    Tensor::from_vec(emb, vec![1, dim])
}

/// A simple MLP denoising network that predicts noise ε from (x_t, t).
///
/// Conditions on the timestep by projecting a sinusoidal embedding and adding it to the input,
/// then running a 2-hidden-layer MLP with SiLU activations.
#[derive(Debug, Clone)]
pub struct DenoiseNet {
    pub data_dim: usize,
    pub time_embed_dim: usize,
    /// Maps the time embedding to `data_dim` for input conditioning.
    pub time_proj: Sequential,
    /// Data -> hidden -> hidden -> data.
    pub net: Sequential,
}

impl DenoiseNet {
    pub fn new(data_dim: usize, hidden_dim: usize, time_embed_dim: usize) -> Self {
        let time_proj = Sequential::new()
            .add(Linear::new(time_embed_dim, hidden_dim, true))
            .add(SiLU)
            .add(Linear::new(hidden_dim, data_dim, true));
        let net = Sequential::new()
            .add(Linear::new(data_dim, hidden_dim, true))
            .add(SiLU)
            .add(Linear::new(hidden_dim, hidden_dim, true))
            .add(SiLU)
            .add(Linear::new(hidden_dim, data_dim, true));
        DenoiseNet { data_dim, time_embed_dim, time_proj, net }
    }

    /// Predict the noise in `x_t` at timestep `t`. Returns a tensor shaped like `x_t`.
    pub fn predict(&self, x_t: &Tensor, t: usize) -> Tensor {
        // Time conditioning: sinusoidal(t) -> data_dim, broadcast-added to the input.
        let temb = sinusoidal_embedding(t as f32, self.time_embed_dim);
        let tcond = self.time_proj.forward(&temb).reshape(&[self.data_dim]); // [data_dim]
        let conditioned = x_t.add(&tcond); // broadcast over batch
        self.net.forward(&conditioned)
    }
}

impl Module for DenoiseNet {
    /// Runs the data MLP only (use [`DenoiseNet::predict`] for timestep-conditioned prediction).
    fn forward(&self, input: &Tensor) -> Tensor {
        self.net.forward(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.time_proj.parameters();
        p.extend(self.net.parameters());
        p
    }
}

/// A full DDPM: noise schedule + denoising network, with training and sampling.
#[derive(Debug, Clone)]
pub struct DDPM {
    pub data_dim: usize,
    pub schedule: NoiseSchedule,
    pub denoise: DenoiseNet,
}

impl DDPM {
    /// Create a DDPM over `data_dim`-dimensional data with `timesteps` denoising steps.
    pub fn new(data_dim: usize, hidden_dim: usize, timesteps: usize, schedule: ScheduleType) -> Self {
        let schedule = NoiseSchedule::new(schedule, timesteps);
        let denoise = DenoiseNet::new(data_dim, hidden_dim, data_dim.clamp(16, 128));
        DDPM { data_dim, schedule, denoise }
    }

    /// Forward (noising) process: `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`.
    pub fn q_sample(&self, x_0: &Tensor, t: usize, noise: &Tensor) -> Tensor {
        let c1 = self.schedule.sqrt_alpha_cumprod[t];
        let c2 = self.schedule.sqrt_one_minus_alpha_cumprod[t];
        let xd = x_0.data();
        let nd = noise.data();
        let out = (&xd * c1) + (&nd * c2);
        Tensor::new(out, x_0.0.read().unwrap().requires_grad)
    }

    /// Predict the noise in `x_t` at timestep `t`.
    pub fn predict_noise(&self, x_t: &Tensor, t: usize) -> Tensor {
        self.denoise.predict(x_t, t)
    }

    /// One DDPM training step: samples a shared timestep `t`, noises the batch, predicts the
    /// noise, and takes an optimizer step on the MSE noise-prediction loss. Returns the loss.
    pub fn train_batch<O: Optimizer>(&self, optimizer: &mut O, x_0: &Tensor) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let t = rng.gen_range(0..self.schedule.timesteps);

        let shape = x_0.shape();
        let n_elems = shape.iter().product::<usize>();
        let noise_data: Vec<f32> = (0..n_elems).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let noise = Tensor::from_vec(noise_data, shape.clone());

        let x_t = self.q_sample(x_0, t, &noise);

        optimizer.zero_grad();
        let predicted = self.predict_noise(&x_t, t);

        let diff = predicted.sub(&noise);
        let loss = diff.mul(&diff).sum();

        loss.backward();
        optimizer.step();

        loss.data().iter().copied().next().unwrap_or(0.0) / (n_elems as f32)
    }

    /// Generate samples from pure noise by iteratively denoising (ancestral sampling).
    pub fn sample(&self, batch_size: usize) -> Tensor {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total = batch_size * self.data_dim;
        let mut x: Vec<f32> = (0..total).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        for t in (0..self.schedule.timesteps).rev() {
            let x_t = Tensor::from_vec(x.clone(), vec![batch_size, self.data_dim]);
            let pred_noise = self.predict_noise(&x_t, t);
            let pn: Vec<f32> = pred_noise.data().iter().copied().collect();

            let alpha = self.schedule.alphas[t];
            let alpha_bar = self.schedule.alpha_cumprod[t];
            let beta = self.schedule.betas[t];

            for i in 0..total {
                let mean = (1.0 / alpha.sqrt()) * (x[i] - (beta / (1.0 - alpha_bar).sqrt()) * pn[i]);
                if t > 0 {
                    let z: f32 = rng.gen::<f32>() * 2.0 - 1.0;
                    x[i] = mean + beta.sqrt() * z;
                } else {
                    x[i] = mean;
                }
            }
        }

        Tensor::from_vec(x, vec![batch_size, self.data_dim])
    }

    /// All learnable parameters (the denoising network).
    pub fn parameters(&self) -> Vec<Tensor> {
        self.denoise.parameters()
    }
}

impl Module for DDPM {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.denoise.forward(input)
    }
    fn parameters(&self) -> Vec<Tensor> {
        self.denoise.parameters()
    }
}
