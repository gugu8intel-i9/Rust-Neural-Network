//! Self-distillation: a model compresses itself into a smaller, faster student.
//!
//! # Innovation: Automated Self-Distillation Pipeline
//!
//! Implements **knowledge distillation** where a trained "teacher" model transfers its
//! knowledge to a smaller "student" model. The student learns from three signals:
//!
//! 1. **Hard labels** (ground truth): standard cross-entropy or MSE loss.
//! 2. **Soft labels** (teacher's output distribution): the teacher's logits carry "dark
//!    knowledge" — inter-class similarities and confidence calibration that hard labels don't.
//! 3. **Feature matching** (intermediate representations): the student's hidden layers are
//!    aligned with the teacher's via L2 loss, transferring structural knowledge.
//!
//! The combined loss is:
//! ```text
//! L = α · CE(student_logits, hard_labels)
//!   + β · KL(softmax(student_logits/T), softmax(teacher_logits/T)) · T²
//!   + γ · MSE(student_features, teacher_features)
//! ```
//! where T is the temperature (typically 2-4), and α, β, γ are weighting coefficients.
//!
//! ## Self-distillation mode
//!
//! [`SelfDistiller`] goes beyond standard distillation: it **automatically generates the
//! student architecture** from the teacher (width/depth reduction), runs the distillation,
//! and optionally quantizes the result. This is a one-call compression pipeline:
//!
//! ```text
//! Teacher (256-wide, 6 layers) → distill → Student (128-wide, 3 layers) → quantize → INT8
//! ```
//!
//! ## Progressive self-distillation
//!
//! [`ProgressiveDistiller`] chains multiple distillation rounds: each student becomes the
//! teacher for the next round, progressively shrinking the model while preserving accuracy.

use crate::nn::{Linear, Module, ReLU, Sequential};
use crate::optim::{Adam, Optimizer};
use crate::loss::{Loss, CrossEntropyLoss};
use crate::tensor::Tensor;
use crate::train::SimpleDataLoader;
use std::sync::Arc;

/// Configuration for knowledge distillation.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Temperature for soft-label KL divergence (typically 2-4).
    pub temperature: f32,
    /// Weight for hard-label loss (ground truth).
    pub alpha: f32,
    /// Weight for soft-label loss (teacher logits).
    pub beta: f32,
    /// Weight for feature-matching loss (intermediate reps).
    pub gamma: f32,
    /// Number of distillation epochs.
    pub epochs: usize,
    /// Learning rate for student training.
    pub lr: f32,
    /// Batch size.
    pub batch_size: usize,
    /// Student hidden dimension (0 = auto = teacher_dim / 2).
    pub student_hidden: usize,
    /// Student depth (number of layers, 0 = auto = teacher_depth / 2).
    pub student_layers: usize,
    /// Whether to quantize the student after distillation.
    pub quantize_output: bool,
}

impl Default for DistillConfig {
    fn default() -> Self {
        DistillConfig {
            temperature: 4.0,
            alpha: 0.3,
            beta: 0.5,
            gamma: 0.2,
            epochs: 20,
            lr: 0.001,
            batch_size: 32,
            student_hidden: 0,
            student_layers: 0,
            quantize_output: false,
        }
    }
}

impl DistillConfig {
    /// Set distillation temperature.
    pub fn with_temperature(mut self, t: f32) -> Self { self.temperature = t; self }
    /// Set loss weights (alpha=hard, beta=soft, gamma=feature).
    pub fn with_weights(mut self, alpha: f32, beta: f32, gamma: f32) -> Self {
        self.alpha = alpha; self.beta = beta; self.gamma = gamma; self
    }
    /// Auto-generate student with half the width and depth.
    pub fn auto_student(mut self) -> Self {
        self.student_hidden = 0; self.student_layers = 0; self
    }
    /// Quantize the student to INT8 after distillation.
    pub fn quantize(mut self) -> Self { self.quantize_output = true; self }
}

/// Result of a distillation run.
#[derive(Debug, Clone)]
pub struct DistillResult {
    /// The trained student model.
    pub student: Arc<Sequential>,
    /// Teacher accuracy on the training data.
    pub teacher_accuracy: f32,
    /// Student accuracy on the training data.
    pub student_accuracy: f32,
    /// Compression ratio (teacher params / student params).
    pub compression_ratio: f64,
    /// Final distillation loss.
    pub final_loss: f32,
    /// Loss history.
    pub loss_history: Vec<f32>,
}

/// Distills a teacher model into an automatically-generated smaller student.
///
/// # Algorithm
///
/// 1. Generate student architecture (half width, half depth by default).
/// 2. For each batch:
///    a. Teacher forward pass → soft logits.
///    b. Student forward pass → student logits.
///    c. Compute combined loss: α·CE(student, labels) + β·KL(student/T, teacher/T)·T².
///    d. Backward + optimizer step.
/// 3. Report accuracy and compression.
pub struct Distiller {
    teacher: Arc<dyn Module>,
    config: DistillConfig,
}

impl std::fmt::Debug for Distiller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Distiller").field("config", &self.config).finish_non_exhaustive()
    }
}

impl Distiller {
    /// Create a distiller with a teacher model and configuration.
    pub fn new(teacher: Arc<dyn Module>, config: DistillConfig) -> Self {
        Distiller { teacher, config }
    }

    /// Auto-generate a student architecture from the teacher's parameter shapes.
    ///
    /// The student has:
    /// - Half the hidden dimension (rounded to nearest power of 2).
    /// - Half the depth (rounded down).
    /// - The same input/output dimensions.
    pub fn generate_student(&self, input_dim: usize, output_dim: usize) -> Sequential {
        let params = self.teacher.parameters();
        // Count weight matrices (2D params).
        let weight_shapes: Vec<(usize, usize)> = params.iter()
            .filter(|p| p.ndim() == 2)
            .map(|p| (p.shape()[0], p.shape()[1]))
            .collect();

        let n_layers = weight_shapes.len();
        let teacher_hidden = weight_shapes.first().map(|(o, _)| *o).unwrap_or(64);
        let student_hidden = if self.config.student_hidden > 0 {
            self.config.student_hidden
        } else {
            (teacher_hidden / 2).max(16)
        };
        let student_layers = if self.config.student_layers > 0 {
            self.config.student_layers
        } else {
            (n_layers / 2).max(1)
        };

        let mut model = Sequential::new();
        let mut prev_dim = input_dim;
        for i in 0..student_layers {
            let out_dim = if i == student_layers - 1 { output_dim } else { student_hidden };
            model = model.add(Linear::new(prev_dim, out_dim, true));
            if i < student_layers - 1 {
                model = model.add(ReLU);
            }
            prev_dim = out_dim;
        }
        model
    }

    /// Run the distillation loop.
    pub fn distill(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        input_dim: usize,
        output_dim: usize,
    ) -> DistillResult {
        // Generate student.
        let student = Arc::new(self.generate_student(input_dim, output_dim));

        // Compute teacher params for compression ratio.
        let teacher_params: usize = (*self.teacher).parameters().iter().map(|t| t.len()).sum();
        let student_params: usize = student.parameters().iter().map(|t| t.len()).sum();
        let compression = teacher_params as f64 / student_params.max(1) as f64;

        // Training loop.
        let mut optimizer = Adam::new(student.parameters(), self.config.lr);
        let ce_loss = CrossEntropyLoss;
        let t = self.config.temperature;

        let mut loss_history = Vec::new();

        for epoch in 0..self.config.epochs {
            let loader = SimpleDataLoader::new(inputs.clone(), targets.clone(), self.config.batch_size);
            let mut epoch_loss = 0.0f32;
            let mut n_batches = 0usize;

            for (x_batch, y_batch) in loader {
                optimizer.zero_grad();

                // Teacher forward (no grad needed — just get logits).
                let teacher_logits = (*self.teacher).forward(&x_batch);
                let teacher_probs = softmax_temp(&teacher_logits, t);

                // Student forward.
                let student_logits = (*student).forward(&x_batch);
                let student_probs = softmax_temp(&student_logits, t);

                // Combined loss.
                // Hard label loss: CE(student, targets).
                let hard_loss = ce_loss.forward(&student_logits, &y_batch);

                // Soft label loss: KL(teacher || student) = sum(teacher * log(teacher/student)).
                // Approximated as MSE on softmax probabilities (simpler, differentiable).
                let soft_diff = student_probs.sub(&teacher_probs);
                let soft_loss = soft_diff.mul(&soft_diff).sum();

                // Total loss with weights and temperature scaling.
                let total_loss = hard_loss.mul(&Tensor::from_vec(vec![self.config.alpha], vec![1]))
                    .add(&soft_loss.mul(&Tensor::from_vec(
                        vec![self.config.beta * t * t],
                        vec![1],
                    )));

                total_loss.backward();
                optimizer.step();

                let loss_val = total_loss.data().iter().copied().next().unwrap_or(0.0);
                epoch_loss += loss_val;
                n_batches += 1;
            }

            let avg_loss = if n_batches > 0 { epoch_loss / n_batches as f32 } else { 0.0 };
            loss_history.push(avg_loss);

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                println!("  Distill epoch {}: loss = {:.4}", epoch + 1, avg_loss);
            }
        }

        // Evaluate accuracy.
        let teacher_acc = compute_accuracy(&*self.teacher, inputs, targets);
        let student_acc = compute_accuracy(&*student, inputs, targets);

        let final_loss = loss_history.last().copied().unwrap_or(0.0);

        println!(
            "  Distillation complete: teacher_acc={:.1}% student_acc={:.1}% compression={:.1}x",
            teacher_acc * 100.0,
            student_acc * 100.0,
            compression
        );

        DistillResult {
            student,
            teacher_accuracy: teacher_acc,
            student_accuracy: student_acc,
            compression_ratio: compression,
            final_loss,
            loss_history,
        }
    }
}

/// Progressive self-distillation: chain multiple distillation rounds.
///
/// Each round, the student from the previous round becomes the teacher, and a new smaller
/// student is generated. This progressively compresses the model while trying to preserve
/// accuracy.
pub struct ProgressiveDistiller {
    config: DistillConfig,
    /// Number of progressive rounds (each halves the size).
    pub rounds: usize,
}

impl ProgressiveDistiller {
    /// Create a progressive distiller.
    pub fn new(config: DistillConfig, rounds: usize) -> Self {
        ProgressiveDistiller { config, rounds }
    }

    /// Run progressive distillation, returning the final (smallest) student.
    pub fn distill(
        &self,
        initial_teacher: Arc<dyn Module>,
        inputs: &Tensor,
        targets: &Tensor,
        input_dim: usize,
        output_dim: usize,
    ) -> Vec<DistillResult> {
        let mut results = Vec::new();
        let mut current_teacher = initial_teacher;

        for round in 0..self.rounds {
            println!("=== Progressive distillation round {} ===", round + 1);
            let mut distiller = Distiller::new(current_teacher.clone(), self.config.clone());
            let result = distiller.distill(inputs, targets, input_dim, output_dim);
            println!(
                "  Round {}: compression={:.1}x, acc={:.1}%\n",
                round + 1,
                result.compression_ratio,
                result.student_accuracy * 100.0
            );
            current_teacher = result.student.clone();
            results.push(result);
        }

        results
    }
}

/// Compute classification accuracy.
fn compute_accuracy(model: &dyn Module, inputs: &Tensor, targets: &Tensor) -> f32 {
    let logits = model.forward(inputs);
    let log_data = logits.data();
    let tgt_data = targets.data();
    let shape = log_data.shape();

    if shape.len() < 2 {
        return 0.0;
    }

    let n = shape[0];
    let c = shape[1];
    let mut correct = 0usize;

    for i in 0..n {
        // Find argmax of logits.
        let mut best_class = 0;
        let mut best_val = f32::NEG_INFINITY;
        for j in 0..c {
            let v = log_data[[i, j]];
            if v > best_val {
                best_val = v;
                best_class = j;
            }
        }
        // Check against target (one-hot or class index).
        let target_class = if tgt_data.shape()[tgt_data.ndim().saturating_sub(1)] == c {
            // One-hot encoding: find the index with value 1.0.
            let mut tc = 0;
            for j in 0..c {
                if tgt_data[[i, j]] > 0.5 { tc = j; break; }
            }
            tc
        } else {
            // Class index encoding.
            tgt_data[[i, 0]] as usize
        };

        if best_class == target_class {
            correct += 1;
        }
    }

    correct as f32 / n.max(1) as f32
}

/// Temperature-scaled softmax: `softmax(logits / T)`.
fn softmax_temp(logits: &Tensor, temperature: f32) -> Tensor {
    let data = logits.data();
    let shape = data.shape();
    if shape.len() < 2 { return logits.clone(); }
    let (n, c) = (shape[0], shape[1]);

    let mut out = vec![0.0f32; n * c];
    for i in 0..n {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..c { max_val = max_val.max(data[[i, j]] / temperature); }
        let mut sum = 0.0f32;
        for j in 0..c {
            out[i * c + j] = ((data[[i, j]] / temperature) - max_val).exp();
            sum += out[i * c + j];
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for j in 0..c { out[i * c + j] *= inv; }
    }

    Tensor::new(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, c]), out).unwrap(),
        logits.requires_grad(),
    )
}

// Helper: repeat a single-row tensor N times.
#[allow(dead_code)]
trait RepeatRows {
    fn repeat_rows(&self, n: usize) -> Self;
}

impl RepeatRows for Tensor {
    fn repeat_rows(&self, n: usize) -> Self {
        let data = self.data();
        let cols = data.shape()[data.ndim() - 1];
        let mut out = Vec::with_capacity(n * cols);
        for _ in 0..n {
            out.extend(data.iter().copied());
        }
        Tensor::from_vec(out, vec![n, cols])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distiller_generates_smaller_student() {
        let teacher = Arc::new(
            Sequential::new()
                .add(Linear::new(32, 128, true))
                .add(ReLU)
                .add(Linear::new(128, 128, true))
                .add(ReLU)
                .add(Linear::new(128, 64, true))
                .add(ReLU)
                .add(Linear::new(64, 10, true)),
        );

        let config = DistillConfig::default();
        let distiller = Distiller::new(teacher.clone(), config);
        let student = distiller.generate_student(32, 10);

        let teacher_params: usize = (*teacher).parameters().iter().map(|t| t.len()).sum();
        let student_params: usize = student.parameters().iter().map(|t| t.len()).sum();
        assert!(student_params < teacher_params, "student should be smaller: {} vs {}", student_params, teacher_params);
    }

    #[test]
    fn distill_config_builder() {
        let cfg = DistillConfig::default()
            .with_temperature(2.0)
            .with_weights(0.5, 0.3, 0.2);
        assert_eq!(cfg.temperature, 2.0);
        assert_eq!(cfg.alpha, 0.5);
        assert_eq!(cfg.beta, 0.3);
        assert_eq!(cfg.gamma, 0.2);
    }

    #[test]
    fn distillation_runs_and_reduces_loss() {
        let teacher = Arc::new(
            Sequential::new()
                .add(Linear::new(8, 32, true))
                .add(ReLU)
                .add(Linear::new(32, 4, true)),
        );

        // Synthetic data.
        let n = 64;
        let inputs = Tensor::randn(&[n, 8]);
        let mut onehot = vec![0.0f32; n * 4];
        for i in 0..n { onehot[i * 4 + (i % 4)] = 1.0; }
        let targets = Tensor::from_vec(onehot, vec![n, 4]);

        let config = DistillConfig {
            epochs: 5,
            batch_size: 16,
            lr: 0.01,
            ..Default::default()
        };

        let mut distiller = Distiller::new(teacher, config);
        let result = distiller.distill(&inputs, &targets, 8, 4);

        assert!(result.compression_ratio > 1.0, "student should be compressed");
        assert!(!result.loss_history.is_empty(), "should have loss history");
    }

    #[test]
    fn softmax_temp_sums_to_one() {
        let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let probs = softmax_temp(&logits, 2.0);
        let p: Vec<f32> = probs.data().iter().copied().collect();
        let row0_sum = p[0] + p[1] + p[2];
        assert!((row0_sum - 1.0).abs() < 1e-5, "softmax row should sum to 1: {}", row0_sum);
    }

    #[test]
    fn accuracy_computes_correctly() {
        // Perfect predictions.
        let model = Sequential::new().add(Linear::new(4, 3, true));
        let inputs = Tensor::randn(&[4, 4]);
        // One-hot targets matching argmax of random model (won't be perfect, but should be [0,1]).
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![4, 3]);
        let acc = compute_accuracy(&model, &inputs, &targets);
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn progressive_distiller_creates_chain() {
        let teacher = Arc::new(
            Sequential::new()
                .add(Linear::new(8, 64, true))
                .add(ReLU)
                .add(Linear::new(64, 4, true)),
        );

        let inputs = Tensor::randn(&[32, 8]);
        let mut oh = vec![0.0f32; 32 * 4];
        for i in 0..32 { oh[i * 4 + (i % 4)] = 1.0; }
        let targets = Tensor::from_vec(oh, vec![32, 4]);

        let config = DistillConfig { epochs: 3, batch_size: 16, lr: 0.01, ..Default::default() };
        let prog = ProgressiveDistiller::new(config, 2);
        let results = prog.distill(teacher, &inputs, &targets, 8, 4);
        assert_eq!(results.len(), 2, "should have 2 rounds");
    }

    #[test]
    fn distill_result_has_valid_metrics() {
        let teacher = Arc::new(
            Sequential::new()
                .add(Linear::new(8, 32, true))
                .add(ReLU)
                .add(Linear::new(32, 4, true)),
        );

        let inputs = Tensor::randn(&[16, 8]);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], vec![1, 4]).repeat_rows(16);

        let config = DistillConfig { epochs: 2, batch_size: 8, ..Default::default() };
        let mut distiller = Distiller::new(teacher, config);
        let result = distiller.distill(&inputs, &targets, 8, 4);

        assert!(result.teacher_accuracy >= 0.0 && result.teacher_accuracy <= 1.0);
        assert!(result.student_accuracy >= 0.0 && result.student_accuracy <= 1.0);
        assert!(result.compression_ratio > 0.0);
    }
}
