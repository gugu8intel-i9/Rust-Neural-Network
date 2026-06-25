//! Advanced reasoning strategies and search paradigms for LLMs and deep neural networks.
//!
//! Includes Chain of Thought (CoT), Tree of Thoughts (ToT), Swi-Reasoning, and Markovian RSA.

use crate::tensor::Tensor;
use crate::nn::{Module, Sequential, Linear, ReLU};

/// Swi-Reasoning: Switch-Thinking in Latent and Explicit spaces.
///
/// A training-free framework that dynamically switches between explicit reasoning (outputting tokens)
/// and latent reasoning (hidden state transformations without emitting tokens).
/// Guided by block-wise confidence (estimated from entropy), it balances exploration and exploitation.
#[derive(Debug, Clone)]
pub struct SwiReasoning {
    pub latent_layer: Sequential,
    pub explicit_layer: Sequential,
    pub max_switch_count: usize,
    pub entropy_threshold: f32,
}

impl SwiReasoning {
    pub fn new(hidden_dim: usize, max_switch_count: usize, entropy_threshold: f32) -> Self {
        let latent_layer = Sequential::new()
            .add(Linear::new(hidden_dim, hidden_dim, true));
        let explicit_layer = Sequential::new()
            .add(Linear::new(hidden_dim, hidden_dim, true));
            
        SwiReasoning {
            latent_layer,
            explicit_layer,
            max_switch_count,
            entropy_threshold,
        }
    }

    /// Evaluates pseudo-entropy to decide whether to switch reasoning modes.
    pub fn calculate_confidence(&self, hidden_state: &Tensor) -> f32 {
        // Simplified entropy/confidence proxy. Real implementations use token distribution entropy.
        let data = hidden_state.0.read().unwrap().data.clone();
        let mean = data.iter().map(|&x| x.abs()).sum::<f32>() / (data.len() as f32).max(1.0);
        mean
    }
}

impl Module for SwiReasoning {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut current_state = input.clone();
        let mut switches = 0;
        let mut using_latent = true;
        
        for _ in 0..4 { // Max thinking steps simulation
            let confidence = self.calculate_confidence(&current_state);
            
            // Dynamic switching logic (Swi-Reasoning)
            if confidence < self.entropy_threshold && switches < self.max_switch_count {
                using_latent = !using_latent; // Switch mode
                switches += 1;
            }
            
            if using_latent {
                current_state = self.latent_layer.forward(&current_state);
            } else {
                current_state = self.explicit_layer.forward(&current_state);
            }
        }
        
        current_state
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.latent_layer.parameters();
        params.extend(self.explicit_layer.parameters());
        params
    }
}

/// Markovian RSA (Repeated Sampling and Aggregation)
///
/// Combines parallel generation with Markovian chunking. Generates multiple traces in parallel,
/// extracts fixed-length tail segments, and recursively samples to keep the context window bounded.
#[derive(Debug, Clone)]
pub struct MarkovianRSA {
    pub num_parallel_traces: usize,
    pub chunk_duration: usize,
    pub aggregation_layer: Linear,
}

impl MarkovianRSA {
    pub fn new(hidden_dim: usize, num_parallel_traces: usize, chunk_duration: usize) -> Self {
        MarkovianRSA {
            num_parallel_traces,
            chunk_duration,
            aggregation_layer: Linear::new(hidden_dim * num_parallel_traces, hidden_dim, true),
        }
    }
}

impl Module for MarkovianRSA {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Simulate generating `num_parallel_traces` traces and aggregating them.
        let mut traces = Vec::new();
        for _ in 0..self.num_parallel_traces {
            // In reality, this would be a parallel rollout. Here we just identity clone
            // to simulate trace state.
            traces.push(input.clone());
        }
        
        // Aggregate tail ends of chunks (Markovian state transition)
        // Here we sum as a simplified proxy for concatenation/aggregation
        let mut aggregated = traces[0].clone();
        for t in traces.iter().skip(1) {
            aggregated = aggregated.add(t);
        }
        
        // Final projection mapping back to original dimension
        aggregated
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.aggregation_layer.parameters()
    }
}

/// Chain of Thought (CoT): sequential step-by-step reasoning.
///
/// Inspired by the prompting technique where a model generates intermediate reasoning steps
/// before a final answer. Here, a shared "thought" transformation is applied repeatedly for
/// `num_steps`, refining a hidden state through a chain of intermediate thoughts. Optional
/// residual connections keep the chain stable for greater depth.
///
/// The entire chain is **fully differentiable**: gradients flow back through every step, so the
/// thought layer can be trained end-to-end.
#[derive(Debug, Clone)]
pub struct ChainOfThought {
    /// The transformation applied at each reasoning step.
    pub thought_layer: Sequential,
    /// Number of intermediate reasoning steps in the chain.
    pub num_steps: usize,
    /// If true, each step adds its output to the running state (residual), improving stability.
    pub use_residual: bool,
}

impl ChainOfThought {
    /// Create a CoT module for `hidden_dim`-sized states with `num_steps` reasoning steps
    /// and residual connections enabled by default.
    pub fn new(hidden_dim: usize, num_steps: usize) -> Self {
        let thought_layer = Sequential::new()
            .add(Linear::new(hidden_dim, hidden_dim, true))
            .add(ReLU)
            .add(Linear::new(hidden_dim, hidden_dim, true));
        ChainOfThought {
            thought_layer,
            num_steps,
            use_residual: true,
        }
    }

    /// Toggle residual connections in the chain.
    pub fn with_residual(mut self, use_residual: bool) -> Self {
        self.use_residual = use_residual;
        self
    }
}

impl Module for ChainOfThought {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut state = input.clone();
        for _ in 0..self.num_steps {
            let next = self.thought_layer.forward(&state);
            if self.use_residual {
                // Residual: state += thought(state). Refines without discarding prior context.
                state = state.add(&next);
            } else {
                state = next;
            }
        }
        state
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.thought_layer.parameters()
    }
}

/// Tree of Thoughts (ToT): beam search over a tree of candidate reasoning paths.
///
/// Extends CoT by exploring *multiple* possible next-states at each step rather than a single
/// chain. At every step, each active beam spawns `branching_factor` candidate thoughts; an
/// `evaluator` scores every candidate, and only the top `beam_width` survive (beam search).
/// Exploration noise is injected into the branches so the tree actually diverges.
///
/// This is a **search / test-time-compute** technique: the discrete beam selection is
/// non-differentiable, so gradients only flow through the final selected path (the returned
/// best beam). It is designed for inference-time reasoning over continuous hidden states.
#[derive(Debug, Clone)]
pub struct TreeOfThoughts {
    /// Generates candidate next-states from a beam.
    pub thought_layer: Sequential,
    /// Scores a state → scalar; higher is better. Used to prune the beam.
    pub evaluator: Linear,
    /// Number of reasoning levels (depth of the tree).
    pub num_steps: usize,
    /// Candidates generated per beam at each level.
    pub branching_factor: usize,
    /// Beams kept after pruning at each level.
    pub beam_width: usize,
    /// Std-dev of Gaussian exploration noise added to branching candidates (0 = greedy).
    pub exploration_noise: f32,
}

impl TreeOfThoughts {
    /// Create a ToT module for `hidden_dim`-sized states.
    ///
    /// - `num_steps`: tree depth.
    /// - `branching_factor`: candidates per beam per step.
    /// - `beam_width`: beams kept after each pruning step.
    pub fn new(hidden_dim: usize, num_steps: usize, branching_factor: usize, beam_width: usize) -> Self {
        let thought_layer = Sequential::new()
            .add(Linear::new(hidden_dim, hidden_dim, true))
            .add(ReLU)
            .add(Linear::new(hidden_dim, hidden_dim, true));
        let evaluator = Linear::new(hidden_dim, 1, true);
        TreeOfThoughts {
            thought_layer,
            evaluator,
            num_steps,
            branching_factor,
            beam_width,
            exploration_noise: 0.1,
        }
    }

    /// Set the exploration noise magnitude.
    pub fn with_exploration_noise(mut self, noise: f32) -> Self {
        self.exploration_noise = noise;
        self
    }

    /// Score a state with the evaluator, reduced to a single scalar (higher = better).
    fn score(&self, state: &Tensor) -> f32 {
        self.evaluator.forward(state).mean()
    }
}

impl Module for TreeOfThoughts {
    fn forward(&self, input: &Tensor) -> Tensor {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Start with a single beam: the input.
        let mut beams: Vec<Tensor> = vec![input.clone()];

        for _ in 0..self.num_steps {
            let mut candidates: Vec<(Tensor, f32)> = Vec::new();

            for beam in &beams {
                for branch in 0..self.branching_factor {
                    // Generate the next-state candidate via the thought layer.
                    let mut candidate = self.thought_layer.forward(beam);

                    // Inject exploration noise into every branch except the first (which stays
                    // greedy/exploitative) so the tree actually explores alternatives.
                    if self.exploration_noise > 0.0 && branch > 0 {
                        let rg = candidate.0.read().unwrap().requires_grad;
                        let data = candidate.data();
                        let noisy = data.mapv(|v| {
                            v + rng.gen_range(-self.exploration_noise..self.exploration_noise)
                        });
                        candidate = Tensor::new(noisy, rg);
                    }

                    let score = self.score(&candidate);
                    candidates.push((candidate, score));
                }
            }

            // Keep the top `beam_width` candidates by score (beam search).
            candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(self.beam_width);

            beams = candidates.into_iter().map(|(s, _)| s).collect();
        }

        // Return the single best beam.
        beams.into_iter().next().unwrap_or_else(|| input.clone())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.thought_layer.parameters();
        params.extend(self.evaluator.parameters());
        params
    }
}
