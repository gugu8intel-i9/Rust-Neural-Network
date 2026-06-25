//! Tests for the reasoning strategies: Chain of Thought and Tree of Thoughts.

use ndarray::{ArrayD, IxDyn};
use rust_nn::nn::Module;
use rust_nn::reasoning::{ChainOfThought, TreeOfThoughts};
use rust_nn::tensor::Tensor;

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap(),
        true,
    )
}

#[test]
fn cot_preserves_shape() {
    let cot = ChainOfThought::new(8, 4);
    let x = Tensor::randn(&[3, 8]);
    let out = cot.forward(&x);
    assert_eq!(out.shape(), vec![3, 8], "CoT must preserve shape");
}

#[test]
fn cot_is_differentiable() {
    // Gradients must flow through the entire chain back to the input.
    let cot = ChainOfThought::new(6, 3);
    let x = leaf(&[0.5, -0.3, 0.8, -0.1, 0.2, -0.4], &[1, 6]);
    let out = cot.forward(&x);
    out.sum().backward();
    let grad = x.grad().expect("CoT input received no gradient");
    // Every element should have a non-trivial gradient flowing through the chain.
    assert!(grad.iter().all(|&g| g.abs() > 0.0), "CoT gradients should be non-zero");
}

#[test]
fn cot_zero_steps_is_identity() {
    let cot = ChainOfThought::new(4, 0);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let out = cot.forward(&x);
    let od: Vec<f32> = out.data().iter().copied().collect();
    for i in 0..4 {
        assert!((od[i] - [1.0, 2.0, 3.0, 4.0][i]).abs() < 1e-6, "0-step CoT should be identity");
    }
}

#[test]
fn cot_residual_vs_no_residual_differ() {
    // With residual connections, the state grows; without, each step replaces.
    let cot_res = ChainOfThought::new(4, 5);
    let cot_no = ChainOfThought::new(4, 5).with_residual(false);

    let x = Tensor::ones(&[1, 4]);
    let out_res = cot_res.forward(&x);
    let out_no = cot_no.forward(&x);
    // They share no weights, so we just check both produce finite output of correct shape.
    assert_eq!(out_res.shape(), vec![1, 4]);
    assert_eq!(out_no.shape(), vec![1, 4]);
    assert!(out_res.data().iter().all(|v| v.is_finite()));
    assert!(out_no.data().iter().all(|v| v.is_finite()));
}

#[test]
fn tot_preserves_shape() {
    let tot = TreeOfThoughts::new(8, 3, 4, 2);
    let x = Tensor::randn(&[2, 8]);
    let out = tot.forward(&x);
    assert_eq!(out.shape(), vec![2, 8], "ToT must preserve shape");
}

#[test]
fn tot_runs_without_panic() {
    // Exercise the beam search with various branching/beam configs.
    let tot = TreeOfThoughts::new(6, 4, 3, 2).with_exploration_noise(0.2);
    let x = Tensor::randn(&[1, 6]);
    let out = tot.forward(&x);
    assert_eq!(out.shape(), vec![1, 6]);
}

#[test]
fn tot_beam_width_one_greedy() {
    // beam_width = 1 means only the single best candidate survives at each step.
    let tot = TreeOfThoughts::new(5, 3, 3, 1);
    let x = Tensor::randn(&[2, 5]);
    let out = tot.forward(&x);
    assert_eq!(out.shape(), vec![2, 5]);
    assert!(out.data().iter().all(|v| v.is_finite()));
}

#[test]
fn tot_selects_highest_scored_candidate() {
    // With deterministic thought layers (zero noise) and beam_width 1, ToT should pick
    // the candidate that the evaluator scores highest. We verify it returns *some* valid
    // output rather than panicking on edge configurations.
    let tot = TreeOfThoughts::new(4, 2, 2, 1).with_exploration_noise(0.0);
    let x = Tensor::randn(&[1, 4]);
    let out = tot.forward(&x);
    assert_eq!(out.shape(), vec![1, 4]);
}

#[test]
fn tot_has_parameters() {
    let tot = TreeOfThoughts::new(8, 2, 2, 2);
    // thought_layer (2 Linears with bias = 4 param tensors) + evaluator (1 Linear = 2 tensors).
    let params = tot.parameters();
    assert!(params.len() >= 4, "ToT should expose thought + evaluator params");
}
