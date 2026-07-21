//! Production robustness tests: edge cases, error handling, and invalid-input rejection.

use rust_nn::error::{checked_add, checked_matmul, checked_reshape, validate_ndims, validate_non_empty};
use rust_nn::tensor::Tensor;

// ==================== Checked operations ====================

#[test]
fn checked_matmul_rejects_shape_mismatch() {
    let a = Tensor::randn(&[2, 3]);
    let b = Tensor::randn(&[4, 5]);
    let result = checked_matmul(&a, &b);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("matmul") || msg.contains("shape mismatch"));
}

#[test]
fn checked_matmul_succeeds_on_valid_input() {
    let a = Tensor::randn(&[2, 3]);
    let b = Tensor::randn(&[3, 4]);
    let result = checked_matmul(&a, &b);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().shape(), vec![2, 4]);
}

#[test]
fn checked_matmul_rejects_non_2d() {
    let a = Tensor::randn(&[2, 3, 4]);
    let b = Tensor::randn(&[4, 5]);
    assert!(checked_matmul(&a, &b).is_err());
}

#[test]
fn checked_reshape_rejects_count_mismatch() {
    let t = Tensor::randn(&[2, 3]);  // 6 elements
    let result = checked_reshape(&t, &[4, 4]); // 16 elements
    assert!(result.is_err());
}

#[test]
fn checked_reshape_succeeds_on_valid() {
    let t = Tensor::randn(&[2, 3]);
    let result = checked_reshape(&t, &[3, 2]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().shape(), vec![3, 2]);
}

#[test]
fn checked_add_rejects_incompatible() {
    let a = Tensor::randn(&[2, 3]);
    let b = Tensor::randn(&[4, 5]);
    assert!(checked_add(&a, &b).is_err());
}

#[test]
fn checked_add_allows_broadcast() {
    let a = Tensor::randn(&[4, 3]);
    let b = Tensor::randn(&[3]);  // broadcastable
    assert!(checked_add(&a, &b).is_ok());
}

#[test]
fn validate_ndims_works() {
    let t = Tensor::randn(&[2, 3, 4]);
    assert!(validate_ndims(&t, "test", 3).is_ok());
    assert!(validate_ndims(&t, "test", 2).is_err());
}

#[test]
fn validate_non_empty_works() {
    let t = Tensor::randn(&[2, 3]);
    assert!(validate_non_empty(&t, "test").is_ok());
}

// ==================== Tensor edge cases ====================

#[test]
fn tensor_zero_dimension() {
    // A tensor with a zero dimension should not panic on creation.
    let t = Tensor::zeros(&[0, 3]);
    assert_eq!(t.len(), 0);
    assert!(t.is_empty());
}

#[test]
fn tensor_scalar() {
    // A 0-dimensional tensor (scalar).
    let t = Tensor::from_vec(vec![42.0], vec![]);
    assert_eq!(t.ndim(), 0);
    // sum should return the scalar itself.
    let s = t.sum();
    assert!((s.data().iter().copied().next().unwrap() - 42.0).abs() < 1e-6);
}

#[test]
fn tensor_single_element() {
    let t = Tensor::from_vec(vec![5.0], vec![1, 1]);
    assert_eq!(t.shape(), vec![1, 1]);
    assert!((t.get(&[0, 0]) - 5.0).abs() < 1e-6);
}

#[test]
fn matmul_1x1() {
    let a = Tensor::from_vec(vec![3.0], vec![1, 1]);
    let b = Tensor::from_vec(vec![4.0], vec![1, 1]);
    let c = a.matmul(&b);
    assert!((c.get(&[0, 0]) - 12.0).abs() < 1e-5);
}

#[test]
fn matmul_large_narrow() {
    // [1, 1000] @ [1000, 1] = [1, 1]
    let a = Tensor::ones(&[1, 1000]);
    let b = Tensor::ones(&[1000, 1]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), vec![1, 1]);
    assert!((c.get(&[0, 0]) - 1000.0).abs() < 1.0);
}

#[test]
fn reshape_roundtrip() {
    let t = Tensor::from_vec((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]);
    let r1 = t.reshape(&[6, 4]);
    let r2 = r1.reshape(&[2, 3, 4]);
    // Values should be identical.
    let orig: Vec<f32> = t.data().iter().copied().collect();
    let back: Vec<f32> = r2.data().iter().copied().collect();
    assert_eq!(orig, back);
}

#[test]
fn relu_negative_input() {
    let t = Tensor::from_vec(vec![-5.0, -3.0, -0.001, 0.0, 0.001, 3.0], vec![6]);
    let r = t.relu();
    let vals: Vec<f32> = r.data().iter().copied().collect();
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 0.0, 0.001, 3.0]);
}

#[test]
fn sigmoid_extreme_values() {
    let t = Tensor::from_vec(vec![-100.0, -1.0, 0.0, 1.0, 100.0], vec![5]);
    let s = t.sigmoid();
    let vals: Vec<f32> = s.data().iter().copied().collect();
    assert!(vals[0] > 0.0 && vals[0] < 1e-40, "sigmoid(-100) should be ~0");
    assert!((vals[2] - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5");
    assert!(vals[4] > 0.999, "sigmoid(100) should be ~1");
    assert!(vals.iter().all(|v| v.is_finite()), "sigmoid should be finite for all inputs");
}

#[test]
fn clamp_out_of_range() {
    let t = Tensor::from_vec(vec![-10.0, -1.0, 0.0, 1.0, 10.0], vec![5]);
    let c = t.clamp(-2.0, 2.0);
    let vals: Vec<f32> = c.data().iter().copied().collect();
    assert_eq!(vals, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn display_formats_tensor() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let s = format!("{t}");
    assert!(s.contains("1"));
    assert!(s.contains("4"));
}

// ==================== Numeric stability ====================

#[test]
fn softmax_sums_to_one() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let s = rust_nn::activations::softmax(&t);
    let sum: f32 = s.data().iter().copied().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1.0, got {sum}");
}

#[test]
fn softmax_extreme_values_stable() {
    let t = Tensor::from_vec(vec![-1000.0, 0.0, 1000.0], vec![3]);
    let s = rust_nn::activations::softmax(&t);
    let vals: Vec<f32> = s.data().iter().copied().collect();
    assert!(vals.iter().all(|v| v.is_finite()), "softmax should be finite for extreme inputs");
    assert!(vals[0] < 1e-10, "softmax(-1000) should be ~0");
    assert!(vals[2] > 0.999, "softmax(1000) should be ~1");
}

#[test]
fn layer_norm_stable_with_extreme_input() {
    use rust_nn::nn::{LayerNorm, Module};
    let ln = LayerNorm::new(4);
    let x = Tensor::from_vec(vec![1e10, -1e10, 1e-10, -1e-10], vec![1, 4]);
    let y = ln.forward(&x);
    assert!(y.data().iter().all(|v| v.is_finite()), "layer_norm should be stable for extreme input");
}

#[test]
fn cross_entropy_stable_with_large_logits() {
    use rust_nn::loss::{CrossEntropyLoss, Loss};
    let logits = Tensor::from_vec(vec![1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0], vec![2, 3]);
    let target = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
    let loss = CrossEntropyLoss.forward(&logits, &target);
    let val = loss.data().iter().copied().next().unwrap();
    assert!(val.is_finite(), "cross_entropy should be finite for large logits, got {val}");
}

// ==================== Serialization robustness ====================

#[test]
fn serialization_rejects_invalid_magic() {
    let result = rust_nn::serialize::deserialize(b"XXXXinvalid data here");
    assert!(result.is_err());
}

#[test]
fn serialization_empty_dataset() {
    let bytes = rust_nn::serialize::serialize(&[]);
    assert!(bytes.starts_with(b"RNNB"));
    let loaded = rust_nn::serialize::deserialize(&bytes).unwrap();
    assert!(loaded.is_empty());
}

// ==================== Determinism ====================

#[test]
fn matmul_is_deterministic() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let c1 = a.matmul(&b);
    let c2 = a.matmul(&b);
    let d1: Vec<f32> = c1.data().iter().copied().collect();
    let d2: Vec<f32> = c2.data().iter().copied().collect();
    assert_eq!(d1, d2, "matmul must be deterministic");
}

// ==================== Thread safety ====================

#[test]
fn tensor_send_sync() {
    // If this compiles, Tensor is Send + Sync (required for the Arc<RwLock> pattern).
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tensor>();
}
