//! Error types for rust-nn.
//!
//! Production-grade error handling. All fallible operations return [`Result<T, RustNnError>`].
//! The error type uses `#[non_exhaustive]` so new variants can be added without breaking
//! downstream code.

use std::fmt;

/// The unified error type for all rust-nn operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RustNnError {
    /// A shape mismatch between two tensors (e.g. matmul with incompatible dimensions).
    ShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
        detail: &'static str,
    },
    /// An input tensor has an invalid number of dimensions.
    InvalidNdims {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A generic invalid input (e.g. zero vocab size, negative learning rate).
    InvalidInput {
        op: &'static str,
        msg: String,
    },
    /// A tensor index is out of bounds.
    IndexOutOfBounds {
        op: &'static str,
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// An empty tensor was provided where a non-empty one was required.
    EmptyTensor {
        op: &'static str,
    },
    /// A serialization or deserialization failure.
    Serialization(String),
    /// A GPU / device error.
    DeviceError(String),
    /// A data loading error (file I/O, network, parsing).
    DataLoad(String),
}

impl RustNnError {
    /// Convenience constructor for shape mismatch.
    pub fn shape_mismatch(op: &'static str, expected: &[usize], actual: &[usize], detail: &'static str) -> Self {
        RustNnError::ShapeMismatch {
            op,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
            detail,
        }
    }

    /// Convenience constructor for invalid ndims.
    pub fn invalid_ndims(op: &'static str, expected: usize, actual: usize) -> Self {
        RustNnError::InvalidNdims { op, expected, actual }
    }
}

impl fmt::Display for RustNnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RustNnError::ShapeMismatch { op, expected, actual, detail } => {
                write!(f, "{}: shape mismatch - expected {:?}, got {:?}. {}", op, expected, actual, detail)
            }
            RustNnError::InvalidNdims { op, expected, actual } => {
                write!(f, "{op}: expected {expected} dimensions, got {actual}")
            }
            RustNnError::InvalidInput { op, msg } => {
                write!(f, "{}: invalid input - {}", op, msg)
            }
            RustNnError::IndexOutOfBounds { op, index, shape } => {
                write!(f, "{op}: index {:?} is out of bounds for shape {:?}", index, shape)
            }
            RustNnError::EmptyTensor { op } => {
                write!(f, "{op}: input tensor is empty")
            }
            RustNnError::Serialization(msg) => {
                write!(f, "serialization error: {msg}")
            }
            RustNnError::DeviceError(msg) => {
                write!(f, "device error: {msg}")
            }
            RustNnError::DataLoad(msg) => {
                write!(f, "data loading error: {msg}")
            }
        }
    }
}

impl std::error::Error for RustNnError {}

/// The Result type alias used throughout rust-nn.
pub type Result<T> = std::result::Result<T, RustNnError>;

/// Convert a `std::result::Result` with a string error into a [`RustNnError`].
pub fn from_io(op: &'static str, e: impl std::fmt::Display) -> RustNnError {
    RustNnError::InvalidInput { op, msg: e.to_string() }
}

// ==================== Checked tensor operations ====================
//
// These are safe (Result-returning) variants of the most dangerous operations.
// The existing methods (matmul, reshape, etc.) still panic on invalid input for backward
// compatibility, but production code should prefer the `_checked` variants.

use crate::tensor::Tensor as TensorType;

/// Checked matrix multiplication. Returns an error on shape mismatch instead of panicking.
///
/// # Example
/// ```ignore
/// use rust_nn::error::checked_matmul;
/// let a = Tensor::randn(&[2, 3]);
/// let b = Tensor::randn(&[4, 5]); // wrong shape
/// match checked_matmul(&a, &b) {
///     Ok(c) => println!("Result: {:?}", c.shape()),
///     Err(e) => eprintln!("Error: {e}"),
/// }
/// ```
pub fn checked_matmul(a: &TensorType, b: &TensorType) -> Result<TensorType> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 {
        return Err(RustNnError::invalid_ndims("checked_matmul", 2, a_shape.len()));
    }
    if b_shape.len() != 2 {
        return Err(RustNnError::invalid_ndims("checked_matmul", 2, b_shape.len()));
    }
    if a_shape[1] != b_shape[0] {
        return Err(RustNnError::shape_mismatch(
            "checked_matmul",
            &[a_shape[1]],
            &[b_shape[0]],
            "A's columns must equal B's rows for A[m,k] @ B[k,n]",
        ));
    }
    Ok(a.matmul(b))
}

/// Checked reshape. Returns an error if the new shape has a different total element count.
pub fn checked_reshape(tensor: &TensorType, shape: &[usize]) -> Result<TensorType> {
    let old_count: usize = tensor.shape().iter().product();
    let new_count: usize = shape.iter().product();
    if old_count != new_count {
        return Err(RustNnError::shape_mismatch(
            "checked_reshape",
            &[old_count],
            &[new_count],
            "total element count must be preserved",
        ));
    }
    Ok(tensor.reshape(shape))
}

/// Checked element-wise add. Returns an error on shape mismatch (after broadcasting attempt).
pub fn checked_add(a: &TensorType, b: &TensorType) -> Result<TensorType> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    // Check broadcastability (trailing dims must match or be 1).
    let max_dims = a_shape.len().max(b_shape.len());
    for i in 0..max_dims {
        let a_dim = if i < a_shape.len() { a_shape[a_shape.len() - 1 - i] } else { 1 };
        let b_dim = if i < b_shape.len() { b_shape[b_shape.len() - 1 - i] } else { 1 };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(RustNnError::shape_mismatch(
                "checked_add",
                &a_shape,
                &b_shape,
                "shapes are not broadcast-compatible",
            ));
        }
    }
    Ok(a.add(b))
}

/// Validate that a tensor has exactly `expected_dims` dimensions.
pub fn validate_ndims(tensor: &TensorType, op: &'static str, expected_dims: usize) -> Result<()> {
    let actual = tensor.ndim();
    if actual != expected_dims {
        return Err(RustNnError::invalid_ndims(op, expected_dims, actual));
    }
    Ok(())
}

/// Validate that a tensor is non-empty.
pub fn validate_non_empty(tensor: &TensorType, op: &'static str) -> Result<()> {
    if tensor.is_empty() {
        return Err(RustNnError::EmptyTensor { op });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_display() {
        let e = RustNnError::shape_mismatch("matmul", &[2, 3], &[4, 5], "A's columns must equal B's rows");
        let s = format!("{e}");
        assert!(s.contains("matmul"));
        assert!(s.contains("shape mismatch"));
    }

    #[test]
    fn invalid_ndims_display() {
        let e = RustNnError::invalid_ndims("layer_norm", 2, 1);
        assert!(format!("{e}").contains("expected 2 dimensions"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(RustNnError::EmptyTensor { op: "test" });
        assert!(format!("{e}").contains("empty"));
    }
}
