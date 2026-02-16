//! ONNX inference abstraction layer for incr.
//!
//! This crate provides a unified interface for running ONNX models across
//! different backends:
//! - `ort` with XNNPACK execution provider for native platforms
//! - `tract` directly for WASM/browser environments

mod backend;
mod error;
mod tensor;

pub use backend::InferenceBackend;
pub use error::InferenceError;
pub use tensor::{InputTensor, OutputTensor, TensorType};

#[cfg(feature = "native")]
pub use backend::ort::OrtBackend;

#[cfg(feature = "wasm")]
pub use backend::tract::TractBackend;

/// Result type for inference operations.
pub type Result<T> = std::result::Result<T, InferenceError>;
