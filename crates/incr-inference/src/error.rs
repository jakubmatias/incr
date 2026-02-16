//! Error types for the inference layer.

use thiserror::Error;

/// Errors that can occur during ONNX inference.
#[derive(Error, Debug)]
pub enum InferenceError {
    /// Failed to load the ONNX model.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Failed to create an inference session.
    #[error("failed to create session: {0}")]
    SessionCreate(String),

    /// Invalid input tensor shape or type.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Inference execution failed.
    #[error("inference failed: {0}")]
    InferenceFailed(String),

    /// Output tensor extraction failed.
    #[error("failed to extract output: {0}")]
    OutputExtraction(String),

    /// I/O error when loading model files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
