//! Inference backend implementations.

#[cfg(feature = "native")]
pub mod ort;

#[cfg(feature = "wasm")]
pub mod tract;

use crate::{InputTensor, OutputTensor, Result};

/// Trait for ONNX inference backends.
///
/// This trait abstracts over different ONNX runtime implementations,
/// allowing the same code to run on native platforms (via ort) and
/// in the browser (via tract).
pub trait InferenceBackend: Send + Sync {
    /// Run inference with the given inputs.
    ///
    /// # Arguments
    /// * `inputs` - Named input tensors
    ///
    /// # Returns
    /// Named output tensors from the model
    fn run(&self, inputs: &[(&str, InputTensor)]) -> Result<Vec<(String, OutputTensor)>>;

    /// Get the input names expected by the model.
    fn input_names(&self) -> &[String];

    /// Get the output names produced by the model.
    fn output_names(&self) -> &[String];
}
