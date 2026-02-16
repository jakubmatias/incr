//! Tract backend for cross-platform ONNX inference.

use std::path::Path;

use ndarray::ArrayD;
use tract_onnx::prelude::*;
use tracing::debug;

use crate::error::InferenceError;
use crate::tensor::{InputTensor, OutputTensor};
use crate::{InferenceBackend, Result};

/// Backend using Tract for cross-platform ONNX inference.
pub struct TractBackend {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl TractBackend {
    /// Load a model from a file path with default input shape (batch=1, channels=3, height=640, width=640).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Default to common OCR input dimensions
        Self::from_file_with_shape(path, &[1, 3, 640, 640])
    }

    /// Load a model from a file path with specified input shape.
    pub fn from_file_with_shape<P: AsRef<Path>>(path: P, input_shape: &[usize]) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading ONNX model with Tract from: {}", path.display());

        // Load as inference model first
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to load model: {}", e)))?;

        // Set input fact with concrete shape to replace dynamic dimensions
        model
            .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to set input shape: {}", e)))?;

        // Now convert to typed model and optimize
        let model = model
            .into_typed()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to type model: {}", e)))?
            .into_optimized()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to optimize: {}", e)))?
            .into_runnable()
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?;

        // Tract doesn't expose input/output names as easily, use indices
        let input_names = vec!["input".to_string()];
        let output_names = vec!["output".to_string()];

        Ok(Self {
            model,
            input_names,
            output_names,
        })
    }

    /// Load a model from bytes with default input shape.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_shape(bytes, &[1, 3, 640, 640])
    }

    /// Load a model from bytes with specified input shape.
    pub fn from_bytes_with_shape(bytes: &[u8], input_shape: &[usize]) -> Result<Self> {
        debug!("Loading ONNX model with Tract from {} bytes", bytes.len());

        // Load as inference model first
        let mut model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to load model: {}", e)))?;

        // Set input fact with concrete shape to replace dynamic dimensions
        model
            .set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), input_shape))
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to set input shape: {}", e)))?;

        // Now convert to typed model and optimize
        let model = model
            .into_typed()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to type model: {}", e)))?
            .into_optimized()
            .map_err(|e| InferenceError::ModelLoad(format!("Failed to optimize: {}", e)))?
            .into_runnable()
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?;

        let input_names = vec!["input".to_string()];
        let output_names = vec!["output".to_string()];

        Ok(Self {
            model,
            input_names,
            output_names,
        })
    }

    fn convert_input(&self, tensor: &InputTensor) -> Result<TValue> {
        match tensor {
            InputTensor::Float32(arr) => {
                let shape: TVec<usize> = arr.shape().iter().cloned().collect();
                let data: Vec<f32> = arr.iter().cloned().collect();
                let tract_tensor = tract_ndarray::ArrayD::from_shape_vec(
                    tract_ndarray::IxDyn(shape.as_slice()),
                    data,
                )
                .map_err(|e| InferenceError::InvalidInput(e.to_string()))?;
                Ok(tract_tensor.into_tvalue())
            }
            InputTensor::Int64(arr) => {
                let shape: TVec<usize> = arr.shape().iter().cloned().collect();
                let data: Vec<i64> = arr.iter().cloned().collect();
                let tract_tensor = tract_ndarray::ArrayD::from_shape_vec(
                    tract_ndarray::IxDyn(shape.as_slice()),
                    data,
                )
                .map_err(|e| InferenceError::InvalidInput(e.to_string()))?;
                Ok(tract_tensor.into_tvalue())
            }
            InputTensor::Int32(arr) => {
                let shape: TVec<usize> = arr.shape().iter().cloned().collect();
                let data: Vec<i32> = arr.iter().cloned().collect();
                let tract_tensor = tract_ndarray::ArrayD::from_shape_vec(
                    tract_ndarray::IxDyn(shape.as_slice()),
                    data,
                )
                .map_err(|e| InferenceError::InvalidInput(e.to_string()))?;
                Ok(tract_tensor.into_tvalue())
            }
            InputTensor::Uint8(arr) => {
                let shape: TVec<usize> = arr.shape().iter().cloned().collect();
                let data: Vec<u8> = arr.iter().cloned().collect();
                let tract_tensor = tract_ndarray::ArrayD::from_shape_vec(
                    tract_ndarray::IxDyn(shape.as_slice()),
                    data,
                )
                .map_err(|e| InferenceError::InvalidInput(e.to_string()))?;
                Ok(tract_tensor.into_tvalue())
            }
            InputTensor::Float64(arr) => {
                let shape: TVec<usize> = arr.shape().iter().cloned().collect();
                let data: Vec<f64> = arr.iter().cloned().collect();
                let tract_tensor = tract_ndarray::ArrayD::from_shape_vec(
                    tract_ndarray::IxDyn(shape.as_slice()),
                    data,
                )
                .map_err(|e| InferenceError::InvalidInput(e.to_string()))?;
                Ok(tract_tensor.into_tvalue())
            }
        }
    }
}

impl InferenceBackend for TractBackend {
    fn run(&self, inputs: &[(&str, InputTensor)]) -> Result<Vec<(String, OutputTensor)>> {
        let tract_inputs: TVec<TValue> = inputs
            .iter()
            .map(|(_, tensor)| self.convert_input(tensor))
            .collect::<Result<TVec<_>>>()?;

        let outputs = self
            .model
            .run(tract_inputs)
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;

        let mut results = Vec::with_capacity(outputs.len());

        for (idx, output) in outputs.iter().enumerate() {
            let name = self.output_names.get(idx)
                .cloned()
                .unwrap_or_else(|| format!("output_{}", idx));

            let tensor = if let Ok(arr) = output.to_array_view::<f32>() {
                let shape: Vec<usize> = arr.shape().to_vec();
                let data: Vec<f32> = arr.iter().cloned().collect();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Float32(arr)
            } else if let Ok(arr) = output.to_array_view::<i64>() {
                let shape: Vec<usize> = arr.shape().to_vec();
                let data: Vec<i64> = arr.iter().cloned().collect();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Int64(arr)
            } else if let Ok(arr) = output.to_array_view::<i32>() {
                let shape: Vec<usize> = arr.shape().to_vec();
                let data: Vec<i32> = arr.iter().cloned().collect();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Int32(arr)
            } else {
                return Err(InferenceError::OutputExtraction(
                    format!("unsupported output type for '{}'", name),
                ));
            };

            results.push((name, tensor));
        }

        Ok(results)
    }

    fn input_names(&self) -> &[String] {
        &self.input_names
    }

    fn output_names(&self) -> &[String] {
        &self.output_names
    }
}
