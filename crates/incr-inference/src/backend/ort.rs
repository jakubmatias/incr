//! ONNX Runtime (ort) backend for native platforms with XNNPACK.

use std::path::Path;
use std::sync::Mutex;

use ndarray::ArrayD;
use ort::ep::XNNPACK;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tracing::debug;

use crate::error::InferenceError;
use crate::tensor::{InputTensor, OutputTensor};
use crate::{InferenceBackend, Result};

/// Backend using ONNX Runtime for native inference.
pub struct OrtBackend {
    session: Mutex<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OrtBackend {
    /// Load a model from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        debug!("Loading ONNX model from: {}", path.display());

        let bytes = std::fs::read(path)
            .map_err(|e| InferenceError::Io(e))?;

        Self::from_bytes_internal(&bytes)
    }

    /// Load a model from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_internal(bytes)
    }

    fn from_bytes_internal(bytes: &[u8]) -> Result<Self> {
        debug!("Loading ONNX model from {} bytes", bytes.len());

        let session = Session::builder()
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?
            .with_execution_providers([XNNPACK::default().build()])
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| InferenceError::SessionCreate(e.to_string()))?
            .commit_from_memory(bytes)
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        debug!("Model inputs: {:?}", input_names);
        debug!("Model outputs: {:?}", output_names);

        Ok(Self {
            session: Mutex::new(session),
            input_names,
            output_names,
        })
    }

    fn convert_input(&self, tensor: &InputTensor) -> Result<ort::session::SessionInputValue<'static>> {
        match tensor {
            InputTensor::Float32(arr) => {
                let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<f32> = arr.iter().cloned().collect();
                Tensor::from_array((shape, data))
                    .map(Into::into)
                    .map_err(|e| InferenceError::InvalidInput(e.to_string()))
            }
            InputTensor::Float64(arr) => {
                let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<f64> = arr.iter().cloned().collect();
                Tensor::from_array((shape, data))
                    .map(Into::into)
                    .map_err(|e| InferenceError::InvalidInput(e.to_string()))
            }
            InputTensor::Int32(arr) => {
                let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<i32> = arr.iter().cloned().collect();
                Tensor::from_array((shape, data))
                    .map(Into::into)
                    .map_err(|e| InferenceError::InvalidInput(e.to_string()))
            }
            InputTensor::Int64(arr) => {
                let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<i64> = arr.iter().cloned().collect();
                Tensor::from_array((shape, data))
                    .map(Into::into)
                    .map_err(|e| InferenceError::InvalidInput(e.to_string()))
            }
            InputTensor::Uint8(arr) => {
                let shape: Vec<i64> = arr.shape().iter().map(|&s| s as i64).collect();
                let data: Vec<u8> = arr.iter().cloned().collect();
                Tensor::from_array((shape, data))
                    .map(Into::into)
                    .map_err(|e| InferenceError::InvalidInput(e.to_string()))
            }
        }
    }
}

impl InferenceBackend for OrtBackend {
    fn run(&self, inputs: &[(&str, InputTensor)]) -> Result<Vec<(String, OutputTensor)>> {
        let ort_inputs: Vec<(&str, ort::session::SessionInputValue<'static>)> = inputs
            .iter()
            .map(|(name, tensor)| {
                let value = self.convert_input(tensor)?;
                Ok((*name, value))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut session = self.session.lock()
            .map_err(|e| InferenceError::InferenceFailed(format!("Failed to lock session: {}", e)))?;

        let outputs = session
            .run(ort_inputs)
            .map_err(|e| InferenceError::InferenceFailed(e.to_string()))?;

        let mut results = Vec::with_capacity(outputs.len());

        for (name, value) in outputs.iter() {
            let tensor = if let Ok(tensor_ref) = value.try_extract_tensor::<f32>() {
                let (shape_ref, data) = tensor_ref;
                let shape: Vec<usize> = shape_ref.iter().map(|&s| s as usize).collect();
                let data_vec: Vec<f32> = data.to_vec();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data_vec)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Float32(arr)
            } else if let Ok(tensor_ref) = value.try_extract_tensor::<i64>() {
                let (shape_ref, data) = tensor_ref;
                let shape: Vec<usize> = shape_ref.iter().map(|&s| s as usize).collect();
                let data_vec: Vec<i64> = data.to_vec();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data_vec)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Int64(arr)
            } else if let Ok(tensor_ref) = value.try_extract_tensor::<i32>() {
                let (shape_ref, data) = tensor_ref;
                let shape: Vec<usize> = shape_ref.iter().map(|&s| s as usize).collect();
                let data_vec: Vec<i32> = data.to_vec();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data_vec)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Int32(arr)
            } else if let Ok(tensor_ref) = value.try_extract_tensor::<f64>() {
                let (shape_ref, data) = tensor_ref;
                let shape: Vec<usize> = shape_ref.iter().map(|&s| s as usize).collect();
                let data_vec: Vec<f64> = data.to_vec();
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data_vec)
                    .map_err(|e| InferenceError::OutputExtraction(e.to_string()))?;
                OutputTensor::Float64(arr)
            } else {
                return Err(InferenceError::OutputExtraction(
                    format!("unsupported output type for '{}'", name),
                ));
            };

            results.push((name.to_string(), tensor));
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
