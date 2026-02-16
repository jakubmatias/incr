//! Tensor types for inference input/output.

use ndarray::{ArrayD, IxDyn};

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
}

/// Input tensor for inference.
#[derive(Debug, Clone)]
pub enum InputTensor {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Uint8(ArrayD<u8>),
}

impl InputTensor {
    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            InputTensor::Float32(arr) => arr.shape(),
            InputTensor::Float64(arr) => arr.shape(),
            InputTensor::Int32(arr) => arr.shape(),
            InputTensor::Int64(arr) => arr.shape(),
            InputTensor::Uint8(arr) => arr.shape(),
        }
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> TensorType {
        match self {
            InputTensor::Float32(_) => TensorType::Float32,
            InputTensor::Float64(_) => TensorType::Float64,
            InputTensor::Int32(_) => TensorType::Int32,
            InputTensor::Int64(_) => TensorType::Int64,
            InputTensor::Uint8(_) => TensorType::Uint8,
        }
    }

    /// Create a Float32 tensor from raw data and shape.
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("shape mismatch");
        InputTensor::Float32(arr)
    }

    /// Create a Uint8 tensor from raw data and shape.
    pub fn from_u8(data: Vec<u8>, shape: Vec<usize>) -> Self {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("shape mismatch");
        InputTensor::Uint8(arr)
    }
}

/// Output tensor from inference.
#[derive(Debug, Clone)]
pub enum OutputTensor {
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Uint8(ArrayD<u8>),
}

impl OutputTensor {
    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            OutputTensor::Float32(arr) => arr.shape(),
            OutputTensor::Float64(arr) => arr.shape(),
            OutputTensor::Int32(arr) => arr.shape(),
            OutputTensor::Int64(arr) => arr.shape(),
            OutputTensor::Uint8(arr) => arr.shape(),
        }
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> TensorType {
        match self {
            OutputTensor::Float32(_) => TensorType::Float32,
            OutputTensor::Float64(_) => TensorType::Float64,
            OutputTensor::Int32(_) => TensorType::Int32,
            OutputTensor::Int64(_) => TensorType::Int64,
            OutputTensor::Uint8(_) => TensorType::Uint8,
        }
    }

    /// Try to get the inner Float32 array.
    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self {
            OutputTensor::Float32(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to get the inner Int64 array.
    pub fn as_i64(&self) -> Option<&ArrayD<i64>> {
        match self {
            OutputTensor::Int64(arr) => Some(arr),
            _ => None,
        }
    }
}
