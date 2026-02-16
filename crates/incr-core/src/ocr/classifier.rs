//! Angle classification for text regions.

use image::DynamicImage;
use tracing::debug;

use crate::error::OcrError;
use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

use super::preprocessing::ImagePreprocessor;

/// Angle classifier for detecting text orientation.
pub struct AngleClassifier<B: InferenceBackend> {
    backend: B,
    preprocessor: ImagePreprocessor,
    threshold: f32,
}

impl<B: InferenceBackend> AngleClassifier<B> {
    /// Create a new angle classifier.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            preprocessor: ImagePreprocessor::new(),
            threshold: 0.9,
        }
    }

    /// Set classification threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Classify the angle of a text region.
    ///
    /// Returns (angle, confidence) where angle is 0 or 180.
    pub fn classify(&self, image: &DynamicImage) -> Result<(i32, f32), OcrError> {
        let tensor = self
            .preprocessor
            .preprocess_for_classification(image)
            .map_err(|e| OcrError::Preprocessing(e.to_string()))?;

        let input = InputTensor::Float32(tensor.into_dyn());

        let outputs = self
            .backend
            .run(&[("x", input)])
            .map_err(|e| OcrError::Recognition(e.to_string()))?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::Recognition("No output from classifier".to_string()))?
            .1;

        let output_arr = match output {
            OutputTensor::Float32(arr) => arr,
            _ => return Err(OcrError::Recognition("Unexpected output type".to_string())),
        };

        // Output is [1, 2] - probabilities for [0°, 180°]
        let probs = output_arr.as_slice().unwrap_or(&[0.5, 0.5]);

        let (angle, confidence) = if probs.len() >= 2 {
            if probs[0] > probs[1] {
                (0, probs[0])
            } else {
                (180, probs[1])
            }
        } else {
            (0, 1.0)
        };

        debug!("Classified angle: {}° (confidence: {:.3})", angle, confidence);

        Ok((angle, confidence))
    }

    /// Classify multiple images in a batch.
    pub fn classify_batch(&self, images: &[DynamicImage]) -> Result<Vec<(i32, f32)>, OcrError> {
        // For simplicity, process one at a time
        // A real implementation would batch the inputs
        images.iter().map(|img| self.classify(img)).collect()
    }

    /// Check if image needs rotation based on classification.
    pub fn needs_rotation(&self, image: &DynamicImage) -> Result<bool, OcrError> {
        let (angle, confidence) = self.classify(image)?;
        Ok(angle == 180 && confidence > self.threshold)
    }

    /// Rotate image if needed based on classification.
    pub fn auto_rotate(&self, image: DynamicImage) -> Result<DynamicImage, OcrError> {
        if self.needs_rotation(&image)? {
            Ok(image.rotate180())
        } else {
            Ok(image)
        }
    }
}
