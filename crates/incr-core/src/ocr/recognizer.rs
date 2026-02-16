//! Text recognition using PaddleOCR recognition model.

use std::path::Path;

use image::DynamicImage;
use ndarray::ArrayD;
use tracing::{debug, trace};

use crate::error::OcrError;
use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

use super::preprocessing::ImagePreprocessor;

/// Text recognizer using PaddleOCR CRNN model.
pub struct TextRecognizer<B: InferenceBackend> {
    backend: B,
    preprocessor: ImagePreprocessor,
    dictionary: Vec<char>,
    threshold: f32,
}

/// Recognition result for a single text region.
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Recognized text.
    pub text: String,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Per-character confidences.
    pub char_scores: Vec<f32>,
}

impl<B: InferenceBackend> TextRecognizer<B> {
    /// Create a new text recognizer.
    pub fn new(backend: B, dictionary: Vec<char>) -> Self {
        Self {
            backend,
            preprocessor: ImagePreprocessor::new(),
            dictionary,
            threshold: 0.5,
        }
    }

    /// Set recognition confidence threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Load dictionary from a file.
    pub fn load_dictionary(path: &Path) -> Result<Vec<char>, OcrError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load dictionary: {}", e)))?;

        // Dictionary file has one character per line
        // First entry is usually blank (for CTC blank token)
        let mut chars: Vec<char> = vec![' ']; // Blank token

        for line in content.lines() {
            if let Some(c) = line.chars().next() {
                chars.push(c);
            }
        }

        debug!("Loaded dictionary with {} characters", chars.len());
        Ok(chars)
    }

    /// Create a default Latin dictionary for Polish text.
    pub fn default_latin_dictionary() -> Vec<char> {
        let mut chars = vec![' ']; // Blank token for CTC

        // Basic ASCII
        for c in '0'..='9' {
            chars.push(c);
        }
        for c in 'A'..='Z' {
            chars.push(c);
        }
        for c in 'a'..='z' {
            chars.push(c);
        }

        // Polish characters
        chars.extend([
            'Ą', 'ą', 'Ć', 'ć', 'Ę', 'ę', 'Ł', 'ł', 'Ń', 'ń', 'Ó', 'ó', 'Ś', 'ś', 'Ź', 'ź', 'Ż',
            'ż',
        ]);

        // Common punctuation and symbols
        chars.extend([
            '.', ',', ';', ':', '!', '?', '-', '_', '/', '\\', '(', ')', '[', ']', '{', '}', '<',
            '>', '@', '#', '$', '%', '^', '&', '*', '+', '=', '|', '~', '`', '\'', '"', ' ',
        ]);

        // Currency and special
        chars.extend(['€', '£', '¥', '§', '©', '®', '°', '²', '³', '½', '¼', '¾']);

        chars
    }

    /// Recognize text in a cropped image.
    pub fn recognize(&self, image: &DynamicImage) -> Result<RecognitionResult, OcrError> {
        let tensor = self
            .preprocessor
            .preprocess_for_recognition(image)
            .map_err(|e| OcrError::Preprocessing(e.to_string()))?;

        let input = InputTensor::Float32(tensor.into_dyn());

        let outputs = self
            .backend
            .run(&[("x", input)])
            .map_err(|e| OcrError::Recognition(e.to_string()))?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::Recognition("No output from model".to_string()))?
            .1;

        let output_arr = match output {
            OutputTensor::Float32(arr) => arr,
            _ => return Err(OcrError::Recognition("Unexpected output type".to_string())),
        };

        self.decode_output(&output_arr)
    }

    /// Recognize text in multiple images (batched).
    pub fn recognize_batch(
        &self,
        images: &[DynamicImage],
    ) -> Result<Vec<RecognitionResult>, OcrError> {
        // For simplicity, process one at a time
        // A real implementation would batch inputs for efficiency
        images.iter().map(|img| self.recognize(img)).collect()
    }

    fn decode_output(&self, output: &ArrayD<f32>) -> Result<RecognitionResult, OcrError> {
        // Output shape is [1, T, num_classes] where T is sequence length
        let shape = output.shape();
        if shape.len() < 3 {
            return Err(OcrError::Recognition(format!(
                "Invalid output shape: {:?}",
                shape
            )));
        }

        let seq_len = shape[1];
        let num_classes = shape[2];

        let mut text = String::new();
        let mut char_scores = Vec::new();
        let mut prev_idx = 0usize;

        // CTC decoding: take argmax at each timestep, remove blanks and duplicates
        for t in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for c in 0..num_classes {
                let val = output[[0, t, c]];
                if val > max_val {
                    max_val = val;
                    max_idx = c;
                }
            }

            // Apply softmax to get probability
            let mut sum_exp = 0.0f32;
            for c in 0..num_classes {
                sum_exp += (output[[0, t, c]] - max_val).exp();
            }
            let confidence = 1.0 / sum_exp;

            // Skip blank token (index 0) and duplicates
            if max_idx != 0 && max_idx != prev_idx {
                if let Some(&c) = self.dictionary.get(max_idx) {
                    text.push(c);
                    char_scores.push(confidence);
                }
            }

            prev_idx = max_idx;
        }

        // Calculate overall confidence
        let avg_confidence = if char_scores.is_empty() {
            0.0
        } else {
            char_scores.iter().sum::<f32>() / char_scores.len() as f32
        };

        trace!("Recognized: '{}' (confidence: {:.3})", text, avg_confidence);

        Ok(RecognitionResult {
            text,
            confidence: avg_confidence,
            char_scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_dictionary() {
        let dict = TextRecognizer::<incr_inference::OrtBackend>::default_latin_dictionary();

        // Should contain Polish characters
        assert!(dict.contains(&'ą'));
        assert!(dict.contains(&'ę'));
        assert!(dict.contains(&'ł'));
        assert!(dict.contains(&'ż'));

        // Should contain digits
        assert!(dict.contains(&'0'));
        assert!(dict.contains(&'9'));

        // Should contain basic punctuation
        assert!(dict.contains(&'.'));
        assert!(dict.contains(&','));
    }
}
