//! Configuration structures for the OCR pipeline.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for the incr pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IncrConfig {
    /// OCR configuration.
    pub ocr: OcrConfig,

    /// PDF processing configuration.
    pub pdf: PdfConfig,

    /// Invoice extraction configuration.
    pub extraction: ExtractionConfig,

    /// Model configuration.
    pub models: ModelConfig,
}

impl Default for IncrConfig {
    fn default() -> Self {
        Self {
            ocr: OcrConfig::default(),
            pdf: PdfConfig::default(),
            extraction: ExtractionConfig::default(),
            models: ModelConfig::default(),
        }
    }
}

/// OCR engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OcrConfig {
    /// Enable text detection.
    pub enable_detection: bool,

    /// Enable angle classification.
    pub enable_classification: bool,

    /// Enable text recognition.
    pub enable_recognition: bool,

    /// Detection score threshold (0.0 - 1.0).
    pub detection_threshold: f32,

    /// Recognition confidence threshold (0.0 - 1.0).
    pub recognition_threshold: f32,

    /// Maximum image dimension (longer side) for processing.
    pub max_image_size: u32,

    /// Batch size for recognition (number of text boxes per batch).
    pub recognition_batch_size: usize,

    /// Use GPU if available.
    pub use_gpu: bool,

    /// Number of CPU threads to use.
    pub num_threads: usize,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            enable_classification: true,
            enable_recognition: true,
            detection_threshold: 0.3,
            recognition_threshold: 0.0, // Disabled - CTC confidence scores are inherently low
            max_image_size: 2048,
            recognition_batch_size: 8,
            use_gpu: false,
            num_threads: 4,
        }
    }
}

/// PDF processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PdfConfig {
    /// DPI for rendering PDF pages to images.
    pub render_dpi: u32,

    /// Process all pages or only the first.
    pub process_all_pages: bool,

    /// Maximum pages to process (0 = unlimited).
    pub max_pages: usize,

    /// Try to extract embedded text before falling back to OCR.
    pub prefer_embedded_text: bool,

    /// Minimum text length to consider PDF as text-based.
    pub min_text_length: usize,
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self {
            render_dpi: 300,
            process_all_pages: true,
            max_pages: 10,
            prefer_embedded_text: true,
            min_text_length: 50,
        }
    }
}

/// Invoice extraction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionConfig {
    /// Enable NIP checksum validation.
    pub validate_nip: bool,

    /// Enable REGON checksum validation.
    pub validate_regon: bool,

    /// Enable IBAN checksum validation.
    pub validate_iban: bool,

    /// Try to correct common OCR errors in numbers.
    pub auto_correct: bool,

    /// Minimum confidence to accept extracted field.
    pub min_field_confidence: f32,

    /// Use ML field classifier in addition to rules.
    pub use_ml_classifier: bool,

    /// Default currency if not detected.
    pub default_currency: String,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            validate_nip: true,
            validate_regon: true,
            validate_iban: true,
            auto_correct: true,
            min_field_confidence: 0.5,
            use_ml_classifier: true,
            default_currency: "PLN".to_string(),
        }
    }
}

/// Model file paths and URLs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Directory containing model files.
    pub model_dir: PathBuf,

    /// Text detection model file name.
    pub detection_model: String,

    /// Angle classification model file name.
    pub classification_model: String,

    /// Text recognition model file name.
    pub recognition_model: String,

    /// Character dictionary file name.
    pub dictionary: String,

    /// Field classifier model file name (optional ML model).
    pub classifier_model: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models"),
            detection_model: "det.onnx".to_string(),
            classification_model: "cls.onnx".to_string(), // Optional
            recognition_model: "latin_rec.onnx".to_string(),
            dictionary: "latin_dict.txt".to_string(),
            classifier_model: None,
        }
    }
}

impl IncrConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })
    }

    /// Save configuration to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let content = serde_json::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        std::fs::write(path, content)
    }

    /// Get full path to a model file.
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        self.models.model_dir.join(model_name)
    }
}
