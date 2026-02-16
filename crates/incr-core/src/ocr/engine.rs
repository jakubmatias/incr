//! Complete OCR engine orchestrating detection, classification, and recognition.

use std::path::Path;
use std::time::Instant;

use image::{DynamicImage, GenericImageView};
use tracing::{debug, info};

use crate::error::OcrError;
use crate::models::config::OcrConfig;
use incr_inference::InferenceBackend;

use super::{
    classifier::AngleClassifier,
    detector::TextDetector,
    layout::{LayoutDetector, LayoutResult},
    preprocessing::ImagePreprocessor,
    recognizer::TextRecognizer,
    OcrResult, TextBox,
};

/// Complete OCR engine combining detection, classification, and recognition.
pub struct OcrEngine<B: InferenceBackend> {
    detector: Option<TextDetector<B>>,
    classifier: Option<AngleClassifier<B>>,
    recognizer: Option<TextRecognizer<B>>,
    layout_detector: Option<LayoutDetector<B>>,
    preprocessor: ImagePreprocessor,
    config: OcrConfig,
}

/// Builder for OcrEngine.
pub struct OcrEngineBuilder<B: InferenceBackend> {
    detector: Option<TextDetector<B>>,
    classifier: Option<AngleClassifier<B>>,
    recognizer: Option<TextRecognizer<B>>,
    layout_detector: Option<LayoutDetector<B>>,
    config: OcrConfig,
}

impl<B: InferenceBackend> OcrEngineBuilder<B> {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            detector: None,
            classifier: None,
            recognizer: None,
            layout_detector: None,
            config: OcrConfig::default(),
        }
    }

    /// Set the text detector.
    pub fn with_detector(mut self, detector: TextDetector<B>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// Set the angle classifier.
    pub fn with_classifier(mut self, classifier: AngleClassifier<B>) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Set the text recognizer.
    pub fn with_recognizer(mut self, recognizer: TextRecognizer<B>) -> Self {
        self.recognizer = Some(recognizer);
        self
    }

    /// Set the layout detector.
    pub fn with_layout_detector(mut self, layout_detector: LayoutDetector<B>) -> Self {
        self.layout_detector = Some(layout_detector);
        self
    }

    /// Set configuration.
    pub fn with_config(mut self, config: OcrConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the OCR engine.
    pub fn build(self) -> OcrEngine<B> {
        OcrEngine {
            detector: self.detector,
            classifier: self.classifier,
            recognizer: self.recognizer,
            layout_detector: self.layout_detector,
            preprocessor: ImagePreprocessor::new().with_max_size(self.config.max_image_size),
            config: self.config,
        }
    }
}

impl<B: InferenceBackend> Default for OcrEngineBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: InferenceBackend> OcrEngine<B> {
    /// Create a new builder.
    pub fn builder() -> OcrEngineBuilder<B> {
        OcrEngineBuilder::new()
    }

    /// Process an image and extract text.
    pub fn process(&self, image: &DynamicImage) -> Result<OcrResult, OcrError> {
        let start = Instant::now();
        let (width, height) = image.dimensions();

        info!("Processing image: {}x{}", width, height);

        // Step 1: Detect text regions
        let detection_result = if let Some(ref detector) = self.detector {
            if self.config.enable_detection {
                detector.detect(image)?
            } else {
                // If detection disabled, treat whole image as one region
                super::detector::DetectionResult {
                    boxes: vec![[
                        0.0,
                        0.0,
                        width as f32,
                        0.0,
                        width as f32,
                        height as f32,
                        0.0,
                        height as f32,
                    ]],
                    scores: vec![1.0],
                    image_size: (width, height),
                }
            }
        } else {
            return Err(OcrError::Detection("No detector configured".to_string()));
        };

        if detection_result.boxes.is_empty() {
            debug!("No text regions detected");
            return Ok(OcrResult::empty(width, height));
        }

        debug!("Detected {} text regions", detection_result.boxes.len());

        // Step 2: Process each detected region
        let mut text_boxes = Vec::with_capacity(detection_result.boxes.len());

        for (bbox, det_score) in detection_result
            .boxes
            .iter()
            .zip(detection_result.scores.iter())
        {
            // Crop the region
            let cropped = self.preprocessor.crop_text_region(image, bbox)?;

            // Step 2a: Classify angle (optional)
            let (rotated, angle) = if let Some(ref classifier) = self.classifier {
                if self.config.enable_classification {
                    let (angle, _conf) = classifier.classify(&cropped)?;
                    let rotated = if angle == 180 {
                        cropped.rotate180()
                    } else {
                        cropped
                    };
                    (rotated, angle)
                } else {
                    (cropped, 0)
                }
            } else {
                (cropped, 0)
            };

            // Step 2b: Recognize text
            let (text, rec_score) = if let Some(ref recognizer) = self.recognizer {
                if self.config.enable_recognition {
                    let result = recognizer.recognize(&rotated)?;
                    (result.text, result.confidence)
                } else {
                    (String::new(), 0.0)
                }
            } else {
                (String::new(), 0.0)
            };

            // Filter by confidence threshold
            if rec_score >= self.config.recognition_threshold || !self.config.enable_recognition {
                text_boxes.push(TextBox {
                    bbox: *bbox,
                    text,
                    detection_score: *det_score,
                    recognition_score: rec_score,
                    angle,
                });
            }
        }

        // Detect layout if available
        let layout = if let Some(ref layout_detector) = self.layout_detector {
            match layout_detector.detect(image) {
                Ok(layout_result) => {
                    use super::{LayoutInfo, RegionBox};

                    let tables: Vec<RegionBox> = layout_result
                        .tables()
                        .iter()
                        .map(|r| RegionBox {
                            region_type: "table".to_string(),
                            bbox: r.bbox,
                            confidence: r.confidence,
                        })
                        .collect();

                    let text_regions: Vec<RegionBox> = layout_result
                        .text_regions()
                        .iter()
                        .map(|r| RegionBox {
                            region_type: format!("{:?}", r.region_type).to_lowercase(),
                            bbox: r.bbox,
                            confidence: r.confidence,
                        })
                        .collect();

                    let figures: Vec<RegionBox> = layout_result
                        .regions
                        .iter()
                        .filter(|r| matches!(r.region_type, super::layout::LayoutType::Figure))
                        .map(|r| RegionBox {
                            region_type: "figure".to_string(),
                            bbox: r.bbox,
                            confidence: r.confidence,
                        })
                        .collect();

                    debug!(
                        "Layout detected: {} tables, {} text regions, {} figures",
                        tables.len(),
                        text_regions.len(),
                        figures.len()
                    );

                    Some(LayoutInfo {
                        tables,
                        text_regions,
                        figures,
                    })
                }
                Err(e) => {
                    debug!("Layout detection failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Sort by reading order
        let mut result = OcrResult {
            boxes: text_boxes,
            text: String::new(),
            processing_time_ms: start.elapsed().as_millis() as u64,
            image_size: (width, height),
            layout,
        };

        result.sort_by_reading_order();

        info!(
            "OCR complete: {} text boxes in {}ms",
            result.boxes.len(),
            result.processing_time_ms
        );

        Ok(result)
    }

    /// Process multiple images.
    pub fn process_batch(&self, images: &[DynamicImage]) -> Result<Vec<OcrResult>, OcrError> {
        images.iter().map(|img| self.process(img)).collect()
    }

    /// Get OCR result as plain text.
    pub fn extract_text(&self, image: &DynamicImage) -> Result<String, OcrError> {
        let result = self.process(image)?;
        Ok(result.text)
    }

    /// Detect layout regions in an image.
    pub fn detect_layout(&self, image: &DynamicImage) -> Result<Option<LayoutResult>, OcrError> {
        if let Some(ref layout_detector) = self.layout_detector {
            let result = layout_detector.detect(image)?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Check if layout detection is available.
    pub fn has_layout_detection(&self) -> bool {
        self.layout_detector.is_some()
    }
}

/// Convenience function to create an OCR engine with models from a directory.
#[cfg(feature = "native")]
pub fn create_engine_from_dir(
    model_dir: &Path,
    config: OcrConfig,
) -> Result<OcrEngine<crate::OrtBackend>, OcrError> {
    use crate::OrtBackend;
    use super::layout::LayoutDetector;

    let det_path = model_dir.join("det.onnx");
    let cls_path = model_dir.join("cls.onnx");
    let rec_path = model_dir.join("latin_rec.onnx");
    let dict_path = model_dir.join("latin_dict.txt");
    let layout_path = model_dir.join("layout.onnx");

    let mut builder = OcrEngine::builder().with_config(config.clone());

    // Load detector
    if config.enable_detection && det_path.exists() {
        let backend = OrtBackend::from_file(&det_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load detector: {}", e)))?;
        builder = builder.with_detector(TextDetector::new(backend));
    }

    // Load classifier
    if config.enable_classification && cls_path.exists() {
        let backend = OrtBackend::from_file(&cls_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load classifier: {}", e)))?;
        builder = builder.with_classifier(AngleClassifier::new(backend));
    }

    // Load recognizer
    if config.enable_recognition && rec_path.exists() {
        let backend = OrtBackend::from_file(&rec_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load recognizer: {}", e)))?;

        let dictionary = if dict_path.exists() {
            TextRecognizer::<OrtBackend>::load_dictionary(&dict_path)?
        } else {
            TextRecognizer::<OrtBackend>::default_latin_dictionary()
        };

        builder = builder.with_recognizer(TextRecognizer::new(backend, dictionary));
    }

    // Load layout detector (PP-Structure)
    if layout_path.exists() {
        let backend = OrtBackend::from_file(&layout_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load layout detector: {}", e)))?;
        builder = builder.with_layout_detector(LayoutDetector::new(backend));
        debug!("Loaded layout detector from {}", layout_path.display());
    }

    Ok(builder.build())
}

/// Create an OCR engine from embedded mobile models.
/// This provides a standalone binary with no external model files needed.
#[cfg(feature = "native")]
pub fn create_engine_from_embedded(
    config: OcrConfig,
) -> Result<OcrEngine<crate::OrtBackend>, OcrError> {
    use crate::models::embedded::EmbeddedModels;
    use crate::OrtBackend;
    use super::layout::LayoutDetector;

    let models = EmbeddedModels::mobile();
    let mut builder = OcrEngine::builder().with_config(config.clone());

    // Load detector from embedded bytes
    if config.enable_detection {
        let backend = OrtBackend::from_bytes(models.detection)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load embedded detector: {}", e)))?;
        builder = builder.with_detector(TextDetector::new(backend));
        debug!("Loaded embedded detector ({} bytes)", models.detection.len());
    }

    // Load recognizer from embedded bytes
    if config.enable_recognition {
        let backend = OrtBackend::from_bytes(models.recognition)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load embedded recognizer: {}", e)))?;

        // Convert dictionary lines to chars (first char of each line)
        let mut dictionary: Vec<char> = vec![' ']; // Blank token
        for line in models.dictionary.lines() {
            if let Some(c) = line.chars().next() {
                dictionary.push(c);
            }
        }

        builder = builder.with_recognizer(TextRecognizer::new(backend, dictionary));
        debug!("Loaded embedded recognizer ({} bytes)", models.recognition.len());
    }

    // Load layout detector from embedded bytes
    if !models.layout.is_empty() {
        let backend = OrtBackend::from_bytes(models.layout)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to load embedded layout detector: {}", e)))?;
        builder = builder.with_layout_detector(LayoutDetector::new(backend));
        debug!("Loaded embedded layout detector ({} bytes)", models.layout.len());
    }

    info!("Created OCR engine from embedded mobile models");
    Ok(builder.build())
}
