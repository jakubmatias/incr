//! Pure Rust OCR engine wrapper using `pure-onnx-ocr`.

use std::path::Path;
use std::time::Instant;

use image::{DynamicImage, GenericImageView};
use tracing::{debug, info};

use crate::error::OcrError;
use crate::models::config::OcrConfig;

use super::{OcrResult, TextBox};

/// OCR engine backed by `pure-onnx-ocr` (pure Rust, no external ONNX Runtime).
pub struct PureOcrEngine {
    engine: pure_onnx_ocr::engine::OcrEngine,
    #[allow(dead_code)]
    config: OcrConfig,
    /// Keep temp dir alive so the temp files aren't deleted.
    _temp_dir: Option<tempfile::TempDir>,
}

impl PureOcrEngine {
    /// Create an engine from model files in a directory.
    pub fn from_dir(model_dir: &Path, config: OcrConfig) -> Result<Self, OcrError> {
        let det_path = model_dir.join("det.onnx");
        let rec_path = model_dir.join("latin_rec.onnx");
        let dict_path = model_dir.join("latin_dict.txt");

        let engine = pure_onnx_ocr::engine::OcrEngineBuilder::new()
            .det_model_path(&det_path)
            .rec_model_path(&rec_path)
            .dictionary_path(&dict_path)
            .build()
            .map_err(|e| OcrError::ModelLoad(format!("pure-onnx-ocr: {}", e)))?;

        info!("Loaded pure-onnx-ocr engine from {}", model_dir.display());

        Ok(Self {
            engine,
            config,
            _temp_dir: None,
        })
    }

    /// Create an engine from embedded model bytes.
    ///
    /// Writes model bytes to temporary files (required by `pure-onnx-ocr`'s
    /// file-path-based API), then loads from those paths. The temp directory
    /// is kept alive for the lifetime of the engine.
    pub fn from_embedded(config: OcrConfig) -> Result<Self, OcrError> {
        use crate::models::embedded::EmbeddedModels;

        let models = EmbeddedModels::mobile();
        let temp_dir = tempfile::tempdir()
            .map_err(|e| OcrError::ModelLoad(format!("failed to create temp dir: {}", e)))?;

        let det_path = temp_dir.path().join("det.onnx");
        let rec_path = temp_dir.path().join("latin_rec.onnx");
        let dict_path = temp_dir.path().join("latin_dict.txt");

        std::fs::write(&det_path, models.detection)
            .map_err(|e| OcrError::ModelLoad(format!("failed to write det model: {}", e)))?;
        std::fs::write(&rec_path, models.recognition)
            .map_err(|e| OcrError::ModelLoad(format!("failed to write rec model: {}", e)))?;
        std::fs::write(&dict_path, models.dictionary)
            .map_err(|e| OcrError::ModelLoad(format!("failed to write dictionary: {}", e)))?;

        debug!(
            "Wrote embedded models to temp dir: {}",
            temp_dir.path().display()
        );

        let engine = pure_onnx_ocr::engine::OcrEngineBuilder::new()
            .det_model_path(&det_path)
            .rec_model_path(&rec_path)
            .dictionary_path(&dict_path)
            .build()
            .map_err(|e| OcrError::ModelLoad(format!("pure-onnx-ocr: {}", e)))?;

        info!("Created pure-onnx-ocr engine from embedded models");

        Ok(Self {
            engine,
            config,
            _temp_dir: Some(temp_dir),
        })
    }

    /// Process an image and extract text with bounding boxes.
    pub fn process(&self, image: &DynamicImage) -> Result<OcrResult, OcrError> {
        let start = Instant::now();
        let (width, height) = image.dimensions();

        info!("Processing image: {}x{}", width, height);

        let results = self
            .engine
            .run_from_image(image)
            .map_err(|e| OcrError::Detection(format!("pure-onnx-ocr: {}", e)))?;

        debug!("pure-onnx-ocr returned {} text regions", results.len());

        let mut text_boxes: Vec<TextBox> = results
            .iter()
            .map(|r| {
                let bbox = polygon_to_bbox(&r.bounding_box);
                let text = if self.config.keep_unk {
                    r.text.clone()
                } else {
                    r.text.replace("[UNK]", " ")
                };
                TextBox {
                    bbox,
                    text,
                    detection_score: r.confidence,
                    recognition_score: r.confidence,
                    angle: 0,
                }
            })
            .collect();

        // Sort by reading order
        text_boxes.sort_by(|a, b| {
            let (_, ay, _, _) = a.rect();
            let (_, by, _, _) = b.rect();
            let row_a = (ay / 20.0) as i32;
            let row_b = (by / 20.0) as i32;
            if row_a != row_b {
                row_a.cmp(&row_b)
            } else {
                let (ax, _, _, _) = a.rect();
                let (bx, _, _, _) = b.rect();
                ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let text = text_boxes
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let processing_time_ms = start.elapsed().as_millis() as u64;

        info!(
            "OCR complete: {} text boxes in {}ms",
            text_boxes.len(),
            processing_time_ms
        );

        Ok(OcrResult {
            boxes: text_boxes,
            text,
            processing_time_ms,
            image_size: (width, height),
            layout: None,
        })
    }

    /// Convenience: extract text only.
    pub fn extract_text(&self, image: &DynamicImage) -> Result<String, OcrError> {
        Ok(self.process(image)?.text)
    }
}

/// Convert a `Polygon<f64>` to our `[f32; 8]` bbox format.
///
/// Extracts the first 4 exterior points (quadrilateral) as
/// `[x1, y1, x2, y2, x3, y3, x4, y4]`.
fn polygon_to_bbox(polygon: &pure_onnx_ocr::Polygon<f64>) -> [f32; 8] {
    let mut bbox = [0.0f32; 8];
    for (i, coord) in polygon.exterior().coords().take(4).enumerate() {
        bbox[i * 2] = coord.x as f32;
        bbox[i * 2 + 1] = coord.y as f32;
    }
    bbox
}
