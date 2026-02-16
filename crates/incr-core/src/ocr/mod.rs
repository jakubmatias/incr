//! OCR pipeline using PaddleOCR models.

mod classifier;
mod detector;
mod engine;
mod layout;
mod preprocessing;
mod recognizer;
mod table;

pub use classifier::AngleClassifier;
pub use detector::TextDetector;
pub use engine::{OcrEngine, OcrEngineBuilder};
pub use layout::{LayoutDetector, LayoutModelType, LayoutRegion, LayoutResult, LayoutType};
pub use preprocessing::ImagePreprocessor;
pub use recognizer::TextRecognizer;
pub use table::{TableCell, TableClassifier, TableRecognizer, TableStructure, TableType};

#[cfg(feature = "native")]
pub use engine::{create_engine_from_dir, create_engine_from_embedded};

use serde::{Deserialize, Serialize};

/// A detected text box with its coordinates and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBox {
    /// Bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4) for quadrilateral.
    pub bbox: [f32; 8],

    /// Recognized text content.
    pub text: String,

    /// Detection confidence score (0.0 - 1.0).
    pub detection_score: f32,

    /// Recognition confidence score (0.0 - 1.0).
    pub recognition_score: f32,

    /// Detected angle (0, 90, 180, 270).
    pub angle: i32,
}

impl TextBox {
    /// Get the center point of the bounding box.
    pub fn center(&self) -> (f32, f32) {
        let x = (self.bbox[0] + self.bbox[2] + self.bbox[4] + self.bbox[6]) / 4.0;
        let y = (self.bbox[1] + self.bbox[3] + self.bbox[5] + self.bbox[7]) / 4.0;
        (x, y)
    }

    /// Get the width of the bounding box.
    pub fn width(&self) -> f32 {
        let dx1 = self.bbox[2] - self.bbox[0];
        let dy1 = self.bbox[3] - self.bbox[1];
        (dx1 * dx1 + dy1 * dy1).sqrt()
    }

    /// Get the height of the bounding box.
    pub fn height(&self) -> f32 {
        let dx1 = self.bbox[6] - self.bbox[0];
        let dy1 = self.bbox[7] - self.bbox[1];
        (dx1 * dx1 + dy1 * dy1).sqrt()
    }

    /// Get the axis-aligned bounding rectangle.
    pub fn rect(&self) -> (f32, f32, f32, f32) {
        let xs = [self.bbox[0], self.bbox[2], self.bbox[4], self.bbox[6]];
        let ys = [self.bbox[1], self.bbox[3], self.bbox[5], self.bbox[7]];

        let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        (min_x, min_y, max_x, max_y)
    }
}

/// Result of OCR processing on an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Detected and recognized text boxes.
    pub boxes: Vec<TextBox>,

    /// Full text (boxes joined with newlines).
    pub text: String,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,

    /// Image dimensions (width, height).
    pub image_size: (u32, u32),

    /// Layout regions detected (if layout detection was enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutInfo>,
}

/// Layout information from PP-Structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutInfo {
    /// Table regions detected.
    pub tables: Vec<RegionBox>,
    /// Text regions detected.
    pub text_regions: Vec<RegionBox>,
    /// Figure regions detected.
    pub figures: Vec<RegionBox>,
}

/// A detected region with bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionBox {
    /// Region type name.
    pub region_type: String,
    /// Bounding box (x1, y1, x2, y2).
    pub bbox: [f32; 4],
    /// Confidence score.
    pub confidence: f32,
}

impl OcrResult {
    /// Create an empty result.
    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            boxes: Vec::new(),
            text: String::new(),
            processing_time_ms: 0,
            image_size: (width, height),
            layout: None,
        }
    }

    /// Sort boxes by reading order (top-to-bottom, left-to-right).
    pub fn sort_by_reading_order(&mut self) {
        self.boxes.sort_by(|a, b| {
            let (_, ay, _, _) = a.rect();
            let (_, by, _, _) = b.rect();

            // Group by approximate vertical position (within 20 pixels)
            let row_a = (ay / 20.0) as i32;
            let row_b = (by / 20.0) as i32;

            if row_a != row_b {
                row_a.cmp(&row_b)
            } else {
                // Same row, sort by x
                let (ax, _, _, _) = a.rect();
                let (bx, _, _, _) = b.rect();
                ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Rebuild full text
        self.text = self
            .boxes
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
    }
}
