//! Layout analysis using PP-Structure layout detection model.
//!
//! Detects document regions: text, title, list, table, figure.

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;
use tracing::debug;

use crate::error::OcrError;
use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

/// Layout region types detected by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutType {
    /// Text paragraph region.
    Text,
    /// Title or heading.
    Title,
    /// Bulleted or numbered list.
    List,
    /// Table region.
    Table,
    /// Figure or image region.
    Figure,
    /// Unknown region type.
    Unknown,
}

impl LayoutType {
    /// Create from class index (PubLayNet model).
    pub fn from_publaynet_class(class: usize) -> Self {
        match class {
            0 => LayoutType::Text,
            1 => LayoutType::Title,
            2 => LayoutType::List,
            3 => LayoutType::Table,
            4 => LayoutType::Figure,
            _ => LayoutType::Unknown,
        }
    }

    /// Create from class index (CDLA model - Chinese Document Layout Analysis).
    pub fn from_cdla_class(class: usize) -> Self {
        // CDLA classes: text, figure, figure_caption, table, table_caption, header, footer, reference, equation
        match class {
            0 => LayoutType::Text,
            1 => LayoutType::Figure,
            2 => LayoutType::Text, // figure_caption -> text
            3 => LayoutType::Table,
            4 => LayoutType::Text, // table_caption -> text
            5 => LayoutType::Title, // header -> title
            6 => LayoutType::Text, // footer -> text
            7 => LayoutType::Text, // reference -> text
            8 => LayoutType::Text, // equation -> text
            _ => LayoutType::Unknown,
        }
    }

    /// Check if this is a table region.
    pub fn is_table(&self) -> bool {
        matches!(self, LayoutType::Table)
    }

    /// Check if this is a text-containing region.
    pub fn is_text(&self) -> bool {
        matches!(self, LayoutType::Text | LayoutType::Title | LayoutType::List)
    }
}

/// A detected layout region.
#[derive(Debug, Clone)]
pub struct LayoutRegion {
    /// Region type.
    pub region_type: LayoutType,
    /// Bounding box (x1, y1, x2, y2).
    pub bbox: [f32; 4],
    /// Detection confidence score.
    pub confidence: f32,
}

impl LayoutRegion {
    /// Get the width of the region.
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    /// Get the height of the region.
    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    /// Get the area of the region.
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Check if a point is inside this region.
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.bbox[0] && x <= self.bbox[2] && y >= self.bbox[1] && y <= self.bbox[3]
    }

    /// Check if this region overlaps with another.
    pub fn overlaps(&self, other: &LayoutRegion) -> bool {
        self.bbox[0] < other.bbox[2]
            && self.bbox[2] > other.bbox[0]
            && self.bbox[1] < other.bbox[3]
            && self.bbox[3] > other.bbox[1]
    }

    /// Calculate IoU (Intersection over Union) with another region.
    pub fn iou(&self, other: &LayoutRegion) -> f32 {
        let x1 = self.bbox[0].max(other.bbox[0]);
        let y1 = self.bbox[1].max(other.bbox[1]);
        let x2 = self.bbox[2].min(other.bbox[2]);
        let y2 = self.bbox[3].min(other.bbox[3]);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// Layout detection result.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Detected layout regions.
    pub regions: Vec<LayoutRegion>,
    /// Image dimensions.
    pub image_size: (u32, u32),
}

impl LayoutResult {
    /// Get all table regions.
    pub fn tables(&self) -> Vec<&LayoutRegion> {
        self.regions
            .iter()
            .filter(|r| r.region_type.is_table())
            .collect()
    }

    /// Get all text regions (text, title, list).
    pub fn text_regions(&self) -> Vec<&LayoutRegion> {
        self.regions
            .iter()
            .filter(|r| r.region_type.is_text())
            .collect()
    }

    /// Get regions sorted by reading order.
    pub fn sorted_by_reading_order(&self) -> Vec<&LayoutRegion> {
        let mut regions: Vec<&LayoutRegion> = self.regions.iter().collect();
        regions.sort_by(|a, b| {
            // Group by approximate vertical position
            let row_a = (a.bbox[1] / 50.0) as i32;
            let row_b = (b.bbox[1] / 50.0) as i32;

            if row_a != row_b {
                row_a.cmp(&row_b)
            } else {
                a.bbox[0]
                    .partial_cmp(&b.bbox[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        regions
    }
}

/// Layout detector using PP-Structure layout model.
pub struct LayoutDetector<B: InferenceBackend> {
    backend: B,
    input_size: (u32, u32),
    confidence_threshold: f32,
    nms_threshold: f32,
    model_type: LayoutModelType,
}

/// Type of layout model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutModelType {
    /// PubLayNet model (English documents).
    PubLayNet,
    /// CDLA model (Chinese documents).
    Cdla,
}

impl<B: InferenceBackend> LayoutDetector<B> {
    /// Create a new layout detector.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            input_size: (800, 608), // PP-PicoDet default
            confidence_threshold: 0.5,
            nms_threshold: 0.5,
            model_type: LayoutModelType::PubLayNet,
        }
    }

    /// Set the model type for class mapping.
    pub fn with_model_type(mut self, model_type: LayoutModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set input size.
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_size = (width, height);
        self
    }

    /// Set confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set NMS threshold.
    pub fn with_nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    /// Detect layout regions in an image.
    pub fn detect(&self, image: &DynamicImage) -> Result<LayoutResult, OcrError> {
        let (orig_width, orig_height) = image.dimensions();

        // Preprocess: resize and normalize
        let (tensor, scale_x, scale_y) = self.preprocess(image)?;

        debug!(
            "Layout detection input: {}x{}, scales: ({:.3}, {:.3})",
            self.input_size.0, self.input_size.1, scale_x, scale_y
        );

        // Create scale factor tensor for PP-PicoDet
        let scale_factor = Array3::from_shape_vec(
            (1, 2, 1),
            vec![scale_y, scale_x],
        )
        .map_err(|e| OcrError::Detection(format!("Failed to create scale tensor: {}", e)))?;

        // Run inference
        let inputs = vec![
            ("image", InputTensor::Float32(tensor.into_dyn())),
            ("scale_factor", InputTensor::Float32(scale_factor.into_dyn())),
        ];

        let outputs = self
            .backend
            .run(&inputs)
            .map_err(|e| OcrError::Detection(format!("Layout inference failed: {}", e)))?;

        // Parse outputs
        let regions = self.post_process(&outputs, scale_x, scale_y, orig_width, orig_height)?;

        debug!("Detected {} layout regions", regions.len());

        Ok(LayoutResult {
            regions,
            image_size: (orig_width, orig_height),
        })
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<(Array3<f32>, f32, f32), OcrError> {
        let (orig_w, orig_h) = image.dimensions();
        let (target_w, target_h) = self.input_size;

        // Resize image
        let resized = image.resize_exact(
            target_w,
            target_h,
            image::imageops::FilterType::Triangle,
        );

        let scale_x = target_w as f32 / orig_w as f32;
        let scale_y = target_h as f32 / orig_h as f32;

        // Convert to CHW format with normalization
        // ImageNet mean/std normalization
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let rgb = resized.to_rgb8();
        let mut tensor = Array3::<f32>::zeros((3, target_h as usize, target_w as usize));

        for y in 0..target_h as usize {
            for x in 0..target_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                tensor[[0, y, x]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
                tensor[[1, y, x]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
                tensor[[2, y, x]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        // Add batch dimension by reshaping to 4D when used
        Ok((tensor, scale_x, scale_y))
    }

    fn post_process(
        &self,
        outputs: &[(String, OutputTensor)],
        scale_x: f32,
        scale_y: f32,
        orig_width: u32,
        orig_height: u32,
    ) -> Result<Vec<LayoutRegion>, OcrError> {
        // PP-PicoDet outputs: [N, 6] where each row is [class_id, score, x1, y1, x2, y2]
        // The coordinates are already scaled by scale_factor

        let output = outputs
            .iter()
            .find(|(name, _)| name.contains("bbox") || name.contains("output"))
            .or_else(|| outputs.first())
            .map(|(_, tensor)| tensor)
            .ok_or_else(|| OcrError::Detection("No output tensor found".to_string()))?;

        let arr = match output {
            OutputTensor::Float32(arr) => arr,
            _ => {
                return Err(OcrError::Detection(
                    "Unexpected output tensor type".to_string(),
                ))
            }
        };

        let shape = arr.shape();
        debug!("Layout output shape: {:?}", shape);

        let mut regions = Vec::new();

        // Handle different output formats
        if shape.len() == 2 && shape[1] == 6 {
            // Standard PicoDet format: [N, 6]
            let num_detections = shape[0];

            for i in 0..num_detections {
                let class_id = arr[[i, 0]] as usize;
                let score = arr[[i, 1]];
                let x1 = arr[[i, 2]] / scale_x;
                let y1 = arr[[i, 3]] / scale_y;
                let x2 = arr[[i, 4]] / scale_x;
                let y2 = arr[[i, 5]] / scale_y;

                if score < self.confidence_threshold {
                    continue;
                }

                let region_type = match self.model_type {
                    LayoutModelType::PubLayNet => LayoutType::from_publaynet_class(class_id),
                    LayoutModelType::Cdla => LayoutType::from_cdla_class(class_id),
                };

                regions.push(LayoutRegion {
                    region_type,
                    bbox: [
                        x1.clamp(0.0, orig_width as f32),
                        y1.clamp(0.0, orig_height as f32),
                        x2.clamp(0.0, orig_width as f32),
                        y2.clamp(0.0, orig_height as f32),
                    ],
                    confidence: score,
                });
            }
        } else if shape.len() == 3 && shape[2] == 6 {
            // Batched format: [B, N, 6]
            let num_detections = shape[1];

            for i in 0..num_detections {
                let class_id = arr[[0, i, 0]] as usize;
                let score = arr[[0, i, 1]];
                let x1 = arr[[0, i, 2]] / scale_x;
                let y1 = arr[[0, i, 3]] / scale_y;
                let x2 = arr[[0, i, 4]] / scale_x;
                let y2 = arr[[0, i, 5]] / scale_y;

                if score < self.confidence_threshold {
                    continue;
                }

                let region_type = match self.model_type {
                    LayoutModelType::PubLayNet => LayoutType::from_publaynet_class(class_id),
                    LayoutModelType::Cdla => LayoutType::from_cdla_class(class_id),
                };

                regions.push(LayoutRegion {
                    region_type,
                    bbox: [
                        x1.clamp(0.0, orig_width as f32),
                        y1.clamp(0.0, orig_height as f32),
                        x2.clamp(0.0, orig_width as f32),
                        y2.clamp(0.0, orig_height as f32),
                    ],
                    confidence: score,
                });
            }
        }

        // Apply NMS
        let regions = self.nms(regions);

        Ok(regions)
    }

    fn nms(&self, mut regions: Vec<LayoutRegion>) -> Vec<LayoutRegion> {
        // Sort by confidence descending
        regions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = Vec::new();

        while let Some(region) = regions.pop() {
            // Check if this region overlaps too much with any kept region of same type
            let dominated = keep.iter().any(|kept: &LayoutRegion| {
                kept.region_type == region.region_type && region.iou(kept) > self.nms_threshold
            });

            if !dominated {
                keep.push(region);
            }
        }

        keep
    }
}
