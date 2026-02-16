//! Text detection using PaddleOCR detection model.


use image::DynamicImage;
use tracing::debug;

use crate::error::OcrError;
use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

use super::preprocessing::ImagePreprocessor;

/// Text detector using PaddleOCR DB model.
pub struct TextDetector<B: InferenceBackend> {
    backend: B,
    preprocessor: ImagePreprocessor,
    threshold: f32,
    box_threshold: f32,
    unclip_ratio: f32,
}

/// Detection result before recognition.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Detected bounding boxes (quadrilaterals).
    pub boxes: Vec<[f32; 8]>,
    /// Detection confidence scores.
    pub scores: Vec<f32>,
    /// Original image size.
    pub image_size: (u32, u32),
}

impl<B: InferenceBackend> TextDetector<B> {
    /// Create a new text detector with the given backend.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            preprocessor: ImagePreprocessor::new(),
            threshold: 0.3,
            box_threshold: 0.6,
            unclip_ratio: 1.5,
        }
    }

    /// Set detection threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set box threshold.
    pub fn with_box_threshold(mut self, threshold: f32) -> Self {
        self.box_threshold = threshold;
        self
    }

    /// Detect text regions in an image.
    pub fn detect(&self, image: &DynamicImage) -> Result<DetectionResult, OcrError> {
        // Preprocess image
        let (tensor, scale_x, scale_y, orig_size) = self
            .preprocessor
            .preprocess_for_detection(image)
            .map_err(|e| OcrError::Preprocessing(e.to_string()))?;

        debug!(
            "Detection input shape: {:?}, scales: ({}, {})",
            tensor.shape(),
            scale_x,
            scale_y
        );

        // Convert to InputTensor
        let input = InputTensor::Float32(tensor.into_dyn());

        // Run inference
        let outputs = self
            .backend
            .run(&[("x", input)])
            .map_err(|e| OcrError::Detection(e.to_string()))?;

        // Extract output
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::Detection("No output from model".to_string()))?
            .1;

        let output_arr = match output {
            OutputTensor::Float32(arr) => arr,
            _ => return Err(OcrError::Detection("Unexpected output type".to_string())),
        };

        debug!("Detection output shape: {:?}", output_arr.shape());

        // Post-process to get bounding boxes
        let (boxes, scores) = self.post_process(&output_arr, scale_x, scale_y, orig_size)?;

        debug!("Detected {} text regions", boxes.len());

        Ok(DetectionResult {
            boxes,
            scores,
            image_size: orig_size,
        })
    }

    fn post_process(
        &self,
        output: &ndarray::ArrayD<f32>,
        scale_x: f32,
        scale_y: f32,
        orig_size: (u32, u32),
    ) -> Result<(Vec<[f32; 8]>, Vec<f32>), OcrError> {
        // Output shape is [1, 1, H, W] - probability map
        let shape = output.shape();
        if shape.len() < 4 {
            return Err(OcrError::Detection(format!(
                "Invalid output shape: {:?}",
                shape
            )));
        }

        let height = shape[2];
        let width = shape[3];

        // Binarize the probability map
        let mut binary = vec![vec![false; width]; height];
        let mut prob_map = vec![vec![0.0f32; width]; height];

        for y in 0..height {
            for x in 0..width {
                let prob = output[[0, 0, y, x]];
                prob_map[y][x] = prob;
                binary[y][x] = prob > self.threshold;
            }
        }

        // Find connected components (simplified contour detection)
        let contours = self.find_contours(&binary, width, height);

        let mut boxes = Vec::new();
        let mut scores = Vec::new();

        for contour in contours {
            if contour.len() < 4 {
                continue;
            }

            // Calculate bounding box and score
            let (bbox, score) = self.get_box_from_contour(&contour, &prob_map, width, height);

            if score < self.box_threshold {
                continue;
            }

            // Scale back to original image coordinates
            let scaled_bbox = [
                bbox[0] / scale_x,
                bbox[1] / scale_y,
                bbox[2] / scale_x,
                bbox[3] / scale_y,
                bbox[4] / scale_x,
                bbox[5] / scale_y,
                bbox[6] / scale_x,
                bbox[7] / scale_y,
            ];

            // Clip to image bounds
            let clipped_bbox = self.clip_bbox(&scaled_bbox, orig_size.0, orig_size.1);

            boxes.push(clipped_bbox);
            scores.push(score);
        }

        Ok((boxes, scores))
    }

    fn find_contours(
        &self,
        binary: &[Vec<bool>],
        width: usize,
        height: usize,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut visited = vec![vec![false; width]; height];
        let mut contours = Vec::new();

        for y in 0..height {
            for x in 0..width {
                if binary[y][x] && !visited[y][x] {
                    let contour = self.flood_fill(binary, &mut visited, x, y, width, height);
                    if contour.len() >= 10 {
                        contours.push(contour);
                    }
                }
            }
        }

        contours
    }

    fn flood_fill(
        &self,
        binary: &[Vec<bool>],
        visited: &mut Vec<Vec<bool>>,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
    ) -> Vec<(usize, usize)> {
        let mut contour = Vec::new();
        let mut stack = vec![(start_x, start_y)];

        while let Some((x, y)) = stack.pop() {
            if x >= width || y >= height || visited[y][x] || !binary[y][x] {
                continue;
            }

            visited[y][x] = true;
            contour.push((x, y));

            // 4-connected neighbors
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x + 1 < width {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y + 1 < height {
                stack.push((x, y + 1));
            }
        }

        contour
    }

    fn get_box_from_contour(
        &self,
        contour: &[(usize, usize)],
        prob_map: &[Vec<f32>],
        _width: usize,
        _height: usize,
    ) -> ([f32; 8], f32) {
        // Find min/max coordinates
        let mut min_x = usize::MAX;
        let mut max_x = 0;
        let mut min_y = usize::MAX;
        let mut max_y = 0;
        let mut score_sum = 0.0f32;

        for &(x, y) in contour {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            score_sum += prob_map[y][x];
        }

        let avg_score = score_sum / contour.len() as f32;

        // Apply unclip ratio to expand the box slightly
        let w = (max_x - min_x) as f32;
        let h = (max_y - min_y) as f32;
        let expand_x = w * (self.unclip_ratio - 1.0) / 2.0;
        let expand_y = h * (self.unclip_ratio - 1.0) / 2.0;

        let x1 = (min_x as f32 - expand_x).max(0.0);
        let y1 = (min_y as f32 - expand_y).max(0.0);
        let x2 = max_x as f32 + expand_x;
        let y2 = max_y as f32 + expand_y;

        // Return as quadrilateral (4 corners: TL, TR, BR, BL)
        let bbox = [x1, y1, x2, y1, x2, y2, x1, y2];

        (bbox, avg_score)
    }

    fn clip_bbox(&self, bbox: &[f32; 8], width: u32, height: u32) -> [f32; 8] {
        let w = width as f32;
        let h = height as f32;

        [
            bbox[0].clamp(0.0, w),
            bbox[1].clamp(0.0, h),
            bbox[2].clamp(0.0, w),
            bbox[3].clamp(0.0, h),
            bbox[4].clamp(0.0, w),
            bbox[5].clamp(0.0, h),
            bbox[6].clamp(0.0, w),
            bbox[7].clamp(0.0, h),
        ]
    }
}
