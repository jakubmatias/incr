//! Image preprocessing for OCR.

use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use ndarray::Array4;
use tracing::debug;

use crate::error::OcrError;

/// Image preprocessor for OCR pipeline.
pub struct ImagePreprocessor {
    /// Maximum image dimension.
    max_size: u32,
    /// Target size for detection model.
    det_target_size: u32,
    /// Target height for recognition model.
    rec_target_height: u32,
    /// Target width for recognition model.
    rec_target_width: u32,
}

impl ImagePreprocessor {
    /// Create a new preprocessor with default settings.
    pub fn new() -> Self {
        Self {
            max_size: 2048,
            det_target_size: 960,
            rec_target_height: 48,
            rec_target_width: 320,
        }
    }

    /// Set maximum image dimension.
    pub fn with_max_size(mut self, size: u32) -> Self {
        self.max_size = size;
        self
    }

    /// Preprocess image for text detection model.
    ///
    /// Returns (preprocessed tensor, scale_x, scale_y, original_size).
    pub fn preprocess_for_detection(
        &self,
        image: &DynamicImage,
    ) -> Result<(Array4<f32>, f32, f32, (u32, u32)), OcrError> {
        let (orig_width, orig_height) = image.dimensions();
        debug!("Original image size: {}x{}", orig_width, orig_height);

        // Resize to fit within max size while maintaining aspect ratio
        let (new_width, new_height) = self.calculate_resize_dimensions(
            orig_width,
            orig_height,
            self.det_target_size,
        );

        let resized = image.resize_exact(
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        );

        // Pad to be divisible by 32 (required by PaddleOCR)
        let pad_width = ((new_width + 31) / 32) * 32;
        let pad_height = ((new_height + 31) / 32) * 32;

        let rgb = resized.to_rgb8();

        // Normalize to [-0.5, 0.5] range and create NCHW tensor
        let mut tensor = Array4::<f32>::zeros((1, 3, pad_height as usize, pad_width as usize));

        // PaddleOCR normalization: (x / 255 - mean) / std
        // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (value - mean[c]) / std[c];
                }
            }
        }

        let scale_x = new_width as f32 / orig_width as f32;
        let scale_y = new_height as f32 / orig_height as f32;

        Ok((tensor, scale_x, scale_y, (orig_width, orig_height)))
    }

    /// Preprocess a cropped text region for recognition.
    pub fn preprocess_for_recognition(
        &self,
        image: &DynamicImage,
    ) -> Result<Array4<f32>, OcrError> {
        let (width, height) = image.dimensions();

        // Calculate target width maintaining aspect ratio
        let aspect_ratio = width as f32 / height as f32;
        let target_width = (self.rec_target_height as f32 * aspect_ratio) as u32;
        let target_width = target_width.min(self.rec_target_width).max(1);

        let resized = image.resize_exact(
            target_width,
            self.rec_target_height,
            image::imageops::FilterType::Lanczos3,
        );

        let rgb = resized.to_rgb8();

        // Create tensor with padding
        let mut tensor = Array4::<f32>::zeros((
            1,
            3,
            self.rec_target_height as usize,
            self.rec_target_width as usize,
        ));

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];

        for y in 0..self.rec_target_height {
            for x in 0..target_width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (value - mean[c]) / std[c];
                }
            }
        }

        Ok(tensor)
    }

    /// Preprocess for angle classification.
    pub fn preprocess_for_classification(
        &self,
        image: &DynamicImage,
    ) -> Result<Array4<f32>, OcrError> {
        // Angle classifier expects 192x48 input
        let target_width = 192u32;
        let target_height = 48u32;

        let resized = image.resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Lanczos3,
        );

        let rgb = resized.to_rgb8();

        let mut tensor = Array4::<f32>::zeros((1, 3, target_height as usize, target_width as usize));

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];

        for y in 0..target_height {
            for x in 0..target_width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let value = pixel[c] as f32 / 255.0;
                    tensor[[0, c, y as usize, x as usize]] = (value - mean[c]) / std[c];
                }
            }
        }

        Ok(tensor)
    }

    /// Crop text region from image using quadrilateral coordinates.
    pub fn crop_text_region(
        &self,
        image: &DynamicImage,
        bbox: &[f32; 8],
    ) -> Result<DynamicImage, OcrError> {
        // Get axis-aligned bounding box
        let xs = [bbox[0], bbox[2], bbox[4], bbox[6]];
        let ys = [bbox[1], bbox[3], bbox[5], bbox[7]];

        let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min).max(0.0) as u32;
        let max_x = xs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            .min(image.width() as f32) as u32;
        let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min).max(0.0) as u32;
        let max_y = ys
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
            .min(image.height() as f32) as u32;

        let width = max_x.saturating_sub(min_x).max(1);
        let height = max_y.saturating_sub(min_y).max(1);

        let cropped = image.crop_imm(min_x, min_y, width, height);
        Ok(cropped)
    }

    /// Apply basic image enhancement for better OCR.
    pub fn enhance(&self, image: &DynamicImage) -> DynamicImage {
        // Convert to grayscale for processing
        let gray = image.to_luma8();

        // Apply adaptive thresholding for better contrast
        let enhanced = self.adaptive_threshold(&gray, 15, 5);

        DynamicImage::ImageLuma8(enhanced)
    }

    fn calculate_resize_dimensions(
        &self,
        width: u32,
        height: u32,
        target_size: u32,
    ) -> (u32, u32) {
        let max_dim = width.max(height);

        if max_dim <= target_size {
            return (width, height);
        }

        let scale = target_size as f32 / max_dim as f32;
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        (new_width.max(1), new_height.max(1))
    }

    fn adaptive_threshold(&self, image: &GrayImage, block_size: u32, c: i32) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut result = GrayImage::new(width, height);

        let half_block = block_size / 2;

        for y in 0..height {
            for x in 0..width {
                // Calculate local mean
                let mut sum = 0u32;
                let mut count = 0u32;

                let y_start = y.saturating_sub(half_block);
                let y_end = (y + half_block + 1).min(height);
                let x_start = x.saturating_sub(half_block);
                let x_end = (x + half_block + 1).min(width);

                for ly in y_start..y_end {
                    for lx in x_start..x_end {
                        sum += image.get_pixel(lx, ly)[0] as u32;
                        count += 1;
                    }
                }

                let mean = (sum / count) as i32;
                let threshold = mean - c;
                let pixel_value = image.get_pixel(x, y)[0] as i32;

                let output = if pixel_value > threshold { 255 } else { 0 };
                result.put_pixel(x, y, Luma([output]));
            }
        }

        result
    }
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_dimensions() {
        let preprocessor = ImagePreprocessor::new();

        // Image smaller than target
        let (w, h) = preprocessor.calculate_resize_dimensions(500, 300, 960);
        assert_eq!((w, h), (500, 300));

        // Image larger than target
        let (w, h) = preprocessor.calculate_resize_dimensions(1920, 1080, 960);
        assert_eq!(w, 960);
        assert!(h < 960);
    }
}
