//! Error types for the incr-core library.

use thiserror::Error;

/// Main error type for the incr library.
#[derive(Error, Debug)]
pub enum IncrError {
    /// PDF processing error.
    #[error("PDF error: {0}")]
    Pdf(#[from] PdfError),

    /// OCR processing error.
    #[error("OCR error: {0}")]
    Ocr(#[from] OcrError),

    /// Invoice extraction error.
    #[error("extraction error: {0}")]
    Extraction(#[from] ExtractionError),

    /// Inference error from the inference layer.
    #[error("inference error: {0}")]
    Inference(#[from] incr_inference::InferenceError),

    /// Image processing error.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Errors related to PDF processing.
#[derive(Error, Debug)]
pub enum PdfError {
    /// Failed to open/parse the PDF file.
    #[error("failed to parse PDF: {0}")]
    Parse(String),

    /// Failed to extract text from PDF.
    #[error("failed to extract text: {0}")]
    TextExtraction(String),

    /// Failed to extract images from PDF.
    #[error("failed to extract images: {0}")]
    ImageExtraction(String),

    /// The PDF is encrypted and cannot be processed.
    #[error("PDF is encrypted")]
    Encrypted,

    /// The PDF is empty or has no pages.
    #[error("PDF has no pages")]
    NoPages,

    /// Invalid page number requested.
    #[error("invalid page number: {0}")]
    InvalidPage(u32),
}

/// Errors related to OCR processing.
#[derive(Error, Debug)]
pub enum OcrError {
    /// Failed to load OCR models.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Text detection failed.
    #[error("text detection failed: {0}")]
    Detection(String),

    /// Text recognition failed.
    #[error("text recognition failed: {0}")]
    Recognition(String),

    /// Image preprocessing failed.
    #[error("preprocessing failed: {0}")]
    Preprocessing(String),

    /// Invalid image format or dimensions.
    #[error("invalid image: {0}")]
    InvalidImage(String),
}

/// Errors related to invoice field extraction.
#[derive(Error, Debug)]
pub enum ExtractionError {
    /// Required field is missing.
    #[error("missing required field: {0}")]
    MissingField(String),

    /// Field validation failed.
    #[error("validation failed for {field}: {reason}")]
    Validation { field: String, reason: String },

    /// Failed to parse a value.
    #[error("failed to parse {field}: {value}")]
    Parse { field: String, value: String },

    /// No invoice data could be extracted.
    #[error("no invoice data found")]
    NoData,
}

/// Result type for the incr library.
pub type Result<T> = std::result::Result<T, IncrError>;
