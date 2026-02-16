//! Invoice field extraction module.

mod parser;
pub mod rules;

pub use parser::{HybridInvoiceParser, InvoiceParser, ExtractionResult};

use crate::error::ExtractionError;
use crate::models::invoice::Invoice;
use crate::ocr::OcrResult;

/// Result type for extraction operations.
pub type Result<T> = std::result::Result<T, ExtractionError>;

/// Trait for invoice field extractors.
pub trait InvoiceExtractor {
    /// Extract invoice data from OCR result.
    fn extract(&self, ocr_result: &OcrResult) -> Result<Invoice>;

    /// Extract invoice data from plain text.
    fn extract_from_text(&self, text: &str) -> Result<Invoice>;
}
