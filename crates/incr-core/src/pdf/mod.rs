//! PDF processing module.

mod extractor;

pub use extractor::{PdfExtractor, PdfContent, PdfPage, ExtractedImage};

use crate::error::PdfError;
use image::DynamicImage;

/// Type of PDF content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfType {
    /// Contains extractable text.
    Text,
    /// Contains only images (scanned document).
    Image,
    /// Contains both text and images.
    Hybrid,
    /// Empty or unreadable.
    Empty,
}

/// Result type for PDF operations.
pub type Result<T> = std::result::Result<T, PdfError>;

/// Trait for PDF processing implementations.
pub trait PdfProcessor {
    /// Load a PDF from bytes.
    fn load(&mut self, data: &[u8]) -> Result<()>;

    /// Get the number of pages in the PDF.
    fn page_count(&self) -> u32;

    /// Analyze the PDF to determine its type.
    fn analyze(&self) -> PdfType;

    /// Extract text from the entire PDF.
    fn extract_text(&self) -> Result<String>;

    /// Extract text from a specific page.
    fn extract_page_text(&self, page: u32) -> Result<String>;

    /// Render a page as an image at the specified DPI.
    fn render_page(&self, page: u32, dpi: u32) -> Result<DynamicImage>;

    /// Extract embedded images from a page.
    fn extract_images(&self, page: u32) -> Result<Vec<DynamicImage>>;
}
