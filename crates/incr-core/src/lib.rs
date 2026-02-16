//! Core library for Polish invoice OCR processing.
//!
//! This crate provides:
//! - PDF processing (text and image extraction)
//! - OCR pipeline using PaddleOCR models
//! - Polish invoice field extraction (NIP, REGON, dates, amounts, VAT)
//! - Invoice data models compatible with KSeF FA(3)

pub mod error;
pub mod models;
pub mod pdf;
pub mod ocr;
pub mod invoice;

pub use error::{IncrError, Result};
pub use models::invoice::{Invoice, InvoiceHeader, InvoiceSummary, Party, LineItem, VatRate};
pub use pdf::{PdfProcessor, PdfContent, PdfType};
pub use ocr::{OcrEngine, OcrResult, TextBox};
#[cfg(feature = "native")]
pub use ocr::{create_engine_from_dir, create_engine_from_embedded};
pub use invoice::{InvoiceParser, InvoiceExtractor, ExtractionResult};

/// Re-export inference types.
pub use incr_inference::{InferenceBackend, InputTensor, OutputTensor};

#[cfg(feature = "native")]
pub use incr_inference::OrtBackend;

#[cfg(feature = "wasm")]
pub use incr_inference::TractBackend;
