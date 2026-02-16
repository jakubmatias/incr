//! Rule-based field extractors for Polish invoices.

pub mod nip;
pub mod regon;
pub mod dates;
pub mod amounts;
pub mod vat;
pub mod iban;
pub mod patterns;

pub use nip::{extract_nip, validate_nip, format_nip, NipExtractor};
pub use regon::{extract_regon, validate_regon, RegonExtractor};
pub use dates::{extract_dates, DateExtractor};
pub use amounts::{extract_amounts, parse_polish_amount, format_polish_amount, AmountExtractor};
pub use vat::{extract_vat_rates, VatExtractor};
pub use iban::{extract_iban, validate_iban, format_iban, IbanExtractor};
pub use patterns::*;


/// Trait for field extractors.
pub trait FieldExtractor {
    /// The type of value this extractor produces.
    type Output;

    /// Extract the field from text.
    fn extract(&self, text: &str) -> Option<Self::Output>;

    /// Extract all occurrences of the field.
    fn extract_all(&self, text: &str) -> Vec<Self::Output>;
}

/// Extraction context with confidence scores.
#[derive(Debug, Clone)]
pub struct ExtractionMatch<T> {
    /// Extracted value.
    pub value: T,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Position in source text.
    pub position: Option<(usize, usize)>,
    /// Source text that was matched.
    pub source: String,
}

impl<T> ExtractionMatch<T> {
    pub fn new(value: T, confidence: f32, source: impl Into<String>) -> Self {
        Self {
            value,
            confidence,
            position: None,
            source: source.into(),
        }
    }

    pub fn with_position(mut self, start: usize, end: usize) -> Self {
        self.position = Some((start, end));
        self
    }
}
