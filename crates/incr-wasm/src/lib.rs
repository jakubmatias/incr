//! WASM bindings for Polish invoice OCR.
//!
//! This crate provides WebAssembly bindings for use in browsers and Node.js.

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;

use incr_core::models::invoice::{Invoice, InvoiceType, VatRate};
use incr_core::invoice::{HybridInvoiceParser, InvoiceParser};

/// Initialize panic hook for better error messages in console.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Version information.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Extract invoice data from text.
///
/// Takes invoice text (from OCR or PDF extraction) and returns structured invoice data.
#[wasm_bindgen]
pub fn extract_invoice_from_text(text: &str) -> Result<JsValue, JsValue> {
    let parser = HybridInvoiceParser::new()
        .with_nip_validation(true)
        .with_regon_validation(true)
        .with_iban_validation(true);

    let result = parser
        .parse(text)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&result.invoice)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Validate a Polish NIP (tax identification number).
#[wasm_bindgen]
pub fn validate_nip(nip: &str) -> bool {
    incr_core::invoice::rules::validate_nip(nip)
}

/// Validate a Polish REGON (statistical number).
#[wasm_bindgen]
pub fn validate_regon(regon: &str) -> bool {
    incr_core::invoice::rules::validate_regon(regon)
}

/// Validate an IBAN (international bank account number).
#[wasm_bindgen]
pub fn validate_iban(iban: &str) -> bool {
    incr_core::invoice::rules::validate_iban(iban)
}

/// Parse a Polish-formatted amount (e.g., "1 234,56").
#[wasm_bindgen]
pub fn parse_polish_amount(amount: &str) -> Option<f64> {
    incr_core::invoice::rules::parse_polish_amount(amount)
        .map(|d| d.to_string().parse().unwrap_or(0.0))
}

/// Invoice extractor class for browser use.
#[wasm_bindgen]
pub struct InvoiceExtractor {
    parser: HybridInvoiceParser,
}

#[wasm_bindgen]
impl InvoiceExtractor {
    /// Create a new invoice extractor.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            parser: HybridInvoiceParser::new(),
        }
    }

    /// Configure NIP validation.
    #[wasm_bindgen]
    pub fn set_validate_nip(&mut self, validate: bool) {
        self.parser = HybridInvoiceParser::new()
            .with_nip_validation(validate);
    }

    /// Extract invoice from text.
    #[wasm_bindgen]
    pub fn extract(&self, text: &str) -> Result<JsValue, JsValue> {
        let result = self.parser
            .parse(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&result.invoice)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get extraction result with metadata.
    #[wasm_bindgen]
    pub fn extract_with_metadata(&self, text: &str) -> Result<JsValue, JsValue> {
        let result = self.parser
            .parse(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        #[derive(serde::Serialize)]
        struct ExtractResult {
            invoice: Invoice,
            raw_text: String,
            warnings: Vec<String>,
            processing_time_ms: u64,
        }

        let output = ExtractResult {
            invoice: result.invoice,
            raw_text: result.raw_text,
            warnings: result.warnings,
            processing_time_ms: result.processing_time_ms,
        };

        serde_wasm_bindgen::to_value(&output)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for InvoiceExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// OCR result from browser-side processing.
#[wasm_bindgen]
pub struct OcrResultJs {
    boxes: Vec<TextBoxJs>,
    text: String,
}

#[wasm_bindgen]
impl OcrResultJs {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            boxes: Vec::new(),
            text: String::new(),
        }
    }

    /// Add a text box to the result.
    #[wasm_bindgen]
    pub fn add_box(
        &mut self,
        text: &str,
        x1: f32, y1: f32,
        x2: f32, y2: f32,
        x3: f32, y3: f32,
        x4: f32, y4: f32,
        confidence: f32,
    ) {
        self.boxes.push(TextBoxJs {
            text: text.to_string(),
            bbox: [x1, y1, x2, y2, x3, y3, x4, y4],
            confidence,
        });
    }

    /// Set the full text.
    #[wasm_bindgen]
    pub fn set_text(&mut self, text: &str) {
        self.text = text.to_string();
    }

    /// Get the full text.
    #[wasm_bindgen]
    pub fn get_text(&self) -> String {
        if self.text.is_empty() {
            self.boxes.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("\n")
        } else {
            self.text.clone()
        }
    }

    /// Extract invoice from this OCR result.
    #[wasm_bindgen]
    pub fn extract_invoice(&self) -> Result<JsValue, JsValue> {
        let text = self.get_text();
        extract_invoice_from_text(&text)
    }
}

impl Default for OcrResultJs {
    fn default() -> Self {
        Self::new()
    }
}

struct TextBoxJs {
    text: String,
    bbox: [f32; 8],
    confidence: f32,
}

/// Utilities for working with Polish invoice data.
#[wasm_bindgen]
pub struct PolishInvoiceUtils;

#[wasm_bindgen]
impl PolishInvoiceUtils {
    /// Format NIP with dashes (XXX-XXX-XX-XX).
    #[wasm_bindgen]
    pub fn format_nip(nip: &str) -> String {
        let digits: String = nip.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() != 10 {
            return nip.to_string();
        }
        format!("{}-{}-{}-{}", &digits[0..3], &digits[3..6], &digits[6..8], &digits[8..10])
    }

    /// Format IBAN in groups of 4.
    #[wasm_bindgen]
    pub fn format_iban(iban: &str) -> String {
        incr_core::invoice::rules::format_iban(iban)
    }

    /// Format amount in Polish style (1 234,56).
    #[wasm_bindgen]
    pub fn format_amount(amount: f64) -> String {
        let s = format!("{:.2}", amount);
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 2 {
            return s;
        }

        let integer_part = parts[0];
        let decimal_part = parts[1];

        let chars: Vec<char> = integer_part.chars().collect();
        let mut formatted = String::new();
        for (i, c) in chars.iter().enumerate() {
            if i > 0 && (chars.len() - i) % 3 == 0 {
                formatted.push(' ');
            }
            formatted.push(*c);
        }

        format!("{},{}", formatted, decimal_part)
    }

    /// Get VAT rate as decimal (e.g., 0.23 for 23%).
    #[wasm_bindgen]
    pub fn vat_rate_as_decimal(rate: &str) -> f64 {
        VatRate::from_str(rate)
            .map(|r| r.as_decimal().to_string().parse().unwrap_or(0.0))
            .unwrap_or(0.0)
    }

    /// Parse a Polish date string.
    #[wasm_bindgen]
    pub fn parse_date(date_str: &str) -> Option<String> {
        use incr_core::invoice::rules::{DateExtractor, FieldExtractor};

        DateExtractor::new()
            .extract(date_str)
            .map(|m| m.value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_validate_nip() {
        assert!(validate_nip("5261040828"));
        assert!(!validate_nip("1234567890"));
    }

    #[wasm_bindgen_test]
    fn test_parse_polish_amount() {
        let amount = parse_polish_amount("1 234,56");
        assert!(amount.is_some());
        assert!((amount.unwrap() - 1234.56).abs() < 0.01);
    }

    #[wasm_bindgen_test]
    fn test_format_nip() {
        assert_eq!(PolishInvoiceUtils::format_nip("5261040828"), "526-104-08-28");
    }

    #[wasm_bindgen_test]
    fn test_format_amount() {
        assert_eq!(PolishInvoiceUtils::format_amount(1234.56), "1 234,56");
    }
}
