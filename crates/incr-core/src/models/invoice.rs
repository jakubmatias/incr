//! Invoice data models compatible with KSeF FA(3) format.

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// A complete invoice representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invoice {
    /// Invoice header information.
    pub header: InvoiceHeader,

    /// Issuer (seller) information.
    pub issuer: Party,

    /// Receiver (buyer) information.
    pub receiver: Party,

    /// Line items on the invoice.
    pub line_items: Vec<LineItem>,

    /// Invoice summary with totals.
    pub summary: InvoiceSummary,

    /// Extraction metadata.
    pub metadata: ExtractionMetadata,
}

/// Invoice header with basic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceHeader {
    /// Invoice number/identifier.
    pub invoice_number: String,

    /// Date the invoice was issued.
    pub issue_date: NaiveDate,

    /// Date of sale/service (may differ from issue date).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sale_date: Option<NaiveDate>,

    /// Payment due date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_date: Option<NaiveDate>,

    /// Type of invoice.
    pub invoice_type: InvoiceType,

    /// Currency code (default: PLN).
    #[serde(default = "default_currency")]
    pub currency: String,

    /// Reference to corrected invoice (for correction invoices).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_of: Option<String>,
}

fn default_currency() -> String {
    "PLN".to_string()
}

/// Type of invoice document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvoiceType {
    /// Standard invoice (faktura VAT).
    Standard,
    /// Correction invoice (faktura korygująca).
    Correction,
    /// Advance payment invoice (faktura zaliczkowa).
    Advance,
    /// Final invoice (faktura końcowa).
    Final,
    /// Proforma invoice.
    Proforma,
    /// Margin invoice (faktura VAT marża).
    Margin,
}

impl Default for InvoiceType {
    fn default() -> Self {
        Self::Standard
    }
}

/// A party (seller or buyer) on the invoice.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Party {
    /// Full legal name.
    pub name: String,

    /// Polish tax identification number (NIP).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nip: Option<String>,

    /// Polish statistical number (REGON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regon: Option<String>,

    /// Full address.
    pub address: Address,

    /// Bank account number (IBAN format preferred).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank_account: Option<String>,

    /// Bank name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bank_name: Option<String>,

    /// Email address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,

    /// Phone number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phone: Option<String>,

    /// Website.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub website: Option<String>,
}

/// Address structure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Address {
    /// Street name and number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub street: Option<String>,

    /// Postal code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postal_code: Option<String>,

    /// City name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// Country (default: Polska).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,

    /// Full address as single string (when parsing fails to separate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl Address {
    /// Check if the address has any data.
    pub fn is_empty(&self) -> bool {
        self.street.is_none()
            && self.postal_code.is_none()
            && self.city.is_none()
            && self.raw.is_none()
    }

    /// Format address as a single string.
    pub fn format(&self) -> String {
        if let Some(raw) = &self.raw {
            return raw.clone();
        }

        let mut parts = Vec::new();
        if let Some(street) = &self.street {
            parts.push(street.clone());
        }
        if let (Some(postal), Some(city)) = (&self.postal_code, &self.city) {
            parts.push(format!("{} {}", postal, city));
        } else if let Some(city) = &self.city {
            parts.push(city.clone());
        }
        if let Some(country) = &self.country {
            if country != "Polska" && country != "Poland" && country != "PL" {
                parts.push(country.clone());
            }
        }
        parts.join(", ")
    }
}

/// A single line item on the invoice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    /// Sequential number on invoice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,

    /// Product/service description.
    pub description: String,

    /// Product code (PKWiU, EAN, SKU, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,

    /// Quantity.
    pub quantity: Decimal,

    /// Unit of measure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,

    /// Unit price (net, before VAT).
    pub unit_price_net: Decimal,

    /// Unit price (gross, with VAT).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit_price_gross: Option<Decimal>,

    /// Applicable VAT rate.
    pub vat_rate: VatRate,

    /// Total net amount for this line.
    pub total_net: Decimal,

    /// VAT amount for this line.
    pub vat_amount: Decimal,

    /// Total gross amount for this line.
    pub total_gross: Decimal,

    /// Discount percentage if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discount_percent: Option<Decimal>,
}

/// Polish VAT rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VatRate {
    /// Standard rate: 23%
    #[serde(rename = "23")]
    Standard23,

    /// Reduced rate: 8%
    #[serde(rename = "8")]
    Reduced8,

    /// Reduced rate: 5%
    #[serde(rename = "5")]
    Reduced5,

    /// Zero rate: 0%
    #[serde(rename = "0")]
    Zero,

    /// Exempt (zwolniony).
    #[serde(rename = "zw")]
    Exempt,

    /// Not subject to VAT (nie podlega).
    #[serde(rename = "np")]
    NotApplicable,

    /// Reverse charge (odwrotne obciążenie).
    #[serde(rename = "oo")]
    ReverseCharge,

    /// Other/custom rate.
    #[serde(untagged)]
    Other(u8),
}

impl VatRate {
    /// Get the VAT rate as a decimal multiplier (e.g., 0.23 for 23%).
    pub fn as_decimal(&self) -> Decimal {
        match self {
            VatRate::Standard23 => Decimal::new(23, 2),
            VatRate::Reduced8 => Decimal::new(8, 2),
            VatRate::Reduced5 => Decimal::new(5, 2),
            VatRate::Zero | VatRate::Exempt | VatRate::NotApplicable | VatRate::ReverseCharge => {
                Decimal::ZERO
            }
            VatRate::Other(rate) => Decimal::new(*rate as i64, 2),
        }
    }

    /// Parse VAT rate from string.
    pub fn from_str(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        let s = s.trim_end_matches('%');

        match &*s {
            "23" => Some(VatRate::Standard23),
            "8" => Some(VatRate::Reduced8),
            "5" => Some(VatRate::Reduced5),
            "0" => Some(VatRate::Zero),
            "zw" | "zw." | "zwolniony" | "zwolnione" => Some(VatRate::Exempt),
            "np" | "np." | "nie podlega" => Some(VatRate::NotApplicable),
            "oo" | "odwrotne obciążenie" => Some(VatRate::ReverseCharge),
            _ => {
                // Try parsing as a number
                s.parse::<u8>().ok().map(VatRate::Other)
            }
        }
    }

    /// Format for display.
    pub fn display(&self) -> String {
        match self {
            VatRate::Standard23 => "23%".to_string(),
            VatRate::Reduced8 => "8%".to_string(),
            VatRate::Reduced5 => "5%".to_string(),
            VatRate::Zero => "0%".to_string(),
            VatRate::Exempt => "zw.".to_string(),
            VatRate::NotApplicable => "np.".to_string(),
            VatRate::ReverseCharge => "oo".to_string(),
            VatRate::Other(rate) => format!("{}%", rate),
        }
    }
}

/// Invoice summary with totals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvoiceSummary {
    /// Total net amount (before VAT).
    pub total_net: Decimal,

    /// Total VAT amount.
    pub total_vat: Decimal,

    /// Total gross amount (after VAT).
    pub total_gross: Decimal,

    /// Breakdown of VAT by rate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vat_breakdown: Vec<VatBreakdown>,

    /// Payment method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method: Option<PaymentMethod>,

    /// Amount already paid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount_paid: Option<Decimal>,

    /// Amount remaining to be paid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount_due: Option<Decimal>,

    /// Amount in words (Polish: słownie).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub amount_in_words: Option<String>,
}

/// VAT breakdown by rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VatBreakdown {
    /// VAT rate.
    pub rate: VatRate,

    /// Net amount at this rate.
    pub net: Decimal,

    /// VAT amount at this rate.
    pub vat: Decimal,

    /// Gross amount at this rate.
    pub gross: Decimal,
}

/// Payment method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentMethod {
    /// Bank transfer (przelew).
    Transfer,
    /// Cash (gotówka).
    Cash,
    /// Credit card (karta).
    Card,
    /// Compensation/barter (kompensata).
    Compensation,
    /// Other method with description.
    Other(String),
}

impl PaymentMethod {
    /// Parse payment method from string.
    pub fn from_str(s: &str) -> Self {
        let s = s.trim().to_lowercase();

        if s.contains("przelew") || s.contains("transfer") || s.contains("bank") {
            PaymentMethod::Transfer
        } else if s.contains("gotówk") || s.contains("cash") || s.contains("gotowk") {
            PaymentMethod::Cash
        } else if s.contains("kart") || s.contains("card") {
            PaymentMethod::Card
        } else if s.contains("kompensat") || s.contains("barter") {
            PaymentMethod::Compensation
        } else {
            PaymentMethod::Other(s)
        }
    }
}

/// Metadata about the extraction process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// Overall extraction confidence (0.0 - 1.0).
    pub confidence: f32,

    /// Source document type.
    pub source_type: SourceType,

    /// Processing time in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,

    /// OCR engine used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr_engine: Option<String>,

    /// Warnings or issues encountered during extraction.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,

    /// Fields that could not be extracted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_fields: Vec<String>,

    /// Field-level confidence scores.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub field_confidence: std::collections::HashMap<String, f32>,
}

/// Source document type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// Text-based PDF (native text extraction).
    TextPdf,
    /// Image-based PDF (requires OCR).
    ImagePdf,
    /// Mixed PDF (some text, some images).
    HybridPdf,
    /// Direct image file.
    Image,
    /// Scanned with PP-Structure layout detection.
    ScannedWithLayout,
    /// Unknown source.
    #[default]
    Unknown,
}

impl Invoice {
    /// Create a new empty invoice with default values.
    pub fn new() -> Self {
        Self {
            header: InvoiceHeader {
                invoice_number: String::new(),
                issue_date: NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
                sale_date: None,
                due_date: None,
                invoice_type: InvoiceType::Standard,
                currency: "PLN".to_string(),
                correction_of: None,
            },
            issuer: Party::default(),
            receiver: Party::default(),
            line_items: Vec::new(),
            summary: InvoiceSummary::default(),
            metadata: ExtractionMetadata::default(),
        }
    }

    /// Validate the invoice data and return any issues found.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.header.invoice_number.is_empty() {
            issues.push("Missing invoice number".to_string());
        }

        if self.issuer.name.is_empty() {
            issues.push("Missing issuer name".to_string());
        }

        if self.issuer.nip.is_none() {
            issues.push("Missing issuer NIP".to_string());
        }

        if self.receiver.name.is_empty() && self.receiver.nip.is_none() {
            issues.push("Missing receiver information".to_string());
        }

        if self.line_items.is_empty() {
            issues.push("No line items".to_string());
        }

        if self.summary.total_gross == Decimal::ZERO {
            issues.push("Total gross is zero".to_string());
        }

        // Validate line item totals
        let calculated_net: Decimal = self.line_items.iter().map(|i| i.total_net).sum();
        let calculated_gross: Decimal = self.line_items.iter().map(|i| i.total_gross).sum();

        if (calculated_net - self.summary.total_net).abs() > Decimal::new(1, 2) {
            issues.push(format!(
                "Line item net total ({}) differs from summary ({})",
                calculated_net, self.summary.total_net
            ));
        }

        if (calculated_gross - self.summary.total_gross).abs() > Decimal::new(1, 2) {
            issues.push(format!(
                "Line item gross total ({}) differs from summary ({})",
                calculated_gross, self.summary.total_gross
            ));
        }

        issues
    }
}

impl Default for Invoice {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vat_rate_parsing() {
        assert_eq!(VatRate::from_str("23%"), Some(VatRate::Standard23));
        assert_eq!(VatRate::from_str("23"), Some(VatRate::Standard23));
        assert_eq!(VatRate::from_str("8%"), Some(VatRate::Reduced8));
        assert_eq!(VatRate::from_str("zw"), Some(VatRate::Exempt));
        assert_eq!(VatRate::from_str("ZW."), Some(VatRate::Exempt));
        assert_eq!(VatRate::from_str("np"), Some(VatRate::NotApplicable));
    }

    #[test]
    fn test_vat_rate_decimal() {
        assert_eq!(VatRate::Standard23.as_decimal(), Decimal::new(23, 2));
        assert_eq!(VatRate::Exempt.as_decimal(), Decimal::ZERO);
    }

    #[test]
    fn test_payment_method_parsing() {
        assert_eq!(
            PaymentMethod::from_str("przelew bankowy"),
            PaymentMethod::Transfer
        );
        assert_eq!(PaymentMethod::from_str("gotówka"), PaymentMethod::Cash);
        assert_eq!(PaymentMethod::from_str("karta płatnicza"), PaymentMethod::Card);
    }

    #[test]
    fn test_address_format() {
        let addr = Address {
            street: Some("ul. Przykładowa 1".to_string()),
            postal_code: Some("00-001".to_string()),
            city: Some("Warszawa".to_string()),
            country: Some("Polska".to_string()),
            raw: None,
        };
        assert_eq!(addr.format(), "ul. Przykładowa 1, 00-001 Warszawa");
    }
}
