//! Hybrid invoice parser combining rule-based and ML extraction.

use std::collections::HashMap;
use std::time::Instant;

use chrono::NaiveDate;
use rust_decimal::Decimal;
use tracing::{debug, info};

use crate::models::invoice::*;
use crate::ocr::OcrResult;

use super::rules::{
    amounts::extract_amounts,
    dates::extract_dates,
    iban::extract_iban,
    nip::NipExtractor,
    patterns::*,
    regon::extract_regon,
    vat::extract_vat_rates,
    FieldExtractor,
};
use super::{InvoiceExtractor, Result};

/// Result of invoice extraction.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// Extracted invoice data.
    pub invoice: Invoice,
    /// Raw extracted text.
    pub raw_text: String,
    /// Extraction warnings.
    pub warnings: Vec<String>,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

/// Trait for invoice parsing.
pub trait InvoiceParser {
    /// Parse invoice from text.
    fn parse(&self, text: &str) -> Result<ExtractionResult>;
}

/// Hybrid invoice parser combining rules and optional ML.
pub struct HybridInvoiceParser {
    /// Whether to validate NIP checksums.
    validate_nip: bool,
    /// Whether to validate REGON checksums.
    validate_regon: bool,
    /// Whether to validate IBAN checksums.
    validate_iban: bool,
    /// Minimum confidence for accepting fields.
    min_confidence: f32,
}

impl HybridInvoiceParser {
    /// Create a new hybrid parser with default settings.
    pub fn new() -> Self {
        Self {
            validate_nip: true,
            validate_regon: true,
            validate_iban: true,
            min_confidence: 0.5,
        }
    }

    /// Set NIP validation.
    pub fn with_nip_validation(mut self, validate: bool) -> Self {
        self.validate_nip = validate;
        self
    }

    /// Set REGON validation.
    pub fn with_regon_validation(mut self, validate: bool) -> Self {
        self.validate_regon = validate;
        self
    }

    /// Set IBAN validation.
    pub fn with_iban_validation(mut self, validate: bool) -> Self {
        self.validate_iban = validate;
        self
    }

    /// Set minimum confidence threshold.
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence;
        self
    }

    fn extract_invoice_number(&self, text: &str) -> Option<String> {
        // Try labeled pattern first
        if let Some(caps) = INVOICE_NUMBER.captures(text) {
            return Some(caps[1].trim().to_string());
        }

        // Try standalone pattern
        if let Some(caps) = INVOICE_NUMBER_STANDALONE.captures(text) {
            return Some(format!("{}/{}", &caps[1], &caps[2]));
        }

        None
    }

    fn extract_parties(&self, text: &str) -> (Party, Party) {
        let mut issuer = Party::default();
        let mut receiver = Party::default();

        // Find seller/buyer section boundaries
        let seller_pos = SELLER_SECTION.find(text).map(|m| m.start());
        let buyer_pos = BUYER_SECTION.find(text).map(|m| m.start());

        // Determine text regions
        let (seller_text, buyer_text) = match (seller_pos, buyer_pos) {
            (Some(s), Some(b)) if s < b => {
                let seller_text = &text[s..b];
                let buyer_text = &text[b..];
                (seller_text, buyer_text)
            }
            (Some(s), Some(b)) => {
                let buyer_text = &text[b..s];
                let seller_text = &text[s..];
                (seller_text, buyer_text)
            }
            (Some(s), None) => (&text[s..], ""),
            (None, Some(b)) => ("", &text[b..]),
            (None, None) => {
                // No clear sections, try to extract from whole text
                (text, text)
            }
        };

        // Extract NIPs
        let nip_extractor = NipExtractor::new().with_validation(self.validate_nip);
        let all_nips = nip_extractor.extract_all(text);

        // Assign first NIP to issuer, second to receiver (common pattern)
        if let Some(nip) = nip_extractor.extract(seller_text) {
            issuer.nip = Some(nip.value);
        } else if !all_nips.is_empty() {
            issuer.nip = Some(all_nips[0].value.clone());
        }

        if let Some(nip) = nip_extractor.extract(buyer_text) {
            receiver.nip = Some(nip.value);
        } else if all_nips.len() > 1 {
            receiver.nip = Some(all_nips[1].value.clone());
        }

        // Extract REGONs
        if let Some(regon) = extract_regon(seller_text) {
            issuer.regon = Some(regon);
        }
        if let Some(regon) = extract_regon(buyer_text) {
            receiver.regon = Some(regon);
        }

        // Extract bank account from issuer section
        if let Some(iban) = extract_iban(seller_text) {
            issuer.bank_account = Some(iban);
        } else if let Some(iban) = extract_iban(text) {
            issuer.bank_account = Some(iban);
        }

        // Extract email and phone
        if let Some(email) = EMAIL.find(seller_text) {
            issuer.email = Some(email.as_str().to_string());
        }
        if let Some(phone) = PHONE.find(seller_text) {
            issuer.phone = Some(phone.as_str().to_string());
        }

        // Extract names (first line after section header)
        issuer.name = self.extract_party_name(seller_text);
        receiver.name = self.extract_party_name(buyer_text);

        // Extract addresses
        issuer.address = self.extract_address(seller_text);
        receiver.address = self.extract_address(buyer_text);

        (issuer, receiver)
    }

    fn extract_party_name(&self, text: &str) -> String {
        // Skip section header and get first non-empty line
        let lines: Vec<&str> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .filter(|l| !SELLER_SECTION.is_match(l) && !BUYER_SECTION.is_match(l))
            .filter(|l| !l.starts_with("NIP") && !l.starts_with("REGON"))
            .collect();

        lines.first().map(|s| s.to_string()).unwrap_or_default()
    }

    fn extract_address(&self, text: &str) -> Address {
        let mut address = Address::default();

        // Look for postal code pattern
        if let Some(caps) = POSTAL_CODE.captures(text) {
            address.postal_code = Some(format!("{}-{}", &caps[1], &caps[2]));

            // City is usually after postal code
            let after_postal = &text[caps.get(0).unwrap().end()..];
            let city: String = after_postal
                .trim()
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .chars()
                .take_while(|c| c.is_alphabetic() || c.is_whitespace() || *c == '-')
                .collect();
            if !city.is_empty() {
                address.city = Some(city);
            }
        }

        // Look for street (ul., al., pl.)
        let street_pattern = regex::Regex::new(r"(?i)(ul\.|al\.|pl\.)\s*[^\n,]+").unwrap();
        if let Some(m) = street_pattern.find(text) {
            address.street = Some(m.as_str().to_string());
        }

        // If structured extraction failed, store raw address
        if address.street.is_none() && address.city.is_none() {
            // Get lines that look like address (after name, before contact info)
            let lines: Vec<&str> = text
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .filter(|l| {
                    !SELLER_SECTION.is_match(l)
                        && !BUYER_SECTION.is_match(l)
                        && !l.starts_with("NIP")
                        && !l.starts_with("REGON")
                        && !EMAIL.is_match(l)
                        && !PHONE.is_match(l)
                })
                .skip(1) // Skip name
                .take(2) // Take up to 2 address lines
                .collect();

            if !lines.is_empty() {
                address.raw = Some(lines.join(", "));
            }
        }

        address
    }

    fn extract_line_items(&self, text: &str) -> Vec<LineItem> {
        let mut items = Vec::new();

        // Look for table-like structure
        // Common patterns:
        // - "Lp. | Nazwa | Ilość | Jm. | Cena | Wartość | VAT"
        // - Lines with quantity, unit price, and amount

        let lines: Vec<&str> = text.lines().collect();
        let mut in_table = false;

        for line in &lines {
            let line = line.trim();

            // Detect table header
            if line.contains("Lp") && (line.contains("Nazwa") || line.contains("Opis")) {
                in_table = true;
                continue;
            }

            // Detect table end (summary lines)
            if in_table && (line.starts_with("Razem") || line.starts_with("SUMA")) {
                break;
            }

            if in_table && !line.is_empty() {
                if let Some(item) = self.parse_line_item(line) {
                    items.push(item);
                }
            }
        }

        // If no table found, try to extract single-item invoice
        if items.is_empty() {
            // Look for description + amount pattern
            let amounts = extract_amounts(text);
            if !amounts.all_amounts.is_empty() {
                // Try to find a description
                let desc_lines: Vec<&str> = text
                    .lines()
                    .filter(|l| {
                        !l.trim().is_empty()
                            && !AMOUNT_PATTERN.is_match(l)
                            && !l.contains("Faktura")
                            && !l.contains("NIP")
                    })
                    .take(1)
                    .collect();

                if let Some(desc) = desc_lines.first() {
                    let total_gross = amounts.total_gross.map(|m| m.value).unwrap_or_default();
                    let total_net = amounts.total_net.map(|m| m.value).unwrap_or(total_gross);
                    let vat_amount = total_gross - total_net;

                    items.push(LineItem {
                        ordinal: Some(1),
                        description: desc.trim().to_string(),
                        code: None,
                        quantity: Decimal::ONE,
                        unit: Some("szt.".to_string()),
                        unit_price_net: total_net,
                        unit_price_gross: Some(total_gross),
                        vat_rate: VatRate::Standard23,
                        total_net,
                        vat_amount,
                        total_gross,
                        discount_percent: None,
                    });
                }
            }
        }

        items
    }

    fn parse_line_item(&self, line: &str) -> Option<LineItem> {
        // Try to parse a tabular line
        // Expected format: ordinal | description | quantity | unit | price | ... | gross

        // Split by common separators
        let parts: Vec<&str> = line.split(|c| c == '|' || c == '\t').collect();

        if parts.len() < 3 {
            // Try whitespace-separated
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return None;
            }
        }

        // Extract amounts from the line
        let amounts_in_line: Vec<Decimal> = AMOUNT_PATTERN
            .captures_iter(line)
            .filter_map(|caps| {
                let int_part = caps[1].replace([' ', '\u{00a0}'], "");
                let dec_part = &caps[2];
                format!("{}.{}", int_part, dec_part).parse().ok()
            })
            .collect();

        if amounts_in_line.is_empty() {
            return None;
        }

        // Try to identify fields by position
        // Usually: ordinal, description, quantity, unit, unit_price, net_total, vat_rate, vat_amount, gross_total

        // Get the description (usually the longest text part)
        let description = parts
            .iter()
            .filter(|p| !p.trim().is_empty())
            .filter(|p| !p.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ','))
            .max_by_key(|p| p.len())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Item".to_string());

        // Determine amounts
        let (total_net, vat_amount, total_gross) = match amounts_in_line.len() {
            1 => (amounts_in_line[0], Decimal::ZERO, amounts_in_line[0]),
            2 => (
                amounts_in_line[0],
                amounts_in_line[1] - amounts_in_line[0],
                amounts_in_line[1],
            ),
            _ => {
                // Assume last is gross, second-to-last is VAT, third-to-last is net
                let gross = *amounts_in_line.last().unwrap();
                let n = amounts_in_line.len();
                let vat = if n >= 2 { amounts_in_line[n - 2] } else { Decimal::ZERO };
                let net = if n >= 3 {
                    amounts_in_line[n - 3]
                } else {
                    gross - vat
                };
                (net, vat, gross)
            }
        };

        // Try to detect VAT rate
        let vat_rate = VAT_RATE
            .captures(line)
            .and_then(|c| VatRate::from_str(&c[1]))
            .unwrap_or(VatRate::Standard23);

        // Extract quantity (first number that's not an amount)
        let quantity = parts
            .iter()
            .find_map(|p| {
                let p = p.trim();
                if p.contains(',') || p.contains('.') {
                    return None;
                }
                p.parse::<i64>()
                    .ok()
                    .filter(|&n| n > 0 && n < 10000)
                    .map(Decimal::from)
            })
            .unwrap_or(Decimal::ONE);

        let unit_price_net = if quantity.is_zero() {
            total_net
        } else {
            total_net / quantity
        };

        Some(LineItem {
            ordinal: parts
                .first()
                .and_then(|p| p.trim().parse().ok()),
            description,
            code: None,
            quantity,
            unit: Some("szt.".to_string()),
            unit_price_net,
            unit_price_gross: Some(total_gross / quantity),
            vat_rate,
            total_net,
            vat_amount,
            total_gross,
            discount_percent: None,
        })
    }

    fn extract_payment_info(&self, text: &str) -> (Option<PaymentMethod>, Option<Decimal>) {
        let payment_method = PAYMENT_METHOD
            .captures(text)
            .map(|c| PaymentMethod::from_str(&c[1]));

        let amount_due = extract_amounts(text).total_gross.map(|m| m.value);

        (payment_method, amount_due)
    }
}

impl Default for HybridInvoiceParser {
    fn default() -> Self {
        Self::new()
    }
}

impl InvoiceParser for HybridInvoiceParser {
    fn parse(&self, text: &str) -> Result<ExtractionResult> {
        let start = Instant::now();
        let mut warnings = Vec::new();

        info!("Parsing invoice from {} characters of text", text.len());

        // Extract invoice number
        let invoice_number = self.extract_invoice_number(text);
        if invoice_number.is_none() {
            warnings.push("Could not extract invoice number".to_string());
        }

        // Extract dates
        let dates = extract_dates(text);
        let has_issue_date = dates.issue_date.is_some();
        let issue_date = dates
            .issue_date
            .map(|m| m.value)
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());

        if !has_issue_date {
            warnings.push("Could not extract issue date".to_string());
        }

        // Extract parties
        let (issuer, receiver) = self.extract_parties(text);

        if issuer.nip.is_none() {
            warnings.push("Could not extract issuer NIP".to_string());
        }

        // Extract line items
        let line_items = self.extract_line_items(text);
        if line_items.is_empty() {
            warnings.push("Could not extract line items".to_string());
        }

        // Extract amounts
        let amounts = extract_amounts(text);
        let total_net = amounts.total_net.map(|m| m.value).unwrap_or_else(|| {
            line_items.iter().map(|i| i.total_net).sum()
        });
        let total_gross = amounts.total_gross.map(|m| m.value).unwrap_or_else(|| {
            line_items.iter().map(|i| i.total_gross).sum()
        });
        let total_vat = amounts.total_vat.map(|m| m.value).unwrap_or_else(|| {
            total_gross - total_net
        });

        // Extract VAT breakdown
        let vat_info = extract_vat_rates(text);

        // Extract payment info
        let (payment_method, amount_due) = self.extract_payment_info(text);

        // Build invoice
        let invoice = Invoice {
            header: InvoiceHeader {
                invoice_number: invoice_number.unwrap_or_else(|| "UNKNOWN".to_string()),
                issue_date,
                sale_date: dates.sale_date.map(|m| m.value),
                due_date: dates.due_date.map(|m| m.value),
                invoice_type: InvoiceType::Standard,
                currency: "PLN".to_string(),
                correction_of: None,
            },
            issuer,
            receiver,
            line_items,
            summary: InvoiceSummary {
                total_net,
                total_vat,
                total_gross,
                vat_breakdown: vat_info.breakdown,
                payment_method,
                amount_paid: None,
                amount_due,
                amount_in_words: None,
            },
            metadata: ExtractionMetadata {
                confidence: 0.0, // Will be calculated
                source_type: SourceType::Unknown,
                processing_time_ms: Some(start.elapsed().as_millis() as u64),
                ocr_engine: None,
                warnings: warnings.clone(),
                missing_fields: Vec::new(),
                field_confidence: HashMap::new(),
            },
        };

        // Calculate overall confidence
        let mut confidence = 1.0f32;
        if invoice.header.invoice_number == "UNKNOWN" {
            confidence -= 0.2;
        }
        if invoice.issuer.nip.is_none() {
            confidence -= 0.2;
        }
        if invoice.line_items.is_empty() {
            confidence -= 0.3;
        }
        if invoice.summary.total_gross.is_zero() {
            confidence -= 0.2;
        }

        let mut invoice = invoice;
        invoice.metadata.confidence = confidence.max(0.0);

        // Validate
        let validation_issues = invoice.validate();
        if !validation_issues.is_empty() {
            warnings.extend(validation_issues);
        }

        debug!(
            "Extracted invoice {} with confidence {:.2}",
            invoice.header.invoice_number, invoice.metadata.confidence
        );

        Ok(ExtractionResult {
            invoice,
            raw_text: text.to_string(),
            warnings,
            processing_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

impl InvoiceExtractor for HybridInvoiceParser {
    fn extract(&self, ocr_result: &OcrResult) -> Result<Invoice> {
        // Check if we have layout information with table regions
        let result = if let Some(ref layout) = ocr_result.layout {
            if !layout.tables.is_empty() {
                // Extract text from table regions for better line item parsing
                let table_text = self.extract_table_text(ocr_result, layout);
                debug!("Extracted {} chars from {} table regions", table_text.len(), layout.tables.len());

                // Parse with table-specific text
                let mut parse_result = self.parse(&ocr_result.text)?;

                // Re-extract line items from table regions if we found any
                if !table_text.is_empty() {
                    let table_items = self.extract_line_items(&table_text);
                    if !table_items.is_empty() {
                        parse_result.invoice.line_items = table_items;
                    }
                }

                parse_result
            } else {
                self.parse(&ocr_result.text)?
            }
        } else {
            self.parse(&ocr_result.text)?
        };

        let mut invoice = result.invoice;
        invoice.metadata.ocr_engine = Some("PaddleOCR".to_string());
        invoice.metadata.processing_time_ms =
            Some(result.processing_time_ms + ocr_result.processing_time_ms);

        // Add layout detection info to metadata
        if ocr_result.layout.is_some() {
            invoice.metadata.source_type = SourceType::ScannedWithLayout;
        }

        Ok(invoice)
    }

    fn extract_from_text(&self, text: &str) -> Result<Invoice> {
        self.parse(text).map(|r| r.invoice)
    }
}

impl HybridInvoiceParser {
    /// Extract text from table regions using OCR box positions.
    fn extract_table_text(&self, ocr_result: &OcrResult, layout: &crate::ocr::LayoutInfo) -> String {
        let mut table_lines: Vec<(f32, String)> = Vec::new();

        for table in &layout.tables {
            // Find text boxes that fall within this table region
            for text_box in &ocr_result.boxes {
                let (bx, by, _, _) = text_box.rect();

                // Check if text box center is within table bounds
                if bx >= table.bbox[0]
                    && bx <= table.bbox[2]
                    && by >= table.bbox[1]
                    && by <= table.bbox[3]
                {
                    table_lines.push((by, text_box.text.clone()));
                }
            }
        }

        // Sort by Y position (top to bottom)
        table_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Group lines by approximate Y position
        let mut grouped_lines: Vec<String> = Vec::new();
        let mut current_y = f32::NEG_INFINITY;
        let mut current_line = String::new();

        for (y, text) in table_lines {
            if (y - current_y).abs() < 15.0 {
                // Same line
                if !current_line.is_empty() {
                    current_line.push_str(" | ");
                }
                current_line.push_str(&text);
            } else {
                // New line
                if !current_line.is_empty() {
                    grouped_lines.push(current_line);
                }
                current_line = text;
                current_y = y;
            }
        }

        if !current_line.is_empty() {
            grouped_lines.push(current_line);
        }

        grouped_lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_invoice() {
        let text = r#"
            FAKTURA VAT nr FV/001/2024

            Sprzedawca:
            ABC Sp. z o.o.
            ul. Przykładowa 1
            00-001 Warszawa
            NIP: 526-104-08-28

            Nabywca:
            XYZ S.A.
            ul. Testowa 10
            00-002 Kraków
            NIP: 675-000-00-06

            Data wystawienia: 15.01.2024
            Termin płatności: 29.01.2024

            Lp. | Nazwa                  | Ilość | Cena netto | Wartość netto | VAT | Wartość brutto
            1   | Usługa konsultingowa   | 1     | 1000,00    | 1000,00       | 23% | 1230,00

            Razem netto: 1 000,00 zł
            VAT 23%: 230,00 zł
            Razem do zapłaty: 1 230,00 zł

            Forma płatności: przelew
        "#;

        let parser = HybridInvoiceParser::new().with_nip_validation(false);
        let result = parser.parse(text).unwrap();

        assert_eq!(result.invoice.header.invoice_number, "FV/001/2024");
        assert!(result.invoice.issuer.nip.is_some());
        assert!(result.invoice.receiver.nip.is_some());
    }

    #[test]
    fn test_extract_invoice_number() {
        let parser = HybridInvoiceParser::new();

        assert_eq!(
            parser.extract_invoice_number("Faktura VAT nr FV/001/2024"),
            Some("FV/001/2024".to_string())
        );

        assert_eq!(
            parser.extract_invoice_number("FV/123/24"),
            Some("123/24".to_string())
        );
    }
}
