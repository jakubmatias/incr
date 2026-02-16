//! Amount extraction for Polish invoices.

use rust_decimal::Decimal;
use std::str::FromStr;

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{AMOUNT_PATTERN, TOTAL_GROSS, TOTAL_NET, TOTAL_VAT};

/// Amount field extractor.
pub struct AmountExtractor;

impl AmountExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AmountExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for AmountExtractor {
    type Output = ExtractionMatch<Decimal>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();

        for caps in AMOUNT_PATTERN.captures_iter(text) {
            let integer_part = caps[1].replace([' ', '\u{00a0}'], "");
            let decimal_part = &caps[2];

            let amount_str = format!("{}.{}", integer_part, decimal_part);
            if let Ok(amount) = Decimal::from_str(&amount_str) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(amount, 0.8, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        results
    }
}

/// Extracted amounts from an invoice.
#[derive(Debug, Clone, Default)]
pub struct InvoiceAmounts {
    /// Total net amount (before VAT).
    pub total_net: Option<ExtractionMatch<Decimal>>,
    /// Total VAT amount.
    pub total_vat: Option<ExtractionMatch<Decimal>>,
    /// Total gross amount (after VAT).
    pub total_gross: Option<ExtractionMatch<Decimal>>,
    /// All detected amounts.
    pub all_amounts: Vec<ExtractionMatch<Decimal>>,
}

/// Extract amounts from invoice text.
pub fn extract_amounts(text: &str) -> InvoiceAmounts {
    let mut result = InvoiceAmounts::default();
    let extractor = AmountExtractor::new();

    // Extract all amounts first
    result.all_amounts = extractor.extract_all(text);

    // Extract labeled amounts
    if let Some(caps) = TOTAL_GROSS.captures(text) {
        if let Some(amount) = parse_polish_amount(&caps[1]) {
            result.total_gross = Some(ExtractionMatch::new(amount, 0.95, &caps[0]));
        }
    }

    if let Some(caps) = TOTAL_NET.captures(text) {
        if let Some(amount) = parse_polish_amount(&caps[1]) {
            result.total_net = Some(ExtractionMatch::new(amount, 0.95, &caps[0]));
        }
    }

    if let Some(caps) = TOTAL_VAT.captures(text) {
        if let Some(amount) = parse_polish_amount(&caps[1]) {
            result.total_vat = Some(ExtractionMatch::new(amount, 0.95, &caps[0]));
        }
    }

    // If we have gross and net but not VAT, calculate it
    if result.total_vat.is_none() {
        if let (Some(gross), Some(net)) = (&result.total_gross, &result.total_net) {
            let vat = gross.value - net.value;
            result.total_vat = Some(ExtractionMatch::new(vat, 0.8, "calculated"));
        }
    }

    // If we have gross and VAT but not net, calculate it
    if result.total_net.is_none() {
        if let (Some(gross), Some(vat)) = (&result.total_gross, &result.total_vat) {
            let net = gross.value - vat.value;
            result.total_net = Some(ExtractionMatch::new(net, 0.8, "calculated"));
        }
    }

    // If we only have gross, try to identify it from the largest amount
    if result.total_gross.is_none() && !result.all_amounts.is_empty() {
        let max_amount = result
            .all_amounts
            .iter()
            .max_by(|a, b| a.value.cmp(&b.value))
            .cloned();
        result.total_gross = max_amount;
    }

    result
}

/// Parse a Polish-formatted amount (e.g., "1 234,56" or "1234.56").
pub fn parse_polish_amount(s: &str) -> Option<Decimal> {
    // Remove spaces and non-breaking spaces
    let cleaned: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
        .collect();

    // Replace comma with period for decimal
    let normalized = if cleaned.contains(',') && !cleaned.contains('.') {
        cleaned.replace(',', ".")
    } else if cleaned.contains(',') && cleaned.contains('.') {
        // Ambiguous case: assume comma is decimal separator if it comes last
        let comma_pos = cleaned.rfind(',');
        let dot_pos = cleaned.rfind('.');
        match (comma_pos, dot_pos) {
            (Some(c), Some(d)) if c > d => cleaned.replace('.', "").replace(',', "."),
            (Some(_), Some(_)) => cleaned.replace(',', ""),
            _ => cleaned,
        }
    } else {
        cleaned
    };

    Decimal::from_str(&normalized).ok()
}

/// Format amount in Polish style (1 234,56 zł).
pub fn format_polish_amount(amount: Decimal) -> String {
    let s = format!("{:.2}", amount);
    let parts: Vec<&str> = s.split('.').collect();

    if parts.len() != 2 {
        return s;
    }

    let integer_part = parts[0];
    let decimal_part = parts[1];

    // Add thousand separators
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_polish_amount() {
        assert_eq!(
            parse_polish_amount("1 234,56"),
            Some(Decimal::from_str("1234.56").unwrap())
        );
        assert_eq!(
            parse_polish_amount("1234,56"),
            Some(Decimal::from_str("1234.56").unwrap())
        );
        assert_eq!(
            parse_polish_amount("1234.56"),
            Some(Decimal::from_str("1234.56").unwrap())
        );
        assert_eq!(
            parse_polish_amount("12 345 678,90"),
            Some(Decimal::from_str("12345678.90").unwrap())
        );
    }

    #[test]
    fn test_format_polish_amount() {
        let amount = Decimal::from_str("1234.56").unwrap();
        assert_eq!(format_polish_amount(amount), "1 234,56");

        let amount = Decimal::from_str("12345678.90").unwrap();
        assert_eq!(format_polish_amount(amount), "12 345 678,90");
    }

    #[test]
    fn test_extract_amounts() {
        let text = r#"
            Wartość netto: 1 000,00 zł
            VAT 23%: 230,00 zł
            Razem do zapłaty: 1 230,00 zł
        "#;

        let amounts = extract_amounts(text);

        assert!(amounts.total_gross.is_some());
        assert_eq!(
            amounts.total_gross.unwrap().value,
            Decimal::from_str("1230.00").unwrap()
        );
    }

    #[test]
    fn test_extract_all_amounts() {
        let extractor = AmountExtractor::new();
        let text = "Cena: 100,00 zł, Razem: 1 234,56 zł";

        let results = extractor.extract_all(text);
        assert_eq!(results.len(), 2);
    }
}
