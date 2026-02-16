//! VAT rate extraction for Polish invoices.

use rust_decimal::Decimal;

use crate::models::invoice::{VatBreakdown, VatRate};

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{VAT_RATE, VAT_BREAKDOWN};
use super::amounts::parse_polish_amount;

/// VAT rate extractor.
pub struct VatExtractor;

impl VatExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VatExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for VatExtractor {
    type Output = ExtractionMatch<VatRate>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for caps in VAT_RATE.captures_iter(text) {
            let rate_str = &caps[1];
            if let Some(rate) = VatRate::from_str(rate_str) {
                let key = format!("{:?}", rate);
                if seen.insert(key) {
                    let full_match = caps.get(0).unwrap();
                    results.push(
                        ExtractionMatch::new(rate, 0.9, full_match.as_str())
                            .with_position(full_match.start(), full_match.end()),
                    );
                }
            }
        }

        results
    }
}

/// Extracted VAT information.
#[derive(Debug, Clone, Default)]
pub struct InvoiceVat {
    /// Detected VAT rates used in the invoice.
    pub rates: Vec<ExtractionMatch<VatRate>>,
    /// VAT breakdown by rate.
    pub breakdown: Vec<VatBreakdown>,
}

/// Extract VAT rates and breakdown from invoice text.
pub fn extract_vat_rates(text: &str) -> InvoiceVat {
    let mut result = InvoiceVat::default();
    let extractor = VatExtractor::new();

    // Extract all VAT rates
    result.rates = extractor.extract_all(text);

    // Try to extract VAT breakdown table
    for caps in VAT_BREAKDOWN.captures_iter(text) {
        let rate_str = &caps[1];
        if let Some(rate) = VatRate::from_str(rate_str) {
            let net = parse_polish_amount(&caps[2]).unwrap_or_default();
            let vat = parse_polish_amount(&caps[3]).unwrap_or_default();
            let gross = net + vat;

            result.breakdown.push(VatBreakdown {
                rate,
                net,
                vat,
                gross,
            });
        }
    }

    result
}

/// Calculate VAT amount from net amount and rate.
pub fn calculate_vat(net: Decimal, rate: VatRate) -> Decimal {
    net * rate.as_decimal()
}

/// Calculate gross amount from net amount and rate.
pub fn calculate_gross(net: Decimal, rate: VatRate) -> Decimal {
    net + calculate_vat(net, rate)
}

/// Calculate net amount from gross amount and rate.
pub fn calculate_net_from_gross(gross: Decimal, rate: VatRate) -> Decimal {
    let divisor = Decimal::ONE + rate.as_decimal();
    if divisor.is_zero() {
        gross
    } else {
        gross / divisor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_extract_vat_rates() {
        let extractor = VatExtractor::new();
        let text = "VAT 23% i VAT 8% oraz zw.";

        let results = extractor.extract_all(text);
        assert_eq!(results.len(), 3);

        let rates: Vec<VatRate> = results.into_iter().map(|r| r.value).collect();
        assert!(rates.contains(&VatRate::Standard23));
        assert!(rates.contains(&VatRate::Reduced8));
        assert!(rates.contains(&VatRate::Exempt));
    }

    #[test]
    fn test_extract_vat_breakdown() {
        let text = r#"
            Stawka    Netto        VAT
            23%       1000,00      230,00
            8%        500,00       40,00
        "#;

        let vat_info = extract_vat_rates(text);
        assert!(!vat_info.breakdown.is_empty());
    }

    #[test]
    fn test_calculate_vat() {
        let net = Decimal::from_str("100.00").unwrap();

        assert_eq!(
            calculate_vat(net, VatRate::Standard23),
            Decimal::from_str("23.00").unwrap()
        );

        assert_eq!(
            calculate_vat(net, VatRate::Reduced8),
            Decimal::from_str("8.00").unwrap()
        );

        assert_eq!(
            calculate_vat(net, VatRate::Exempt),
            Decimal::ZERO
        );
    }

    #[test]
    fn test_calculate_gross() {
        let net = Decimal::from_str("100.00").unwrap();

        assert_eq!(
            calculate_gross(net, VatRate::Standard23),
            Decimal::from_str("123.00").unwrap()
        );
    }

    #[test]
    fn test_calculate_net_from_gross() {
        let gross = Decimal::from_str("123.00").unwrap();

        let net = calculate_net_from_gross(gross, VatRate::Standard23);
        // Allow small rounding differences
        assert!((net - Decimal::from_str("100.00").unwrap()).abs() < Decimal::from_str("0.01").unwrap());
    }
}
