//! NIP (Polish Tax Identification Number) extraction and validation.

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{NIP_PATTERN, NIP_STANDALONE};

/// NIP field extractor.
pub struct NipExtractor {
    validate: bool,
}

impl NipExtractor {
    /// Create a new NIP extractor.
    pub fn new() -> Self {
        Self { validate: true }
    }

    /// Set whether to validate NIP checksums.
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

impl Default for NipExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for NipExtractor {
    type Output = ExtractionMatch<String>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();

        // Try labeled pattern first (higher confidence)
        for caps in NIP_PATTERN.captures_iter(text) {
            let nip = format!(
                "{}{}{}{}",
                &caps[1], &caps[2], &caps[3], &caps[4]
            );

            if !self.validate || validate_nip(&nip) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(nip, 0.95, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        // Try standalone pattern (lower confidence)
        for caps in NIP_STANDALONE.captures_iter(text) {
            let nip = format!(
                "{}{}{}{}",
                &caps[1], &caps[2], &caps[3], &caps[4]
            );

            // Skip if already found with labeled pattern
            if results.iter().any(|r| r.value == nip) {
                continue;
            }

            if !self.validate || validate_nip(&nip) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(nip, 0.7, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        results
    }
}

/// Extract NIP from text.
pub fn extract_nip(text: &str) -> Option<String> {
    NipExtractor::new().extract(text).map(|m| m.value)
}

/// Validate a Polish NIP using the checksum algorithm.
///
/// NIP format: 10 digits where the last digit is a checksum.
/// Weights: 6, 5, 7, 2, 3, 4, 5, 6, 7
pub fn validate_nip(nip: &str) -> bool {
    let digits: Vec<u32> = nip
        .chars()
        .filter(|c| c.is_ascii_digit())
        .filter_map(|c| c.to_digit(10))
        .collect();

    if digits.len() != 10 {
        return false;
    }

    let weights = [6, 5, 7, 2, 3, 4, 5, 6, 7];
    let sum: u32 = digits
        .iter()
        .take(9)
        .zip(weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    let checksum = sum % 11;

    // If checksum is 10, the NIP is invalid
    if checksum == 10 {
        return false;
    }

    checksum == digits[9]
}

/// Format NIP with dashes (XXX-XXX-XX-XX).
pub fn format_nip(nip: &str) -> String {
    let digits: String = nip.chars().filter(|c| c.is_ascii_digit()).collect();

    if digits.len() != 10 {
        return nip.to_string();
    }

    format!(
        "{}-{}-{}-{}",
        &digits[0..3],
        &digits[3..6],
        &digits[6..8],
        &digits[8..10]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_nip_valid() {
        // Known valid NIPs
        assert!(validate_nip("5261040828")); // Example valid NIP
        assert!(validate_nip("123-456-32-18")); // With dashes
        assert!(validate_nip("123 456 32 18")); // With spaces
    }

    #[test]
    fn test_validate_nip_invalid() {
        assert!(!validate_nip("1234567890")); // Invalid checksum
        assert!(!validate_nip("123456789")); // Too short
        assert!(!validate_nip("12345678901")); // Too long
    }

    #[test]
    fn test_extract_nip_labeled() {
        let text = "Sprzedawca: ABC Sp. z o.o.\nNIP: 526-104-08-28\nWarszawa";
        let nip = extract_nip(text);
        assert_eq!(nip, Some("5261040828".to_string()));
    }

    #[test]
    fn test_extract_nip_standalone() {
        let text = "Firma ABC, 526-104-08-28, ul. Przykładowa 1";
        let extractor = NipExtractor::new();
        let results = extractor.extract_all(text);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_format_nip() {
        assert_eq!(format_nip("5261040828"), "526-104-08-28");
        assert_eq!(format_nip("526-104-08-28"), "526-104-08-28");
    }
}
