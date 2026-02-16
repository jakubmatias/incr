//! REGON (Polish Statistical Number) extraction and validation.

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{REGON_PATTERN, REGON_STANDALONE};

/// REGON field extractor.
pub struct RegonExtractor {
    validate: bool,
}

impl RegonExtractor {
    /// Create a new REGON extractor.
    pub fn new() -> Self {
        Self { validate: true }
    }

    /// Set whether to validate REGON checksums.
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

impl Default for RegonExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for RegonExtractor {
    type Output = ExtractionMatch<String>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();

        // Try labeled pattern first
        for caps in REGON_PATTERN.captures_iter(text) {
            let regon = caps[1].to_string();

            if !self.validate || validate_regon(&regon) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(regon, 0.95, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        // Try standalone pattern
        for caps in REGON_STANDALONE.captures_iter(text) {
            let regon = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();

            if regon.is_empty() {
                continue;
            }

            // Skip if already found
            if results.iter().any(|r| r.value == regon) {
                continue;
            }

            if !self.validate || validate_regon(&regon) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(regon, 0.6, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        results
    }
}

/// Extract REGON from text.
pub fn extract_regon(text: &str) -> Option<String> {
    RegonExtractor::new().extract(text).map(|m| m.value)
}

/// Validate a Polish REGON using the checksum algorithm.
///
/// REGON can be 9 or 14 digits.
/// - 9 digits: weights [8, 9, 2, 3, 4, 5, 6, 7]
/// - 14 digits: first 9 validated as above, then [2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8]
pub fn validate_regon(regon: &str) -> bool {
    let digits: Vec<u32> = regon
        .chars()
        .filter(|c| c.is_ascii_digit())
        .filter_map(|c| c.to_digit(10))
        .collect();

    match digits.len() {
        9 => validate_regon_9(&digits),
        14 => validate_regon_14(&digits),
        _ => false,
    }
}

fn validate_regon_9(digits: &[u32]) -> bool {
    let weights = [8, 9, 2, 3, 4, 5, 6, 7];
    let sum: u32 = digits
        .iter()
        .take(8)
        .zip(weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    let checksum = sum % 11;
    let expected = if checksum == 10 { 0 } else { checksum };

    expected == digits[8]
}

fn validate_regon_14(digits: &[u32]) -> bool {
    // First validate the 9-digit base
    if !validate_regon_9(&digits[..9].to_vec()) {
        return false;
    }

    // Then validate the full 14-digit number
    let weights = [2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8];
    let sum: u32 = digits
        .iter()
        .take(13)
        .zip(weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    let checksum = sum % 11;
    let expected = if checksum == 10 { 0 } else { checksum };

    expected == digits[13]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_regon_9_valid() {
        // Example valid 9-digit REGON
        assert!(validate_regon("123456785"));
    }

    #[test]
    fn test_validate_regon_invalid() {
        assert!(!validate_regon("123456789")); // Invalid checksum
        assert!(!validate_regon("12345678")); // Too short
        assert!(!validate_regon("1234567890")); // Wrong length (10)
    }

    #[test]
    fn test_extract_regon_labeled() {
        let text = "REGON: 123456785\nWarszawa";
        let regon = extract_regon(text);
        assert!(regon.is_some());
    }

    #[test]
    fn test_extract_regon_case_insensitive() {
        let text = "regon 123456785";
        let extractor = RegonExtractor::new().with_validation(false);
        let results = extractor.extract_all(text);
        assert!(!results.is_empty());
    }
}
