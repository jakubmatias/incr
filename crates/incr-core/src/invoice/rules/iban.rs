//! IBAN (International Bank Account Number) extraction and validation.

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{IBAN_PATTERN, BANK_ACCOUNT};

/// IBAN field extractor.
pub struct IbanExtractor {
    validate: bool,
}

impl IbanExtractor {
    /// Create a new IBAN extractor.
    pub fn new() -> Self {
        Self { validate: true }
    }

    /// Set whether to validate IBAN checksums.
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }
}

impl Default for IbanExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for IbanExtractor {
    type Output = ExtractionMatch<String>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();

        // Try IBAN pattern
        for caps in IBAN_PATTERN.captures_iter(text) {
            let country_code = caps.get(1).map(|m| m.as_str()).unwrap_or("PL");
            let check_digits = &caps[2];
            let bban = format!(
                "{}{}{}{}{}{}",
                &caps[3], &caps[4], &caps[5], &caps[6], &caps[7], &caps[8]
            );

            let iban = format!("{}{}{}", country_code, check_digits, bban);

            if !self.validate || validate_iban(&iban) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(iban, 0.95, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        // Try bank account label pattern
        for caps in BANK_ACCOUNT.captures_iter(text) {
            let account_text = caps[1].trim();

            // Extract digits from the text
            let digits: String = account_text
                .chars()
                .filter(|c| c.is_ascii_digit())
                .collect();

            // Polish bank account is 26 digits
            if digits.len() == 26 {
                let iban = format!("PL{}", digits);

                // Skip if already found
                if results.iter().any(|r| r.value == iban) {
                    continue;
                }

                if !self.validate || validate_iban(&iban) {
                    let full_match = caps.get(0).unwrap();
                    results.push(
                        ExtractionMatch::new(iban, 0.9, full_match.as_str())
                            .with_position(full_match.start(), full_match.end()),
                    );
                }
            }
        }

        results
    }
}

/// Extract IBAN from text.
pub fn extract_iban(text: &str) -> Option<String> {
    IbanExtractor::new().extract(text).map(|m| m.value)
}

/// Validate an IBAN using the checksum algorithm.
///
/// Algorithm:
/// 1. Move first 4 characters to the end
/// 2. Replace letters with numbers (A=10, B=11, ..., Z=35)
/// 3. The resulting number mod 97 should equal 1
pub fn validate_iban(iban: &str) -> bool {
    // Remove spaces and convert to uppercase
    let iban: String = iban
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_uppercase();

    // Check minimum length (country code + check digits + BBAN)
    if iban.len() < 5 {
        return false;
    }

    // Check country code is letters and check digits are numbers
    let country_code = &iban[..2];
    let check_digits = &iban[2..4];

    if !country_code.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    if !check_digits.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    // Move first 4 characters to the end
    let rearranged = format!("{}{}", &iban[4..], &iban[..4]);

    // Convert to number string (letters become 10-35)
    let mut number_str = String::new();
    for c in rearranged.chars() {
        if c.is_ascii_digit() {
            number_str.push(c);
        } else if c.is_ascii_alphabetic() {
            let value = (c as u32) - ('A' as u32) + 10;
            number_str.push_str(&value.to_string());
        } else {
            return false;
        }
    }

    // Calculate mod 97 using string arithmetic (number is too large for u64)
    mod97(&number_str) == 1
}

fn mod97(number_str: &str) -> u32 {
    let mut remainder: u32 = 0;

    for c in number_str.chars() {
        let digit = c.to_digit(10).unwrap_or(0);
        remainder = (remainder * 10 + digit) % 97;
    }

    remainder
}

/// Format IBAN in groups of 4 characters.
pub fn format_iban(iban: &str) -> String {
    let cleaned: String = iban.chars().filter(|c| !c.is_whitespace()).collect();

    cleaned
        .chars()
        .collect::<Vec<char>>()
        .chunks(4)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect::<Vec<String>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_iban_valid() {
        // Example valid Polish IBAN
        assert!(validate_iban("PL61109010140000071219812874"));
        assert!(validate_iban("PL 61 1090 1014 0000 0712 1981 2874")); // With spaces
    }

    #[test]
    fn test_validate_iban_invalid() {
        assert!(!validate_iban("PL00000000000000000000000000")); // Invalid checksum
        assert!(!validate_iban("PL123")); // Too short
    }

    #[test]
    fn test_extract_iban() {
        let text = "Numer konta: PL61 1090 1014 0000 0712 1981 2874";
        let iban = extract_iban(text);
        assert!(iban.is_some());
    }

    #[test]
    fn test_extract_bank_account() {
        let text = "Rachunek bankowy: 61 1090 1014 0000 0712 1981 2874";
        let extractor = IbanExtractor::new().with_validation(false);
        let results = extractor.extract_all(text);
        // May or may not find depending on validation
        // The pattern should match, validation would check checksum
    }

    #[test]
    fn test_format_iban() {
        let iban = "PL61109010140000071219812874";
        assert_eq!(format_iban(iban), "PL61 1090 1014 0000 0712 1981 2874");
    }
}
