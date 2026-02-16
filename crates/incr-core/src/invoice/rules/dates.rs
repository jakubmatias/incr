//! Date extraction for Polish invoices.

use chrono::NaiveDate;

use super::{ExtractionMatch, FieldExtractor};
use super::patterns::{DATE_DMY, DATE_YMD, DATE_POLISH_LONG, ISSUE_DATE, SALE_DATE, DUE_DATE};

/// Date field extractor.
pub struct DateExtractor;

impl DateExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DateExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldExtractor for DateExtractor {
    type Output = ExtractionMatch<NaiveDate>;

    fn extract(&self, text: &str) -> Option<Self::Output> {
        self.extract_all(text).into_iter().next()
    }

    fn extract_all(&self, text: &str) -> Vec<Self::Output> {
        let mut results = Vec::new();

        // DD.MM.YYYY or DD/MM/YYYY or DD-MM-YYYY
        for caps in DATE_DMY.captures_iter(text) {
            let day: u32 = caps[1].parse().unwrap_or(0);
            let month: u32 = caps[2].parse().unwrap_or(0);
            let year: i32 = parse_year(&caps[3]);

            if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(date, 0.9, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        // YYYY-MM-DD or YYYY/MM/DD
        for caps in DATE_YMD.captures_iter(text) {
            let year: i32 = caps[1].parse().unwrap_or(0);
            let month: u32 = caps[2].parse().unwrap_or(0);
            let day: u32 = caps[3].parse().unwrap_or(0);

            if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                // Skip if already found
                if results.iter().any(|r| r.value == date) {
                    continue;
                }

                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(date, 0.9, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        // Polish long format: "15 stycznia 2024"
        for caps in DATE_POLISH_LONG.captures_iter(text) {
            let day: u32 = caps[1].parse().unwrap_or(0);
            let month = polish_month_to_number(&caps[2]);
            let year: i32 = caps[3].parse().unwrap_or(0);

            if let Some(date) = NaiveDate::from_ymd_opt(year, month, day) {
                // Skip if already found
                if results.iter().any(|r| r.value == date) {
                    continue;
                }

                let full_match = caps.get(0).unwrap();
                results.push(
                    ExtractionMatch::new(date, 0.95, full_match.as_str())
                        .with_position(full_match.start(), full_match.end()),
                );
            }
        }

        results
    }
}

/// Extracted dates from an invoice.
#[derive(Debug, Clone, Default)]
pub struct InvoiceDates {
    /// Issue date (data wystawienia).
    pub issue_date: Option<ExtractionMatch<NaiveDate>>,
    /// Sale/delivery date (data sprzedaży).
    pub sale_date: Option<ExtractionMatch<NaiveDate>>,
    /// Due date (termin płatności).
    pub due_date: Option<ExtractionMatch<NaiveDate>>,
}

/// Extract all labeled dates from invoice text.
pub fn extract_dates(text: &str) -> InvoiceDates {
    let mut result = InvoiceDates::default();
    let date_extractor = DateExtractor::new();

    // Extract issue date
    if let Some(caps) = ISSUE_DATE.captures(text) {
        let date_text = &caps[1];
        if let Some(date) = date_extractor.extract(date_text) {
            result.issue_date = Some(ExtractionMatch::new(date.value, 0.95, date_text));
        }
    }

    // Extract sale date
    if let Some(caps) = SALE_DATE.captures(text) {
        let date_text = &caps[1];
        if let Some(date) = date_extractor.extract(date_text) {
            result.sale_date = Some(ExtractionMatch::new(date.value, 0.95, date_text));
        }
    }

    // Extract due date
    if let Some(caps) = DUE_DATE.captures(text) {
        let date_text = &caps[1];
        if let Some(date) = date_extractor.extract(date_text) {
            result.due_date = Some(ExtractionMatch::new(date.value, 0.95, date_text));
        }
    }

    // If no labeled dates found, try to extract any dates
    if result.issue_date.is_none() {
        let all_dates = date_extractor.extract_all(text);
        if let Some(first) = all_dates.into_iter().next() {
            result.issue_date = Some(first);
        }
    }

    result
}

fn parse_year(s: &str) -> i32 {
    let year: i32 = s.parse().unwrap_or(0);
    if year < 100 {
        // Two-digit year: assume 2000s for 00-50, 1900s for 51-99
        if year <= 50 {
            2000 + year
        } else {
            1900 + year
        }
    } else {
        year
    }
}

fn polish_month_to_number(month: &str) -> u32 {
    match month.to_lowercase().as_str() {
        "stycznia" => 1,
        "lutego" => 2,
        "marca" => 3,
        "kwietnia" => 4,
        "maja" => 5,
        "czerwca" => 6,
        "lipca" => 7,
        "sierpnia" => 8,
        "września" => 9,
        "października" => 10,
        "listopada" => 11,
        "grudnia" => 12,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_date_dmy() {
        let extractor = DateExtractor::new();

        let result = extractor.extract("15.01.2024");
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_extract_date_ymd() {
        let extractor = DateExtractor::new();

        let result = extractor.extract("2024-01-15");
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_extract_date_polish_long() {
        let extractor = DateExtractor::new();

        let result = extractor.extract("15 stycznia 2024");
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }

    #[test]
    fn test_extract_labeled_dates() {
        let text = r#"
            Faktura VAT nr FV/001/2024
            Data wystawienia: 15.01.2024
            Data sprzedaży: 10.01.2024
            Termin płatności: 29.01.2024
        "#;

        let dates = extract_dates(text);

        assert!(dates.issue_date.is_some());
        assert_eq!(
            dates.issue_date.unwrap().value,
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap()
        );

        assert!(dates.sale_date.is_some());
        assert_eq!(
            dates.sale_date.unwrap().value,
            NaiveDate::from_ymd_opt(2024, 1, 10).unwrap()
        );

        assert!(dates.due_date.is_some());
        assert_eq!(
            dates.due_date.unwrap().value,
            NaiveDate::from_ymd_opt(2024, 1, 29).unwrap()
        );
    }

    #[test]
    fn test_two_digit_year() {
        let extractor = DateExtractor::new();

        let result = extractor.extract("15.01.24");
        assert!(result.is_some());
        assert_eq!(result.unwrap().value, NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
    }
}
