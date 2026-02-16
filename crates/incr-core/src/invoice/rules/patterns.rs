//! Common regex patterns for Polish invoice extraction.

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    // NIP patterns (Polish tax ID)
    pub static ref NIP_PATTERN: Regex = Regex::new(
        r"(?i)(?:NIP|N\.I\.P\.?)[\s:]*(\d{3})[- ]?(\d{3})[- ]?(\d{2})[- ]?(\d{2})"
    ).unwrap();

    pub static ref NIP_STANDALONE: Regex = Regex::new(
        r"\b(\d{3})[- ]?(\d{3})[- ]?(\d{2})[- ]?(\d{2})\b"
    ).unwrap();

    // REGON patterns (Polish statistical ID)
    pub static ref REGON_PATTERN: Regex = Regex::new(
        r"(?i)(?:REGON|REG\.?)[\s:]*(\d{9}|\d{14})"
    ).unwrap();

    pub static ref REGON_STANDALONE: Regex = Regex::new(
        r"\b(\d{9})\b|\b(\d{14})\b"
    ).unwrap();

    // Polish date patterns
    pub static ref DATE_DMY: Regex = Regex::new(
        r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4}|\d{2})\b"
    ).unwrap();

    pub static ref DATE_YMD: Regex = Regex::new(
        r"\b(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})\b"
    ).unwrap();

    pub static ref DATE_POLISH_LONG: Regex = Regex::new(
        r"(\d{1,2})\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+(\d{4})"
    ).unwrap();

    // Labeled dates
    pub static ref ISSUE_DATE: Regex = Regex::new(
        r"(?i)(?:data\s+(?:wystawienia|faktury)|wystawion[ao]?\s+dnia?)[\s:]*(.+?)(?:\n|$)"
    ).unwrap();

    pub static ref SALE_DATE: Regex = Regex::new(
        r"(?i)(?:data\s+sprzeda[żz]y|data\s+dostawy|data\s+wykonania)[\s:]*(.+?)(?:\n|$)"
    ).unwrap();

    pub static ref DUE_DATE: Regex = Regex::new(
        r"(?i)(?:termin\s+p[łl]atno[śs]ci|termin\s+zap[łl]aty|p[łl]atne?\s+do)[\s:]*(.+?)(?:\n|$)"
    ).unwrap();

    // Amount patterns (Polish format: 1 234,56 or 1234.56)
    pub static ref AMOUNT_PATTERN: Regex = Regex::new(
        r"(\d{1,3}(?:[\s\u{00a0}]?\d{3})*)[,.](\d{2})\b"
    ).unwrap();

    pub static ref AMOUNT_WITH_CURRENCY: Regex = Regex::new(
        r"(\d{1,3}(?:[\s\u{00a0}]?\d{3})*)[,.](\d{2})\s*(PLN|zł|EUR|€|USD|\$|GBP|£)"
    ).unwrap();

    // Total amounts
    pub static ref TOTAL_GROSS: Regex = Regex::new(
        r"(?i)(?:razem|suma|do\s+zap[łl]aty|kwota\s+brutto|warto[śs][ćc]\s+brutto)[\s:]*(\d{1,3}(?:[\s\u{00a0}]?\d{3})*[,.]\d{2})"
    ).unwrap();

    pub static ref TOTAL_NET: Regex = Regex::new(
        r"(?i)(?:netto|warto[śs][ćc]\s+netto|razem\s+netto)[\s:]*(\d{1,3}(?:[\s\u{00a0}]?\d{3})*[,.]\d{2})"
    ).unwrap();

    pub static ref TOTAL_VAT: Regex = Regex::new(
        r"(?i)(?:VAT|podatek|kwota\s+VAT|razem\s+VAT)[\s:]*(\d{1,3}(?:[\s\u{00a0}]?\d{3})*[,.]\d{2})"
    ).unwrap();

    // VAT rate patterns
    pub static ref VAT_RATE: Regex = Regex::new(
        r"(?i)(23|8|5|0|zw\.?|np\.?|oo)%?"
    ).unwrap();

    pub static ref VAT_BREAKDOWN: Regex = Regex::new(
        r"(?i)(23|8|5|0|zw\.?|np\.?)%?\s*(\d{1,3}(?:[\s\u{00a0}]?\d{3})*[,.]\d{2})\s*(\d{1,3}(?:[\s\u{00a0}]?\d{3})*[,.]\d{2})"
    ).unwrap();

    // IBAN pattern (Polish format: PL + 26 digits)
    pub static ref IBAN_PATTERN: Regex = Regex::new(
        r"(?i)(?:IBAN[\s:]*)?(PL)?[\s]?(\d{2})[\s]?(\d{4})[\s]?(\d{4})[\s]?(\d{4})[\s]?(\d{4})[\s]?(\d{4})[\s]?(\d{4})"
    ).unwrap();

    pub static ref BANK_ACCOUNT: Regex = Regex::new(
        r"(?i)(?:(?:nr|numer)\s+(?:konta|rachunku)|rachunek\s+bankowy|konto)[\s:]*(.+?)(?:\n|$)"
    ).unwrap();

    // Invoice number patterns
    pub static ref INVOICE_NUMBER: Regex = Regex::new(
        r"(?i)(?:faktura\s+(?:VAT\s+)?(?:nr|numer)|nr\s+faktury|numer\s+faktury)[\s:]*([A-Za-z0-9/\-_]+)"
    ).unwrap();

    pub static ref INVOICE_NUMBER_STANDALONE: Regex = Regex::new(
        r"(?i)(?:FV|F|FA|FVS)[\s/\-]?(\d{1,6})[/\-](\d{2,4})"
    ).unwrap();

    // Party identification
    pub static ref SELLER_SECTION: Regex = Regex::new(
        r"(?i)(?:sprzedawca|wystawca|dostawca)[\s:]*"
    ).unwrap();

    pub static ref BUYER_SECTION: Regex = Regex::new(
        r"(?i)(?:nabywca|kupuj[aą]cy|odbiorca|zamawiaj[aą]cy)[\s:]*"
    ).unwrap();

    // Payment method
    pub static ref PAYMENT_METHOD: Regex = Regex::new(
        r"(?i)(?:forma\s+p[łl]atno[śs]ci|spos[óo]b\s+p[łl]atno[śs]ci|metoda\s+p[łl]atno[śs]ci)[\s:]*(\w+)"
    ).unwrap();

    // Postal code pattern
    pub static ref POSTAL_CODE: Regex = Regex::new(
        r"\b(\d{2})-(\d{3})\b"
    ).unwrap();

    // Email pattern
    pub static ref EMAIL: Regex = Regex::new(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ).unwrap();

    // Phone pattern (Polish format)
    pub static ref PHONE: Regex = Regex::new(
        r"(?:\+48[\s\-]?)?(?:\d{3}[\s\-]?\d{3}[\s\-]?\d{3}|\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})"
    ).unwrap();
}
