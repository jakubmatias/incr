# incr - Polish Invoice OCR

A CLI tool for extracting structured data from Polish invoices (faktury VAT). Works 100% offline with bundled OCR models.

## Features

- **Standalone Binary** - OCR models embedded in the executable, no external dependencies
- **100% Offline** - All processing runs locally, no data leaves your machine
- **Polish Invoice Support** - NIP, REGON, IBAN validation with Polish number/date formats
- **PDF & Image Support** - Process text-based PDFs, scanned documents, and images (PNG, JPG, TIFF)
- **PP-Structure Layout** - Document layout analysis for tables and text regions
- **Batch Processing** - Process multiple files with glob patterns

## Installation

### Download Binary

Download the latest release from [GitHub Releases](https://github.com/jakubmatias/incr/releases).

The binary includes embedded OCR models (~19MB) - no additional downloads required.

### Build from Source

```bash
# Clone the repository
git clone https://github.com/jakubmatias/incr.git
cd incr

# Build the CLI (requires ONNX Runtime)
cargo build --release -p incr-cli

# Binary at target/release/incr
```

#### Build Requirements

- Rust 1.85+
- ONNX Runtime 1.24+ (set `ORT_LIB_LOCATION` environment variable)

```bash
# macOS with Homebrew
brew install onnxruntime
ORT_LIB_LOCATION=/usr/local/Cellar/onnxruntime/1.24.1/lib cargo build --release -p incr-cli
```

## Usage

### Process a Single Invoice

```bash
# Output JSON to stdout
incr process invoice.pdf

# Save to file
incr process invoice.pdf -o result.json

# Text summary
incr process invoice.pdf -f text

# CSV output
incr process invoice.pdf -f csv
```

### Process Images

```bash
# Process scanned invoice
incr process scan.png

# Process with confidence scores
incr process scan.jpg --show-confidence
```

### Batch Processing

```bash
# Process all PDFs in a directory
incr batch "invoices/*.pdf" --output-dir results/

# Process with summary CSV
incr batch "*.pdf" --output-dir results/ --summary

# Continue on errors
incr batch "*.pdf" --output-dir results/ --continue-on-error

# Process images
incr batch "scans/*.png" --output-dir results/
```

### Model Management

The binary includes embedded mobile models. For higher accuracy, download server models:

```bash
# List available model variants
incr models list

# Download server models (88MB, better accuracy)
incr models download -v server

# Check model status
incr models status

# Switch active variant
incr models use server
```

#### Model Variants

| Variant  | Size   | Description                                    |
| -------- | ------ | ---------------------------------------------- |
| `mobile` | ~19MB  | Embedded in binary, good for most invoices     |
| `server` | ~103MB | Higher accuracy detection model (Downloadable) |

## Output Formats

### JSON (default)

```bash
incr process invoice.pdf -f json
```

```json
{
  "header": {
    "invoice_number": "FV/001/2024",
    "issue_date": "2024-01-15",
    "sale_date": "2024-01-10",
    "due_date": "2024-01-29",
    "currency": "PLN"
  },
  "issuer": {
    "name": "ABC Sp. z o.o.",
    "nip": "5261040828",
    "address": {
      "street": "ul. Przykładowa 1",
      "postal_code": "00-001",
      "city": "Warszawa"
    }
  },
  "receiver": {
    "name": "XYZ S.A.",
    "nip": "6750000000"
  },
  "summary": {
    "total_net": 1000.0,
    "total_vat": 230.0,
    "total_gross": 1230.0
  },
  "metadata": {
    "confidence": 0.92,
    "source_type": "text_pdf"
  }
}
```

### Text Summary

```bash
incr process invoice.pdf -f text
```

```
Invoice: FV/001/2024
Date: 2024-01-15

Issuer:
  ABC Sp. z o.o.
  NIP: 5261040828
  ul. Przykładowa 1, 00-001 Warszawa

Receiver:
  XYZ S.A.
  NIP: 6750000000

Summary:
  Net:   1000.00 PLN
  VAT:   230.00 PLN
  Gross: 1230.00 PLN

Payment due: 2024-01-29
```

### CSV

```bash
incr process invoice.pdf -f csv
```

## CLI Commands

| Command                | Description                              |
| ---------------------- | ---------------------------------------- |
| `process <file>`       | Process a single invoice (PDF or image)  |
| `batch <pattern>`      | Process multiple files with glob pattern |
| `models list`          | List available model variants            |
| `models download`      | Download OCR models                      |
| `models status`        | Check installed models                   |
| `models use <variant>` | Switch active model variant              |
| `models clean`         | Remove downloaded models                 |

## Polish Field Validation

### NIP (Tax ID)

- 10-digit number with checksum validation
- Formats: `5261040828`, `526-104-08-28`, `526 104 08 28`

### REGON (Statistical ID)

- 9 or 14 digits with checksum validation

### IBAN (Bank Account)

- Polish format: `PL` + 26 digits
- Full checksum validation

### VAT Rates

- Standard: 23%
- Reduced: 8%, 5%
- Zero: 0%
- Exempt: `zw` (zwolniony)
- Not applicable: `np` (nie podlega)

### Amount Formats

- Polish: `1 234,56` or `1234,56`
- International: `1234.56`

## Project Structure

```
incr/
├── crates/
│   ├── incr-core/           # Core library (PDF, OCR, extraction)
│   ├── incr-inference/      # ONNX inference abstraction
│   ├── incr-cli/            # Command-line interface
│   └── incr-wasm/           # WebAssembly bindings (experimental)
├── models/
│   ├── mobile/              # Mobile models (embedded)
│   └── server/              # Server models (downloadable)
└── dictionaries/            # Character dictionaries
```

## Configuration

Create `~/.config/incr/config.toml` or use `--config` flag:

```toml
[ocr]
detection_threshold = 0.3
recognition_threshold = 0.5
max_image_size = 2048

[pdf]
prefer_embedded_text = true
min_text_length = 100

[extraction]
validate_nip = true
validate_regon = true
validate_iban = true
default_currency = "PLN"
```

## Development

```bash
# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug incr process invoice.pdf

# Build release
cargo build --release -p incr-cli
```

## License

MIT OR Apache-2.0

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR models
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
