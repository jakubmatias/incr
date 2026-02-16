//! Process command - extract data from a single invoice file.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use console::style;
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{debug, info, warn};

use incr_core::models::config::IncrConfig;
use incr_core::models::invoice::Invoice;
use incr_core::invoice::{HybridInvoiceParser, InvoiceParser};
use incr_core::pdf::{PdfExtractor, PdfProcessor, PdfType};

use super::models::{get_active_variant, get_variant_dir};

/// Arguments for the process command.
#[derive(Args)]
pub struct ProcessArgs {
    /// Input file (PDF or image)
    #[arg(required = true)]
    input: PathBuf,

    /// Output file (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output format
    #[arg(short, long, value_enum, default_value = "json")]
    format: OutputFormat,

    /// Model directory
    #[arg(short, long)]
    model_dir: Option<PathBuf>,

    /// Skip OCR and use only PDF text extraction
    #[arg(long)]
    text_only: bool,

    /// Show extraction confidence scores
    #[arg(long)]
    show_confidence: bool,

    /// Validate extracted data
    #[arg(long)]
    validate: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    /// JSON output
    Json,
    /// CSV output
    Csv,
    /// Plain text summary
    Text,
}

pub async fn run(args: ProcessArgs, config_path: Option<&str>) -> anyhow::Result<()> {
    let start = Instant::now();

    // Load configuration
    let config = if let Some(path) = config_path {
        IncrConfig::from_file(std::path::Path::new(path))?
    } else {
        IncrConfig::default()
    };

    // Check input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file not found: {}", args.input.display());
    }

    // Determine file type
    let extension = args.input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    info!("Processing file: {}", args.input.display());

    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let invoice = match extension.as_str() {
        "pdf" => process_pdf(&args, &config, &pb).await?,
        "png" | "jpg" | "jpeg" | "tiff" | "bmp" => process_image(&args, &config, &pb).await?,
        _ => anyhow::bail!("Unsupported file format: {}", extension),
    };

    pb.finish_with_message("Done");

    // Validate if requested
    if args.validate {
        let issues = invoice.validate();
        if !issues.is_empty() {
            eprintln!("{}", style("Validation issues:").yellow());
            for issue in &issues {
                eprintln!("  - {}", issue);
            }
        }
    }

    // Format output
    let output = format_invoice(&invoice, args.format)?;

    // Write output
    if let Some(output_path) = &args.output {
        fs::write(output_path, &output)?;
        println!(
            "{} Output written to {}",
            style("✓").green(),
            output_path.display()
        );
    } else {
        println!("{}", output);
    }

    // Show summary
    if args.show_confidence {
        println!();
        println!(
            "{} Extraction confidence: {:.1}%",
            style("ℹ").blue(),
            invoice.metadata.confidence * 100.0
        );
        if let Some(time_ms) = invoice.metadata.processing_time_ms {
            println!(
                "{} Processing time: {}ms",
                style("ℹ").blue(),
                time_ms
            );
        }
    }

    debug!("Total processing time: {:?}", start.elapsed());

    Ok(())
}

async fn process_pdf(
    args: &ProcessArgs,
    config: &IncrConfig,
    pb: &ProgressBar,
) -> anyhow::Result<Invoice> {
    pb.set_message("Loading PDF...");
    pb.set_position(10);

    let data = fs::read(&args.input)?;
    let mut extractor = PdfExtractor::new();
    extractor.load(&data)?;

    let page_count = extractor.page_count();
    debug!("PDF has {} pages", page_count);

    pb.set_message("Analyzing PDF...");
    pb.set_position(20);

    let pdf_type = extractor.analyze();
    debug!("PDF type: {:?}", pdf_type);

    let text = match pdf_type {
        PdfType::Text | PdfType::Hybrid if config.pdf.prefer_embedded_text || args.text_only => {
            pb.set_message("Extracting text...");
            pb.set_position(40);
            let extracted = extractor.extract_text()?;

            // For hybrid PDFs, check if we got enough text
            if pdf_type == PdfType::Hybrid && extracted.len() < config.pdf.min_text_length {
                warn!("Hybrid PDF has insufficient embedded text, falling back to OCR");
                try_ocr_pdf(&extractor, args, config, pb).unwrap_or(extracted)
            } else {
                extracted
            }
        }
        PdfType::Image | PdfType::Hybrid if !args.text_only => {
            pb.set_message("Running OCR...");
            pb.set_position(40);

            try_ocr_pdf(&extractor, args, config, pb)?
        }
        PdfType::Empty => {
            anyhow::bail!("PDF appears to be empty");
        }
        _ => {
            // text_only flag set but PDF is image-based
            anyhow::bail!("PDF is image-based but --text-only flag was set. Remove flag to use OCR.");
        }
    };

    if text.trim().is_empty() {
        anyhow::bail!("No text could be extracted from the PDF");
    }

    pb.set_message("Extracting invoice data...");
    pb.set_position(70);

    let parser = HybridInvoiceParser::new()
        .with_nip_validation(config.extraction.validate_nip)
        .with_regon_validation(config.extraction.validate_regon)
        .with_iban_validation(config.extraction.validate_iban);

    let result = parser.parse(&text)?;
    let mut invoice = result.invoice;

    invoice.metadata.source_type = match pdf_type {
        PdfType::Text => incr_core::models::invoice::SourceType::TextPdf,
        PdfType::Image => incr_core::models::invoice::SourceType::ImagePdf,
        PdfType::Hybrid => incr_core::models::invoice::SourceType::HybridPdf,
        PdfType::Empty => incr_core::models::invoice::SourceType::Unknown,
    };

    pb.set_position(100);

    Ok(invoice)
}

/// Try to run OCR on a PDF by extracting images.
fn try_ocr_pdf(
    extractor: &PdfExtractor,
    args: &ProcessArgs,
    config: &IncrConfig,
    pb: &ProgressBar,
) -> anyhow::Result<String> {
    // Get model directory (use active variant if not specified)
    let model_dir = args.model_dir.clone().unwrap_or_else(|| {
        get_variant_dir(get_active_variant())
    });

    // Check if models exist
    let det_model = model_dir.join(&config.models.detection_model);
    let rec_model = model_dir.join(&config.models.recognition_model);

    if !det_model.exists() || !rec_model.exists() {
        // Fall back to text extraction if models not available
        warn!("OCR models not found at {}, falling back to text extraction", model_dir.display());
        return Ok(extractor.extract_text()?);
    }

    // Extract images from all PDF pages
    pb.set_message("Extracting images from PDF...");
    pb.set_position(35);

    let page_count = extractor.page_count();
    let mut all_images = Vec::new();

    for page in 1..=page_count {
        match extractor.extract_images(page) {
            Ok(images) => all_images.extend(images),
            Err(e) => {
                warn!("Failed to extract images from page {}: {}", page, e);
            }
        }
    }

    if all_images.is_empty() {
        warn!("No images found in PDF, falling back to text extraction");
        return Ok(extractor.extract_text()?);
    }

    debug!("Extracted {} images from PDF", all_images.len());

    // Process each image with OCR
    let mut all_text = Vec::new();
    let total_images = all_images.len();

    for (i, image) in all_images.iter().enumerate() {
        pb.set_message(format!("OCR on image {}/{}", i + 1, total_images));
        pb.set_position(40 + ((i as u64 * 25) / total_images as u64));

        // Run OCR on the image directly
        match run_ocr(image, &model_dir, config, pb) {
            Ok(text) if !text.trim().is_empty() => {
                all_text.push(text);
            }
            Ok(_) => {
                debug!("No text detected in image {}", i + 1);
            }
            Err(e) => {
                warn!("OCR failed for image {}: {}", i + 1, e);
            }
        }
    }

    if all_text.is_empty() {
        anyhow::bail!("No text detected in any PDF images");
    }

    Ok(all_text.join("\n\n"))
}

async fn process_image(
    args: &ProcessArgs,
    config: &IncrConfig,
    pb: &ProgressBar,
) -> anyhow::Result<Invoice> {
    pb.set_message("Loading image...");
    pb.set_position(10);

    let image = image::open(&args.input)?;

    pb.set_message("Running OCR...");
    pb.set_position(30);

    // Get model directory (use active variant if not specified)
    let model_dir = args.model_dir.clone().unwrap_or_else(|| {
        get_variant_dir(get_active_variant())
    });

    // Check if models exist
    let det_model = model_dir.join(&config.models.detection_model);
    let rec_model = model_dir.join(&config.models.recognition_model);

    if !det_model.exists() || !rec_model.exists() {
        let active = get_active_variant();
        anyhow::bail!(
            "OCR models not found at {}.\n\n\
             Run 'incr models download -v {}' to download {} models.",
            model_dir.display(),
            active,
            active
        );
    }

    // Run OCR
    let text = run_ocr(&image, &model_dir, config, pb)?;

    if text.trim().is_empty() {
        anyhow::bail!("No text detected in image");
    }

    pb.set_message("Extracting invoice data...");
    pb.set_position(70);

    let parser = HybridInvoiceParser::new()
        .with_nip_validation(config.extraction.validate_nip)
        .with_regon_validation(config.extraction.validate_regon)
        .with_iban_validation(config.extraction.validate_iban);

    let result = parser.parse(&text)?;
    let mut invoice = result.invoice;

    invoice.metadata.source_type = incr_core::models::invoice::SourceType::Image;

    pb.set_position(100);

    Ok(invoice)
}

/// Run OCR on an image using embedded or external models.
fn run_ocr(
    image: &DynamicImage,
    model_dir: &PathBuf,
    config: &IncrConfig,
    pb: &ProgressBar,
) -> anyhow::Result<String> {
    use incr_core::{create_engine_from_dir, create_engine_from_embedded};

    pb.set_message("Loading OCR models...");
    pb.set_position(35);

    // Try external models first if model_dir exists, otherwise use embedded
    let det_model = model_dir.join(&config.models.detection_model);
    let engine = if det_model.exists() {
        debug!("Using external models from {}", model_dir.display());
        create_engine_from_dir(model_dir, config.ocr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to load OCR models: {}", e))?
    } else {
        debug!("Using embedded mobile models");
        create_engine_from_embedded(config.ocr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to load embedded OCR models: {}", e))?
    };

    pb.set_message("Detecting text regions...");
    pb.set_position(45);

    let result = engine
        .process(image)
        .map_err(|e| anyhow::anyhow!("OCR failed: {}", e))?;

    pb.set_message("OCR complete");
    pb.set_position(60);

    debug!(
        "OCR detected {} text boxes in {}ms",
        result.boxes.len(),
        result.processing_time_ms
    );

    Ok(result.text)
}

fn format_invoice(invoice: &Invoice, format: OutputFormat) -> anyhow::Result<String> {
    match format {
        OutputFormat::Json => {
            Ok(serde_json::to_string(invoice)?)
        }
        OutputFormat::Csv => {
            format_csv(invoice)
        }
        OutputFormat::Text => {
            format_text(invoice)
        }
    }
}

fn format_csv(invoice: &Invoice) -> anyhow::Result<String> {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write header
    wtr.write_record([
        "invoice_number",
        "issue_date",
        "issuer_name",
        "issuer_nip",
        "receiver_name",
        "receiver_nip",
        "total_net",
        "total_vat",
        "total_gross",
        "currency",
    ])?;

    // Write data
    wtr.write_record([
        &invoice.header.invoice_number,
        &invoice.header.issue_date.to_string(),
        &invoice.issuer.name,
        &invoice.issuer.nip.clone().unwrap_or_default(),
        &invoice.receiver.name,
        &invoice.receiver.nip.clone().unwrap_or_default(),
        &invoice.summary.total_net.to_string(),
        &invoice.summary.total_vat.to_string(),
        &invoice.summary.total_gross.to_string(),
        &invoice.header.currency,
    ])?;

    let data = String::from_utf8(wtr.into_inner()?)?;
    Ok(data)
}

fn format_text(invoice: &Invoice) -> anyhow::Result<String> {
    let mut output = String::new();

    output.push_str(&format!("Invoice: {}\n", invoice.header.invoice_number));
    output.push_str(&format!("Date: {}\n", invoice.header.issue_date));
    output.push_str("\n");

    output.push_str("Issuer:\n");
    output.push_str(&format!("  {}\n", invoice.issuer.name));
    if let Some(nip) = &invoice.issuer.nip {
        output.push_str(&format!("  NIP: {}\n", nip));
    }
    output.push_str(&format!("  {}\n", invoice.issuer.address.format()));
    output.push_str("\n");

    output.push_str("Receiver:\n");
    output.push_str(&format!("  {}\n", invoice.receiver.name));
    if let Some(nip) = &invoice.receiver.nip {
        output.push_str(&format!("  NIP: {}\n", nip));
    }
    output.push_str("\n");

    output.push_str("Summary:\n");
    output.push_str(&format!("  Net:   {} {}\n", invoice.summary.total_net, invoice.header.currency));
    output.push_str(&format!("  VAT:   {} {}\n", invoice.summary.total_vat, invoice.header.currency));
    output.push_str(&format!("  Gross: {} {}\n", invoice.summary.total_gross, invoice.header.currency));

    if let Some(due_date) = invoice.header.due_date {
        output.push_str(&format!("\nPayment due: {}\n", due_date));
    }

    Ok(output)
}
