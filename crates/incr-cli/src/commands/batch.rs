//! Batch processing command for multiple invoice files.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use console::style;
use glob::glob;
use image::DynamicImage;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::{debug, error, warn};

use incr_core::models::config::IncrConfig;
use incr_core::models::invoice::Invoice;
use incr_core::invoice::{HybridInvoiceParser, InvoiceParser};
use incr_core::pdf::{PdfExtractor, PdfProcessor};
use incr_core::{create_engine_from_dir, create_engine_from_embedded};

use super::models::{get_active_variant, get_variant_dir};

/// Arguments for the batch command.
#[derive(Args)]
pub struct BatchArgs {
    /// Input files or glob pattern
    #[arg(required = true)]
    input: String,

    /// Output directory
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Output format for each file
    #[arg(short, long, value_enum, default_value = "json")]
    format: super::process::OutputFormat,

    /// Also generate a summary CSV
    #[arg(long)]
    summary: bool,

    /// Number of parallel workers
    #[arg(short = 'j', long, default_value = "4")]
    jobs: usize,

    /// Continue on error
    #[arg(long)]
    continue_on_error: bool,

    /// Model directory
    #[arg(short, long)]
    model_dir: Option<PathBuf>,
}

/// Result of processing a single file.
struct ProcessResult {
    path: PathBuf,
    invoice: Option<Invoice>,
    error: Option<String>,
    processing_time_ms: u64,
}

pub async fn run(args: BatchArgs, config_path: Option<&str>) -> anyhow::Result<()> {
    let start = Instant::now();

    // Load configuration
    let config = if let Some(path) = config_path {
        IncrConfig::from_file(std::path::Path::new(path))?
    } else {
        IncrConfig::default()
    };

    // Expand glob pattern
    let files: Vec<PathBuf> = glob(&args.input)?
        .filter_map(|r| r.ok())
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            matches!(ext.to_lowercase().as_str(), "pdf" | "png" | "jpg" | "jpeg" | "tiff")
        })
        .collect();

    if files.is_empty() {
        anyhow::bail!("No matching files found for pattern: {}", args.input);
    }

    println!(
        "{} Found {} files to process",
        style("ℹ").blue(),
        files.len()
    );

    // Create output directory if specified
    if let Some(ref output_dir) = args.output_dir {
        fs::create_dir_all(output_dir)?;
    }

    // Set up progress bars
    let multi_progress = MultiProgress::new();
    let overall_pb = multi_progress.add(ProgressBar::new(files.len() as u64));
    overall_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Process files (simplified sequential processing for now)
    let mut results = Vec::with_capacity(files.len());
    let parser = HybridInvoiceParser::new()
        .with_nip_validation(config.extraction.validate_nip)
        .with_regon_validation(config.extraction.validate_regon)
        .with_iban_validation(config.extraction.validate_iban);

    for path in files {
        let file_start = Instant::now();
        let result = process_single_file(&path, &parser, &args, &config);

        let processing_time_ms = file_start.elapsed().as_millis() as u64;

        match result {
            Ok(invoice) => {
                results.push(ProcessResult {
                    path: path.clone(),
                    invoice: Some(invoice),
                    error: None,
                    processing_time_ms,
                });
            }
            Err(e) => {
                let error_msg = e.to_string();
                if args.continue_on_error {
                    warn!("Failed to process {}: {}", path.display(), error_msg);
                    results.push(ProcessResult {
                        path: path.clone(),
                        invoice: None,
                        error: Some(error_msg),
                        processing_time_ms,
                    });
                } else {
                    error!("Failed to process {}: {}", path.display(), error_msg);
                    anyhow::bail!("Processing failed: {}", error_msg);
                }
            }
        }

        overall_pb.inc(1);
    }

    overall_pb.finish_with_message("Complete");

    // Write outputs
    let successful: Vec<_> = results.iter().filter(|r| r.invoice.is_some()).collect();
    let failed: Vec<_> = results.iter().filter(|r| r.error.is_some()).collect();

    for result in &successful {
        if let (Some(invoice), Some(output_dir)) = (&result.invoice, &args.output_dir) {
            let output_name = result.path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("invoice");

            let extension = match args.format {
                super::process::OutputFormat::Json => "json",
                super::process::OutputFormat::Csv => "csv",
                super::process::OutputFormat::Text => "txt",
            };

            let output_path = output_dir.join(format!("{}.{}", output_name, extension));

            let content = match args.format {
                super::process::OutputFormat::Json => serde_json::to_string(invoice)?,
                super::process::OutputFormat::Csv => format_invoice_csv(invoice)?,
                super::process::OutputFormat::Text => format_invoice_text(invoice),
            };

            fs::write(&output_path, content)?;
            debug!("Wrote output to {}", output_path.display());
        }
    }

    // Generate summary if requested
    if args.summary {
        let summary_path = args.output_dir
            .as_ref()
            .map(|d| d.join("summary.csv"))
            .unwrap_or_else(|| PathBuf::from("summary.csv"));

        write_summary(&summary_path, &results)?;
        println!(
            "{} Summary written to {}",
            style("✓").green(),
            summary_path.display()
        );
    }

    // Print summary
    println!();
    println!(
        "{} Processed {} files in {:?}",
        style("✓").green(),
        results.len(),
        start.elapsed()
    );
    println!(
        "   {} successful, {} failed",
        style(successful.len()).green(),
        style(failed.len()).red()
    );

    if !failed.is_empty() {
        println!();
        println!("{}", style("Failed files:").red());
        for result in &failed {
            println!(
                "  - {}: {}",
                result.path.display(),
                result.error.as_deref().unwrap_or("unknown error")
            );
        }
    }

    Ok(())
}

fn process_single_file(
    path: &PathBuf,
    parser: &HybridInvoiceParser,
    args: &BatchArgs,
    config: &IncrConfig,
) -> anyhow::Result<Invoice> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "pdf" => {
            let data = fs::read(path)?;
            let mut extractor = PdfExtractor::new();
            extractor.load(&data)?;

            let text = extractor.extract_text()?;
            if text.trim().is_empty() {
                anyhow::bail!("No text extracted from PDF");
            }

            let result = parser.parse(&text)?;
            Ok(result.invoice)
        }
        "png" | "jpg" | "jpeg" | "webp" | "tiff" | "tif" | "bmp" => {
            // Process image with OCR
            let image = image::open(path)?;
            let text = run_ocr_on_image(&image, args, config)?;

            if text.trim().is_empty() {
                anyhow::bail!("No text detected in image");
            }

            let result = parser.parse(&text)?;
            let mut invoice = result.invoice;
            invoice.metadata.source_type = incr_core::models::invoice::SourceType::Image;
            Ok(invoice)
        }
        _ => {
            anyhow::bail!("Unsupported file format: {}", extension);
        }
    }
}

fn run_ocr_on_image(
    image: &DynamicImage,
    args: &BatchArgs,
    config: &IncrConfig,
) -> anyhow::Result<String> {
    // Get model directory
    let model_dir = args.model_dir.clone().unwrap_or_else(|| {
        get_variant_dir(get_active_variant())
    });

    // Try external models first, then embedded
    let det_model = model_dir.join(&config.models.detection_model);
    let engine = if det_model.exists() {
        debug!("Using external models from {}", model_dir.display());
        create_engine_from_dir(&model_dir, config.ocr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to load OCR models: {}", e))?
    } else {
        debug!("Using embedded mobile models");
        create_engine_from_embedded(config.ocr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to load embedded OCR models: {}", e))?
    };

    let result = engine
        .process(image)
        .map_err(|e| anyhow::anyhow!("OCR failed: {}", e))?;

    debug!(
        "OCR detected {} text boxes in {}ms",
        result.boxes.len(),
        result.processing_time_ms
    );

    Ok(result.text)
}

fn write_summary(path: &PathBuf, results: &[ProcessResult]) -> anyhow::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record([
        "filename",
        "status",
        "invoice_number",
        "issue_date",
        "issuer_name",
        "issuer_nip",
        "total_gross",
        "currency",
        "confidence",
        "processing_time_ms",
        "error",
    ])?;

    for result in results {
        let filename = result.path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        if let Some(invoice) = &result.invoice {
            wtr.write_record([
                filename,
                "success",
                &invoice.header.invoice_number,
                &invoice.header.issue_date.to_string(),
                &invoice.issuer.name,
                &invoice.issuer.nip.clone().unwrap_or_default(),
                &invoice.summary.total_gross.to_string(),
                &invoice.header.currency,
                &format!("{:.2}", invoice.metadata.confidence),
                &result.processing_time_ms.to_string(),
                "",
            ])?;
        } else {
            wtr.write_record([
                filename,
                "error",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                &result.processing_time_ms.to_string(),
                result.error.as_deref().unwrap_or(""),
            ])?;
        }
    }

    wtr.flush()?;
    Ok(())
}

fn format_invoice_csv(invoice: &Invoice) -> anyhow::Result<String> {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write header
    wtr.write_record([
        "invoice_number",
        "issue_date",
        "sale_date",
        "due_date",
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
        &invoice.header.sale_date.map(|d| d.to_string()).unwrap_or_default(),
        &invoice.header.due_date.map(|d| d.to_string()).unwrap_or_default(),
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

fn format_invoice_text(invoice: &Invoice) -> String {
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

    output
}
