//! Models command - download and manage OCR models.

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};
use console::style;
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Arguments for the models command.
#[derive(Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    command: ModelsCommand,
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// List available models
    List,

    /// Download models
    Download(DownloadArgs),

    /// Check model status
    Status(StatusArgs),

    /// Remove downloaded models
    Clean(CleanArgs),

    /// Set the active model variant
    Use(UseArgs),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelVariant {
    /// Mobile models - smaller, faster (~10MB)
    Mobile,
    /// Server models - better detection accuracy (~96MB)
    Server,
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelVariant::Mobile => write!(f, "mobile"),
            ModelVariant::Server => write!(f, "server"),
        }
    }
}

#[derive(Args)]
struct DownloadArgs {
    /// Model variant to download
    #[arg(short, long, value_enum, default_value = "mobile")]
    variant: ModelVariant,

    /// Output directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Force re-download even if files exist
    #[arg(long)]
    force: bool,

    /// Use mirror URL (for users in China)
    #[arg(long)]
    mirror: bool,
}

#[derive(Args)]
struct StatusArgs {
    /// Check specific variant only
    #[arg(short, long, value_enum)]
    variant: Option<ModelVariant>,
}

#[derive(Args)]
struct CleanArgs {
    /// Clean specific variant only
    #[arg(short, long, value_enum)]
    variant: Option<ModelVariant>,

    /// Clean all variants
    #[arg(long)]
    all: bool,
}

#[derive(Args)]
struct UseArgs {
    /// Variant to set as active
    #[arg(value_enum)]
    variant: ModelVariant,
}

/// Model information with download URLs.
#[derive(Clone)]
struct ModelInfo {
    filename: &'static str,
    size_bytes: u64,
    description: &'static str,
    url: &'static str,
    mirror_url: &'static str,
}

/// Model variant configuration
struct VariantConfig {
    detection: ModelInfo,
    recognition: ModelInfo,
    dictionary: ModelInfo,
    layout: Option<ModelInfo>,
    table: Option<ModelInfo>,
}

fn get_variant_config(variant: ModelVariant) -> VariantConfig {
    // Models are downloaded from: https://github.com/jakubmatias/incr/tree/main/models
    match variant {
        ModelVariant::Mobile => VariantConfig {
            detection: ModelInfo {
                filename: "det.onnx",
                size_bytes: 4_500_000,
                description: "PP-OCRv3 mobile detection",
                url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/det.onnx",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/det.onnx",
            },
            recognition: ModelInfo {
                filename: "latin_rec.onnx",
                size_bytes: 7_500_000,
                description: "Latin recognition",
                url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/latin_rec.onnx",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/latin_rec.onnx",
            },
            dictionary: ModelInfo {
                filename: "latin_dict.txt",
                size_bytes: 2_000,
                description: "Latin character dictionary",
                url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/latin_dict.txt",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/mobile/latin_dict.txt",
            },
            layout: None,
            table: None,
        },
        ModelVariant::Server => VariantConfig {
            detection: ModelInfo {
                filename: "det.onnx",
                size_bytes: 84_000_000,
                description: "PP-OCRv5 server detection",
                url: "https://github.com/jakubmatias/incr/raw/main/models/server/det.onnx",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/server/det.onnx",
            },
            recognition: ModelInfo {
                filename: "latin_rec.onnx",
                size_bytes: 7_500_000,
                description: "Latin recognition",
                url: "https://github.com/jakubmatias/incr/raw/main/models/server/latin_rec.onnx",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/server/latin_rec.onnx",
            },
            dictionary: ModelInfo {
                filename: "latin_dict.txt",
                size_bytes: 2_000,
                description: "Latin character dictionary",
                url: "https://github.com/jakubmatias/incr/raw/main/models/server/latin_dict.txt",
                mirror_url: "https://github.com/jakubmatias/incr/raw/main/models/server/latin_dict.txt",
            },
            layout: None,
            table: None,
        },
    }
}

/// Get the model directory for a specific variant
pub fn get_variant_dir(variant: ModelVariant) -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("incr")
        .join("models")
        .join(variant.to_string())
}

/// Get the active variant from config file
pub fn get_active_variant() -> ModelVariant {
    let config_path = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("incr")
        .join("active_variant");

    if let Ok(content) = fs::read_to_string(&config_path) {
        match content.trim() {
            "server" => ModelVariant::Server,
            _ => ModelVariant::Mobile,
        }
    } else {
        ModelVariant::Mobile
    }
}

/// Set the active variant
fn set_active_variant(variant: ModelVariant) -> anyhow::Result<()> {
    let config_dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("incr");
    fs::create_dir_all(&config_dir)?;

    let config_path = config_dir.join("active_variant");
    fs::write(&config_path, variant.to_string())?;
    Ok(())
}

pub async fn run(args: ModelsArgs) -> anyhow::Result<()> {
    match args.command {
        ModelsCommand::List => list_models(),
        ModelsCommand::Download(download_args) => download_models(download_args).await,
        ModelsCommand::Status(status_args) => check_status(status_args),
        ModelsCommand::Clean(clean_args) => clean_models(clean_args),
        ModelsCommand::Use(use_args) => use_variant(use_args),
    }
}

fn list_models() -> anyhow::Result<()> {
    println!("{}", style("Available Model Variants").bold());
    println!();

    let active = get_active_variant();

    for variant in [ModelVariant::Mobile, ModelVariant::Server] {
        let config = get_variant_config(variant);
        let is_active = variant == active;
        let active_marker = if is_active { " (active)" } else { "" };

        let mut total_size = config.detection.size_bytes + config.recognition.size_bytes + config.dictionary.size_bytes;
        if let Some(ref layout) = config.layout {
            total_size += layout.size_bytes;
        }
        if let Some(ref table) = config.table {
            total_size += table.size_bytes;
        }

        let desc = match variant {
            ModelVariant::Mobile => "- faster, smaller",
            ModelVariant::Server => "- better detection accuracy",
        };

        println!(
            "{} {} {}{}",
            style(format!("▸ {}", variant)).bold().cyan(),
            format_size(total_size),
            style(desc).dim(),
            style(active_marker).green().bold()
        );

        // Core OCR models
        for model in [&config.detection, &config.recognition, &config.dictionary] {
            println!(
                "    {:<20} {:>10}  {}",
                model.filename,
                format_size(model.size_bytes),
                model.description
            );
        }

        // Structure models (PP-Structure)
        if let Some(ref layout) = config.layout {
            println!(
                "    {:<20} {:>10}  {}",
                layout.filename,
                format_size(layout.size_bytes),
                layout.description
            );
        }
        if let Some(ref table) = config.table {
            println!(
                "    {:<20} {:>10}  {}",
                table.filename,
                format_size(table.size_bytes),
                table.description
            );
        }
        println!();
    }

    println!("Commands:");
    println!("  incr models download -v mobile    Download mobile models (~18MB)");
    println!("  incr models download -v server    Download server models (~103MB)");
    println!("  incr models use <variant>         Switch active variant");

    Ok(())
}

fn use_variant(args: UseArgs) -> anyhow::Result<()> {
    let variant_dir = get_variant_dir(args.variant);

    // Check if variant is downloaded
    let config = get_variant_config(args.variant);
    let det_exists = variant_dir.join(&config.detection.filename).exists();
    let rec_exists = variant_dir.join(&config.recognition.filename).exists();

    if !det_exists || !rec_exists {
        println!(
            "{} {} models not downloaded yet.",
            style("⚠").yellow(),
            args.variant
        );
        println!("Run: incr models download -v {}", args.variant);
        return Ok(());
    }

    set_active_variant(args.variant)?;
    println!(
        "{} Switched to {} models",
        style("✓").green(),
        style(args.variant.to_string()).cyan().bold()
    );

    Ok(())
}

async fn download_models(args: DownloadArgs) -> anyhow::Result<()> {
    let variant = args.variant;
    let config = get_variant_config(variant);

    let output_dir = args.output.unwrap_or_else(|| get_variant_dir(variant));
    fs::create_dir_all(&output_dir)?;

    println!(
        "{} Downloading {} models to {}",
        style("ℹ").blue(),
        style(variant.to_string()).cyan().bold(),
        output_dir.display()
    );
    println!();

    let client = reqwest::Client::builder()
        .user_agent("incr-cli/0.1.0")
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let multi_progress = MultiProgress::new();
    let mut success_count = 0;
    let mut skip_count = 0;
    let mut error_count = 0;

    // Collect all models to download
    let mut models: Vec<&ModelInfo> = vec![&config.detection, &config.recognition, &config.dictionary];
    if let Some(ref layout) = config.layout {
        models.push(layout);
    }
    if let Some(ref table) = config.table {
        models.push(table);
    }

    for model in models {
        let path = output_dir.join(model.filename);

        // Check if already exists
        if path.exists() && !args.force {
            let metadata = fs::metadata(&path)?;
            // Check if file size is reasonable (at least 50% of expected)
            if metadata.len() > model.size_bytes / 2 {
                println!(
                    "  {} {} (already exists, {})",
                    style("✓").green(),
                    model.filename,
                    format_size(metadata.len())
                );
                skip_count += 1;
                continue;
            }
        }

        // Select URL based on mirror flag
        let url = if args.mirror {
            model.mirror_url
        } else {
            model.url
        };

        // Create progress bar
        let pb = multi_progress.add(ProgressBar::new(model.size_bytes));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.green} {msg:<30} [{bar:25.cyan/blue}] {bytes}/{total_bytes}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_message(model.filename.to_string());

        // Download
        match download_file(&client, url, &path, &pb).await {
            Ok(()) => {
                pb.finish_with_message(format!("{} {}", style("✓").green(), model.filename));
                success_count += 1;
            }
            Err(e) => {
                pb.finish_with_message(format!("{} {} - {}", style("✗").red(), model.filename, e));
                error_count += 1;

                // Try mirror if primary failed
                if !args.mirror {
                    println!(
                        "    {} Trying mirror...",
                        style("↻").yellow()
                    );
                    let pb2 = multi_progress.add(ProgressBar::new(model.size_bytes));
                    pb2.set_style(
                        ProgressStyle::default_bar()
                            .template("    {spinner:.green} {msg:<28} [{bar:23.cyan/blue}] {bytes}/{total_bytes}")
                            .unwrap()
                            .progress_chars("=>-"),
                    );
                    pb2.set_message(format!("(mirror) {}", model.filename));

                    match download_file(&client, model.mirror_url, &path, &pb2).await {
                        Ok(()) => {
                            pb2.finish_with_message(format!(
                                "{} {} (from mirror)",
                                style("✓").green(),
                                model.filename
                            ));
                            error_count -= 1;
                            success_count += 1;
                        }
                        Err(e2) => {
                            pb2.finish_with_message(format!(
                                "{} {} - mirror also failed: {}",
                                style("✗").red(),
                                model.filename,
                                e2
                            ));
                        }
                    }
                }
            }
        }
    }

    println!();

    // Summary
    if error_count == 0 {
        println!(
            "{} {} models downloaded successfully!",
            style("✓").green().bold(),
            variant
        );
        if skip_count > 0 {
            println!(
                "   {} downloaded, {} already present",
                success_count, skip_count
            );
        }

        // Set as active if this is the first download
        let active = get_active_variant();
        if active != variant {
            println!();
            println!(
                "{} To use these models, run: incr models use {}",
                style("ℹ").blue(),
                variant
            );
        }
    } else {
        println!(
            "{} Download completed with errors",
            style("⚠").yellow().bold()
        );
        println!(
            "   {} downloaded, {} skipped, {} failed",
            success_count, skip_count, error_count
        );
        println!();
        println!("For failed downloads, you can:");
        println!("  1. Retry with: incr models download -v {} --force", variant);
        println!("  2. Try mirror: incr models download -v {} --mirror", variant);
    }

    // Verify all models
    println!();
    check_status(StatusArgs { variant: Some(variant) })?;

    Ok(())
}

async fn download_file(
    client: &reqwest::Client,
    url: &str,
    path: &PathBuf,
    pb: &ProgressBar,
) -> anyhow::Result<()> {
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {}", response.status());
    }

    // Get content length if available
    if let Some(content_length) = response.content_length() {
        pb.set_length(content_length);
    }

    // Create temp file first
    let temp_path = path.with_extension("tmp");
    let mut file = File::create(&temp_path)?;

    // Stream download with progress
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);

    // Rename temp to final
    fs::rename(&temp_path, path)?;

    Ok(())
}

fn check_status(args: StatusArgs) -> anyhow::Result<()> {
    let active = get_active_variant();

    println!("{}", style("Model Status").bold());
    println!("Active variant: {}", style(active.to_string()).cyan().bold());
    println!();

    let variants: Vec<ModelVariant> = if let Some(v) = args.variant {
        vec![v]
    } else {
        vec![ModelVariant::Mobile, ModelVariant::Server]
    };

    for variant in variants {
        let config = get_variant_config(variant);
        let model_dir = get_variant_dir(variant);
        let is_active = variant == active;

        let active_marker = if is_active {
            style(" ◀ active").green().to_string()
        } else {
            String::new()
        };

        println!(
            "{} {}{}",
            style(format!("▸ {}", variant)).bold(),
            model_dir.display(),
            active_marker
        );

        // Collect all models to check
        let mut models: Vec<&ModelInfo> = vec![&config.detection, &config.recognition, &config.dictionary];
        if let Some(ref layout) = config.layout {
            models.push(layout);
        }
        if let Some(ref table) = config.table {
            models.push(table);
        }

        let mut all_present = true;
        let mut total_size: u64 = 0;

        for model in models {
            let path = model_dir.join(model.filename);
            let (status, size_str, valid) = if path.exists() {
                let metadata = fs::metadata(&path)?;
                let size = metadata.len();
                total_size += size;

                let valid = size > model.size_bytes / 2;
                if valid {
                    (style("✓").green(), format_size(size), true)
                } else {
                    (
                        style("⚠").yellow(),
                        format!("{} (incomplete?)", format_size(size)),
                        false,
                    )
                }
            } else {
                all_present = false;
                (style("✗").red(), "missing".to_string(), false)
            };

            if !valid {
                all_present = false;
            }

            println!("    {} {:<25} {:>10}", status, model.filename, size_str);
        }

        if all_present {
            println!(
                "    {} Ready ({} total)",
                style("✓").green(),
                format_size(total_size)
            );
        } else {
            println!(
                "    {} Run 'incr models download -v {}' to download",
                style("⚠").yellow(),
                variant
            );
        }
        println!();
    }

    Ok(())
}

fn clean_models(args: CleanArgs) -> anyhow::Result<()> {
    let variants: Vec<ModelVariant> = if args.all {
        vec![ModelVariant::Mobile, ModelVariant::Server]
    } else if let Some(v) = args.variant {
        vec![v]
    } else {
        println!(
            "{} Specify --all to remove all models or -v <variant> for specific variant",
            style("ℹ").blue()
        );
        return Ok(());
    };

    let mut total_removed = 0;
    let mut total_freed: u64 = 0;

    for variant in variants {
        let model_dir = get_variant_dir(variant);

        if !model_dir.exists() {
            continue;
        }

        println!(
            "{} Cleaning {} models...",
            style("⚠").yellow(),
            variant
        );

        let config = get_variant_config(variant);

        // Collect all models to clean
        let mut models: Vec<&ModelInfo> = vec![&config.detection, &config.recognition, &config.dictionary];
        if let Some(ref layout) = config.layout {
            models.push(layout);
        }
        if let Some(ref table) = config.table {
            models.push(table);
        }

        for model in models {
            let path = model_dir.join(model.filename);
            if path.exists() {
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                fs::remove_file(&path)?;
                total_removed += 1;
                total_freed += size;
                println!("  {} Removed {}", style("✓").green(), model.filename);
            }
        }

        // Also remove any .tmp files
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "tmp").unwrap_or(false) {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }

    if total_removed > 0 {
        println!();
        println!(
            "{} Removed {} files, freed {}",
            style("✓").green(),
            total_removed,
            format_size(total_freed)
        );
    } else {
        println!("{} No model files to remove.", style("ℹ").blue());
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1}KB", bytes as f64 / 1_000.0)
    } else {
        format!("{}B", bytes)
    }
}
