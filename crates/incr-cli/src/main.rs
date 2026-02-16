//! CLI application for Polish invoice OCR processing.

mod commands;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use commands::{batch, config, models, process};

/// Polish invoice OCR - Extract structured data from Polish invoices
#[derive(Parser)]
#[command(name = "incr")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Path to config file
    #[arg(short, long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a single invoice file
    Process(process::ProcessArgs),

    /// Process multiple invoice files
    Batch(batch::BatchArgs),

    /// Manage OCR models
    Models(models::ModelsArgs),

    /// Manage configuration
    Config(config::ConfigArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Set up logging based on verbosity
    let level = match cli.verbose {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    // Execute command
    match cli.command {
        Commands::Process(args) => process::run(args, cli.config.as_deref()).await,
        Commands::Batch(args) => batch::run(args, cli.config.as_deref()).await,
        Commands::Models(args) => models::run(args).await,
        Commands::Config(args) => config::run(args).await,
    }
}
