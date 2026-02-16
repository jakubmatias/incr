//! Config command - manage configuration.

use std::fs;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use console::style;

use incr_core::models::config::IncrConfig;

/// Arguments for the config command.
#[derive(Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    command: ConfigCommand,
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Show current configuration
    Show,

    /// Initialize a new configuration file
    Init(InitArgs),

    /// Get a specific configuration value
    Get {
        /// Configuration key (e.g., "ocr.detection_threshold")
        key: String,
    },

    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// New value
        value: String,
    },

    /// Show configuration file path
    Path,
}

#[derive(Args)]
struct InitArgs {
    /// Output path for configuration file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Overwrite existing file
    #[arg(long)]
    force: bool,
}

pub async fn run(args: ConfigArgs) -> anyhow::Result<()> {
    match args.command {
        ConfigCommand::Show => show_config(),
        ConfigCommand::Init(init_args) => init_config(init_args),
        ConfigCommand::Get { key } => get_config(&key),
        ConfigCommand::Set { key, value } => set_config(&key, &value),
        ConfigCommand::Path => show_path(),
    }
}

fn default_config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("incr")
        .join("config.json")
}

fn show_config() -> anyhow::Result<()> {
    let config_path = default_config_path();

    let config = if config_path.exists() {
        IncrConfig::from_file(&config_path)?
    } else {
        println!(
            "{} No config file found, showing defaults.",
            style("ℹ").blue()
        );
        IncrConfig::default()
    };

    println!("{}", serde_json::to_string_pretty(&config)?);

    Ok(())
}

fn init_config(args: InitArgs) -> anyhow::Result<()> {
    let output_path = args.output.unwrap_or_else(default_config_path);

    if output_path.exists() && !args.force {
        anyhow::bail!(
            "Config file already exists at {}. Use --force to overwrite.",
            output_path.display()
        );
    }

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let config = IncrConfig::default();
    config.save(&output_path)?;

    println!(
        "{} Created configuration file at {}",
        style("✓").green(),
        output_path.display()
    );

    Ok(())
}

fn get_config(key: &str) -> anyhow::Result<()> {
    let config_path = default_config_path();

    let config = if config_path.exists() {
        IncrConfig::from_file(&config_path)?
    } else {
        IncrConfig::default()
    };

    // Convert config to JSON for key lookup
    let json = serde_json::to_value(&config)?;

    // Navigate the key path
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = &json;

    for part in &parts {
        current = current.get(part).ok_or_else(|| {
            anyhow::anyhow!("Configuration key not found: {}", key)
        })?;
    }

    println!("{}", serde_json::to_string_pretty(current)?);

    Ok(())
}

fn set_config(key: &str, value: &str) -> anyhow::Result<()> {
    let config_path = default_config_path();

    let mut config = if config_path.exists() {
        IncrConfig::from_file(&config_path)?
    } else {
        // Create parent directory if needed
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        IncrConfig::default()
    };

    // Parse the value
    let parsed_value: serde_json::Value = serde_json::from_str(value)
        .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));

    // Convert config to JSON, modify, and convert back
    let mut json = serde_json::to_value(&config)?;

    // Navigate and set the key
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = &mut json;

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            // Last part - set the value
            if let Some(obj) = current.as_object_mut() {
                obj.insert((*part).to_string(), parsed_value.clone());
            } else {
                anyhow::bail!("Cannot set value at non-object path");
            }
        } else {
            // Navigate deeper
            current = current.get_mut(*part).ok_or_else(|| {
                anyhow::anyhow!("Configuration path not found: {}", key)
            })?;
        }
    }

    // Convert back to config
    config = serde_json::from_value(json)?;
    config.save(&config_path)?;

    println!(
        "{} Set {} = {}",
        style("✓").green(),
        key,
        serde_json::to_string(&parsed_value)?
    );

    Ok(())
}

fn show_path() -> anyhow::Result<()> {
    let config_path = default_config_path();

    println!("Configuration file: {}", config_path.display());

    if config_path.exists() {
        println!("Status: {}", style("exists").green());
    } else {
        println!("Status: {}", style("not created").yellow());
        println!();
        println!("Run 'incr config init' to create a configuration file.");
    }

    Ok(())
}
