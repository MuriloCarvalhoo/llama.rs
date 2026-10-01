//! Automação do llama.rs: sync com o upstream, oráculo e CI.
//! Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md

mod ci;
mod paths;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::paths::Paths;

#[derive(Parser)]
#[command(name = "xtask", about = "Automação do llama.rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// fmt + clippy + testes
    Ci,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let paths = Paths::from_manifest();
    let result = match cli.cmd {
        Cmd::Ci => ci::run(&paths),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::FAILURE
        }
    }
}
