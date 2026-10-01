//! Automação do llama.rs: sync com o upstream, oráculo e CI.
//! Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md

mod check_map;
mod ci;
mod git;
mod map;
mod paths;
mod state;
#[cfg(test)]
mod testutil;

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
    /// fmt + clippy + testes + check-map
    Ci,
    /// Confere se todo arquivo do upstream casa uma regra de sync/map.toml
    CheckMap {
        /// Revisão do upstream (padrão: o cursor de UPSTREAM.toml)
        #[arg(long)]
        rev: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let paths = Paths::from_manifest();
    let result = match cli.cmd {
        Cmd::Ci => ci::run(&paths),
        Cmd::CheckMap { rev } => check_map::run(&paths, rev.as_deref()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::FAILURE
        }
    }
}
