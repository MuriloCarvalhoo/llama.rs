//! Automação do llama.rs: sync com o upstream, oráculo e CI.
//! Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md

mod check_map;
mod ci;
mod git;
mod map;
mod paths;
mod state;
mod sync;
mod task;
#[cfg(test)]
mod testutil;

use std::path::PathBuf;
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
    /// Gera tarefas de porte para os commits novos da master do upstream
    Sync {
        /// Só mostra a classificação dos commits, sem gravar nada
        #[arg(long)]
        dry_run: bool,
    },
    /// Opera uma tarefa de sync/pending/
    Task {
        #[command(subcommand)]
        cmd: TaskCmd,
    },
}

#[derive(Subcommand)]
enum TaskCmd {
    /// Copia os arquivos vendor da tarefa na versão do commit dela
    Apply { file: PathBuf },
    /// Fecha a tarefa e recalcula o synced
    Done { file: PathBuf },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let paths = Paths::from_manifest();
    let result = match cli.cmd {
        Cmd::Ci => ci::run(&paths),
        Cmd::CheckMap { rev } => check_map::run(&paths, rev.as_deref()),
        Cmd::Sync { dry_run } => sync::run(&paths, dry_run),
        Cmd::Task {
            cmd: TaskCmd::Apply { file },
        } => task::apply(&paths, &file),
        Cmd::Task {
            cmd: TaskCmd::Done { file },
        } => task::done(&paths, &file),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::FAILURE
        }
    }
}
