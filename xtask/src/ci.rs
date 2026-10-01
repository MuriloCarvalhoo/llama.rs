//! `cargo xtask ci`: o portão de toda tarefa de porte e de sync.

use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::paths::Paths;

const CARGO_STEPS: [&[&str]; 3] = [
    &["fmt", "--all", "--check"],
    &[
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ],
    &["test", "--workspace"],
];

pub fn run(paths: &Paths) -> Result<()> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    for args in CARGO_STEPS {
        let line = args.join(" ");
        eprintln!("==> cargo {line}");
        let status = Command::new(&cargo)
            .args(args)
            .current_dir(paths.root())
            .status()
            .with_context(|| format!("executando cargo {line}"))?;
        if !status.success() {
            bail!("falhou: cargo {line}");
        }
    }
    Ok(())
}
