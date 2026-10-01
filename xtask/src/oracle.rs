//! `cargo xtask oracle-build`: compila o upstream numa revisão para servir de referência.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::git;
use crate::paths::Paths;
use crate::state::State;

/// Artefatos que os testes de comparação usam (spec §5).
const EXPECTED: [&str; 10] = [
    "bin/libggml-base.so",
    "bin/libggml-cpu.so",
    "bin/libggml-vulkan.so",
    "bin/libllama.so",
    "bin/llama-tokenize",
    "bin/llama-cli",
    "bin/llama-server",
    "bin/llama-perplexity",
    "bin/llama-bench",
    "bin/test-backend-ops",
];

pub fn cmake_args(src: &Path, build: &Path) -> Vec<String> {
    let mut args = vec![
        "-S".to_owned(),
        src.display().to_string(),
        "-B".to_owned(),
        build.display().to_string(),
    ];
    args.extend(
        [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_VULKAN=ON",
            "-DLLAMA_BUILD_TESTS=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_OPENSSL=OFF",
        ]
        .map(str::to_owned),
    );
    args
}

fn cmake(args: &[String]) -> Result<()> {
    let status = Command::new("cmake")
        .args(args)
        .status()
        .context("executando cmake (está instalado?)")?;
    if !status.success() {
        bail!("cmake {} falhou ({status})", args.join(" "));
    }
    Ok(())
}

/// Devolve o diretório de build da revisão; só compila se ainda não houver build completo.
pub fn run(paths: &Paths, rev: Option<&str>) -> Result<PathBuf> {
    let state = State::load(&paths.upstream_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    let sha = git.rev_parse(rev.unwrap_or(&state.upstream.synced))?;
    let sha7 = git::short(&sha);
    let src = paths.upstream_dir().join(format!("src-{sha7}"));
    let build = paths.upstream_dir().join(format!("build-{sha7}"));
    let stamp = build.join(".ok");
    if stamp.exists() {
        println!("oráculo já compilado: {}", build.display());
        return Ok(build);
    }
    if !src.exists() {
        git.worktree_add(&src, &sha)?;
    }
    cmake(&cmake_args(&src, &build))?;
    let jobs = std::thread::available_parallelism().map_or(4, usize::from);
    cmake(&[
        "--build".to_owned(),
        build.display().to_string(),
        "--parallel".to_owned(),
        jobs.to_string(),
    ])?;
    let missing: Vec<&str> = EXPECTED
        .into_iter()
        .filter(|f| !build.join(f).exists())
        .collect();
    if !missing.is_empty() {
        bail!("build sem: {}", missing.join(", "));
    }
    std::fs::write(&stamp, &sha).with_context(|| format!("gravando {}", stamp.display()))?;
    println!("oráculo pronto: {}", build.display());
    Ok(build)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmake_liga_vulkan_testes_e_libs_compartilhadas() {
        let args = cmake_args(Path::new("/s"), Path::new("/b"));
        assert_eq!(&args[..4], ["-S", "/s", "-B", "/b"]);
        for flag in [
            "-DGGML_VULKAN=ON",
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLAMA_BUILD_TESTS=ON",
        ] {
            assert!(args.iter().any(|a| a == flag), "falta {flag}");
        }
    }
}
