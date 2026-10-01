//! `UPSTREAM.toml`: onde o porte está em relação ao upstream.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct State {
    pub upstream: Upstream,
}

#[derive(Deserialize)]
pub struct Upstream {
    pub repo: String,
    /// Último commit para o qual o `xtask sync` já gerou tarefas.
    pub cursor: String,
}

impl State {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("lendo {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("{} inválido", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn le_upstream_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("UPSTREAM.toml");
        std::fs::write(
            &path,
            "[upstream]\nrepo = \"r\"\ncursor = \"c\"\nsynced = \"s\"\n",
        )
        .unwrap();
        let state = State::load(&path).unwrap();
        assert_eq!(state.upstream.repo, "r");
        assert_eq!(state.upstream.cursor, "c");
    }
}
