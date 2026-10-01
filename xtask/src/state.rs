//! `UPSTREAM.toml`: onde o porte está em relação ao upstream.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const HEADER: &str = "# Lido e gravado pelo `cargo xtask`. Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md §4.\n\n";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub upstream: Upstream,
    #[serde(default)]
    pub crates: BTreeMap<String, CrateState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upstream {
    pub repo: String,
    /// Último commit para o qual o `xtask sync` já gerou tarefas.
    pub cursor: String,
    /// Todas as tarefas até este commit estão concluídas.
    pub synced: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrateState {
    /// SHA que o porte inicial desta crate mira.
    pub baseline: String,
    pub status: CrateStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CrateStatus {
    Porting,
    Synced,
}

impl State {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("lendo {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("{} inválido", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let body = toml::to_string_pretty(self).context("serializando UPSTREAM.toml")?;
        std::fs::write(path, format!("{HEADER}{body}"))
            .with_context(|| format!("gravando {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grava_e_le_de_volta() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("UPSTREAM.toml");
        let mut state = State {
            upstream: Upstream {
                repo: "https://example.com/u".into(),
                cursor: "a".repeat(40),
                synced: "b".repeat(40),
            },
            crates: BTreeMap::new(),
        };
        state.crates.insert(
            "ggml-cpu".into(),
            CrateState {
                baseline: "c".repeat(40),
                status: CrateStatus::Porting,
            },
        );
        state.save(&path).unwrap();
        assert_eq!(State::load(&path).unwrap(), state);
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("[crates.ggml-cpu]")
        );
    }

    #[test]
    fn campo_ou_tabela_desconhecida_e_erro() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("UPSTREAM.toml");
        let base = format!(
            "[upstream]\nrepo = \"r\"\ncursor = \"{}\"\nsynced = \"{}\"\n",
            "a".repeat(40),
            "b".repeat(40)
        );
        let casos = [
            // tabela no singular
            format!("{base}[crate.ggml]\nbaseline = \"c\"\nstatus = \"porting\"\n"),
            // campo desconhecido em upstream
            base.replace("repo = \"r\"", "repo = \"r\"\nextra = 1"),
            // campo desconhecido em crate
            format!("{base}[crates.ggml]\nbaseline = \"c\"\nstatus = \"porting\"\nextra = 1\n"),
        ];
        for text in casos {
            std::fs::write(&path, &text).unwrap();
            assert!(State::load(&path).is_err(), "devia falhar:\n{text}");
        }
    }
}
