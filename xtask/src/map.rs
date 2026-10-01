//! `sync/map.toml`: cada caminho do upstream recebe uma ação. Vale a primeira regra que casar.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use globset::{GlobBuilder, GlobMatcher};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Port,
    Vendor,
    Ignore,
}

#[derive(Deserialize)]
struct RawMap {
    #[serde(default)]
    rule: Vec<RawRule>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRule {
    path: String,
    action: Action,
    #[serde(rename = "crate")]
    krate: Option<String>,
}

pub struct Rule {
    pub action: Action,
    /// Crate dona do arquivo; `None` só para `ignore`.
    pub krate: Option<String>,
    matcher: GlobMatcher,
}

pub struct SyncMap {
    rules: Vec<Rule>,
}

impl SyncMap {
    pub fn parse(text: &str) -> Result<Self> {
        let raw: RawMap = toml::from_str(text).context("sync/map.toml inválido")?;
        let mut rules = Vec::with_capacity(raw.rule.len());
        for r in raw.rule {
            match (r.action, r.krate.is_some()) {
                (Action::Ignore, true) => bail!("regra `{}`: ignore não leva crate", r.path),
                (Action::Port | Action::Vendor, false) => bail!("regra `{}`: falta crate", r.path),
                _ => {}
            }
            // `*` não atravessa `/`; `**` atravessa.
            let matcher = GlobBuilder::new(&r.path)
                .literal_separator(true)
                .build()
                .with_context(|| format!("glob inválido: {}", r.path))?
                .compile_matcher();
            rules.push(Rule {
                action: r.action,
                krate: r.krate,
                matcher,
            });
        }
        Ok(Self { rules })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("lendo {}", path.display()))?;
        Self::parse(&text)
    }

    /// Nomes de crate citados pelas regras.
    pub fn crates(&self) -> BTreeSet<&str> {
        self.rules
            .iter()
            .filter_map(|r| r.krate.as_deref())
            .collect()
    }

    /// A primeira regra que casa com `path`.
    pub fn classify(&self, path: &str) -> Option<&Rule> {
        self.rules.iter().find(|r| r.matcher.is_match(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP: &str = r#"
[[rule]]
path = "src/models/qwen35.cpp"
action = "port"
crate = "llama"

[[rule]]
path = "src/models/*.cpp"
action = "ignore"

[[rule]]
path = "ggml/src/ggml-vulkan/vulkan-shaders/**"
action = "vendor"
crate = "ggml-vulkan"

[[rule]]
path = "ggml/include/{ggml.h,gguf.h}"
action = "port"
crate = "ggml"
"#;

    fn classify<'m>(map: &'m SyncMap, path: &str) -> Option<(Action, Option<&'m str>)> {
        map.classify(path).map(|r| (r.action, r.krate.as_deref()))
    }

    #[test]
    fn primeira_regra_vence() {
        let map = SyncMap::parse(MAP).unwrap();
        assert_eq!(
            classify(&map, "src/models/qwen35.cpp"),
            Some((Action::Port, Some("llama")))
        );
        assert_eq!(
            classify(&map, "src/models/gemma.cpp"),
            Some((Action::Ignore, None))
        );
    }

    #[test]
    fn asterisco_nao_atravessa_barra() {
        let map = SyncMap::parse(MAP).unwrap();
        assert_eq!(classify(&map, "src/models/sub/x.cpp"), None);
    }

    #[test]
    fn dois_asteriscos_atravessam_barra() {
        let map = SyncMap::parse(MAP).unwrap();
        assert_eq!(
            classify(&map, "ggml/src/ggml-vulkan/vulkan-shaders/a/b.comp"),
            Some((Action::Vendor, Some("ggml-vulkan")))
        );
    }

    #[test]
    fn chaves_alternam() {
        let map = SyncMap::parse(MAP).unwrap();
        assert_eq!(
            classify(&map, "ggml/include/gguf.h"),
            Some((Action::Port, Some("ggml")))
        );
        assert_eq!(classify(&map, "ggml/include/ggml-metal.h"), None);
    }

    #[test]
    fn crates_lista_os_nomes_das_regras() {
        let map = SyncMap::parse(MAP).unwrap();
        let nomes: Vec<&str> = map.crates().into_iter().collect();
        assert_eq!(nomes, vec!["ggml", "ggml-vulkan", "llama"]);
    }

    #[test]
    fn ignore_com_crate_e_erro() {
        let text = "[[rule]]\npath = \"a\"\naction = \"ignore\"\ncrate = \"x\"\n";
        let Err(e) = SyncMap::parse(text) else {
            panic!("devia falhar")
        };
        assert!(e.to_string().contains("ignore não leva crate"));
    }

    #[test]
    fn port_sem_crate_e_erro() {
        let Err(e) = SyncMap::parse("[[rule]]\npath = \"a\"\naction = \"port\"\n") else {
            panic!("devia falhar")
        };
        assert!(e.to_string().contains("falta crate"));
    }
}
