//! Tarefas de sync em `sync/pending/`: um commit do upstream × uma crate.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::git::{self, Git};
use crate::paths::Paths;
use crate::state::State;

const OPEN: &str = "+++\n";
const CLOSE: &str = "\n+++\n";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskMeta {
    pub seq: u32,
    pub sha: String,
    #[serde(rename = "crate")]
    pub krate: String,
    pub subject: String,
    /// Arquivos a traduzir para Rust (inclui os removidos no upstream).
    pub port: Vec<String>,
    /// Arquivos a copiar verbatim para `vendor/upstream/`.
    pub vendor: Vec<String>,
    /// Arquivos vendor removidos no upstream.
    pub vendor_deleted: Vec<String>,
}

impl TaskMeta {
    pub fn file_name(&self) -> String {
        format!(
            "{:05}-{}-{}.md",
            self.seq,
            git::short(&self.sha),
            self.krate
        )
    }
}

/// Front matter TOML entre `+++`, seguido do corpo em Markdown.
pub fn render(meta: &TaskMeta, body: &str) -> Result<String> {
    let front = toml::to_string(meta).context("serializando tarefa")?;
    Ok(format!("{OPEN}{front}+++\n\n{body}"))
}

pub fn parse_meta(text: &str) -> Result<TaskMeta> {
    let rest = text.strip_prefix(OPEN).context("tarefa sem front matter")?;
    let (front, _) = rest
        .split_once(CLOSE)
        .context("front matter sem fechamento")?;
    toml::from_str(front).context("front matter inválido")
}

pub fn read_meta(path: &Path) -> Result<TaskMeta> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("lendo {}", path.display()))?;
    parse_meta(&text).with_context(|| format!("em {}", path.display()))
}

/// Tarefas pendentes, em ordem de `seq`.
pub fn pending(dir: &Path) -> Result<Vec<(TaskMeta, PathBuf)>> {
    let mut tasks = Vec::new();
    if !dir.exists() {
        return Ok(tasks);
    }
    for entry in std::fs::read_dir(dir).with_context(|| format!("lendo {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "md") {
            tasks.push((read_meta(&path)?, path));
        }
    }
    tasks.sort_unstable_by_key(|(meta, _)| meta.seq);
    Ok(tasks)
}

/// Tudo antes da tarefa pendente mais antiga está feito; sem pendências, tudo até o cursor.
pub fn recompute_synced(git: &Git, cursor: &str, pending_dir: &Path) -> Result<String> {
    match pending(pending_dir)?.first() {
        Some((oldest, _)) => git.rev_parse(&format!("{}^", oldest.sha)),
        None => Ok(cursor.to_owned()),
    }
}

/// `cargo xtask task apply`: traz os arquivos vendor da tarefa na versão do commit dela.
pub fn apply(paths: &Paths, file: &Path) -> Result<()> {
    let meta = read_meta(file)?;
    let state = State::load(&paths.upstream_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    let vendor = paths.vendor_dir();
    for path in &meta.vendor {
        let dest = vendor.join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("criando {}", parent.display()))?;
        }
        std::fs::write(&dest, git.show(&meta.sha, path)?)
            .with_context(|| format!("gravando {}", dest.display()))?;
        println!("vendor: {path}");
    }
    for path in &meta.vendor_deleted {
        let dest = vendor.join(path);
        if dest.exists() {
            std::fs::remove_file(&dest).with_context(|| format!("removendo {}", dest.display()))?;
            println!("vendor removido: {path}");
        }
    }
    Ok(())
}

/// `cargo xtask task done`: fecha a tarefa e recalcula o `synced`.
pub fn done(paths: &Paths, file: &Path) -> Result<()> {
    let meta = read_meta(file)?;
    let pending_dir = paths.pending_dir();
    let older: Vec<String> = pending(&pending_dir)?
        .iter()
        .filter(|(m, _)| m.krate == meta.krate && m.seq < meta.seq)
        .map(|(m, _)| m.file_name())
        .collect();
    if !older.is_empty() {
        bail!(
            "feche antes as tarefas anteriores da crate {}: {}",
            meta.krate,
            older.join(", ")
        );
    }
    std::fs::remove_file(file).with_context(|| format!("removendo {}", file.display()))?;
    let mut state = State::load(&paths.upstream_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    state.upstream.synced = recompute_synced(&git, &state.upstream.cursor, &pending_dir)?;
    state.save(&paths.upstream_toml())?;
    println!("synced = {}", git::short(&state.upstream.synced));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Fixture;

    #[test]
    fn front_matter_ida_e_volta() {
        let meta = TaskMeta {
            seq: 7,
            sha: "0123456789abcdef".into(),
            krate: "ggml-vulkan".into(),
            subject: "vulkan: \"x\" +++".into(),
            port: vec!["a.cpp".into()],
            vendor: vec![],
            vendor_deleted: vec![],
        };
        let text = render(&meta, "# corpo\n").unwrap();
        assert_eq!(parse_meta(&text).unwrap(), meta);
        assert_eq!(meta.file_name(), "00007-0123456-ggml-vulkan.md");
    }

    #[test]
    fn apply_e_done_seguem_a_ordem_dos_commits() {
        let fx = Fixture::new();
        fx.upstream.write("shaders/t.comp", "v1\n");
        let c1 = fx.upstream.commit("c1");
        fx.upstream.write("shaders/t.comp", "v2\n");
        fx.upstream.remove("shaders/s.comp");
        let c2 = fx.upstream.commit("c2");
        let paths = fx.paths();
        crate::sync::run(&paths, false).unwrap();
        let tasks = pending(&paths.pending_dir()).unwrap();
        assert_eq!(tasks.len(), 2);
        let (first, second) = (&tasks[0].1, &tasks[1].1);

        let Err(e) = done(&paths, second) else {
            panic!("fechar fora de ordem devia falhar")
        };
        assert!(e.to_string().contains("tarefas anteriores"));

        let t = paths.vendor_dir().join("shaders/t.comp");
        apply(&paths, first).unwrap();
        assert_eq!(std::fs::read_to_string(&t).unwrap(), "v1\n");
        done(&paths, first).unwrap();
        assert_eq!(
            State::load(&paths.upstream_toml()).unwrap().upstream.synced,
            c1
        );

        let s = paths.vendor_dir().join("shaders/s.comp");
        std::fs::write(&s, "antigo\n").unwrap();
        apply(&paths, second).unwrap();
        assert_eq!(std::fs::read_to_string(&t).unwrap(), "v2\n");
        assert!(!s.exists());
        done(&paths, second).unwrap();
        assert_eq!(
            State::load(&paths.upstream_toml()).unwrap().upstream.synced,
            c2
        );
    }
}
