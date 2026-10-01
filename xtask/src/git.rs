//! Chamadas ao `git` do sistema. Sem libgit2: o xtask só lê o histórico e faz fetch.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Os 7 primeiros caracteres de um SHA.
pub fn short(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

/// `git` sem prompt de credencial: rodando sem terminal, um prompt seria um travamento.
fn git_command() -> Command {
    let mut cmd = Command::new("git");
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    cmd
}

/// O clone do upstream em `.upstream/llama.cpp`; clona na primeira chamada.
///
/// Clone completo, sem `--filter`: um clone parcial busca objeto por objeto sob demanda, e
/// um clone parcial interrompido vira um laço de buscas que não termina. O clone é feito em
/// `<dir>.tmp` e renomeado no fim, então um clone interrompido nunca passa por pronto.
pub fn open_upstream(clone_dir: &Path, url: &str) -> Result<Git> {
    if clone_dir.join(".git").exists() {
        return Ok(Git::new(clone_dir));
    }
    let mut tmp = clone_dir.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).with_context(|| format!("removendo {}", tmp.display()))?;
    }
    if let Some(parent) = clone_dir.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("criando {}", parent.display()))?;
    }
    eprintln!("clonando {url} em {}...", clone_dir.display());
    let out = git_command()
        .args(["clone", "--quiet", url])
        .arg(&tmp)
        .output()
        .context("executando git clone")?;
    if !out.status.success() {
        bail!(
            "git clone {url}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    std::fs::rename(&tmp, clone_dir)
        .with_context(|| format!("renomeando {} para {}", tmp.display(), clone_dir.display()))?;
    Ok(Git::new(clone_dir))
}

pub struct Git {
    dir: PathBuf,
}

impl Git {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn run(&self, args: &[&str]) -> Result<Vec<u8>> {
        let out = git_command()
            .arg("-C")
            .arg(&self.dir)
            .args(args)
            .output()
            .with_context(|| format!("executando git {}", args.join(" ")))?;
        if !out.status.success() {
            bail!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(out.stdout)
    }

    fn run_text(&self, args: &[&str]) -> Result<String> {
        String::from_utf8(self.run(args)?).context("saída do git não é UTF-8")
    }

    /// Todos os arquivos da árvore em `rev`.
    pub fn ls_tree(&self, rev: &str) -> Result<Vec<String>> {
        let out = self.run_text(&["ls-tree", "-r", "-z", "--name-only", rev])?;
        Ok(out
            .split('\0')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempRepo;

    #[test]
    fn open_upstream_faz_clone_completo_e_descarta_clone_interrompido() {
        let upstream = TempRepo::init();
        upstream.write("a.txt", "1");
        let sha = upstream.commit("c1");
        let root = tempfile::tempdir().unwrap();
        let clone_dir = root.path().join("llama.cpp");
        let leftover = root.path().join("llama.cpp.tmp");
        std::fs::create_dir_all(leftover.join(".git")).unwrap();

        let git = open_upstream(&clone_dir, upstream.path().to_str().unwrap()).unwrap();

        assert!(!leftover.exists());
        assert_eq!(git.ls_tree(&sha).unwrap(), vec!["a.txt"]);
        // Clone parcial busca objeto por objeto sob demanda: não pode ser promisor.
        let Err(_) = git.run(&["config", "--get", "remote.origin.promisor"]) else {
            panic!("clone não pode ser parcial")
        };
    }

    #[test]
    fn ls_tree_lista_arquivos_da_revisao() {
        let repo = TempRepo::init();
        repo.write("a.txt", "1");
        repo.write("dir/b.txt", "2");
        let sha = repo.commit("c1");
        let git = Git::new(repo.path());
        assert_eq!(git.ls_tree(&sha).unwrap(), vec!["a.txt", "dir/b.txt"]);
    }
}
