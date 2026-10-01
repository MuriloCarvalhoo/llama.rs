//! Chamadas ao `git` do sistema. Sem libgit2: o xtask só lê o histórico e faz fetch.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Uma mudança de arquivo num commit, relativa ao primeiro pai.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Arquivo criado ou modificado.
    Upsert(String),
    Deleted(String),
}

impl Change {
    pub fn path(&self) -> &str {
        match self {
            Self::Upsert(p) | Self::Deleted(p) => p,
        }
    }
}

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

    pub fn fetch(&self) -> Result<()> {
        self.run(&["fetch", "--quiet", "origin"]).map(drop)
    }

    /// SHA completo de `rev`.
    pub fn rev_parse(&self, rev: &str) -> Result<String> {
        let spec = format!("{rev}^{{commit}}");
        Ok(self
            .run_text(&["rev-parse", "--verify", &spec])?
            .trim()
            .to_owned())
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

    /// Commits depois de `from` até `to`, do mais antigo ao mais novo, pelo primeiro pai.
    pub fn commits_between(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let range = format!("{from}..{to}");
        let out = self.run_text(&["rev-list", "--reverse", "--first-parent", &range])?;
        Ok(out.lines().map(str::to_owned).collect())
    }

    /// Arquivos que `sha` mudou em relação ao primeiro pai.
    pub fn changes(&self, sha: &str) -> Result<Vec<Change>> {
        let parent = format!("{sha}^");
        let out = self.run_text(&["diff", "--no-renames", "--name-status", "-z", &parent, sha])?;
        let mut fields = out.split('\0').filter(|s| !s.is_empty());
        let mut changes = Vec::new();
        while let (Some(status), Some(path)) = (fields.next(), fields.next()) {
            let path = path.to_owned();
            changes.push(if status == "D" {
                Change::Deleted(path)
            } else {
                Change::Upsert(path)
            });
        }
        Ok(changes)
    }

    pub fn subject(&self, sha: &str) -> Result<String> {
        Ok(self
            .run_text(&["log", "-1", "--format=%s", sha])?
            .trim()
            .to_owned())
    }

    /// Diff de `sha` contra o primeiro pai, restrito a `paths`.
    pub fn diff(&self, sha: &str, paths: &[&str]) -> Result<String> {
        let parent = format!("{sha}^");
        let mut args = vec!["diff", "--no-renames", parent.as_str(), sha, "--"];
        args.extend_from_slice(paths);
        Ok(String::from_utf8_lossy(&self.run(&args)?).into_owned())
    }

    /// Conteúdo de `path` em `sha`.
    pub fn show(&self, sha: &str, path: &str) -> Result<Vec<u8>> {
        self.run(&["show", &format!("{sha}:{path}")])
    }

    /// Checkout de `sha` em `dest`, sem branch.
    pub fn worktree_add(&self, dest: &Path, sha: &str) -> Result<()> {
        let dest = dest.to_str().context("caminho não é UTF-8")?;
        self.run(&["worktree", "add", "--quiet", "--detach", dest, sha])
            .map(drop)
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

    #[test]
    fn commits_e_mudancas_em_ordem() {
        let repo = TempRepo::init();
        repo.write("a.txt", "1\n");
        repo.write("b.txt", "1\n");
        let c1 = repo.commit("c1");
        repo.write("a.txt", "2\n");
        repo.write("n.txt", "novo\n");
        let c2 = repo.commit("c2");
        repo.remove("b.txt");
        let c3 = repo.commit("c3: remove b");
        let git = Git::new(repo.path());

        assert_eq!(
            git.commits_between(&c1, &c3).unwrap(),
            vec![c2.clone(), c3.clone()]
        );
        assert_eq!(
            git.changes(&c2).unwrap(),
            vec![
                Change::Upsert("a.txt".into()),
                Change::Upsert("n.txt".into())
            ]
        );
        assert_eq!(
            git.changes(&c3).unwrap(),
            vec![Change::Deleted("b.txt".into())]
        );
        assert_eq!(git.subject(&c3).unwrap(), "c3: remove b");
        assert!(git.diff(&c2, &["a.txt"]).unwrap().contains("+2"));
        assert_eq!(git.rev_parse("HEAD").unwrap(), c3);
    }

    #[test]
    fn show_le_arquivo_na_revisao() {
        let repo = TempRepo::init();
        repo.write("a.txt", "v1");
        let c1 = repo.commit("c1");
        repo.write("a.txt", "v2");
        repo.commit("c2");
        assert_eq!(Git::new(repo.path()).show(&c1, "a.txt").unwrap(), b"v1");
    }
}
