# M0 — Workspace e `cargo xtask`: plano de implementação

> **Para agentes:** SUB-SKILL OBRIGATÓRIA: use superpowers:subagent-driven-development (recomendado) ou superpowers:executing-plans para executar este plano tarefa a tarefa. Os passos usam checkbox (`- [ ]`).

**Objetivo:** workspace Rust 1.99 com um `cargo xtask` que classifica todo arquivo do upstream (`check-map`), gera e fecha tarefas de sync (`sync`, `task apply`, `task done`), compila o oráculo (`oracle-build`) e serve de portão de qualidade (`ci`).

**Arquitetura:** um crate binário `xtask/`, sem depender do resto do workspace. Cada módulo tem uma responsabilidade: `map` (regras de `sync/map.toml`), `state` (`UPSTREAM.toml`), `git` (chamadas ao git do sistema), `check_map`, `sync`, `task`, `oracle` e `ci`. Os testes usam repositórios git temporários e não acessam a rede.

**Stack:** Rust 1.99.0 (edition 2024) · anyhow 1.0.104 · clap 4.6.7 · globset 0.4.20 · serde 1.0.229 · toml 1.1.6 · tempfile 3.27.0

**Spec:** `docs/superpowers/specs/2026-10-01-llama-rs-port-design.md` (§3 arquitetura, §4 sync, §5.1 oráculo).

**Validação prévia:** em 2026-10-01, os 4 estágios deste plano foram montados num rascunho com rustc 1.99.0. Cada estágio passou `cargo xtask ci` (fmt, clippy `-D warnings`, testes). O `check-map` classificou 3669 de 3669 arquivos do upstream em `ec7630a6`.

## Restrições globais

- Toolchain fixa em `1.99.0` (`rust-toolchain.toml`), edition 2024, resolver 3.
- Dependências na última versão publicada no crates.io. As versões acima são as de 2026-10-01.
- `unsafe_code = "deny"`. `unwrap_used`, `expect_used` e `panic` = deny, liberados só em testes (via `clippy.toml`).
- Comentários, mensagens e nomes de teste em português.
- O clone do upstream é sempre **completo** (sem `--filter`), em `.upstream/llama.cpp`, criado em `llama.cpp.tmp` e renomeado no fim. Motivo: em 2026-10-01, um clone parcial interrompido entrou num laço de buscas objeto a objeto e travou o shell duas vezes.
- Todo `git` roda com `GIT_TERMINAL_PROMPT=0`.
- Comandos longos (primeiro clone, `sync`, `oracle-build`) rodam com `timeout` e com a saída num arquivo. Nunca deixe stderr sem limite no terminal.
- Neste computador, o hook `rtk` reescreve `cargo` e engole o `--`. Em comando com `--`, use `rtk proxy cargo ...` ou o alias `cargo xtask ...`, que não usa `--`.
- No M0, **nunca** rode `cargo xtask sync` sem `--dry-run`. Ainda não há crates registradas, e o `cursor` passaria do baseline `ec7630a6`.
- Cada commit termina com as linhas de atribuição que a sessão executora indicar.

## Mapa de arquivos

| Arquivo | Responsabilidade | Tarefa |
|---|---|---|
| `Cargo.toml`, `rust-toolchain.toml`, `clippy.toml`, `.cargo/config.toml`, `.gitignore` | workspace, toolchain, lints, alias `cargo xtask` | 1 |
| `xtask/Cargo.toml`, `xtask/src/main.rs` | CLI (clap); cada tarefa acrescenta subcomandos | 1–4 |
| `xtask/src/paths.rs` | caminhos do repositório | 1–4 |
| `xtask/src/ci.rs` | `cargo xtask ci` | 1–2 |
| `xtask/src/map.rs` | regras de `sync/map.toml` (primeira que casa vence) | 2 |
| `xtask/src/state.rs` | leitura e escrita de `UPSTREAM.toml` | 2–3 |
| `xtask/src/git.rs` | git do sistema: clone, ls-tree, histórico, diff, show, worktree | 2–4 |
| `xtask/src/testutil.rs` | repositórios git temporários e fixture de sync (só em teste) | 2–3 |
| `xtask/src/check_map.rs` | `cargo xtask check-map` | 2 |
| `xtask/src/sync.rs` | `cargo xtask sync [--dry-run]` | 3 |
| `xtask/src/task.rs` | formato das tarefas, `task apply`, `task done` | 3 |
| `xtask/src/oracle.rs` | `cargo xtask oracle-build` | 4 |
| `sync/map.toml`, `UPSTREAM.toml` | classificação do upstream e estado do sync | 2 |

---

### Tarefa 1: Workspace, toolchain e `cargo xtask ci`

**Arquivos:**
- Criar: `Cargo.toml`, `rust-toolchain.toml`, `clippy.toml`, `.cargo/config.toml`, `xtask/Cargo.toml`, `xtask/src/main.rs`, `xtask/src/paths.rs`, `xtask/src/ci.rs`
- Modificar: `.gitignore` (acrescentar uma linha)

**Interfaces:**
- Produz: `Paths::new(root: impl Into<PathBuf>) -> Paths`, `Paths::from_manifest() -> Paths`, `Paths::root(&self) -> &Path`, `ci::run(&Paths) -> anyhow::Result<()>`, o alias `cargo xtask`.

- [ ] **Passo 1: Instalar a toolchain**

Run: `rustup toolchain install 1.99.0 --profile minimal --component rustfmt --component clippy && rustc +1.99.0 --version`
Esperado: `rustc 1.99.0 (b940084d7 2026-09-28)`

- [ ] **Passo 2: Arquivos do workspace**

`Cargo.toml`:

```toml
[workspace]
resolver = "3"
members = ["xtask"]

[workspace.package]
edition = "2024"
rust-version = "1.99"
publish = false

[workspace.dependencies]
anyhow = "1.0.104"
clap = { version = "4.6.7", features = ["derive"] }
globset = "0.4.20"
serde = { version = "1.0.229", features = ["derive"] }
tempfile = "3.27.0"
toml = "1.1.6"

[workspace.lints.rust]
unsafe_code = "deny"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
undocumented_unsafe_blocks = "deny"
cast_possible_truncation = "warn"
cast_sign_loss = "warn"
cast_possible_wrap = "warn"
indexing_slicing = "warn"

# docs/rust-praticas-da-documentacao.md §2.1. panic = "abort" fica de fora: quebra #[should_panic].
[profile.release]
lto = "thin"
codegen-units = 1
```

`rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.99.0"
components = ["rustfmt", "clippy"]
```

`clippy.toml`:

```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
allow-panic-in-tests = true
allow-indexing-slicing-in-tests = true
```

`.cargo/config.toml`:

```toml
[alias]
xtask = "run --quiet --package xtask --"

[build]
rustflags = ["-C", "target-cpu=native"]
```

`xtask/Cargo.toml`:

```toml
[package]
name = "xtask"
version = "0.0.0"
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
anyhow.workspace = true
clap.workspace = true
globset.workspace = true
serde.workspace = true
toml.workspace = true

[dev-dependencies]
tempfile.workspace = true

[lints]
workspace = true
```

Acrescente ao fim de `.gitignore`:

```gitignore

# Clone do upstream, worktrees e builds do oráculo (cargo xtask)
/.upstream/
```

- [ ] **Passo 3: Escrever o teste que falha**

`xtask/src/main.rs`:

```rust
//! Automação do llama.rs: sync com o upstream, oráculo e CI.
//! Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md

mod ci;
mod paths;

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
    /// fmt + clippy + testes
    Ci,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let paths = Paths::from_manifest();
    let result = match cli.cmd {
        Cmd::Ci => ci::run(&paths),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::FAILURE
        }
    }
}
```

`xtask/src/ci.rs`:

```rust
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
```

`xtask/src/paths.rs`, por enquanto só com o teste:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raiz_e_o_workspace() {
        let root = Paths::from_manifest();
        assert!(root.root().join("rust-toolchain.toml").exists());
    }
}
```

- [ ] **Passo 4: Rodar e ver falhar**

Run: `cargo test -p xtask`
Esperado: erro de compilação `cannot find type 'Paths'` (ou `cannot find struct`).

- [ ] **Passo 5: Implementar `Paths`**

Acima do módulo de testes em `xtask/src/paths.rs`:

```rust
//! Caminhos do repositório usados pelo xtask.

use std::path::{Path, PathBuf};

pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Raiz do workspace: o diretório pai de `xtask/`.
    pub fn from_manifest() -> Self {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self::new(manifest.parent().unwrap_or(manifest))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}
```

- [ ] **Passo 6: Rodar o portão**

Run: `cargo xtask ci`
Esperado: `==> cargo fmt --all --check`, `==> cargo clippy ...`, `==> cargo test --workspace`, `test result: ok. 1 passed`. Saída com código 0. Se o `fmt --check` falhar, rode `cargo fmt --all` e repita.

- [ ] **Passo 7: Commit**

```bash
git add Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml .cargo/config.toml .gitignore xtask/
git commit -m "build: workspace com Rust 1.99 e cargo xtask ci"
```

---

### Tarefa 2: `sync/map.toml` e `cargo xtask check-map`

**Arquivos:**
- Criar: `sync/map.toml`, `UPSTREAM.toml`, `xtask/src/map.rs`, `xtask/src/state.rs`, `xtask/src/git.rs`, `xtask/src/testutil.rs`, `xtask/src/check_map.rs`
- Modificar (conteúdo completo abaixo): `xtask/src/main.rs`, `xtask/src/paths.rs`, `xtask/src/ci.rs`

**Interfaces:**
- Consome: `Paths` e `ci::run` da Tarefa 1.
- Produz:
  - `map::Action { Port, Vendor, Ignore }`, `map::Rule { action: Action, krate: Option<String> }`, `map::SyncMap::{parse(&str), load(&Path)} -> Result<SyncMap>`, `SyncMap::classify(&self, &str) -> Option<&Rule>`
  - `state::State { upstream: Upstream }`, `state::Upstream { repo: String, cursor: String }`, `State::load(&Path) -> Result<State>`
  - `git::short(&str) -> &str`, `git::open_upstream(clone_dir: &Path, url: &str) -> Result<Git>`, `git::Git::{new, ls_tree(&self, rev) -> Result<Vec<String>>}`
  - `testutil::TempRepo::{init, path, git, write, commit}`
  - `check_map::{unmatched, summarize, run(&Paths, Option<&str>)}`
  - `Paths::{upstream_toml, map_toml, upstream_clone}`

- [ ] **Passo 1: Escrever os testes que falham**

Crie cada arquivo abaixo só com o módulo de testes. Os testes de `git` usam o `TempRepo`; crie `xtask/src/testutil.rs` já completo:

```rust
//! Repositórios git temporários para os testes.

use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

pub struct TempRepo {
    dir: TempDir,
}

impl TempRepo {
    pub fn init() -> Self {
        let repo = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        repo.git(&["init", "-q", "-b", "master"]);
        repo
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(self.path())
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    pub fn write(&self, rel: &str, content: &str) {
        let path = self.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    /// Commit de tudo; devolve o SHA.
    pub fn commit(&self, msg: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", msg]);
        self.git(&["rev-parse", "HEAD"])
    }
}
```

`xtask/src/map.rs`:

```rust
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
```

`xtask/src/state.rs`:

```rust
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
```

`xtask/src/git.rs`:

```rust
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
```

`xtask/src/check_map.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const MAP: &str = "[[rule]]\npath = \"src/*.cpp\"\naction = \"port\"\ncrate = \"llama\"\n\n[[rule]]\npath = \"*\"\naction = \"ignore\"\n";

    fn files(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn lista_so_os_sem_regra() {
        let map = SyncMap::parse(MAP).unwrap();
        let files = files(&["src/a.cpp", "README.md", "docs/x.md"]);
        assert_eq!(unmatched(&map, &files), vec!["docs/x.md"]);
    }

    #[test]
    fn conta_por_acao_e_crate() {
        let map = SyncMap::parse(MAP).unwrap();
        let files = files(&["src/a.cpp", "src/b.cpp", "README.md"]);
        let counts = summarize(&map, &files);
        assert_eq!(counts[&(Action::Port, Some("llama"))], 2);
        assert_eq!(counts[&(Action::Ignore, None)], 1);
    }
}
```

Substitua `xtask/src/main.rs` (registra os módulos e o subcomando `check-map`):

```rust
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
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `cargo test -p xtask`
Esperado: erros de compilação (`SyncMap`, `State`, `Git`, `open_upstream`, `unmatched` e `check_map::run` inexistentes).

- [ ] **Passo 3: Implementar**

Acima do módulo de testes de `xtask/src/map.rs`:

```rust
//! `sync/map.toml`: cada caminho do upstream recebe uma ação. Vale a primeira regra que casar.

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

    /// A primeira regra que casa com `path`.
    pub fn classify(&self, path: &str) -> Option<&Rule> {
        self.rules.iter().find(|r| r.matcher.is_match(path))
    }
}
```

Acima do módulo de testes de `xtask/src/state.rs`:

```rust
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
```

Acima do módulo de testes de `xtask/src/git.rs`:

```rust
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
```

Acima do módulo de testes de `xtask/src/check_map.rs`:

```rust
//! `cargo xtask check-map`: todo arquivo do upstream precisa casar uma regra.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::git;
use crate::map::{Action, SyncMap};
use crate::paths::Paths;
use crate::state::State;

/// Arquivos sem regra.
pub fn unmatched<'f>(map: &SyncMap, files: &'f [String]) -> Vec<&'f str> {
    files
        .iter()
        .map(String::as_str)
        .filter(|f| map.classify(f).is_none())
        .collect()
}

/// Quantos arquivos caem em cada (ação, crate).
pub fn summarize<'m>(
    map: &'m SyncMap,
    files: &[String],
) -> BTreeMap<(Action, Option<&'m str>), usize> {
    let mut counts = BTreeMap::new();
    for rule in files.iter().filter_map(|f| map.classify(f)) {
        *counts
            .entry((rule.action, rule.krate.as_deref()))
            .or_insert(0) += 1;
    }
    counts
}

pub fn run(paths: &Paths, rev: Option<&str>) -> Result<()> {
    let state = State::load(&paths.upstream_toml())?;
    let map = SyncMap::load(&paths.map_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    let rev = rev.unwrap_or(&state.upstream.cursor);
    let files = git.ls_tree(rev)?;
    let missing = unmatched(&map, &files);
    if !missing.is_empty() {
        for f in &missing {
            eprintln!("sem regra: {f}");
        }
        bail!(
            "{} de {} arquivos sem regra em sync/map.toml",
            missing.len(),
            files.len()
        );
    }
    for ((action, krate), n) in summarize(&map, &files) {
        println!("{n:>5}  {action:?} {}", krate.unwrap_or(""));
    }
    println!(
        "ok: {} arquivos de {} classificados",
        files.len(),
        git::short(rev)
    );
    Ok(())
}
```

Substitua `xtask/src/paths.rs`:

```rust
//! Caminhos do repositório usados pelo xtask.

use std::path::{Path, PathBuf};

pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Raiz do workspace: o diretório pai de `xtask/`.
    pub fn from_manifest() -> Self {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self::new(manifest.parent().unwrap_or(manifest))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn upstream_toml(&self) -> PathBuf {
        self.root.join("UPSTREAM.toml")
    }

    pub fn map_toml(&self) -> PathBuf {
        self.root.join("sync/map.toml")
    }

    pub fn upstream_clone(&self) -> PathBuf {
        self.root.join(".upstream/llama.cpp")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raiz_e_o_workspace() {
        let root = Paths::from_manifest();
        assert!(root.root().join("rust-toolchain.toml").exists());
    }
}
```

Substitua `xtask/src/ci.rs` (agora termina com o `check-map`):

```rust
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
    eprintln!("==> check-map");
    crate::check_map::run(paths, None)
}
```

- [ ] **Passo 4: Rodar os testes**

Run: `cargo test -p xtask`
Esperado: `test result: ok. 12 passed`

- [ ] **Passo 5: Classificação do upstream e estado inicial**

`UPSTREAM.toml`:

```toml
# Lido e gravado pelo `cargo xtask`. Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md §4.

[upstream]
repo = "https://github.com/ggml-org/llama.cpp"
cursor = "ec7630a640789c393694fb194f1bbbf0369fc62d"
synced = "ec7630a640789c393694fb194f1bbbf0369fc62d"
```

`sync/map.toml` (cada regra segue o escopo do spec §2; o arquivo é a decisão de escopo, então revise-o como código):

```toml
# sync/map.toml — classifica cada arquivo de ggml-org/llama.cpp.
# Vale a primeira regra que casar. `*` não atravessa `/`; `**` atravessa; `{a,b}` alterna.
# Ações: port (traduzir para Rust) · vendor (cópia verbatim em vendor/upstream/) · ignore.
# Diretório do escopo listado arquivo a arquivo: um arquivo novo nele faz o check-map falhar.
# Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md §2 (escopo) e §4 (sync).

# ── API pública da libllama ───────────────────────────────────────────────
[[rule]]
path = "include/{llama.h,llama-cpp.h}"
action = "port"
crate = "llama"

# ── ggml: headers públicos ────────────────────────────────────────────────
[[rule]]
path = "ggml/include/{ggml.h,ggml-alloc.h,ggml-backend.h,ggml-cpp.h,gguf.h}"
action = "port"
crate = "ggml"

[[rule]]
path = "ggml/include/ggml-cpu.h"
action = "port"
crate = "ggml-cpu"

[[rule]]
path = "ggml/include/ggml-vulkan.h"
action = "port"
crate = "ggml-vulkan"

[[rule]]
path = "ggml/include/ggml-cuda.h"
action = "vendor"
crate = "ggml-hip"

# Outros backends e o otimizador de treino.
[[rule]]
path = "ggml/include/{ggml-blas.h,ggml-cann.h,ggml-et.h,ggml-hexagon.h,ggml-metal.h,ggml-opencl.h,ggml-openvino.h,ggml-opt.h,ggml-rpc.h,ggml-sycl.h,ggml-virtgpu.h,ggml-webgpu.h,ggml-zdnn.h,ggml-zendnn.h}"
action = "ignore"

# ── ggml: núcleo ──────────────────────────────────────────────────────────
# Fora: otimizador de treino, carga dinâmica de backends (aqui o registro é estático), build.
[[rule]]
path = "ggml/src/{ggml-opt.cpp,ggml-backend-dl.cpp,ggml-backend-dl.h,ggml-version.h.in,CMakeLists.txt}"
action = "ignore"

[[rule]]
path = "ggml/src/*.{c,cpp,h}"
action = "port"
crate = "ggml"

[[rule]]
path = "ggml/{CMakeLists.txt,.gitignore,cmake/**}"
action = "ignore"

# ── ggml-cpu: só x86 (AVX2) e o caminho genérico ──────────────────────────
[[rule]]
path = "ggml/src/ggml-cpu/arch/x86/**"
action = "port"
crate = "ggml-cpu"

[[rule]]
path = "ggml/src/ggml-cpu/arch/**"
action = "ignore"

# AMX (Sapphire Rapids+), KleidiAI (ARM), SpacemiT (RISC-V), HBM (Xeon Max): ausentes no Broadwell.
[[rule]]
path = "ggml/src/ggml-cpu/{amx/**,kleidiai/**,spacemit/**,cmake/**,CMakeLists.txt,hbm.cpp,hbm.h}"
action = "ignore"

[[rule]]
path = "ggml/src/ggml-cpu/{llamafile,tiled}/*.{cpp,h}"
action = "port"
crate = "ggml-cpu"

[[rule]]
path = "ggml/src/ggml-cpu/*.{c,cpp,h}"
action = "port"
crate = "ggml-cpu"

# ── ggml-vulkan: host em Rust, shaders verbatim ───────────────────────────
[[rule]]
path = "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"
action = "port"
crate = "ggml-vulkan"

[[rule]]
path = "ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt"
action = "ignore"

[[rule]]
path = "ggml/src/ggml-vulkan/vulkan-shaders/**"
action = "vendor"
crate = "ggml-vulkan"

[[rule]]
path = "ggml/src/ggml-vulkan/{CMakeLists.txt,cmake/**}"
action = "ignore"

[[rule]]
path = "ggml/src/ggml-vulkan/*.{cpp,h}"
action = "port"
crate = "ggml-vulkan"

# ── ggml-hip: ilha C++ verbatim (M6; o spec próprio do M6 pode reclassificar) ─
[[rule]]
path = "ggml/src/{ggml-cuda,ggml-hip}/**"
action = "vendor"
crate = "ggml-hip"

# ── ggml: backends fora do escopo ─────────────────────────────────────────
[[rule]]
path = "ggml/src/{ggml-blas,ggml-cann,ggml-et,ggml-hexagon,ggml-metal,ggml-musa,ggml-opencl,ggml-openvino,ggml-rpc,ggml-sycl,ggml-virtgpu,ggml-webgpu,ggml-zdnn,ggml-zendnn}/**"
action = "ignore"

# ── llama (src/) ──────────────────────────────────────────────────────────
# Caches específicos do DeepSeek e a quantização (tool fora do escopo).
[[rule]]
path = "src/{llama-kv-cache-dsa,llama-kv-cache-dsa-iswa,llama-kv-cache-dsv4,llama-kv-cache-msa,llama-quant}.{cpp,h}"
action = "ignore"

[[rule]]
path = "src/models/{models.h,llama.cpp,qwen2.cpp,qwen3.cpp,qwen35.cpp,qwen3vl.cpp,delta-net-base.cpp}"
action = "port"
crate = "llama"

# Arquiteturas fora do escopo (D5). Arquitetura nova no upstream também cai aqui.
[[rule]]
path = "src/models/*.cpp"
action = "ignore"

[[rule]]
path = "src/{CMakeLists.txt,llama-version.h.in}"
action = "ignore"

[[rule]]
path = "src/*.{cpp,h}"
action = "port"
crate = "llama"

# ── common/ ───────────────────────────────────────────────────────────────
# Fora: llguidance (biblioteca externa opcional), imatrix (quantização), build.
[[rule]]
path = "common/{CMakeLists.txt,build-info.cpp.in,llguidance.cpp,imatrix-loader.cpp,imatrix-loader.h,parsers/sources.cmake}"
action = "ignore"

[[rule]]
path = "common/**/README.md"
action = "ignore"

[[rule]]
path = "common/**/*.{cpp,h,hpp}"
action = "port"
crate = "common"

# ── tools no escopo (D6) ──────────────────────────────────────────────────
[[rule]]
path = "tools/cli/*.{cpp,h}"
action = "port"
crate = "llama-cli"

[[rule]]
path = "tools/completion/*.cpp"
action = "port"
crate = "llama-completion"

[[rule]]
path = "tools/llama-bench/*.cpp"
action = "port"
crate = "llama-bench"

[[rule]]
path = "tools/perplexity/*.cpp"
action = "port"
crate = "llama-perplexity"

# Suíte pytest do upstream: roda contra o binário Rust (spec §5.2, camada 7).
[[rule]]
path = "tools/server/tests/**"
action = "vendor"
crate = "llama-server"

[[rule]]
path = "tools/server/*.{cpp,h}"
action = "port"
crate = "llama-server"

[[rule]]
path = "tools/mtmd/{tests/**,test-1.jpeg,test-2.mp3,test-3.mp4,tests.sh}"
action = "vendor"
crate = "mtmd"

# mtmd-cli é um binário à parte; legacy-models e debug são ferramentas de conversão.
[[rule]]
path = "tools/mtmd/{mtmd-cli.cpp,deprecation-warning.cpp,legacy-models/**,debug/**}"
action = "ignore"

# Só os encoders de visão da família Qwen.
[[rule]]
path = "tools/mtmd/models/{models.h,qwen2vl.cpp,qwen3vl.cpp}"
action = "port"
crate = "mtmd"

[[rule]]
path = "tools/mtmd/models/*.cpp"
action = "ignore"

[[rule]]
path = "tools/mtmd/*.{cpp,h}"
action = "port"
crate = "mtmd"

# Demais tools, READMEs, CMakeLists e a web UI (tools/ui; decidir no M4).
[[rule]]
path = "tools/**"
action = "ignore"

# ── tests/: cada teste vai para a crate que ele exercita ──────────────────
[[rule]]
path = "tests/{test-backend-ops,test-gguf,test-alloc,test-rope,test-quantize-fns,test-double-float,test-col2im-1d}.cpp"
action = "port"
crate = "ggml"

[[rule]]
path = "tests/{test-barrier,test-tiled-mulmat}.cpp"
action = "port"
crate = "ggml-cpu"

[[rule]]
path = "tests/{test-tokenizer-0,test-tokenizer-1-bpe,test-tokenizer-1-spm,test-unicode,test-sampling,test-backend-sampler,test-grammar-parser,test-grammar-integration,test-llama-grammar,test-llama-archs,test-batch-alloc,test-model-load-cancel,test-autorelease,test-save-load-state,test-state-restore-fragmented,test-recurrent-state-rollback,test-thread-safety}.cpp"
action = "port"
crate = "llama"

[[rule]]
path = "tests/{test-chat,test-chat-analysis,test-chat-auto-parser,test-chat-peg-parser,test-chat-template,test-jinja,test-json-schema,test-json-schema-to-grammar,test-peg-parser,test-arg-parser,test-log,test-reasoning-budget,test-gbnf-validator,test-model-resolution}.cpp"
action = "port"
crate = "common"

[[rule]]
path = "tests/peg-parser/*.{cpp,h}"
action = "port"
crate = "common"

[[rule]]
path = "tests/test-mtmd-impl.cpp"
action = "port"
crate = "mtmd"

# Fora: treino, RPC, llguidance, quantização, fusão (decidir no M3), metadados do HF,
# teste que exige modelo grande, harness C e scripts.
[[rule]]
path = "tests/{test-opt.cpp,test-rpc-multi-server.cpp,test-rpc-multi-server.sh,test-grammar-llguidance.cpp,test-quantize-perf.cpp,test-quantize-stats.cpp,test-quant-type-selection.cpp,test-fusion.cpp,test-export-graph-ops.cpp,test-gguf-model-data.cpp,gguf-model-data.cpp,gguf-model-data.h,test-rset-release.cpp,test-c.c,test-mtmd-c-api.c,testing.h,test-lora-conversion-inference.sh,test-tokenizer-0.py,test-tokenizer-0.sh,test-tokenizer-random.py,test-tokenizers-repo.sh,CMakeLists.txt,.gitignore,fusion/**,snapshots/**}"
action = "ignore"

# ── models/: fixtures de tokenizer e templates de chat ────────────────────
[[rule]]
path = "models/ggml-vocab-{llama-spm,llama-bpe,qwen2,qwen35}.gguf*"
action = "vendor"
crate = "llama"

[[rule]]
path = "models/templates/**"
action = "vendor"
crate = "common"

[[rule]]
path = "models/**"
action = "ignore"

# ── fora do escopo: conversão Python, exemplos, CI, docs, libs C++ de terceiros ─
[[rule]]
path = "{.devops,.gemini,.github,.pi,app,benches,ci,cmake,conversion,docs,examples,gguf-py,grammars,licenses,media,pocs,requirements,scripts,skills,vendor}/**"
action = "ignore"

# Arquivos da raiz: README, scripts de conversão, configs de build e lint.
[[rule]]
path = "*"
action = "ignore"
```

- [ ] **Passo 6: Rodar o check-map no upstream real**

O primeiro uso clona o upstream (~380 MB, alguns minutos):

Run: `mkdir -p target && timeout 900 cargo xtask check-map > target/check-map.log 2>&1; echo "exit=$?"; tail -22 target/check-map.log`
Esperado: `exit=0` e o resumo abaixo. Linhas `sem regra: <caminho>` indicam um arquivo sem classificação: classifique-o no `sync/map.toml` seguindo o spec §2 e repita.

```
  110  Port common
   27  Port ggml
   33  Port ggml-cpu
    8  Port ggml-vulkan
   81  Port llama
    2  Port llama-bench
    8  Port llama-cli
    2  Port llama-completion
    2  Port llama-perplexity
   25  Port llama-server
   20  Port mtmd
   72  Vendor common
  286  Vendor ggml-hip
  192  Vendor ggml-vulkan
   12  Vendor llama
   40  Vendor llama-server
    8  Vendor mtmd
 2741  Ignore
ok: 3669 arquivos de ec7630a classificados
```

- [ ] **Passo 7: Rodar o portão**

Run: `cargo xtask ci`
Esperado: código 0, `test result: ok. 12 passed` e `ok: 3669 arquivos de ec7630a classificados`.

- [ ] **Passo 8: Commit**

```bash
git add sync/map.toml UPSTREAM.toml xtask/
git commit -m "feat(xtask): check-map e classificação completa do upstream"
```

---

### Tarefa 3: `cargo xtask sync`, `task apply` e `task done`

**Arquivos:**
- Criar: `xtask/src/sync.rs`, `xtask/src/task.rs`
- Modificar (conteúdo completo abaixo): `xtask/src/main.rs`, `xtask/src/paths.rs`, `xtask/src/state.rs`, `xtask/src/git.rs`, `xtask/src/testutil.rs`

**Interfaces:**
- Consome: `SyncMap`, `State::load`, `open_upstream`, `Git`, `TempRepo`, `Paths` (Tarefas 1–2).
- Produz:
  - `state::State { upstream, crates: BTreeMap<String, CrateState> }`, `Upstream { repo, cursor, synced }`, `CrateState { baseline: String, status: CrateStatus }`, `CrateStatus { Porting, Synced }`, `State::save(&self, &Path)`
  - `git::Change { Upsert(String), Deleted(String) }` + `Change::path`; `Git::{fetch, rev_parse, commits_between, changes, subject, diff, show}`
  - `task::TaskMeta { seq: u32, sha, krate, subject, port, vendor, vendor_deleted }`, `TaskMeta::file_name`, `task::{render, parse_meta, read_meta, pending, recompute_synced, apply, done}`
  - `sync::{CratePlan, plan_commit, rust_modules, run(&Paths, dry_run: bool)}`
  - `testutil::{FIXTURE_MAP, Fixture}`, `TempRepo::remove`
  - `Paths::{pending_dir, vendor_dir, crates_dir}`
  - Formato da tarefa: front matter TOML entre `+++` + Markdown; nome `<seq:05>-<sha7>-<crate>.md`

- [ ] **Passo 1: Base de teste e estado completo**

Substitua `xtask/src/testutil.rs` (ganha `remove` e o `Fixture`):

```rust
//! Repositórios git temporários para os testes.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

use crate::paths::Paths;
use crate::state::{CrateState, CrateStatus, State, Upstream};

pub struct TempRepo {
    dir: TempDir,
}

impl TempRepo {
    pub fn init() -> Self {
        let repo = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        repo.git(&["init", "-q", "-b", "master"]);
        repo
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    pub fn git(&self, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(self.path())
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(["-c", "commit.gpgsign=false"])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    pub fn write(&self, rel: &str, content: &str) {
        let path = self.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    pub fn remove(&self, rel: &str) {
        std::fs::remove_file(self.path().join(rel)).unwrap();
    }

    /// Commit de tudo; devolve o SHA.
    pub fn commit(&self, msg: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", msg]);
        self.git(&["rev-parse", "HEAD"])
    }
}

pub const FIXTURE_MAP: &str = r#"
[[rule]]
path = "keep/*.cpp"
action = "port"
crate = "k"

[[rule]]
path = "shaders/**"
action = "vendor"
crate = "k"

[[rule]]
path = "other/*.cpp"
action = "port"
crate = "outra"

[[rule]]
path = "skip/**"
action = "ignore"
"#;

/// Um upstream falso e uma raiz do llama.rs com map, UPSTREAM.toml e a crate `k` registrada.
pub struct Fixture {
    pub upstream: TempRepo,
    pub root: TempDir,
    pub base: String,
}

impl Fixture {
    pub fn new() -> Self {
        let upstream = TempRepo::init();
        upstream.write("keep/a.cpp", "int a = 1;\n");
        upstream.write("shaders/s.comp", "void main() {}\n");
        upstream.write("skip/x.md", "x\n");
        let base = upstream.commit("base");

        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("sync")).unwrap();
        std::fs::write(root.path().join("sync/map.toml"), FIXTURE_MAP).unwrap();
        let mut crates = BTreeMap::new();
        crates.insert(
            "k".to_owned(),
            CrateState {
                baseline: base.clone(),
                status: CrateStatus::Porting,
            },
        );
        let state = State {
            upstream: Upstream {
                repo: upstream.path().to_str().unwrap().to_owned(),
                cursor: base.clone(),
                synced: base.clone(),
            },
            crates,
        };
        state.save(&root.path().join("UPSTREAM.toml")).unwrap();
        Self {
            upstream,
            root,
            base,
        }
    }

    pub fn paths(&self) -> Paths {
        Paths::new(self.root.path())
    }
}
```

Substitua `xtask/src/state.rs` (ganha `synced`, `crates` e `save`):

```rust
//! `UPSTREAM.toml`: onde o porte está em relação ao upstream.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const HEADER: &str = "# Lido e gravado pelo `cargo xtask`. Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md §4.\n\n";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub upstream: Upstream,
    #[serde(default)]
    pub crates: BTreeMap<String, CrateState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Upstream {
    pub repo: String,
    /// Último commit para o qual o `xtask sync` já gerou tarefas.
    pub cursor: String,
    /// Todas as tarefas até este commit estão concluídas.
    pub synced: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
}
```

- [ ] **Passo 2: Escrever os testes que falham**

`xtask/src/sync.rs`, só com o módulo de testes:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{FIXTURE_MAP, Fixture};

    #[test]
    fn plan_agrupa_por_crate_e_descarta_ignore() {
        let map = SyncMap::parse(FIXTURE_MAP).unwrap();
        let changes = vec![
            Change::Upsert("keep/a.cpp".into()),
            Change::Upsert("shaders/n.comp".into()),
            Change::Deleted("shaders/s.comp".into()),
            Change::Upsert("skip/x.md".into()),
        ];
        let plans = plan_commit(&map, &changes).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(
            plans["k"],
            CratePlan {
                port: vec!["keep/a.cpp".into()],
                vendor: vec!["shaders/n.comp".into()],
                vendor_deleted: vec!["shaders/s.comp".into()],
            }
        );
    }

    #[test]
    fn plan_falha_em_caminho_sem_regra() {
        let map = SyncMap::parse(FIXTURE_MAP).unwrap();
        let Err(e) = plan_commit(&map, &[Change::Upsert("novo/x.c".into())]) else {
            panic!("devia falhar")
        };
        assert!(e.to_string().contains("novo/x.c"));
    }

    #[test]
    fn indice_de_modulos_le_os_marcadores() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("k/src/a.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(
            &file,
            "//! upstream: keep/a.cpp\n//! upstream: keep/a.h\nfn x() {}\n",
        )
        .unwrap();
        let index = rust_modules(dir.path()).unwrap();
        assert_eq!(index["keep/a.cpp"], vec![file.clone()]);
        assert_eq!(index["keep/a.h"], vec![file]);
    }

    #[test]
    fn sync_gera_tarefa_so_para_crate_registrada() {
        let fx = Fixture::new();
        fx.upstream.write("keep/a.cpp", "int a = 2;\n");
        fx.upstream.write("shaders/t.comp", "void main() { }\n");
        fx.upstream.write("other/o.cpp", "int o;\n");
        let c1 = fx.upstream.commit("muda a, cria t e o");
        fx.upstream.write("skip/x.md", "y\n");
        let c2 = fx.upstream.commit("só docs");
        let paths = fx.paths();

        run(&paths, false).unwrap();

        let tasks = task::pending(&paths.pending_dir()).unwrap();
        assert_eq!(tasks.len(), 1);
        let (meta, file) = &tasks[0];
        assert_eq!(
            (meta.seq, meta.sha.as_str(), meta.krate.as_str()),
            (1, c1.as_str(), "k")
        );
        assert_eq!(meta.port, vec!["keep/a.cpp"]);
        assert_eq!(meta.vendor, vec!["shaders/t.comp"]);
        let text = std::fs::read_to_string(file).unwrap();
        assert!(text.contains("+int a = 2;"));
        assert!(text.contains("sem módulo Rust"));
        let state = State::load(&paths.upstream_toml()).unwrap();
        assert_eq!(state.upstream.cursor, c2);
        assert_eq!(state.upstream.synced, fx.base);
    }

    #[test]
    fn sync_para_no_commit_sem_regra_e_guarda_o_progresso() {
        let fx = Fixture::new();
        fx.upstream.write("keep/a.cpp", "int a = 3;\n");
        let c1 = fx.upstream.commit("ok");
        fx.upstream.write("novo/x.c", "x\n");
        fx.upstream.commit("caminho sem regra");
        let paths = fx.paths();

        let Err(e) = run(&paths, false) else {
            panic!("devia falhar")
        };
        assert!(format!("{e:#}").contains("novo/x.c"));
        assert_eq!(
            State::load(&paths.upstream_toml()).unwrap().upstream.cursor,
            c1
        );
    }
}
```

`xtask/src/task.rs`, só com o módulo de testes:

```rust
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
```

Substitua `xtask/src/main.rs` (registra `sync`, `task` e os subcomandos):

```rust
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
```

- [ ] **Passo 3: Rodar e ver falhar**

Run: `cargo test -p xtask`
Esperado: erros de compilação (`plan_commit`, `rust_modules`, `TaskMeta`, `Change`, `pending_dir` e similares inexistentes).

- [ ] **Passo 4: Implementar**

Substitua `xtask/src/git.rs` (ganha `Change`, `fetch`, `rev_parse`, `commits_between`, `changes`, `subject`, `diff`, `show` e os testes deles):

```rust
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
```

Substitua `xtask/src/paths.rs`:

```rust
//! Caminhos do repositório usados pelo xtask.

use std::path::{Path, PathBuf};

pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Raiz do workspace: o diretório pai de `xtask/`.
    pub fn from_manifest() -> Self {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self::new(manifest.parent().unwrap_or(manifest))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn upstream_toml(&self) -> PathBuf {
        self.root.join("UPSTREAM.toml")
    }

    pub fn map_toml(&self) -> PathBuf {
        self.root.join("sync/map.toml")
    }

    pub fn pending_dir(&self) -> PathBuf {
        self.root.join("sync/pending")
    }

    pub fn vendor_dir(&self) -> PathBuf {
        self.root.join("vendor/upstream")
    }

    pub fn crates_dir(&self) -> PathBuf {
        self.root.join("crates")
    }

    pub fn upstream_clone(&self) -> PathBuf {
        self.root.join(".upstream/llama.cpp")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raiz_e_o_workspace() {
        let root = Paths::from_manifest();
        assert!(root.root().join("rust-toolchain.toml").exists());
    }
}
```

Acima do módulo de testes de `xtask/src/task.rs`:

```rust
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
```

Acima do módulo de testes de `xtask/src/sync.rs`:

```rust
//! `cargo xtask sync`: transforma commits novos do upstream em tarefas de porte.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::git::{self, Change, Git, short};
use crate::map::{Action, SyncMap};
use crate::paths::Paths;
use crate::state::State;
use crate::task::{self, TaskMeta};

/// O que um commit pede para uma crate.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CratePlan {
    pub port: Vec<String>,
    pub vendor: Vec<String>,
    pub vendor_deleted: Vec<String>,
}

/// Agrupa as mudanças de um commit por crate; `ignore` some. Caminho sem regra é erro.
pub fn plan_commit(map: &SyncMap, changes: &[Change]) -> Result<BTreeMap<String, CratePlan>> {
    let mut plans: BTreeMap<String, CratePlan> = BTreeMap::new();
    for change in changes {
        let path = change.path();
        let Some(rule) = map.classify(path) else {
            bail!("sem regra em sync/map.toml: {path}");
        };
        let Some(krate) = &rule.krate else { continue };
        let plan = plans.entry(krate.clone()).or_default();
        match (rule.action, change) {
            (Action::Port, _) => plan.port.push(path.to_owned()),
            (Action::Vendor, Change::Upsert(_)) => plan.vendor.push(path.to_owned()),
            (Action::Vendor, Change::Deleted(_)) => plan.vendor_deleted.push(path.to_owned()),
            (Action::Ignore, _) => {}
        }
    }
    Ok(plans)
}

/// Caminho do upstream → arquivos Rust que o declaram com `//! upstream: <caminho>`.
pub fn rust_modules(crates_dir: &Path) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let mut index: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    let mut stack = vec![crates_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&dir).with_context(|| format!("lendo {}", dir.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("lendo {}", path.display()))?;
                for upstream in text
                    .lines()
                    .filter_map(|l| l.strip_prefix("//! upstream: "))
                {
                    index
                        .entry(upstream.trim().to_owned())
                        .or_default()
                        .push(path.clone());
                }
            }
        }
    }
    Ok(index)
}

/// Gera as tarefas de um commit para as crates registradas em `UPSTREAM.toml`.
struct Generator<'a> {
    git: &'a Git,
    map: &'a SyncMap,
    state: &'a State,
    modules: &'a BTreeMap<String, Vec<PathBuf>>,
    root: &'a Path,
    pending_dir: &'a Path,
}

impl Generator<'_> {
    /// Devolve quantas tarefas criou; `seq` avança uma por tarefa.
    fn commit(&self, sha: &str, seq: &mut u32) -> Result<usize> {
        let plans = plan_commit(self.map, &self.git.changes(sha)?)?;
        let subject = self.git.subject(sha)?;
        let mut created = 0;
        for (krate, plan) in plans {
            // Crate ainda não registrada: ela nasce com baseline posterior a este commit.
            if !self.state.crates.contains_key(&krate) {
                continue;
            }
            let diff = if plan.port.is_empty() {
                String::new()
            } else {
                let files: Vec<&str> = plan.port.iter().map(String::as_str).collect();
                self.git.diff(sha, &files)?
            };
            let meta = TaskMeta {
                seq: *seq,
                sha: sha.to_owned(),
                krate,
                subject: subject.clone(),
                port: plan.port,
                vendor: plan.vendor,
                vendor_deleted: plan.vendor_deleted,
            };
            let body = self.body(&meta, &diff)?;
            let path = self.pending_dir.join(meta.file_name());
            std::fs::write(&path, task::render(&meta, &body)?)
                .with_context(|| format!("gravando {}", path.display()))?;
            *seq += 1;
            created += 1;
        }
        Ok(created)
    }

    fn body(&self, meta: &TaskMeta, diff: &str) -> Result<String> {
        let sha7 = short(&meta.sha);
        let name = meta.file_name();
        let mut body = String::with_capacity(diff.len() + 1024);
        writeln!(body, "# {} (upstream {sha7})\n", meta.subject)?;
        writeln!(body, "Crate `{}` · commit `{}`\n", meta.krate, meta.sha)?;
        if !meta.port.is_empty() {
            writeln!(body, "## Portar\n")?;
            for path in &meta.port {
                match self.modules.get(path) {
                    Some(files) => {
                        for file in files {
                            let rel = file.strip_prefix(self.root).unwrap_or(file);
                            writeln!(body, "- `{path}` → `{}`", rel.display())?;
                        }
                    }
                    None => writeln!(
                        body,
                        "- `{path}` → **sem módulo Rust**: criar o módulo com `//! upstream: {path}`"
                    )?,
                }
            }
            writeln!(body)?;
        }
        if !meta.vendor.is_empty() || !meta.vendor_deleted.is_empty() {
            writeln!(
                body,
                "## Vendor\n\nRodar `cargo xtask task apply sync/pending/{name}`:\n"
            )?;
            for path in &meta.vendor {
                writeln!(body, "- `{path}`")?;
            }
            for path in &meta.vendor_deleted {
                writeln!(body, "- `{path}` (removido)")?;
            }
            writeln!(body)?;
        }
        writeln!(body, "## Fechar\n")?;
        writeln!(body, "1. `cargo xtask ci` verde")?;
        writeln!(body, "2. `cargo xtask task done sync/pending/{name}`")?;
        writeln!(
            body,
            "3. commit `sync({}): {} (upstream {sha7})`",
            meta.krate, meta.subject
        )?;
        if !diff.is_empty() {
            writeln!(
                body,
                "\n## Diff (só os arquivos a portar)\n\n~~~diff\n{diff}~~~"
            )?;
        }
        Ok(body)
    }
}

pub fn run(paths: &Paths, dry_run: bool) -> Result<()> {
    let mut state = State::load(&paths.upstream_toml())?;
    let map = SyncMap::load(&paths.map_toml())?;
    let git = git::open_upstream(&paths.upstream_clone(), &state.upstream.repo)?;
    git.fetch()?;
    let target = git.rev_parse("origin/master")?;
    let commits = git.commits_between(&state.upstream.cursor, &target)?;
    println!(
        "{} commits novos ({}..{})",
        commits.len(),
        short(&state.upstream.cursor),
        short(&target)
    );
    if dry_run {
        return report(&git, &map, &state, &commits);
    }

    let pending_dir = paths.pending_dir();
    std::fs::create_dir_all(&pending_dir)
        .with_context(|| format!("criando {}", pending_dir.display()))?;
    let modules = rust_modules(&paths.crates_dir())?;
    let mut seq = task::pending(&pending_dir)?
        .last()
        .map_or(1, |(meta, _)| meta.seq + 1);
    let mut created = 0;
    let mut outcome = Ok(());
    {
        let generator = Generator {
            git: &git,
            map: &map,
            state: &state,
            modules: &modules,
            root: paths.root(),
            pending_dir: &pending_dir,
        };
        let mut cursor = None;
        for sha in &commits {
            match generator.commit(sha, &mut seq) {
                Ok(n) => {
                    created += n;
                    cursor = Some(sha);
                }
                Err(e) => {
                    outcome = Err(e.context(format!("no commit {}", short(sha))));
                    break;
                }
            }
        }
        if let Some(sha) = cursor {
            state.upstream.cursor.clone_from(sha);
        }
    }
    // Grava o progresso mesmo se um commit falhou: a próxima execução continua dele.
    state.upstream.synced = task::recompute_synced(&git, &state.upstream.cursor, &pending_dir)?;
    state.save(&paths.upstream_toml())?;
    println!(
        "{created} tarefas novas em sync/pending/ · cursor {} · synced {}",
        short(&state.upstream.cursor),
        short(&state.upstream.synced)
    );
    outcome
}

/// `--dry-run`: o que cada commit pede, por crate, sem gravar nada.
fn report(git: &Git, map: &SyncMap, state: &State, commits: &[String]) -> Result<()> {
    let mut per_crate: BTreeMap<String, usize> = BTreeMap::new();
    let mut in_scope = 0;
    for sha in commits {
        let plans = plan_commit(map, &git.changes(sha)?)
            .with_context(|| format!("no commit {}", short(sha)))?;
        if plans.is_empty() {
            continue;
        }
        in_scope += 1;
        println!("{} {}", short(sha), git.subject(sha)?);
        for (krate, plan) in &plans {
            let note = if state.crates.contains_key(krate) {
                ""
            } else {
                "  (crate não registrada: sem tarefa)"
            };
            println!(
                "    {krate}: {} port, {} vendor{note}",
                plan.port.len(),
                plan.vendor.len() + plan.vendor_deleted.len()
            );
            *per_crate.entry(krate.clone()).or_default() += 1;
        }
    }
    println!("\n{in_scope} de {} commits tocam o escopo", commits.len());
    for (krate, n) in per_crate {
        println!("{n:>5}  {krate}");
    }
    Ok(())
}
```

- [ ] **Passo 5: Rodar os testes**

Run: `cargo test -p xtask`
Esperado: `test result: ok. 21 passed`

- [ ] **Passo 6: Dry-run contra a master real**

Run: `timeout 900 cargo xtask sync --dry-run > target/sync-dry-run.log 2>&1; echo "exit=$?"; head -3 target/sync-dry-run.log; tail -15 target/sync-dry-run.log; git diff --exit-code UPSTREAM.toml && echo ESTADO_INTACTO`
Esperado:
- primeira linha `N commits novos (ec7630a..<sha7>)`;
- commits com linhas `<crate>: X port, Y vendor  (crate não registrada: sem tarefa)`;
- no fim, `M de N commits tocam o escopo` e a contagem por crate;
- `ESTADO_INTACTO`.

Se sair `sem regra em sync/map.toml: <caminho>`, é um arquivo criado na master depois do baseline: classifique-o no `sync/map.toml` (spec §2), rode `cargo xtask check-map --rev origin/master` e repita o dry-run.

- [ ] **Passo 7: Rodar o portão**

Run: `cargo xtask ci`
Esperado: código 0, `test result: ok. 21 passed` e `ok: 3669 arquivos de ec7630a classificados`.

- [ ] **Passo 8: Commit**

```bash
git add xtask/ sync/map.toml
git commit -m "feat(xtask): sync gera tarefas por commit e crate; task apply/done"
```

---

### Tarefa 4: `cargo xtask oracle-build`

**Arquivos:**
- Criar: `xtask/src/oracle.rs`
- Modificar (conteúdo completo abaixo): `xtask/src/main.rs`, `xtask/src/paths.rs`, `xtask/src/git.rs`

**Interfaces:**
- Consome: `State`, `open_upstream`, `Git::rev_parse`, `git::short`, `Paths` (Tarefas 1–3).
- Produz: `oracle::cmake_args(src: &Path, build: &Path) -> Vec<String>`, `oracle::run(&Paths, rev: Option<&str>) -> Result<PathBuf>` (diretório `.upstream/build-<sha7>`, marcado com `.ok` quando completo), `Git::worktree_add(&self, dest: &Path, sha: &str)`, `Paths::upstream_dir`.

- [ ] **Passo 1: Escrever o teste que falha**

`xtask/src/oracle.rs`, só com o módulo de testes:

```rust
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
```

Substitua `xtask/src/main.rs` (versão final, com `oracle-build`):

```rust
//! Automação do llama.rs: sync com o upstream, oráculo e CI.
//! Design: docs/superpowers/specs/2026-10-01-llama-rs-port-design.md

mod check_map;
mod ci;
mod git;
mod map;
mod oracle;
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
    /// Compila o upstream numa revisão para servir de oráculo
    OracleBuild {
        /// Revisão do upstream (padrão: o synced de UPSTREAM.toml)
        rev: Option<String>,
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
        Cmd::OracleBuild { rev } => oracle::run(&paths, rev.as_deref()).map(drop),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erro: {e:#}");
            ExitCode::FAILURE
        }
    }
}
```

- [ ] **Passo 2: Rodar e ver falhar**

Run: `cargo test -p xtask oracle`
Esperado: erro de compilação (`cmake_args` e `oracle::run` inexistentes).

- [ ] **Passo 3: Implementar**

Acima do módulo de testes de `xtask/src/oracle.rs`:

```rust
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
```

Substitua `xtask/src/paths.rs` (versão final, com `upstream_dir`):

```rust
//! Caminhos do repositório usados pelo xtask.

use std::path::{Path, PathBuf};

pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Raiz do workspace: o diretório pai de `xtask/`.
    pub fn from_manifest() -> Self {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self::new(manifest.parent().unwrap_or(manifest))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn upstream_toml(&self) -> PathBuf {
        self.root.join("UPSTREAM.toml")
    }

    pub fn map_toml(&self) -> PathBuf {
        self.root.join("sync/map.toml")
    }

    pub fn pending_dir(&self) -> PathBuf {
        self.root.join("sync/pending")
    }

    pub fn vendor_dir(&self) -> PathBuf {
        self.root.join("vendor/upstream")
    }

    pub fn crates_dir(&self) -> PathBuf {
        self.root.join("crates")
    }

    pub fn upstream_dir(&self) -> PathBuf {
        self.root.join(".upstream")
    }

    pub fn upstream_clone(&self) -> PathBuf {
        self.root.join(".upstream/llama.cpp")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raiz_e_o_workspace() {
        let root = Paths::from_manifest();
        assert!(root.root().join("rust-toolchain.toml").exists());
    }
}
```

Substitua `xtask/src/git.rs` (versão final, com `worktree_add`):

```rust
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
```

- [ ] **Passo 4: Rodar os testes e o portão**

Run: `cargo xtask ci`
Esperado: código 0, `test result: ok. 22 passed` e `ok: 3669 arquivos de ec7630a classificados`.

- [ ] **Passo 5: Pré-requisitos do build do upstream**

Run: `cmake --version | head -1; glslc --version | head -1; pkg-config --modversion vulkan`
Esperado: as três versões impressas. Se faltar alguma, pare e avise o usuário (instalação do sistema).

- [ ] **Passo 6: Compilar o oráculo no baseline**

Leva de 10 a 30 minutos (os 188 shaders Vulkan e os testes). Rode em segundo plano:

Run: `timeout 5400 cargo xtask oracle-build > target/oracle-build.log 2>&1; echo "exit=$?"; tail -3 target/oracle-build.log`
Esperado: `exit=0` e `oráculo pronto: <raiz>/.upstream/build-ec7630a`.

- [ ] **Passo 7: Fumaça do oráculo**

Run: `.upstream/build-ec7630a/bin/test-backend-ops -o ADD -b Vulkan0 2>&1 | tail -5; timeout 60 cargo xtask oracle-build | tail -1`
Esperado: linhas `ADD(...)` com `OK` e nenhuma com `FAIL`. Na segunda chamada, `oráculo já compilado: ...`, porque o cache por SHA evita recompilar.

- [ ] **Passo 8: Commit**

```bash
git add xtask/
git commit -m "feat(xtask): oracle-build compila o upstream por SHA com cache"
```

---

## M0 pronto quando

- [ ] `cargo xtask ci` sai com código 0 (fmt, clippy `-D warnings`, 22 testes, check-map 3669/3669).
- [ ] `cargo xtask sync --dry-run` lista os commits `ec7630a..origin/master` por crate, sem alterar `UPSTREAM.toml`.
- [ ] `.upstream/build-ec7630a/.ok` existe, e `test-backend-ops` roda no Vulkan.
- [ ] Os 4 commits estão no branch.

Próximo plano: M1. Registra as crates `ggml`, `ggml-cpu` e `llama` em `UPSTREAM.toml` com `baseline = cursor` e começa pelos contratos (spec §6.1).
