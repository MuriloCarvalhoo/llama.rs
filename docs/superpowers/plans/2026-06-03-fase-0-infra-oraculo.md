# Fase 0 — Infra + Harness Oráculo: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Workspace Cargo com lints rigorosos, scripts de gate, build do llama.cpp C++ como oráculo somente-leitura, modelos de teste baixados, e crate `oracle` em Rust que captura referências (tokens, texto greedy, dump de tensors) para validação diferencial das fases seguintes.

**Architecture:** Workspace na raiz do repo com um único crate inicial (`oracle/`). O oráculo C++ é compilado out-of-tree em `build-oracle/` (o diretório `llama.cpp/` nunca é modificado). O crate `oracle` shella para os binários `llama-tokenize`, `llama-cli` e `llama-eval-callback` e grava artefatos de referência em `refs/`.

**Tech Stack:** Rust 1.96 (edition 2024), thiserror, serde_json, cmake 4.3 + gcc 16 (oráculo), bash (scripts).

**Fatos verificados nesta máquina:** cargo 1.96.0, cmake 4.3.3, gcc 16.1.1, 56 cores, sem remote git. `llama-tokenize --ids` imprime `[1, 2, 3]` (verificado em `tools/tokenize/tokenize.cpp:30`). Binário `llama-eval-callback` existe (`examples/eval-callback/CMakeLists.txt:1`).

---

## Estrutura de arquivos da fase

| Arquivo | Responsabilidade |
|---|---|
| `Cargo.toml` | Workspace, deps compartilhadas, lints |
| `rust-toolchain.toml` | Pin do toolchain |
| `clippy.toml` | Exceções de lint para testes |
| `.gitignore` | target, build-oracle, models, tensors |
| `scripts/gate.sh` | Gate de validação (fmt, clippy, test, cobertura) |
| `scripts/build-oracle.sh` | Compila o oráculo C++ |
| `scripts/get-model.sh` | Baixa modelos de teste |
| `oracle/Cargo.toml` | Crate do harness |
| `oracle/src/lib.rs` | Raiz do crate (forbid unsafe, módulos) |
| `oracle/src/error.rs` | Tipo de erro do harness |
| `oracle/src/parse.rs` | Parser da saída `--ids` (puro, testável sem binários) |
| `oracle/src/runner.rs` | Execução dos binários do oráculo |
| `oracle/src/main.rs` | Bin de captura: gera `refs/*` |
| `refs/` | Artefatos de referência (tokens.json e greedy.txt commitados) |

---

### Task 1: Workspace Cargo + lints

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `clippy.toml`, `.gitignore`
- Create: `oracle/Cargo.toml`, `oracle/src/lib.rs`

- [ ] **Step 1: Criar `Cargo.toml` na raiz**

```toml
[workspace]
resolver = "3"
members = ["oracle"]

[workspace.package]
edition = "2024"

[workspace.dependencies]
thiserror = "2"
serde_json = "1"

[workspace.lints.rust]
unsafe_code = "deny"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
cast_possible_truncation = "deny"
cast_sign_loss = "deny"
cast_possible_wrap = "deny"
indexing_slicing = "warn"
```

(Crates de kernel futuros — `ggml-cpu`, `ggml-vulkan` — definirão sua própria tabela `[lints]` sem `workspace = true` para liberar `unsafe` localmente.)

- [ ] **Step 2: Criar `rust-toolchain.toml`**

```toml
[toolchain]
channel = "1.96.0"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 3: Criar `clippy.toml`**

```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
allow-panic-in-tests = true
```

- [ ] **Step 4: Criar `.gitignore`**

```gitignore
/target
/build-oracle
/models
/refs/tensors.txt
```

- [ ] **Step 5: Criar `oracle/Cargo.toml`**

```toml
[package]
name = "oracle"
version = "0.1.0"
edition.workspace = true

[dependencies]
thiserror.workspace = true
serde_json.workspace = true

[lints]
workspace = true
```

- [ ] **Step 6: Criar `oracle/src/lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Harness diferencial: executa o llama.cpp C++ (oráculo) e captura
//! tokens, texto greedy e dumps de tensors como referência.
```

- [ ] **Step 7: Verificar que o workspace compila e passa lints**

Run: `cargo build && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
Expected: build verde, sem warnings.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml .gitignore oracle/
git commit -m "chore: workspace Cargo com lints rigorosos e crate oracle vazio"
```

---

### Task 2: Script de gate

**Files:**
- Create: `scripts/gate.sh`

- [ ] **Step 1: Criar `scripts/gate.sh`**

```bash
#!/usr/bin/env bash
# Gate de validação por tarefa — itens 2, 3 e 5 do gate da spec.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

if command -v cargo-llvm-cov >/dev/null 2>&1; then
    cargo llvm-cov --workspace --fail-under-lines 80
else
    echo "AVISO: cargo-llvm-cov não instalado — cobertura não verificada (cargo install cargo-llvm-cov --locked)"
fi
echo "GATE OK"
```

- [ ] **Step 2: Tornar executável e rodar**

Run: `chmod +x scripts/gate.sh && ./scripts/gate.sh`
Expected: termina com `GATE OK` (com o aviso de cobertura se cargo-llvm-cov não estiver instalado).

- [ ] **Step 3: Instalar cargo-llvm-cov (uma vez)**

Run: `cargo install cargo-llvm-cov --locked && rustup component add llvm-tools`
Expected: instalação concluída. Rodar `./scripts/gate.sh` de novo — cobertura agora executa (workspace quase vazio passa trivialmente).

- [ ] **Step 4: Commit**

```bash
git add scripts/gate.sh
git commit -m "chore: script de gate (fmt, clippy, test, cobertura)"
```

---

### Task 3: Build do oráculo C++

**Files:**
- Create: `scripts/build-oracle.sh`

- [ ] **Step 1: Criar `scripts/build-oracle.sh`**

```bash
#!/usr/bin/env bash
# Compila o llama.cpp upstream (somente leitura) out-of-tree como oráculo.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S llama.cpp -B build-oracle \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF
cmake --build build-oracle -j"$(nproc)" \
    --target llama-cli llama-tokenize llama-eval-callback
build-oracle/bin/llama-cli --version
```

- [ ] **Step 2: Rodar o build**

Run: `chmod +x scripts/build-oracle.sh && ./scripts/build-oracle.sh`
Expected: compila (alguns minutos com 56 cores) e imprime a versão do llama-cli ao final. Se o cmake falhar por opção inexistente (`LLAMA_CURL`), remover a flag e rodar de novo — nomes de opções mudam entre versões do upstream.

- [ ] **Step 3: Verificar que os 3 binários existem**

Run: `ls build-oracle/bin/llama-cli build-oracle/bin/llama-tokenize build-oracle/bin/llama-eval-callback`
Expected: os 3 caminhos listados.

- [ ] **Step 4: Verificar que `llama.cpp/` não foi modificado**

Run: `git -C llama.cpp status --porcelain | head`
Expected: saída vazia (build foi 100% out-of-tree).

- [ ] **Step 5: Commit**

```bash
git add scripts/build-oracle.sh
git commit -m "chore: script de build do oráculo C++ out-of-tree"
```

---

### Task 4: Modelos de teste

**Files:**
- Create: `scripts/get-model.sh`

- [ ] **Step 1: Criar `scripts/get-model.sh`**

```bash
#!/usr/bin/env bash
# Baixa os modelos de teste:
#  - stories260K: arch llama minúscula (usada pelo CI do próprio llama.cpp) — debug camada a camada
#  - qwen2.5-0.5b q8_0: modelo realista para validação ponta a ponta
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

STORIES_URL="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

[ -f models/stories260K.gguf ] || curl -fL --retry 3 -o models/stories260K.gguf "$STORIES_URL"
[ -f models/qwen2.5-0.5b-instruct-q8_0.gguf ] || curl -fL --retry 3 -o models/qwen2.5-0.5b-instruct-q8_0.gguf "$QWEN_URL"
ls -lh models/
```

- [ ] **Step 2: Rodar o download**

Run: `chmod +x scripts/get-model.sh && ./scripts/get-model.sh`
Expected: `models/stories260K.gguf` (~1–2 MB) e `models/qwen2.5-0.5b-instruct-q8_0.gguf` (~650 MB). Se alguma URL retornar 404 (arquivos no HF mudam de nome), localizar o substituto com `curl -sL "https://huggingface.co/api/models/ggml-org/models/tree/main/tinyllamas"` e ajustar a URL no script.

- [ ] **Step 3: Smoke test do oráculo com o modelo**

Run: `build-oracle/bin/llama-tokenize -m models/stories260K.gguf -p "Once upon a time" --ids --log-disable`
Expected: uma linha no formato `[1, 80, 59, ...]` (IDs exatos variam com o tokenizer do modelo).

- [ ] **Step 4: Commit**

```bash
git add scripts/get-model.sh
git commit -m "chore: script de download dos modelos de teste"
```

---

### Task 5: Parser da saída do oráculo (TDD puro)

**Files:**
- Create: `oracle/src/parse.rs`, `oracle/src/error.rs`
- Modify: `oracle/src/lib.rs`

- [ ] **Step 1: Criar `oracle/src/error.rs`**

```rust
#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("falha ao executar {0}: {1}")]
    Io(String, #[source] std::io::Error),
    #[error("{0} terminou com status {1}")]
    NonZero(String, i32),
    #[error("saída do oráculo não reconhecida: {0:?}")]
    Parse(String),
}
```

- [ ] **Step 2: Criar `oracle/src/parse.rs` SOMENTE com os testes (RED)**

```rust
use crate::error::OracleError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bracketed_id_list() {
        let ids = parse_token_ids("[1, 15043, 3186]").unwrap();
        assert_eq!(ids, vec![1, 15043, 3186]);
    }

    #[test]
    fn parses_ids_with_surrounding_log_noise() {
        let out = "load: vocab loaded\n[1, 2, 3]\n";
        assert_eq!(parse_token_ids(out).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn parses_empty_list() {
        assert_eq!(parse_token_ids("[]").unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn rejects_output_without_brackets() {
        assert!(matches!(
            parse_token_ids("error: model not found"),
            Err(OracleError::Parse(_))
        ));
    }

    #[test]
    fn rejects_non_numeric_entries() {
        assert!(matches!(parse_token_ids("[1, x, 3]"), Err(OracleError::Parse(_))));
    }
}
```

E registrar os módulos em `oracle/src/lib.rs` (substituir o conteúdo):

```rust
#![forbid(unsafe_code)]
//! Harness diferencial: executa o llama.cpp C++ (oráculo) e captura
//! tokens, texto greedy e dumps de tensors como referência.

mod error;
mod parse;

pub use error::OracleError;
pub use parse::parse_token_ids;
```

- [ ] **Step 3: Rodar e ver falhar (não compila — função não existe)**

Run: `cargo test -p oracle`
Expected: FAIL — `cannot find function parse_token_ids`.

- [ ] **Step 4: Implementar `parse_token_ids` em `oracle/src/parse.rs` (GREEN) — acima do `mod tests`**

```rust
/// Extrai os IDs da saída `--ids` do llama-tokenize (formato `[1, 2, 3]`,
/// possivelmente cercado de logs em outras linhas).
pub fn parse_token_ids(output: &str) -> Result<Vec<i64>, OracleError> {
    let err = || OracleError::Parse(output.to_owned());
    let start = output.find('[').ok_or_else(err)?;
    let end = output.rfind(']').ok_or_else(err)?;
    let inner = output.get(start.saturating_add(1)..end).ok_or_else(err)?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|tok| {
            tok.trim()
                .parse::<i64>()
                .map_err(|_| OracleError::Parse(tok.to_owned()))
        })
        .collect()
}
```

- [ ] **Step 5: Rodar e ver passar**

Run: `cargo test -p oracle`
Expected: 5 testes PASS.

- [ ] **Step 6: Gate completo + commit**

```bash
./scripts/gate.sh
git add oracle/
git commit -m "feat: parser da saída --ids do oráculo (TDD)"
```

---

### Task 6: Runner do oráculo

**Files:**
- Create: `oracle/src/runner.rs`, `oracle/tests/oracle_integration.rs`
- Modify: `oracle/src/lib.rs`

- [ ] **Step 1: Escrever o teste de integração primeiro — `oracle/tests/oracle_integration.rs`**

```rust
//! Testes que exigem o oráculo compilado (scripts/build-oracle.sh)
//! e o modelo baixado (scripts/get-model.sh). Rodar com:
//!     cargo test -p oracle -- --ignored

use oracle::Oracle;

fn oracle_under_test() -> Oracle {
    Oracle::new("build-oracle/bin", "models/stories260K.gguf")
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn tokenize_returns_nonempty_ids() {
    let ids = oracle_under_test().tokenize("Once upon a time").unwrap();
    assert!(!ids.is_empty());
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn tokenize_is_deterministic() {
    let o = oracle_under_test();
    assert_eq!(o.tokenize("hello").unwrap(), o.tokenize("hello").unwrap());
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn greedy_generation_is_deterministic() {
    let o = oracle_under_test();
    let a = o.generate_greedy("Once upon a time", 16).unwrap();
    let b = o.generate_greedy("Once upon a time", 16).unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, b);
}

#[test]
#[ignore = "requer oráculo C++ compilado e modelo baixado"]
fn missing_binary_is_reported_as_io_error() {
    let o = Oracle::new("caminho/inexistente", "models/stories260K.gguf");
    assert!(matches!(o.tokenize("x"), Err(oracle::OracleError::Io(_, _))));
}
```

(Os testes de integração rodam com cwd na raiz do pacote `oracle/`? Não — o cargo define cwd do teste como o diretório do **pacote**. Por isso os caminhos relativos `build-oracle/bin` precisam de um nível acima: usar `Oracle::new("../build-oracle/bin", "../models/stories260K.gguf")` se o teste falhar com `Io`. O Step 4 valida qual dos dois é o correto e fixa.)

- [ ] **Step 2: Rodar e ver falhar (não compila — `Oracle` não existe)**

Run: `cargo test -p oracle -- --ignored`
Expected: FAIL — `cannot find struct Oracle`.

- [ ] **Step 3: Implementar `oracle/src/runner.rs`**

```rust
use std::path::PathBuf;
use std::process::Command;

use crate::error::OracleError;
use crate::parse::parse_token_ids;

/// Executa os binários do llama.cpp compilado (o oráculo).
pub struct Oracle {
    bin_dir: PathBuf,
    model: PathBuf,
}

impl Oracle {
    pub fn new(bin_dir: impl Into<PathBuf>, model: impl Into<PathBuf>) -> Self {
        Self { bin_dir: bin_dir.into(), model: model.into() }
    }

    /// Tokeniza `text` com o tokenizer do oráculo. Equivale a:
    /// `llama-tokenize -m <model> -p <text> --ids --log-disable`
    pub fn tokenize(&self, text: &str) -> Result<Vec<i64>, OracleError> {
        let out = self.run(
            "llama-tokenize",
            &["-m", &self.model_arg(), "-p", text, "--ids", "--log-disable"],
        )?;
        parse_token_ids(&out.stdout)
    }

    /// Gera `n_tokens` com sampling greedy determinístico; retorna o texto.
    pub fn generate_greedy(&self, prompt: &str, n_tokens: u32) -> Result<String, OracleError> {
        let n = n_tokens.to_string();
        let out = self.run(
            "llama-cli",
            &[
                "-m", &self.model_arg(),
                "-p", prompt,
                "-n", &n,
                "--temp", "0",
                "--seed", "42",
                "-no-cnv",
                "--no-display-prompt",
                "--simple-io",
            ],
        )?;
        Ok(out.stdout)
    }

    /// Dump dos tensors intermediários do forward pass
    /// (saída do llama-eval-callback, stdout+stderr concatenados).
    pub fn dump_tensors(&self, prompt: &str) -> Result<String, OracleError> {
        let out = self.run(
            "llama-eval-callback",
            &["-m", &self.model_arg(), "-p", prompt, "-n", "1"],
        )?;
        let mut full = out.stdout;
        full.push_str(&out.stderr);
        Ok(full)
    }

    fn model_arg(&self) -> String {
        self.model.to_string_lossy().into_owned()
    }

    fn run(&self, bin: &str, args: &[&str]) -> Result<RunOutput, OracleError> {
        let path = self.bin_dir.join(bin);
        let out = Command::new(&path)
            .args(args)
            .output()
            .map_err(|e| OracleError::Io(bin.to_owned(), e))?;
        if !out.status.success() {
            return Err(OracleError::NonZero(bin.to_owned(), out.status.code().unwrap_or(-1)));
        }
        Ok(RunOutput {
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

struct RunOutput {
    stdout: String,
    stderr: String,
}
```

E atualizar `oracle/src/lib.rs` (substituir o conteúdo):

```rust
#![forbid(unsafe_code)]
//! Harness diferencial: executa o llama.cpp C++ (oráculo) e captura
//! tokens, texto greedy e dumps de tensors como referência.

mod error;
mod parse;
mod runner;

pub use error::OracleError;
pub use parse::parse_token_ids;
pub use runner::Oracle;
```

- [ ] **Step 4: Rodar os testes de integração e ver passar**

Run: `cargo test -p oracle -- --ignored`
Expected: 4 testes PASS. Dois ajustes possíveis e como resolver:
- Falha com `Io`: o cwd dos testes é `oracle/` — trocar os caminhos do helper para `../build-oracle/bin` e `../models/stories260K.gguf`.
- Falha no greedy porque alguma flag do llama-cli mudou de nome (ex.: `-no-cnv`): rodar `build-oracle/bin/llama-cli --help | grep -i conv` e ajustar a flag no runner. O contrato dos testes não muda.

- [ ] **Step 5: Gate completo + commit**

```bash
./scripts/gate.sh
git add oracle/
git commit -m "feat: runner do oráculo (tokenize, greedy, dump de tensors)"
```

---

### Task 7: Bin de captura de referências

**Files:**
- Create: `oracle/src/main.rs`
- Create (gerados e commitados): `refs/tokens.json`, `refs/greedy.txt`

- [ ] **Step 1: Criar `oracle/src/main.rs`**

```rust
#![forbid(unsafe_code)]
//! Gera os artefatos de referência em refs/ a partir do oráculo C++.
//! Uso (na raiz do workspace): cargo run -p oracle
//! Env opcionais: ORACLE_BIN_DIR (default build-oracle/bin),
//!                ORACLE_MODEL  (default models/stories260K.gguf)

use std::fs;

use oracle::Oracle;

const PROMPT: &str = "Once upon a time";
const N_TOKENS: u32 = 32;
const CORPUS: &[&str] = &[
    "Once upon a time",
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Era uma vez uma menina",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bin_dir =
        std::env::var("ORACLE_BIN_DIR").unwrap_or_else(|_| "build-oracle/bin".to_owned());
    let model =
        std::env::var("ORACLE_MODEL").unwrap_or_else(|_| "models/stories260K.gguf".to_owned());
    let oracle = Oracle::new(bin_dir, &model);

    fs::create_dir_all("refs")?;

    let tokens: Vec<serde_json::Value> = CORPUS
        .iter()
        .map(|text| Ok(serde_json::json!({ "text": text, "ids": oracle.tokenize(text)? })))
        .collect::<Result<_, oracle::OracleError>>()?;
    fs::write(
        "refs/tokens.json",
        serde_json::to_string_pretty(&serde_json::json!({ "model": model, "cases": tokens }))?,
    )?;

    fs::write("refs/greedy.txt", oracle.generate_greedy(PROMPT, N_TOKENS)?)?;
    fs::write("refs/tensors.txt", oracle.dump_tensors(PROMPT)?)?;

    println!("refs/ atualizadas (tokens.json, greedy.txt, tensors.txt)");
    Ok(())
}
```

- [ ] **Step 2: Rodar a captura (na raiz do workspace)**

Run: `cargo run -p oracle`
Expected: imprime `refs/ atualizadas ...`. Conferir: `refs/tokens.json` no formato `{"model": "...", "cases": [{"text": "...", "ids": [1, 2, 3]}, ...]}`; `refs/greedy.txt` com texto não-vazio; `refs/tensors.txt` com dumps de tensors.

- [ ] **Step 3: Conferir determinismo e gitignore**

Run: `cargo run -p oracle && git status --porcelain refs/ && git diff --stat refs/ 2>/dev/null`
Expected: rodou 2ª vez sem mudar conteúdo; apenas `refs/tokens.json` e `refs/greedy.txt` aparecem como novos (tensors.txt ignorado pelo .gitignore).

- [ ] **Step 4: Gate completo + commit**

```bash
./scripts/gate.sh
git add oracle/src/main.rs refs/tokens.json refs/greedy.txt
git commit -m "feat: captura de referências do oráculo (tokens, greedy, tensors)"
```

---

### Task 8: Hook de formatação automática

**Files:**
- Modify: `.claude/settings.json` (criar se não existir; **ler o conteúdo atual antes e mesclar** — não sobrescrever chaves existentes)

- [ ] **Step 1: Ler o estado atual**

Run: `ls .claude/ && cat .claude/settings.json 2>/dev/null || echo "(sem settings.json)"`
Expected: ver o que existe para mesclar sem perder configuração.

- [ ] **Step 2: Adicionar o hook PostToolUse (mesclando com o JSON existente, se houver)**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "f=$(jq -r '.tool_input.file_path // empty'); case \"$f\" in *.rs) cd \"$CLAUDE_PROJECT_DIR\" && cargo fmt --all 2>/dev/null ;; esac; true"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Verificar que o JSON é válido**

Run: `jq . .claude/settings.json`
Expected: JSON impresso sem erro.

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.json
git commit -m "chore: hook PostToolUse de cargo fmt em edits de arquivos .rs"
```

---

## Critério de aceite da Fase 0 (da spec)

- [ ] `./scripts/gate.sh` termina com `GATE OK` (com cobertura ativa)
- [ ] `./scripts/build-oracle.sh` produz os 3 binários sem modificar `llama.cpp/`
- [ ] `cargo test -p oracle -- --ignored` passa (oráculo + modelo funcionando ponta a ponta)
- [ ] `cargo run -p oracle` (2×) regenera `refs/` deterministicamente — `git diff refs/` vazio na 2ª execução
