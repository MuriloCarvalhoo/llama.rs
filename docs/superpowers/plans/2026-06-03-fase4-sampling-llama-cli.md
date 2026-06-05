# Fase 4 — Sampling + `llama-cli` End-to-End — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Criar `crates/llama-sampling` com estratégias de amostragem (greedy, temperatura, top-k, top-p) e `crates/llama-cli` com um binário que gera texto ponta a ponta; gate: saída greedy no stories260K idêntica ao oráculo C++.

**Architecture:** `llama-sampling` expõe `Sampler { Greedy, Temperature, TopK, TopP }` com `sample(logits, rng) -> usize`. `llama-model::Model::generate` adiciona geração com qualquer `Sampler`; `generate_greedy` permanece para retro-compatibilidade com o oracle gate da Fase 2. `llama-cli` lê args via `clap`, carrega modelo, chama `generate`, imprime resultado. Gate diferencial via `tests/greedy_gate.rs` que compara output com `refs/greedy.txt`.

**Tech Stack:** `rand = "0.9"` (`SmallRng`, `seed_from_u64`, `Rng::random()`), `clap = "4"` (derive), deps workspace existentes.

---

## File Structure

- Create: `crates/llama-sampling/Cargo.toml`
- Create: `crates/llama-sampling/src/lib.rs`
- Create: `crates/llama-sampling/src/sampler.rs`
- Create: `crates/llama-cli/Cargo.toml`
- Create: `crates/llama-cli/src/main.rs`
- Create: `crates/llama-cli/src/lib.rs`
- Create: `crates/llama-cli/src/args.rs`
- Create: `crates/llama-cli/src/runner.rs`
- Create: `crates/llama-cli/tests/greedy_gate.rs`
- Create: `crates/llama-cli/tests/args_test.rs`
- Modify: `Cargo.toml` (workspace) — membros + `rand` + `clap` + `llama-sampling`
- Modify: `crates/llama-model/Cargo.toml` — deps `llama-sampling` + `rand`
- Modify: `crates/llama-model/src/generate.rs` — adicionar `Model::generate` com `Sampler`

---

## Task 0: Workspace scaffold + `crates/llama-sampling` shell

**Files:**
- Modify: `Cargo.toml` (workspace)
- Create: `crates/llama-sampling/Cargo.toml`
- Create: `crates/llama-sampling/src/lib.rs`
- Create: `crates/llama-sampling/src/sampler.rs` (stubs + greedy)

- [ ] **Step 1: Atualizar `Cargo.toml` raiz**

Em `[workspace]`, atualizar `members`:
```toml
members = ["oracle", "crates/gguf", "crates/llama-tokenizer", "crates/llama-model", "crates/ggml-cpu", "crates/llama-sampling", "crates/llama-cli"]
```

Em `[workspace.dependencies]`, adicionar:
```toml
rand = "0.9"
clap = { version = "4", features = ["derive"] }
llama-sampling = { path = "crates/llama-sampling" }
llama-cli = { path = "crates/llama-cli" }
```

- [ ] **Step 2: Criar `crates/llama-sampling/Cargo.toml`**

```toml
[package]
name = "llama-sampling"
version = "0.1.0"
edition.workspace = true

[dependencies]
rand.workspace = true

[lints]
workspace = true
```

- [ ] **Step 3: Criar `crates/llama-sampling/src/lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Estratégias de amostragem para inferência de LLMs.

mod sampler;
pub use sampler::Sampler;
```

- [ ] **Step 4: Criar `crates/llama-sampling/src/sampler.rs` (stubs + greedy)**

```rust
//! Estratégias de amostragem: greedy, temperatura, top-k, top-p.
#![allow(clippy::indexing_slicing)]

use rand::Rng;

/// Estratégia de amostragem para selecionar o próximo token a partir de logits.
#[derive(Clone, Debug)]
pub enum Sampler {
    /// Argmax — determinístico, equivale a temperatura zero.
    Greedy,
    /// Multinomial com rescala de logits por `1/temp`. Se `temp == 0.0` → greedy.
    Temperature { temp: f32 },
    /// Mantém os `k` maiores logits antes de amostrar. Se `temp == 0.0` → greedy.
    TopK { k: usize, temp: f32 },
    /// Mantém o menor conjunto de tokens com prob. acumulada >= `p` antes de amostrar.
    TopP { p: f32, temp: f32 },
}

impl Sampler {
    /// Retorna o índice do token amostrado dado o vetor de logits.
    pub fn sample(&self, logits: &[f32], rng: &mut impl Rng) -> usize {
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::Temperature { temp } => {
                todo!("implementado na Task 1: temp={temp}")
            }
            Sampler::TopK { k, temp } => {
                todo!("implementado na Task 2: k={k} temp={temp}")
            }
            Sampler::TopP { p, temp } => {
                todo!("implementado na Task 2: p={p} temp={temp}")
            }
        }
    }
}

pub(crate) fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

pub(crate) fn sample_multinomial(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.random();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
        assert_eq!(Sampler::Greedy.sample(&logits, &mut rng()), 3);
    }

    #[test]
    fn greedy_single_token() {
        assert_eq!(Sampler::Greedy.sample(&[1.0f32], &mut rng()), 0);
    }

    #[test]
    fn argmax_picks_max_index() {
        assert_eq!(argmax(&[0.0, 1.0, 0.5]), 1);
    }

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn softmax_with_negative_logits() {
        let probs = softmax(&[-1.0, -2.0, -3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn sample_multinomial_single_prob() {
        let mut r = SmallRng::seed_from_u64(1);
        assert_eq!(sample_multinomial(&[1.0], &mut r), 0);
    }
}
```

- [ ] **Step 5: Verificar build**

Run: `cargo build -p llama-sampling`
Expected: PASS (warnings de `todo!` aceitáveis).

- [ ] **Step 6: Rodar testes**

Run: `cargo test -p llama-sampling -- --nocapture`
Expected: PASS (greedy_returns_argmax, greedy_single_token, argmax_picks_max_index, softmax_sums_to_one, softmax_with_negative_logits, sample_multinomial_single_prob).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock crates/llama-sampling
git commit -m "chore(llama-sampling): scaffold crate com Sampler + stubs (Fase 4 Task 0)"
```

---

## Task 1: `Sampler::Temperature`

**Files:**
- Modify: `crates/llama-sampling/src/sampler.rs`

- [ ] **Step 1: Escrever os testes (RED)**

Adicionar em `mod tests` (após os testes existentes):

```rust
#[test]
fn temperature_zero_is_greedy() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
    let result = Sampler::Temperature { temp: 0.0 }.sample(&logits, &mut rng());
    assert_eq!(result, 3, "temp=0 deve ser greedy");
}

#[test]
fn temperature_skewed_picks_dominant() {
    // Logits muito concentrados: token 3 (100.0) deve vencer sempre
    let logits = vec![0.0f32, 0.0, 0.0, 100.0, 0.0];
    let mut r = rng();
    let choices: Vec<usize> = (0..20)
        .map(|_| Sampler::Temperature { temp: 1.0 }.sample(&logits, &mut r))
        .collect();
    assert!(choices.iter().all(|&t| t == 3), "esperado sempre 3, got {choices:?}");
}

#[test]
fn temperature_uniform_logits_shows_variety() {
    let logits = vec![1.0f32; 10];
    let mut r = SmallRng::seed_from_u64(99);
    let choices: Vec<usize> = (0..50)
        .map(|_| Sampler::Temperature { temp: 1.0 }.sample(&logits, &mut r))
        .collect();
    let unique: std::collections::HashSet<usize> = choices.into_iter().collect();
    assert!(unique.len() > 1, "esperado variedade com logits uniformes");
}
```

Run: `cargo test -p llama-sampling -- temperature 2>&1 | head -5`
Expected: **FAIL** (todo! panic).

- [ ] **Step 2: Implementar `Sampler::Temperature`**

Substituir o arm `Sampler::Temperature { temp }` dentro do `match` em `sample`:

```rust
Sampler::Temperature { temp } => {
    if *temp == 0.0 {
        return argmax(logits);
    }
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
    let probs = softmax(&scaled);
    sample_multinomial(&probs, rng)
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p llama-sampling`
Expected: PASS (todos os testes até aqui).

- [ ] **Step 4: Commit**

```bash
git add crates/llama-sampling/src/sampler.rs
git commit -m "feat(llama-sampling): Sampler::Temperature (Fase 4 Task 1)"
```

---

## Task 2: `Sampler::TopK` + `Sampler::TopP`

**Files:**
- Modify: `crates/llama-sampling/src/sampler.rs`

- [ ] **Step 1: Escrever os testes (RED)**

Adicionar em `mod tests`:

```rust
#[test]
fn topk_one_is_greedy() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
    let result = Sampler::TopK { k: 1, temp: 1.0 }.sample(&logits, &mut rng());
    assert_eq!(result, 3, "k=1 deve ser greedy");
}

#[test]
fn topk_zero_temp_is_greedy() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
    let result = Sampler::TopK { k: 40, temp: 0.0 }.sample(&logits, &mut rng());
    assert_eq!(result, 3);
}

#[test]
fn topk_returns_valid_index() {
    let logits = vec![100.0f32, 0.0, 0.0, 0.0, 0.0];
    let mut r = rng();
    for _ in 0..20 {
        let t = Sampler::TopK { k: 2, temp: 1.0 }.sample(&logits, &mut r);
        assert!(t < logits.len(), "índice {t} fora do range");
    }
}

#[test]
fn topp_zero_temp_is_greedy() {
    let logits = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
    let result = Sampler::TopP { p: 1.0, temp: 0.0 }.sample(&logits, &mut rng());
    assert_eq!(result, 3);
}

#[test]
fn topp_tight_picks_dominant() {
    // p tiny -> apenas o token mais provável (tok2=100.0) é mantido
    let logits = vec![0.0f32, 0.0, 100.0, 0.0, 0.0];
    let mut r = rng();
    for _ in 0..10 {
        let t = Sampler::TopP { p: 0.001, temp: 1.0 }.sample(&logits, &mut r);
        assert_eq!(t, 2, "com p tiny, só tok2 deve ser escolhido");
    }
}

#[test]
fn topp_wide_shows_variety() {
    let logits = vec![1.0f32; 10];
    let mut r = SmallRng::seed_from_u64(77);
    let choices: Vec<usize> = (0..50)
        .map(|_| Sampler::TopP { p: 0.95, temp: 1.0 }.sample(&logits, &mut r))
        .collect();
    let unique: std::collections::HashSet<usize> = choices.into_iter().collect();
    assert!(unique.len() > 1, "esperado variedade com logits uniformes e p=0.95");
}
```

Run: `cargo test -p llama-sampling -- topk topp 2>&1 | head -5`
Expected: **FAIL** (todo! panic).

- [ ] **Step 2: Implementar `Sampler::TopK`**

Substituir o arm `Sampler::TopK { k, temp }`:

```rust
Sampler::TopK { k, temp } => {
    if *temp == 0.0 {
        return argmax(logits);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate((*k).max(1));
    let scaled: Vec<f32> = indexed.iter().map(|(_, v)| v / temp).collect();
    let probs = softmax(&scaled);
    let pick = sample_multinomial(&probs, rng);
    indexed.get(pick).map_or(0, |t| t.0)
}
```

- [ ] **Step 3: Implementar `Sampler::TopP`**

Substituir o arm `Sampler::TopP { p, temp }`:

```rust
Sampler::TopP { p, temp } => {
    if *temp == 0.0 {
        return argmax(logits);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let scaled: Vec<f32> = indexed.iter().map(|(_, v)| v / temp).collect();
    let probs = softmax(&scaled);
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &pr) in probs.iter().enumerate() {
        cumsum += pr;
        if cumsum >= *p {
            cutoff = i + 1;
            break;
        }
    }
    let kept = &probs[..cutoff];
    let total: f32 = kept.iter().sum();
    let normalized: Vec<f32> = kept.iter().map(|&pr| pr / total).collect();
    let pick = sample_multinomial(&normalized, rng);
    indexed.get(pick).map_or(0, |t| t.0)
}
```

- [ ] **Step 4: Rodar todos os testes**

Run: `cargo test -p llama-sampling`
Expected: PASS (todos os ~14 testes).

- [ ] **Step 5: Commit**

```bash
git add crates/llama-sampling/src/sampler.rs
git commit -m "feat(llama-sampling): Sampler::TopK + Sampler::TopP (Fase 4 Task 2)"
```

---

## Task 3: `Model::generate` em `llama-model`

**Files:**
- Modify: `crates/llama-model/Cargo.toml`
- Modify: `crates/llama-model/src/generate.rs`

- [ ] **Step 1: Adicionar deps em `crates/llama-model/Cargo.toml`**

```toml
[dependencies]
thiserror.workspace = true
gguf.workspace = true
llama-tokenizer.workspace = true
ggml-cpu.workspace = true
llama-sampling.workspace = true
rand.workspace = true
```

- [ ] **Step 2: Escrever o teste (RED)**

Adicionar ao final de `crates/llama-model/src/generate.rs` um novo bloco de testes:

```rust
#[cfg(test)]
mod generate_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use llama_sampling::Sampler;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::path::Path;

    fn load() -> Option<(Model, llama_tokenizer::Tokenizer)> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = gguf::GgufFile::parse(&bytes).ok()?;
        let model = Model::load(&f, &bytes).ok()?;
        let tok = llama_tokenizer::Tokenizer::from_gguf(&f).ok()?;
        Some((model, tok))
    }

    #[test]
    fn generate_with_greedy_sampler_matches_generate_greedy() {
        let Some((model, tok)) = load() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let mut rng = SmallRng::seed_from_u64(42);
        let out_gen = model
            .generate(&tok, "Once upon a time", 8, &Sampler::Greedy, &mut rng)
            .unwrap();
        let out_greedy = model.generate_greedy(&tok, "Once upon a time", 8).unwrap();
        assert_eq!(
            out_gen, out_greedy,
            "Sampler::Greedy deve produzir mesma saída que generate_greedy"
        );
    }
}
```

Run: `cargo test -p llama-model generate_tests 2>&1 | head -5`
Expected: **FAIL** (método `generate` não existe).

- [ ] **Step 3: Adicionar imports ao topo de `generate.rs`**

No início de `crates/llama-model/src/generate.rs`, garantir os imports:

```rust
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::Rng;

use crate::error::ModelError;
use crate::model::Model;
```

- [ ] **Step 4: Implementar `Model::generate`**

Adicionar o seguinte método **antes** de `generate_greedy` no bloco `impl Model`:

```rust
/// Gera ate `n_tokens` usando a estrategia `sampler`.
/// Retorna o texto gerado (sem o prompt), parando em EOS.
pub fn generate(
    &self,
    tokenizer: &Tokenizer,
    prompt: &str,
    n_tokens: usize,
    sampler: &Sampler,
    rng: &mut impl Rng,
) -> Result<String, ModelError> {
    let prompt_ids = tokenizer.encode(prompt, true);
    let mut cache = self.new_cache();

    let logits = self.forward(&prompt_ids, &mut cache)?;
    let first_idx = sampler.sample(&logits, rng);
    let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;

    let mut generated = Vec::with_capacity(n_tokens);
    while generated.len() < n_tokens {
        if next == self.config.eos_id {
            break;
        }
        generated.push(next);
        let logits = self.forward(&[next], &mut cache)?;
        let idx = sampler.sample(&logits, rng);
        next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
    }

    Ok(tokenizer.decode(&generated))
}
```

- [ ] **Step 5: Rodar o teste novo**

Run: `cargo test -p llama-model generate_tests -- --nocapture`
Expected: PASS.

- [ ] **Step 6: Verificar que o gate da Fase 2 continua verde**

Run: `cargo test -p llama-model --test oracle_forward -- --nocapture`
Expected: PASS — sequência greedy inalterada.

- [ ] **Step 7: Commit**

```bash
git add crates/llama-model/Cargo.toml crates/llama-model/src/generate.rs
git commit -m "feat(llama-model): Model::generate com Sampler (Fase 4 Task 3)"
```

---

## Task 4: Scaffold `crates/llama-cli` + args

**Files:**
- Create: `crates/llama-cli/Cargo.toml`
- Create: `crates/llama-cli/src/args.rs`
- Create: `crates/llama-cli/src/runner.rs` (stub)
- Create: `crates/llama-cli/src/lib.rs`
- Create: `crates/llama-cli/src/main.rs`
- Create: `crates/llama-cli/tests/args_test.rs`

- [ ] **Step 1: Criar `crates/llama-cli/Cargo.toml`**

```toml
[package]
name = "llama-cli"
version = "0.1.0"
edition.workspace = true

[[bin]]
name = "llama-cli"
path = "src/main.rs"

[dependencies]
thiserror.workspace = true
gguf.workspace = true
llama-tokenizer.workspace = true
llama-model.workspace = true
llama-sampling.workspace = true
rand.workspace = true
clap = { version = "4", features = ["derive"] }

[lints]
workspace = true
```

- [ ] **Step 2: Criar `crates/llama-cli/src/args.rs`**

```rust
//! Argumentos de linha de comando para `llama-cli`.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "llama-cli", about = "Inferencia de LLMs em Rust (llama-rs)")]
pub struct Args {
    /// Caminho para o modelo GGUF
    #[arg(short, long)]
    pub model: PathBuf,

    /// Texto de entrada (prompt)
    #[arg(short, long, default_value = "")]
    pub prompt: String,

    /// Numero maximo de tokens a gerar
    #[arg(short = 'n', long, default_value_t = 128)]
    pub n_predict: usize,

    /// Semente aleatoria para amostragem reproduzivel
    #[arg(short, long, default_value_t = 42)]
    pub seed: u64,

    /// Temperatura de amostragem (0.0 = greedy deterministico)
    #[arg(long, default_value_t = 0.8)]
    pub temp: f32,

    /// Top-K -- manter K candidatos antes de amostrar (0 = desabilitado)
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Top-P / nucleus -- prob. acumulada minima (1.0 = desabilitado)
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,

    /// Suprimir o prompt da saida (equivale a --no-display-prompt do llama.cpp)
    #[arg(long)]
    pub no_display_prompt: bool,
}
```

- [ ] **Step 3: Criar `crates/llama-cli/src/runner.rs` (stub)**

```rust
//! Logica de geracao reutilizavel (separada do `main` para testes).

use crate::args::Args;

/// Carrega o modelo e gera texto conforme `args`. Retorna o texto gerado (sem o prompt).
pub fn generate_text(_args: &Args) -> Result<String, Box<dyn std::error::Error>> {
    todo!("implementado na Task 5")
}
```

- [ ] **Step 4: Criar `crates/llama-cli/src/lib.rs`**

```rust
#![forbid(unsafe_code)]
//! Biblioteca auxiliar do `llama-cli` -- expoe `generate_text` para testes de integracao.

pub mod args;
mod runner;

pub use runner::generate_text;
```

- [ ] **Step 5: Criar `crates/llama-cli/src/main.rs`**

```rust
#![forbid(unsafe_code)]

use clap::Parser;

use llama_cli::args::Args;
use llama_cli::generate_text;

fn main() {
    let args = Args::parse();

    if !args.no_display_prompt {
        print!("{}", args.prompt);
    }

    match generate_text(&args) {
        Ok(text) => print!("{text}"),
        Err(e) => {
            eprintln!("Erro: {e}");
            std::process::exit(1);
        }
    }
}
```

- [ ] **Step 6: Verificar que compila**

Run: `cargo build -p llama-cli`
Expected: PASS (warning de `todo!` aceitavel).

- [ ] **Step 7: Testar `--help`**

Run: `cargo run -p llama-cli -- --help`
Expected: help impresso com model, prompt, n-predict, seed, temp, top-k, top-p, no-display-prompt listados.

- [ ] **Step 8: Escrever os testes de args (RED entao GREEN)**

Criar `crates/llama-cli/tests/args_test.rs`:

```rust
//! Testes de parsing de args -- nao precisam de modelo no disco.
#![allow(clippy::unwrap_used)]

use clap::Parser;
use llama_cli::args::Args;

#[test]
fn default_args_parse() {
    let args = Args::try_parse_from(["llama-cli", "--model", "/tmp/m.gguf"]).unwrap();
    assert_eq!(args.n_predict, 128);
    assert_eq!(args.seed, 42);
    assert!((args.temp - 0.8).abs() < 1e-6);
    assert_eq!(args.top_k, 40);
    assert!((args.top_p - 0.9).abs() < 1e-6);
    assert!(!args.no_display_prompt);
}

#[test]
fn greedy_mode_args() {
    let args = Args::try_parse_from([
        "llama-cli",
        "--model", "/tmp/m.gguf",
        "--temp", "0",
        "--no-display-prompt",
        "-n", "32",
    ])
    .unwrap();
    assert_eq!(args.temp, 0.0);
    assert!(args.no_display_prompt);
    assert_eq!(args.n_predict, 32);
}

#[test]
fn custom_sampling_args() {
    let args = Args::try_parse_from([
        "llama-cli",
        "--model", "/tmp/m.gguf",
        "--temp", "0.5",
        "--top-k", "20",
        "--top-p", "0.85",
        "--seed", "1234",
    ])
    .unwrap();
    assert!((args.temp - 0.5).abs() < 1e-6);
    assert_eq!(args.top_k, 20);
    assert!((args.top_p - 0.85).abs() < 1e-6);
    assert_eq!(args.seed, 1234);
}
```

Run: `cargo test -p llama-cli --test args_test`
Expected: PASS (parsing funciona sem modelo no disco).

- [ ] **Step 9: Commit**

```bash
git add crates/llama-cli
git commit -m "chore(llama-cli): scaffold crate + args clap (Fase 4 Task 4)"
```

---

## Task 5: Runner `generate_text` + oracle gate

**Files:**
- Modify: `crates/llama-cli/src/runner.rs`
- Create: `crates/llama-cli/tests/greedy_gate.rs`

- [ ] **Step 1: Escrever o teste oracle (RED)**

Criar `crates/llama-cli/tests/greedy_gate.rs`:

```rust
//! Gate diferencial: saida greedy do llama-cli deve ser identica ao oraculo C++.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use llama_cli::{args::Args, generate_text};

const MODEL: &str = "../../models/stories260K.gguf";
const REFS: &str = "../../refs/greedy.txt";
const PROMPT: &str = "Once upon a time";

#[test]
fn greedy_matches_oracle_reference() {
    if !Path::new(MODEL).exists() {
        eprintln!("modelo ausente -- pulando");
        return;
    }
    let Ok(reference) = std::fs::read_to_string(REFS) else {
        eprintln!("refs/greedy.txt ausente -- pulando");
        return;
    };

    let args = Args {
        model: MODEL.into(),
        prompt: PROMPT.to_owned(),
        n_predict: 32,
        seed: 42,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        no_display_prompt: true,
    };

    let output = generate_text(&args).expect("generate_text falhou");
    let reference_trimmed = reference.trim_end_matches('\n');

    eprintln!("got: {output:?}");
    eprintln!("ref: {reference_trimmed:?}");

    assert_eq!(
        output, reference_trimmed,
        "\n  got: {output:?}\n  ref: {reference_trimmed:?}"
    );
}

#[test]
fn topp_sampler_does_not_panic() {
    if !Path::new(MODEL).exists() {
        return;
    }
    let args = Args {
        model: MODEL.into(),
        prompt: "Once".to_owned(),
        n_predict: 2,
        seed: 1,
        temp: 0.5,
        top_k: 0,
        top_p: 0.8,
        no_display_prompt: true,
    };
    generate_text(&args).expect("nao deve falhar com TopP");
}
```

Run: `cargo test -p llama-cli --test greedy_gate -- --nocapture 2>&1 | head -5`
Expected: **FAIL** (todo! panic).

- [ ] **Step 2: Implementar `generate_text` em `runner.rs`**

Substituir todo o conteudo de `crates/llama-cli/src/runner.rs`:

```rust
//! Logica de geracao reutilizavel (separada do `main` para testes de integracao).

use gguf::GgufFile;
use llama_model::Model;
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Carrega o modelo e gera texto conforme `args`. Retorna o texto gerado (sem o prompt).
pub fn generate_text(args: &Args) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let text = model.generate(&tokenizer, &args.prompt, args.n_predict, &sampler, &mut rng)?;
    Ok(text)
}

fn choose_sampler(args: &Args) -> Sampler {
    if args.temp == 0.0 {
        return Sampler::Greedy;
    }
    if args.top_k > 0 {
        return Sampler::TopK { k: args.top_k, temp: args.temp };
    }
    if args.top_p < 1.0 {
        return Sampler::TopP { p: args.top_p, temp: args.temp };
    }
    Sampler::Temperature { temp: args.temp }
}
```

- [ ] **Step 3: Rodar o gate diferencial**

Run: `cargo test -p llama-cli --test greedy_gate -- --nocapture`
Expected: PASS -- output == refs/greedy.txt (trimmed).

Se falhar com divergencia de texto, verificar se `argmax` em `llama-sampling/src/sampler.rs` e `argmax` em `llama-model/src/ops.rs` usam a mesma ordem de comparacao. Ambos devem usar `total_cmp`. Rodar:
```bash
cargo test -p llama-model generate_tests::generate_with_greedy_sampler_matches_generate_greedy -- --nocapture
```

- [ ] **Step 4: Rodar todos os testes do crate**

Run: `cargo test -p llama-cli -- --nocapture`
Expected: PASS (args_test + greedy_gate).

- [ ] **Step 5: Testar o binario ponta a ponta**

Run: `cargo run -p llama-cli -- -m models/stories260K.gguf -p "Once upon a time" -n 32 --temp 0 --no-display-prompt`
Expected: texto impresso igual ao conteudo de `refs/greedy.txt`.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-cli/src/runner.rs crates/llama-cli/tests/greedy_gate.rs
git commit -m "feat(llama-cli): generate_text + gate diferencial vs oraculo (Fase 4 Task 5)"
```

---

## Task 6: Gate de qualidade

**Files:** nenhum novo (ajustes de lint se necessario).

- [ ] **Step 1: fmt**

Run: `cargo fmt --all` e depois `cargo fmt --all --check`
Expected: sem diferencas.

- [ ] **Step 2: clippy estrito**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS.

Issues comuns e resolucoes:
- `clippy::float_cmp` em `args.temp == 0.0` em `choose_sampler` -- adicionar `#[allow(clippy::float_cmp)]` na funcao `choose_sampler` se necessario.
- `clippy::indexing_slicing` em `&probs[..cutoff]` no TopP -- o `#![allow(clippy::indexing_slicing)]` no topo de `sampler.rs` ja cobre isso.
- `clippy::struct_excessive_bools` em `Args` -- apenas `no_display_prompt` e bool; improvavel disparar mas adicionar `#[allow]` em `Args` se necessario.
- `clippy::too_many_arguments` em `Model::generate` -- 5 params, dentro do limite de 7.

- [ ] **Step 3: workspace completo**

Run: `cargo test --workspace`
Expected: PASS (oracle_forward, quant_load, greedy_gate, args_test, todos os testes unitarios).

- [ ] **Step 4: gate completo**

Run: `./scripts/gate.sh`
Expected: `GATE OK`.

- [ ] **Step 5: Commit final (se houve ajustes)**

```bash
git add -A
git commit -m "chore(fase4): gate verde (fmt + clippy + cobertura) (Fase 4 Task 6)"
```

---

## Riscos conhecidos

1. **`rand` 0.9 API.** O plano usa `seed_from_u64` e `rng.random::<f32>()`, ambos estaveis em 0.9. Se `rand` 0.9 nao estiver no cache local, tentar `rand = "0.8"` com `gen::<f32>()` e `thread_rng()`. Se houver erro de build, usar `build-error-resolver`.

2. **Divergencia de `argmax` greedy.** Se `greedy_gate` divergir, a causa mais provavel e que `argmax` em `llama-sampling` e `argmax` em `llama-model/src/ops.rs` diferem em caso de empate. Verificar se `ops.rs` usa `partial_cmp` (nao-deterministo para NaN) enquanto `sampler.rs` usa `total_cmp`. Corrigir `ops.rs` se necessario.

3. **`clap` e lints do workspace.** O derive de `clap` pode gerar codigo que aciona `cast_possible_truncation` ou `expect_used`. Se `cargo clippy` falhar em codigo gerado, adicionar `#[allow(...)]` pontual em `Args` -- nao relaxar o workspace.

4. **`Model::generate` vs `generate_greedy` -- identidade semantica.** Ambos devem produzir saida identica com `Sampler::Greedy`. Se divergirem, verificar se o `argmax` em `ops.rs` e o `argmax` em `sampler.rs` sao equivalentes. O teste `generate_with_greedy_sampler_matches_generate_greedy` detecta essa divergencia cedo.

---

## Self-Review

- **Cobertura da spec:** Sampling (Tasks 0-2), `Model::generate` (Task 3), `llama-cli` binario (Tasks 4-5), gate diferencial greedy identico ao C++ (Task 5 Step 3). Todos os 4 samplers implementados com TDD.
- **Placeholders:** nenhum TODO/TBD -- todo step tem codigo concreto.
- **Consistencia de tipos:** `Sampler::sample(logits: &[f32], rng: &mut impl Rng) -> usize` definido em Task 0 e usado em Task 1/2; `Model::generate(tokenizer, prompt, n, sampler: &Sampler, rng: &mut impl Rng) -> Result<String, ModelError>` definido em Task 3 e chamado em Task 5; `generate_text(args: &Args) -> Result<String, Box<dyn Error>>` definido em Task 4 stub e implementado em Task 5; `Args` struct definido em Task 4 e usado identicamente em `greedy_gate.rs` Task 5.
- **Gate preservado:** `oracle_forward.rs` da Fase 2 usa `generate_greedy` que nao e alterado -- Step 6 da Task 3 verifica explicitamente.
