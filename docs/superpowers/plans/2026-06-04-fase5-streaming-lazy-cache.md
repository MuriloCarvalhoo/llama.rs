# Fase 5 — Streaming Output + Lazy Weight Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar as ~57 alocações de `Vec<f32>` por decode step adicionando `OnceCell<Vec<f32>>` em `RawTensor` (lazy memoização: zero-cost após o primeiro uso), e adicionar saída de tokens em streaming ao `llama-cli` com flag `--timings` que imprime tok/s ao final.

**Architecture:** `RawTensor` ganha `f32_cache: std::cell::OnceCell<Vec<f32>>`; `dequant_to_f32()` passa a retornar `&[f32]` — na primeira chamada dequantiza e armazena, nas subsequentes retorna borrow direto. `model.rs` é atualizado mecanicamente (remover `&` nos callers). Novo método `Model::generate_streaming` chama um callback `FnMut(&str)` a cada token decodificado individualmente; `generate` e `generate_greedy` ficam intactos (decodificam ao final, mantêm compatibilidade com o gate). Em `llama-cli`, `runner.rs` recebe `run_generate(args, callback) -> Result<Timing, ...>`; `generate_text` vira wrapper; `main.rs` usa streaming com flush por token.

**Tech Stack:** `std::cell::OnceCell` (stable desde Rust 1.82), `std::time::Instant`, deps workspace existentes.

---

## File Structure

- Modify: `crates/llama-model/src/weights.rs` — `RawTensor` com `OnceCell` + retorno `&[f32]`
- Modify: `crates/llama-model/src/model.rs` — update callers em `forward()` e testes
- Modify: `crates/llama-model/src/generate.rs` — novo `generate_streaming` + testes
- Modify: `crates/llama-cli/src/args.rs` — flag `--timings`
- Modify: `crates/llama-cli/src/runner.rs` — `Timing` struct + `run_generate`
- Modify: `crates/llama-cli/src/lib.rs` — re-exportar `run_generate` e `Timing`
- Modify: `crates/llama-cli/src/main.rs` — usar streaming + imprimir timings
- Modify: `crates/llama-cli/tests/args_test.rs` — testar `--timings`

---

## Task 0: `RawTensor` com `OnceCell` + update `model.rs`

**Files:**
- Modify: `crates/llama-model/src/weights.rs`
- Modify: `crates/llama-model/src/model.rs`

**Contexto:** Cada chamada a `forward()` faz hoje 11+ chamadas a `dequant_to_f32()` por camada — cada uma aloca um `Vec<f32>` novo. Com `OnceCell`, a primeira chamada dequantiza e armazena; as seguintes retornam `&[f32]` direto da cache, sem alloc.

- [ ] **Step 1: Escrever o teste RED (segunda chamada deve reusar a cache)**

Adicionar ao final do `mod tests` existente em `crates/llama-model/src/weights.rs`:

```rust
#[test]
fn dequant_cache_second_call_returns_same_pointer() {
    let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = GgufFile::parse(&bytes).unwrap();
    let cfg = LlamaConfig::from_gguf(&f).unwrap();
    let w = Weights::from_gguf(&f, &bytes, &cfg).unwrap();
    let ptr1 = w.token_embd.dequant_to_f32().unwrap().as_ptr();
    let ptr2 = w.token_embd.dequant_to_f32().unwrap().as_ptr();
    assert_eq!(ptr1, ptr2, "segunda chamada deve reusar a cache (mesmo ponteiro)");
}
```

Run: `cargo test -p llama-model weights::tests::dequant_cache 2>&1 | head -5`
Expected: **FAIL** (`dequant_to_f32` ainda retorna `Vec<f32>` → ponteiros diferentes).

- [ ] **Step 2: Atualizar `weights.rs` — adicionar `OnceCell` a `RawTensor`**

Substituir o bloco de struct `RawTensor` e seus métodos (mantendo tudo que vem depois — `LayerWeights`, `Weights`, `tensor_raw`, etc.):

```rust
//! Pesos quantizados do GGUF armazenados em bytes raw; dequantizados sob demanda.

use std::cell::OnceCell;

use ggml_cpu::dequant_to_f32 as dequant_impl;
use gguf::{GgufFile, TensorInfo};

use crate::config::LlamaConfig;
use crate::error::ModelError;

/// Tensor raw: bytes tal como lidos do GGUF + tipo de dado para dequant.
/// Primeira chamada a `dequant_to_f32` dequantiza e cacheia em memória;
/// chamadas subsequentes retornam `&[f32]` sem realocar.
pub(crate) struct RawTensor {
    pub bytes: Vec<u8>,
    pub ty: gguf::GgmlType,
    f32_cache: OnceCell<Vec<f32>>,
}

impl RawTensor {
    pub(crate) fn new(bytes: Vec<u8>, ty: gguf::GgmlType) -> Self {
        Self { bytes, ty, f32_cache: OnceCell::new() }
    }

    /// Número de elementos lógicos (não de bytes).
    pub fn n_elements(&self) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let bs = self.ty.block_size() as usize;
        #[allow(clippy::cast_possible_truncation)]
        let ts = self.ty.type_size() as usize;
        if ts == 0 {
            return 0;
        }
        (self.bytes.len() / ts) * bs
    }

    /// Bytes raw (footprint de RAM — quantizado, sem dequant).
    pub fn memory_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Dequantiza para f32 e cacheia. Primeira chamada: O(n). Subsequentes: O(1).
    pub fn dequant_to_f32(&self) -> Result<&[f32], ModelError> {
        let v = self.f32_cache.get_or_try_init(|| {
            dequant_impl(&self.bytes, self.ty).map_err(ModelError::from)
        })?;
        Ok(v)
    }
}
```

Também atualizar `tensor_raw` para usar `RawTensor::new`:

```rust
fn tensor_raw(f: &GgufFile, bytes: &[u8], name: &str) -> Result<RawTensor, ModelError> {
    let info: &TensorInfo = f
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))?;
    let raw = f.tensor_data(bytes, info)?;
    Ok(RawTensor::new(raw.to_vec(), info.ggml_type))
}
```

- [ ] **Step 3: Rodar só o teste de cache (model.rs ainda quebra — esperado)**

Run: `cargo test -p llama-model weights::tests::dequant_cache -- --nocapture 2>&1 | head -10`
Expected: PASS para o teste novo; erros de compilação em `model.rs` são esperados.

- [ ] **Step 4: Atualizar `model.rs` — remover `&` nos callers de pesos**

A mudança é puramente mecânica: `dequant_to_f32()` agora retorna `&[f32]`, então não precisa de `&` adicional ao passar para funções que já aceitam `&[f32]`.

Aplicar as seguintes substituições em `forward()`:

```rust
// ANTES → DEPOIS (remover & apenas nos locais marcados)

// Linha token_embd:
embedding_lookup(&token_embd, tokens, c.n_embd)?
// →
embedding_lookup(token_embd, tokens, c.n_embd)?

// Linha attn_in (mul_rows):
mul_rows(&normed, &attn_norm, c.n_embd)
// →
mul_rows(&normed, attn_norm, c.n_embd)

// Linhas q, k, v (matmul com pesos de atenção):
matmul(&attn_q_w, &attn_in, ...)  →  matmul(attn_q_w, &attn_in, ...)
matmul(&attn_k_w, &attn_in, ...)  →  matmul(attn_k_w, &attn_in, ...)
matmul(&attn_v_w, &attn_in, ...)  →  matmul(attn_v_w, &attn_in, ...)

// Linha attn_out:
matmul(&attn_out_w, &attn, ...)  →  matmul(attn_out_w, &attn, ...)

// Linha ffn_in (mul_rows):
mul_rows(&normed, &ffn_norm, ...)  →  mul_rows(&normed, ffn_norm, ...)

// Linhas gate, up, ffn_out:
matmul(&ffn_gate_w, &ffn_in, ...)  →  matmul(ffn_gate_w, &ffn_in, ...)
matmul(&ffn_up_w, &ffn_in, ...)    →  matmul(ffn_up_w, &ffn_in, ...)
matmul(&ffn_down_w, &act, ...)     →  matmul(ffn_down_w, &act, ...)

// Linhas output:
mul_rows(&normed, &output_norm, ...)  →  mul_rows(&normed, output_norm, ...)
matmul(&output_w, last, ...)          →  matmul(output_w, last, ...)
```

Aplicar as mesmas substituições nos **testes dentro de `model.rs`** (`mod tests`):

```rust
// ANTES:
let token_embd = m.weights.token_embd.dequant_to_f32().unwrap();
let x = embedding_lookup(&token_embd, &tokens, c.n_embd).unwrap();
let attn_norm = lw.attn_norm.dequant_to_f32().unwrap();
let attn_q_w = lw.attn_q.dequant_to_f32().unwrap();
let attn_in = mul_rows(&normed, &attn_norm, c.n_embd);
let mut q = matmul(&attn_q_w, &attn_in, c.n_embd, c.n_embd, n_tok);

// DEPOIS:
let token_embd = m.weights.token_embd.dequant_to_f32().unwrap();
let x = embedding_lookup(token_embd, &tokens, c.n_embd).unwrap();
let attn_norm = lw.attn_norm.dequant_to_f32().unwrap();
let attn_q_w = lw.attn_q.dequant_to_f32().unwrap();
let attn_in = mul_rows(&normed, attn_norm, c.n_embd);
let mut q = matmul(attn_q_w, &attn_in, c.n_embd, c.n_embd, n_tok);
```

- [ ] **Step 5: Verificar build e rodar todos os testes do crate**

Run: `cargo test -p llama-model`
Expected: PASS (incluindo o novo `dequant_cache_second_call_returns_same_pointer`).

- [ ] **Step 6: Verificar que o gate da Fase 2 continua verde**

Run: `cargo test -p llama-model --test oracle_forward -- --nocapture`
Expected: PASS — sequência greedy idêntica a `refs/greedy.txt`.

- [ ] **Step 7: Commit**

```bash
git add crates/llama-model/src/weights.rs crates/llama-model/src/model.rs
git commit -m "perf(llama-model): RawTensor lazy OnceCell cache — zero-alloc apos 1o forward (Fase 5 Task 0)"
```

---

## Task 1: `Model::generate_streaming`

**Files:**
- Modify: `crates/llama-model/src/generate.rs`

**Contexto:** `generate_streaming` decodifica cada token individualmente com `tokenizer.decode(&[token_id])` e chama o callback imediatamente. `generate` e `generate_greedy` permanecem intactos — eles decodificam ao final em batch, mantendo o gate greedy verde.

- [ ] **Step 1: Escrever os testes (RED)**

Adicionar ao bloco `mod generate_tests` em `generate.rs`:

```rust
#[test]
fn generate_streaming_calls_callback_for_each_token() {
    let Some((model, tok)) = load() else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(42);
    let mut pieces: Vec<String> = Vec::new();
    model
        .generate_streaming(
            &tok,
            "Once upon a time",
            8,
            &Sampler::Greedy,
            &mut rng,
            &mut |piece| pieces.push(piece.to_owned()),
        )
        .unwrap();
    assert!(!pieces.is_empty(), "callback deve ser chamado pelo menos uma vez");
    assert!(pieces.len() <= 8, "no maximo 8 callbacks: {pieces:?}");
}

#[test]
fn generate_streaming_zero_tokens_calls_no_callback() {
    let Some((model, tok)) = load() else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(0);
    let mut count = 0usize;
    model
        .generate_streaming(
            &tok,
            "Hello",
            0,
            &Sampler::Greedy,
            &mut rng,
            &mut |_| count += 1,
        )
        .unwrap();
    assert_eq!(count, 0, "n_tokens=0 nao deve chamar o callback");
}
```

Run: `cargo test -p llama-model generate_tests::generate_streaming 2>&1 | head -5`
Expected: **FAIL** (método `generate_streaming` não existe).

- [ ] **Step 2: Implementar `generate_streaming`**

Adicionar **antes** do método `generate` no bloco `impl Model` em `generate.rs`:

```rust
/// Gera até `n_tokens` chamando `on_token` a cada token decodificado individualmente.
/// Para em EOS ou quando `n_tokens` for atingido.
/// Nota: decodifica token a token — pode diferir de `generate` (que decodifica em batch)
/// em modelos BPE com byte-fallback. Use `generate` para comparação com o gate.
pub fn generate_streaming(
    &self,
    tokenizer: &Tokenizer,
    prompt: &str,
    n_tokens: usize,
    sampler: &Sampler,
    rng: &mut impl Rng,
    on_token: &mut impl FnMut(&str),
) -> Result<(), ModelError> {
    let prompt_ids = tokenizer.encode(prompt, true);
    let mut cache = self.new_cache();

    let logits = self.forward(&prompt_ids, &mut cache)?;
    let first_idx = sampler.sample(&logits, rng);
    let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;

    let mut count = 0usize;
    while count < n_tokens {
        if next == self.config.eos_id {
            break;
        }
        let piece = tokenizer.decode(&[next]);
        on_token(&piece);
        count += 1;
        let logits = self.forward(&[next], &mut cache)?;
        let idx = sampler.sample(&logits, rng);
        next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
    }

    Ok(())
}
```

- [ ] **Step 3: Rodar os testes**

Run: `cargo test -p llama-model generate_tests -- --nocapture`
Expected: PASS (3 testes: `generate_with_greedy_sampler_matches_generate_greedy` + os 2 novos).

- [ ] **Step 4: Verificar que o gate continua verde**

Run: `cargo test -p llama-model --test oracle_forward -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-model/src/generate.rs
git commit -m "feat(llama-model): generate_streaming com callback por token (Fase 5 Task 1)"
```

---

## Task 2: `run_generate` + streaming + `--timings` em `llama-cli`

**Files:**
- Modify: `crates/llama-cli/src/args.rs`
- Modify: `crates/llama-cli/src/runner.rs`
- Modify: `crates/llama-cli/src/lib.rs`
- Modify: `crates/llama-cli/src/main.rs`
- Modify: `crates/llama-cli/tests/args_test.rs`

- [ ] **Step 1: Escrever os testes RED para o novo flag**

Adicionar em `crates/llama-cli/tests/args_test.rs`:

```rust
#[test]
fn timings_flag_default_false() {
    let args = Args::try_parse_from(["llama-cli", "--model", "/tmp/m.gguf"]).unwrap();
    assert!(!args.timings, "timings deve ser false por padrao");
}

#[test]
fn timings_flag_enabled() {
    let args =
        Args::try_parse_from(["llama-cli", "--model", "/tmp/m.gguf", "--timings"]).unwrap();
    assert!(args.timings);
}
```

Run: `cargo test -p llama-cli --test args_test -- timings 2>&1 | head -5`
Expected: **FAIL** (campo `timings` não existe em `Args`).

- [ ] **Step 2: Adicionar `--timings` a `Args`**

Em `crates/llama-cli/src/args.rs`, adicionar após o campo `no_display_prompt`:

```rust
    /// Imprimir tempo de geracao (tokens/seg) ao final para stderr
    #[arg(long)]
    pub timings: bool,
```

Run: `cargo test -p llama-cli --test args_test`
Expected: PASS (todos os 5 testes, incluindo os 2 novos).

- [ ] **Step 3: Adicionar `Timing` e `run_generate` a `runner.rs`**

Substituir todo o conteúdo de `crates/llama-cli/src/runner.rs`:

```rust
//! Logica de geracao reutilizavel — streaming e buffered.

use std::time::Instant;

use gguf::GgufFile;
use llama_model::Model;
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::args::Args;

/// Metricas coletadas durante a geracao.
pub struct Timing {
    pub n_tokens: usize,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

/// Carrega o modelo e chama `on_token` para cada token gerado. Retorna metricas de tempo.
pub fn run_generate(
    args: &Args,
    on_token: &mut impl FnMut(&str),
) -> Result<Timing, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let mut n_tokens = 0usize;
    let start = Instant::now();

    model.generate_streaming(
        &tokenizer,
        &args.prompt,
        args.n_predict,
        &sampler,
        &mut rng,
        &mut |piece| {
            on_token(piece);
            n_tokens += 1;
        },
    )?;

    let elapsed_secs = start.elapsed().as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let tokens_per_sec =
        if elapsed_secs > 0.0 { n_tokens as f64 / elapsed_secs } else { 0.0 };

    Ok(Timing { n_tokens, elapsed_secs, tokens_per_sec })
}

/// Carrega o modelo e retorna o texto completo como String (sem streaming).
/// Mantém compatibilidade com `greedy_gate.rs` e outros testes de integracao.
pub fn generate_text(args: &Args) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&args.model)?;
    let f = GgufFile::parse(&bytes)?;
    let model = Model::load(&f, &bytes)?;
    let tokenizer = Tokenizer::from_gguf(&f)?;
    let sampler = choose_sampler(args);
    let mut rng = SmallRng::seed_from_u64(args.seed);

    let text =
        model.generate(&tokenizer, &args.prompt, args.n_predict, &sampler, &mut rng)?;
    Ok(text)
}

#[allow(clippy::float_cmp)]
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

- [ ] **Step 4: Atualizar `lib.rs` para re-exportar os novos símbolos**

Substituir o conteúdo de `crates/llama-cli/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! Biblioteca auxiliar do `llama-cli`.

pub mod args;
mod runner;

pub use runner::generate_text;
pub use runner::run_generate;
pub use runner::Timing;
```

- [ ] **Step 5: Atualizar `main.rs` para usar streaming**

Substituir o conteúdo de `crates/llama-cli/src/main.rs`:

```rust
#![forbid(unsafe_code)]

use std::io::{self, Write};

use clap::Parser;

use llama_cli::args::Args;
use llama_cli::run_generate;

fn main() {
    let args = Args::parse();

    if !args.no_display_prompt {
        print!("{}", args.prompt);
        let _ = io::stdout().flush();
    }

    match run_generate(&args, &mut |piece| {
        print!("{piece}");
        let _ = io::stdout().flush();
    }) {
        Ok(timing) => {
            println!();
            if args.timings {
                eprintln!(
                    "{} tokens, {:.2} tok/s",
                    timing.n_tokens, timing.tokens_per_sec
                );
            }
        }
        Err(e) => {
            eprintln!("Erro: {e}");
            std::process::exit(1);
        }
    }
}
```

- [ ] **Step 6: Rodar todos os testes do crate**

Run: `cargo test -p llama-cli -- --nocapture`
Expected: PASS (`args_test` + `greedy_gate`; `greedy_gate` usa `generate_text` que não mudou).

- [ ] **Step 7: Escrever teste de integração para `run_generate` em `greedy_gate.rs`**

Adicionar ao final de `crates/llama-cli/tests/greedy_gate.rs`:

```rust
#[test]
fn run_generate_streaming_does_not_panic() {
    if !Path::new(MODEL).exists() {
        eprintln!("modelo ausente — pulando");
        return;
    }
    let args = Args {
        model: MODEL.into(),
        prompt: PROMPT.to_owned(),
        n_predict: 4,
        seed: 42,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        no_display_prompt: true,
        timings: false,
    };
    let mut pieces: Vec<String> = Vec::new();
    llama_cli::run_generate(&args, &mut |p| pieces.push(p.to_owned()))
        .expect("run_generate nao deve falhar");
    assert!(!pieces.is_empty(), "esperado pelo menos um token");
}
```

Run: `cargo test -p llama-cli --test greedy_gate -- --nocapture`
Expected: PASS (todos os 3 testes).

- [ ] **Step 8: Testar streaming e timings no binário**

Run: `cargo run -p llama-cli -- -m models/stories260K.gguf -p "Once upon a time" -n 16 --temp 0 --no-display-prompt --timings`
Expected: tokens impressos conforme gerados, então `16 tokens, X.XX tok/s` em stderr.

- [ ] **Step 9: Commit**

```bash
git add crates/llama-cli/src/args.rs crates/llama-cli/src/runner.rs \
        crates/llama-cli/src/lib.rs crates/llama-cli/src/main.rs \
        crates/llama-cli/tests/greedy_gate.rs crates/llama-cli/tests/args_test.rs
git commit -m "feat(llama-cli): streaming output + --timings tok/s + run_generate (Fase 5 Task 2)"
```

---

## Task 3: Gate de qualidade

**Files:** nenhum novo (ajustes de lint se necessário).

- [ ] **Step 1: fmt**

Run: `cargo fmt --all` e então `cargo fmt --all --check`
Expected: sem diferenças.

- [ ] **Step 2: clippy estrito**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS.

Issues comuns e resoluções:
- `clippy::cast_precision_loss` em `n_tokens as f64` — já coberto pelo `#[allow]` adicionado em `runner.rs`.
- `dead_code` em `Timing::elapsed_secs` — campo público, improvável. Se ocorrer, adicionar `#[allow(dead_code)]` no campo.
- `clippy::significant_drop_in_scrutinee` no `match run_generate(...)` em `main.rs` — improvável. Se ocorrer, extrair para `let result = run_generate(...); match result { ... }`.
- Qualquer lint em `OnceCell` usage em `weights.rs` — improvável; `OnceCell` é API estável.

- [ ] **Step 3: Workspace completo**

Run: `cargo test --workspace`
Expected: PASS (oracle_forward, quant_load, greedy_gate, args_test, todos os unitários).

- [ ] **Step 4: Gate completo**

Run: `./scripts/gate.sh`
Expected: `GATE OK`.

- [ ] **Step 5: Commit final (se houve ajustes)**

```bash
git add -A
git commit -m "chore(fase5): gate verde (fmt + clippy + cobertura) (Fase 5 Task 3)"
```

---

## Riscos conhecidos

1. **Divergência de decode token-a-token vs. batch.** `generate_streaming` chama `tokenizer.decode(&[token_id])` por token; `generate` chama `tokenizer.decode(&all_ids)` ao final. Para tokens BPE com byte-fallback, o decode isolado pode produzir resultados diferentes. O `greedy_gate.rs` usa `generate_text` → `model.generate()` (batch), então o gate oficial NÃO é afetado. O output via CLI pode divergir visualmente — aceitável neste estágio; corrigível na Fase 6 com buffer de byte-fallback.

2. **`OnceCell` não é `Sync`.** `RawTensor` com `OnceCell<Vec<f32>>` é `Send` mas não `Sync`. Se `Model` precisar ser compartilhado entre threads no futuro, trocar `OnceCell` por `std::sync::OnceLock`. Por ora, inferência single-thread — sem impacto.

3. **Primeira chamada a `forward()` ainda paga dequant.** Com `OnceCell`, o primeiro decode da sessão dequantiza todos os tensores. Para stories260K F32 (≈250KB de pesos), o custo é desprezível. Para modelos Q8_0 maiores, será mais visível no primeiro token. `Model::warmup(&self)` pode ser adicionado na Fase 6 para pré-popular todos os caches eagerly.

4. **Struct `Args` em `greedy_gate.rs` precisa do campo `timings: false`.** O teste `run_generate_streaming_does_not_panic` usa `Args { ..., timings: false }`. Se o teste `greedy_gate_matches_oracle_reference` usar struct literal (não `Args::try_parse_from`), ele precisará ser atualizado para incluir `timings: false`. Verificar e ajustar se necessário.

---

## Self-Review

- **Cobertura da spec:** Lazy cache via `OnceCell` (Task 0), `generate_streaming` (Task 1), streaming + `--timings` + `run_generate` (Task 2). Todos os gates anteriores (oracle_forward, greedy_gate) preservados.
- **Placeholders:** nenhum TODO/TBD — todo step tem código concreto.
- **Consistência de tipos:**
  - `RawTensor::dequant_to_f32() -> Result<&[f32], ModelError>` definido em Task 0 e usado mecanicamente em `model.rs` (Steps 4).
  - `Model::generate_streaming(..., on_token: &mut impl FnMut(&str)) -> Result<(), ModelError>` definido em Task 1 e chamado em `run_generate` (Task 2 Step 3).
  - `Timing { n_tokens: usize, elapsed_secs: f64, tokens_per_sec: f64 }` definido em Task 2 e retornado por `run_generate`.
  - `Args::timings: bool` adicionado em Task 2 Step 2 e referenciado em `main.rs`, `args_test.rs`, e `greedy_gate.rs` (Step 7).
- **Retrocompatibilidade:** `generate` e `generate_greedy` inalterados → gate greedy verde. `generate_text` inalterado → `greedy_gate.rs` precisa apenas adicionar `timings: false` no struct literal (Task 2, Step 7).
