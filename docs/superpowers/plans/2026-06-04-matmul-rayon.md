# matmul paralelo (rayon + target-cpu=native) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paralelizar `matmul` com rayon e ativar auto-vetorização AVX2 via `-C target-cpu=native`, reduzindo a latência de decode do Qwen2 0.5B.

**Architecture:** Três mudanças cirúrgicas: (1) `.cargo/config.toml` novo com `rustflags` para release, (2) `rayon` adicionado como dependência workspace + llama-model, (3) loop de `n_out` em `matmul` trocado por `par_iter_mut`. Nenhuma assinatura pública muda.

**Tech Stack:** `rayon = "1"`, `rustc` com `-C target-cpu=native` (AVX2/FMA automático).

---

## File Structure

- Create: `.cargo/config.toml` — rustflags de release
- Modify: `Cargo.toml` (workspace) — adicionar `rayon` em `[workspace.dependencies]`
- Modify: `crates/llama-model/Cargo.toml` — adicionar `rayon.workspace = true`
- Modify: `crates/llama-model/src/ops.rs` — `use rayon::prelude::*` + reescrever `matmul`

---

## Task 0: `.cargo/config.toml` + dependência rayon

**Files:**
- Create: `.cargo/config.toml`
- Modify: `Cargo.toml` (raiz do workspace)
- Modify: `crates/llama-model/Cargo.toml`

- [ ] **Step 1: Criar `.cargo/config.toml`**

```toml
[profile.release]
rustflags = ["-C", "target-cpu=native"]
```

- [ ] **Step 2: Adicionar rayon em `[workspace.dependencies]` no `Cargo.toml` da raiz**

Abrir `Cargo.toml` (raiz). Localizar o bloco `[workspace.dependencies]` e acrescentar:

```toml
rayon = "1"
```

- [ ] **Step 3: Adicionar rayon em `crates/llama-model/Cargo.toml`**

No bloco `[dependencies]`, acrescentar:

```toml
rayon.workspace = true
```

- [ ] **Step 4: Verificar que o workspace compila**

```bash
cargo check --workspace
```

Esperado: zero erros.

- [ ] **Step 5: Commit**

```bash
git add .cargo/config.toml Cargo.toml crates/llama-model/Cargo.toml
git commit -m "chore: rayon workspace dep + target-cpu=native em release"
```

---

## Task 1: `matmul` paralelo com rayon

**Files:**
- Modify: `crates/llama-model/src/ops.rs`

- [ ] **Step 1: Verificar que os testes existentes passam antes de qualquer mudança**

```bash
cargo test -p llama-model ops::tests 2>&1 | grep -E "^test result"
```

Esperado: `test result: ok`.

- [ ] **Step 2: Adicionar `use rayon::prelude::*` em `ops.rs`**

No topo do arquivo `crates/llama-model/src/ops.rs`, após os comentários existentes:

```rust
use rayon::prelude::*;
```

- [ ] **Step 3: Substituir `matmul` pela versão com `par_iter_mut`**

Localizar a função `matmul` (linhas ~51–62) e substituir o corpo completo:

```rust
pub(crate) fn matmul(w: &[f32], x: &[f32], n_in: usize, n_out: usize, n_tok: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * n_out];
    for t in 0..n_tok {
        let xrow = &x[t * n_in..(t + 1) * n_in];
        let orow = &mut out[t * n_out..(t + 1) * n_out];
        orow.par_iter_mut().enumerate().for_each(|(j, o)| {
            let wrow = &w[j * n_in..(j + 1) * n_in];
            *o = wrow.iter().zip(xrow.iter()).map(|(&a, &b)| a * b).sum();
        });
    }
    out
}
```

- [ ] **Step 4: Rodar testes de corretude**

```bash
cargo test -p llama-model 2>&1 | grep -E "^test result"
```

Esperado: todos `ok`, zero falhas.

- [ ] **Step 5: Rodar o oráculo de geração**

```bash
cargo test -p llama-model --test oracle_forward 2>&1 | tail -3
```

Esperado: `test result: ok. 1 passed`.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-model/src/ops.rs
git commit -m "perf(llama-model): matmul paralelo com rayon (par_iter_mut sobre n_out)"
```

---

## Task 2: Benchmark

**Files:** nenhum arquivo novo

- [ ] **Step 1: Build release**

```bash
cargo build --release -p llama-cli 2>&1 | grep -E "^(warning|error)" | head -5
```

Esperado: nenhum erro.

- [ ] **Step 2: Rodar benchmark**

```bash
bash scripts/benchmark.sh
```

Esperado: tok/s do Qwen2 0.5B substancialmente maior que 1.6. stories260K deve manter-se acima de 3000 tok/s.

- [ ] **Step 3: Commit com resultado**

```bash
git commit --allow-empty -m "perf: benchmark rayon — <stories_tps> tok/s stories, <qwen_tps> tok/s qwen2"
```

Substituir `<stories_tps>` e `<qwen_tps>` pelos valores medidos no Step 2.
