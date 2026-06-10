# Fase 7 — Forward Pass End-to-End nas 2× MI50 (decode na GPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir os 8 matmuls do passo de *decode* (n_tok=1) pelas 2× AMD MI50 via row-split Vulkan, gerando tokens reais na GPU com saída token-idêntica à CPU, e expor isso na CLI via `--gpu`.

**Architecture:** A orquestração do forward continua em `llama-model` (onde vivem RMSNorm, RoPE, attention, SwiGLU, KV cache — todos reusados sem cópia). Definimos uma trait `GpuMatmul` em `llama-model`; o `llama-vulkan` a implementa envolvendo o `DualGpuMatmul` já testado. Apenas o *decode* (n_tok=1) usa GPU; o *prefill* do prompt (n_tok>1) permanece na CPU porque o shader é matvec. Os pesos Q8_0 vêm direto do GGUF (`GpuRawWeights`), pois o `Model` descarta os bytes raw após o repack. Este é um marco **correção-primeiro**: os pesos são re-enviados à VRAM a cada matmul (lento mas correto); residência em VRAM é otimização da Fase 8.

**Tech Stack:** Rust 2024, `llama-vulkan` (ash/shaderc, já implementado nas Tasks 0–7 da Fase 6), trait object `&dyn GpuMatmul`, feature flag `gpu` em `llama-model` e `llama-cli`.

**Contexto de assinaturas reais (confirmadas no código):**
- `DualGpuMatmul::new(ctx: &VulkanContext) -> Result<Self, DualGpuError>` (borrows ctx)
- `DualGpuMatmul::matvec_q8_0(&self, w_bytes: &[u8], x_f32: &[f32], n_in: usize, n_out: usize) -> Result<Vec<f32>, DualGpuError>`
- `RawTensor` é `pub(crate)` e **descarta** `bytes` após repack Q8_0 → GPU lê do GGUF
- `gguf::GgufFile { pub tensors: Vec<TensorInfo> }`, `TensorInfo { pub name: String, pub ggml_type: GgmlType }`, `GgufFile::tensor_data(&self, bytes, &TensorInfo) -> Result<&[u8], GgufError>`
- `LlamaConfig { pub n_embd, n_layer, n_head, n_head_kv, head_dim, n_ff, rope_dim, rms_eps, freq_base, vocab, ctx, bos_id, eos_id }`
- `ops` (pub(crate)): `quantize_q8_0_split(x, n_in, n_tok) -> Vec<u8>`, `matmul_q8_0_actq(w_bytes, x_q8, n_in, n_out, n_tok) -> Vec<f32>`, `argmax(&[f32]) -> usize`
- Qwen2.5-0.5B: n_embd=896, kv_dim=128, n_ff=4864, vocab=151936 — todos n_in múltiplos de 32

---

## Mapeamento de Arquivos

### Criar
- `crates/llama-model/src/gpu.rs` — trait `GpuMatmul`, `GpuRawWeights`, `forward_gpu`, `generate_streaming_gpu`, helpers de decode (tudo `#[cfg(feature = "gpu")]`)
- `crates/llama-vulkan/src/backend.rs` — `DualGpuBackend` implementando `llama_model::GpuMatmul`

### Modificar
- `crates/llama-model/Cargo.toml` — adicionar `[features] gpu = []`
- `crates/llama-model/src/lib.rs` — `#[cfg(feature="gpu")] mod gpu;` + re-exports
- `crates/llama-model/src/error.rs` — variante `ModelError::Gpu(String)`
- `crates/llama-model/src/model.rs` — helpers `pub(crate)` de acesso a pesos CPU
- `crates/llama-vulkan/Cargo.toml` — dep `llama-model = { workspace = true, features = ["gpu"] }`
- `crates/llama-vulkan/src/lib.rs` — `mod backend; pub use backend::DualGpuBackend;`
- `crates/llama-vulkan/tests/integration.rs` — teste no hardware real
- `crates/llama-cli/Cargo.toml` — `gpu = ["llama-vulkan", "llama-model/gpu"]`
- `crates/llama-cli/src/runner.rs` — usar `generate_streaming_gpu` quando `--gpu` e ≥2 MI50

---

## Task 1: Feature `gpu`, trait `GpuMatmul` e erro

**Files:**
- Modify: `crates/llama-model/Cargo.toml`
- Modify: `crates/llama-model/src/error.rs`
- Create: `crates/llama-model/src/gpu.rs`
- Modify: `crates/llama-model/src/lib.rs`

- [ ] **Step 1: Adicionar feature `gpu` ao Cargo.toml**

Editar `crates/llama-model/Cargo.toml`, adicionar ao final (criar a seção se não existir):
```toml
[features]
gpu = []
```

- [ ] **Step 2: Adicionar variante de erro**

Em `crates/llama-model/src/error.rs`, adicionar uma variante ao enum `ModelError` (manter o estilo `thiserror` existente):
```rust
    #[error("Erro no backend GPU: {0}")]
    Gpu(String),
```

- [ ] **Step 3: Criar `gpu.rs` com a trait (sem lógica de forward ainda)**

Criar `crates/llama-model/src/gpu.rs`:
```rust
//! Backend GPU para o passo de decode. A orquestração do forward vive aqui
//! (reusa RMSNorm/RoPE/attention/SwiGLU); apenas os matmuls vão para a GPU.

use crate::error::ModelError;

/// Multiplicação matriz-vetor Q8_0 executada na GPU.
///
/// `w_bytes`: pesos Q8_0 row-major, `n_out × (n_in/32 × 34)` bytes.
/// `x`: ativações f32 de tamanho `n_in`.
/// Retorna `y` de tamanho `n_out`.
pub trait GpuMatmul {
    fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, ModelError>;
}
```

- [ ] **Step 4: Registrar o módulo em lib.rs**

Em `crates/llama-model/src/lib.rs`, adicionar:
```rust
#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::{GpuMatmul, GpuRawWeights};
```

Nota: `GpuRawWeights` ainda não existe — será criado na Task 2. Este re-export falha a compilação **com** a feature até a Task 2; sem a feature, compila normalmente.

- [ ] **Step 5: Verificar que compila sem a feature (default)**

Run: `cargo build -p llama-model 2>&1 | tail -5`
Expected: PASS (o módulo `gpu` é excluído sem a feature).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-model/Cargo.toml crates/llama-model/src/error.rs crates/llama-model/src/gpu.rs crates/llama-model/src/lib.rs
git commit -m "feat(model): feature gpu + trait GpuMatmul + ModelError::Gpu"
```

---

## Task 2: `GpuRawWeights` — extrai pesos Q8_0 do GGUF

**Files:**
- Modify: `crates/llama-model/src/gpu.rs`

Contexto: o `Model` descarta os bytes Q8_0 após o repack, então a GPU precisa de uma cópia própria lida do GGUF. Validamos que cada peso é Q8_0 com `n_in % 32 == 0` (requisito do shader matvec); caso contrário, erro explícito nomeando o tensor.

- [ ] **Step 1: Escrever o teste (RED)**

Adicionar ao final de `crates/llama-model/src/gpu.rs`:
```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;
    use crate::config::LlamaConfig;
    use gguf::GgufFile;
    use std::path::Path;

    fn load_qwen() -> Option<(Vec<u8>, GgufFile, LlamaConfig)> {
        let bytes = std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let cfg = LlamaConfig::from_gguf(&f).ok()?;
        Some((bytes, f, cfg))
    }

    #[test]
    fn gpu_raw_weights_extrai_todas_as_camadas() {
        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).expect("from_gguf falhou");
        assert_eq!(w.layers.len(), cfg.n_layer);
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        let row_bytes_q = (cfg.n_embd / 32) * 34;
        assert_eq!(w.layers[0].attn_q.len(), cfg.n_embd * row_bytes_q);
        assert_eq!(w.layers[0].attn_k.len(), kv_dim * row_bytes_q);
        assert_eq!(w.output.len(), cfg.vocab * row_bytes_q);
        eprintln!("GpuRawWeights OK — {} camadas", w.layers.len());
    }
}
```

- [ ] **Step 2: Rodar o teste — deve falhar na compilação**

Run: `cargo test -p llama-model --features gpu gpu_raw_weights -- --nocapture 2>&1 | head -10`
Expected: FAIL — `GpuRawWeights` não existe.

- [ ] **Step 3: Implementar `GpuRawWeights`**

Adicionar em `crates/llama-model/src/gpu.rs`, antes do bloco `#[cfg(test)]`:
```rust
use crate::config::LlamaConfig;
use gguf::{GgmlType, GgufFile};

/// Pesos Q8_0 por camada, em bytes raw lidos do GGUF (cópia própria da GPU).
pub struct GpuLayerRaw {
    pub attn_q: Vec<u8>,
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_output: Vec<u8>,
    pub ffn_gate: Vec<u8>,
    pub ffn_up: Vec<u8>,
    pub ffn_down: Vec<u8>,
}

/// Todos os pesos Q8_0 que o decode envia à GPU.
pub struct GpuRawWeights {
    pub layers: Vec<GpuLayerRaw>,
    pub output: Vec<u8>,
}

impl GpuRawWeights {
    /// Lê e valida os pesos Q8_0 do GGUF. Erro se algum tensor não for Q8_0
    /// ou tiver `n_in % 32 != 0` (incompatível com o shader matvec wave64).
    pub fn from_gguf(
        f: &GgufFile,
        bytes: &[u8],
        cfg: &LlamaConfig,
    ) -> Result<Self, ModelError> {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;

        // Lê um tensor Q8_0 validando tipo e n_in. n_in é a dimensão de entrada
        // (contígua/row-major): cada linha tem (n_in/32)*34 bytes.
        let read = |name: &str, n_in: usize, n_out: usize| -> Result<Vec<u8>, ModelError> {
            let info = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| ModelError::Gpu(format!("tensor {name} ausente")))?;
            if info.ggml_type != GgmlType::Q8_0 {
                return Err(ModelError::Gpu(format!(
                    "tensor {name} não é Q8_0 (é {:?}) — GPU exige Q8_0",
                    info.ggml_type
                )));
            }
            if n_in % 32 != 0 {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: n_in={n_in} não é múltiplo de 32"
                )));
            }
            let raw = f
                .tensor_data(bytes, info)
                .map_err(|e| ModelError::Gpu(e.to_string()))?;
            let expected = n_out * (n_in / 32) * 34;
            if raw.len() != expected {
                return Err(ModelError::Gpu(format!(
                    "tensor {name}: {} bytes, esperado {expected}",
                    raw.len()
                )));
            }
            Ok(raw.to_vec())
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            layers.push(GpuLayerRaw {
                attn_q: read(&p("attn_q.weight"), cfg.n_embd, cfg.n_embd)?,
                attn_k: read(&p("attn_k.weight"), cfg.n_embd, kv_dim)?,
                attn_v: read(&p("attn_v.weight"), cfg.n_embd, kv_dim)?,
                attn_output: read(&p("attn_output.weight"), cfg.n_embd, cfg.n_embd)?,
                ffn_gate: read(&p("ffn_gate.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_up: read(&p("ffn_up.weight"), cfg.n_embd, cfg.n_ff)?,
                ffn_down: read(&p("ffn_down.weight"), cfg.n_ff, cfg.n_embd)?,
            });
        }
        let output = read("output.weight", cfg.n_embd, cfg.vocab)?;
        Ok(Self { layers, output })
    }
}
```

Nota: se `output.weight` não existir (pesos amarrados a `token_embd`), o erro explícito documenta isso — fora de escopo deste marco.

- [ ] **Step 4: Garantir dep `gguf` no llama-model**

Verificar `crates/llama-model/Cargo.toml` — `gguf` já é dep (usado em `weights.rs`). Nenhum ajuste esperado.

- [ ] **Step 5: Rodar o teste**

Run: `cargo test -p llama-model --features gpu gpu_raw_weights -- --nocapture 2>&1 | tail -10`
Expected: PASS "GpuRawWeights OK — 24 camadas" (ou "qwen ausente — pulando").

- [ ] **Step 6: Commit**

```bash
git add crates/llama-model/src/gpu.rs
git commit -m "feat(model): GpuRawWeights extrai+valida pesos Q8_0 do GGUF"
```

---

## Task 3: `forward_gpu` (decode) com validação via mock CPU

**Files:**
- Modify: `crates/llama-model/src/gpu.rs`
- Modify: `crates/llama-model/src/model.rs`

Contexto: `forward_gpu` espelha `Model::forward` para n_tok=1, trocando os 8 matmuls por `gpu.matvec_q8_0`. Validamos **sem hardware** com um mock que reproduz exatamente a matemática da CPU (`quantize_q8_0_split` + `matmul_q8_0_actq`): assim `forward_gpu(mock)` deve ser **bit-idêntico** a `forward`.

- [ ] **Step 1: Garantir `LlamaConfig: Clone`**

Verificar `crates/llama-model/src/config.rs`. Se `LlamaConfig` não derivar `Clone`, adicionar `#[derive(Clone)]` (os testes clonam a config).

- [ ] **Step 2: Escrever o teste com mock (RED)**

Adicionar dentro de `mod tests` em `gpu.rs`:
```rust
    /// Mock que replica a matemática Q8_0 da CPU (ativações quantizadas).
    /// Faz forward_gpu(mock) == forward bit-a-bit.
    struct CpuMockMatmul;
    impl GpuMatmul for CpuMockMatmul {
        fn matvec_q8_0(
            &self,
            w_bytes: &[u8],
            x: &[f32],
            n_in: usize,
            n_out: usize,
        ) -> Result<Vec<f32>, ModelError> {
            let x_q8 = crate::ops::quantize_q8_0_split(x, n_in, 1);
            Ok(crate::ops::matmul_q8_0_actq(w_bytes, &x_q8, n_in, n_out, 1))
        }
    }

    #[test]
    fn forward_gpu_mock_identico_a_forward_cpu() {
        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let model = crate::model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
        let mock = CpuMockMatmul;

        // Prefill idêntico nos dois caches (CPU), depois 1 decode.
        let mut c_cpu = model.new_cache();
        let mut c_gpu = model.new_cache();
        let prompt = [cfg.bos_id, 9707u32];
        let _ = model.forward(&prompt, &mut c_cpu).unwrap();
        let _ = model.forward(&prompt, &mut c_gpu).unwrap();
        let next = cfg.bos_id;

        let logits_cpu = model.forward(&[next], &mut c_cpu).unwrap();
        let logits_gpu = model.forward_gpu(&[next], &mut c_gpu, &mock, &w).unwrap();

        assert_eq!(logits_cpu.len(), logits_gpu.len());
        for (i, (a, b)) in logits_cpu.iter().zip(logits_gpu.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit[{i}]: cpu={a} gpu={b}");
        }
        eprintln!("forward_gpu(mock) == forward CPU — {} logits", logits_gpu.len());
    }
```

- [ ] **Step 3: Rodar — deve falhar (forward_gpu não existe)**

Run: `cargo test -p llama-model --features gpu forward_gpu_mock -- --nocapture 2>&1 | head -10`
Expected: FAIL — método `forward_gpu` não existe.

- [ ] **Step 4: Implementar os helpers de acesso a pesos CPU em `model.rs`**

Adicionar em `crates/llama-model/src/model.rs`, dentro de `impl Model`:
```rust
    /// Embedding table dequantizada (para forward_gpu).
    #[cfg(feature = "gpu")]
    pub(crate) fn token_embd_f32(&self) -> Result<&[f32], ModelError> {
        self.weights.token_embd.dequant_to_f32()
    }

    /// (attn_norm, ffn_norm) dequantizados da camada `l`.
    #[cfg(feature = "gpu")]
    pub(crate) fn layer_norms_f32(&self, l: usize) -> Result<(&[f32], &[f32]), ModelError> {
        let lw = &self.weights.layers[l];
        Ok((lw.attn_norm.dequant_to_f32()?, lw.ffn_norm.dequant_to_f32()?))
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn output_norm_f32(&self) -> Result<&[f32], ModelError> {
        self.weights.output_norm.dequant_to_f32()
    }

    /// Soma os biases Q/K/V (Qwen2) se presentes. No-op em Llama.
    #[cfg(feature = "gpu")]
    pub(crate) fn add_layer_biases(
        &self,
        l: usize,
        q: &mut [f32],
        k: &mut [f32],
        v: &mut [f32],
        kv_dim: usize,
        n_tok: usize,
    ) -> Result<(), ModelError> {
        let lw = &self.weights.layers[l];
        if let Some(b) = &lw.attn_q_bias {
            add_bias(q, b.dequant_to_f32()?, self.config.n_embd, n_tok);
        }
        if let Some(b) = &lw.attn_k_bias {
            add_bias(k, b.dequant_to_f32()?, kv_dim, n_tok);
        }
        if let Some(b) = &lw.attn_v_bias {
            add_bias(v, b.dequant_to_f32()?, kv_dim, n_tok);
        }
        Ok(())
    }
```

Nota: `add_bias` já existe como `fn` livre em `model.rs` (linha ~15). `KvCache::len/append/advance/k_slice/v_slice` já são usados por `forward` no mesmo arquivo.

- [ ] **Step 5: Implementar `forward_gpu` em `gpu.rs`**

Adicionar em `crates/llama-model/src/gpu.rs` (antes de `mod tests`):
```rust
use crate::attention::{KvCache, attention};
use crate::model::Model;
use crate::ops::{embedding_lookup, rmsnorm_and_scale, rope_norm, swiglu};

impl Model {
    /// Forward de **decode** (n_tok=1) com os 8 matmuls na GPU.
    /// RMSNorm/RoPE/attention/SwiGLU/bias permanecem na CPU.
    /// Retorna os logits do token (tamanho `vocab`).
    pub(crate) fn forward_gpu(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        if tokens.len() != 1 {
            return Err(ModelError::Gpu(format!(
                "forward_gpu exige n_tok=1 (decode), recebeu {}",
                tokens.len()
            )));
        }
        let n_tok = 1usize;
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let token_embd = self.token_embd_f32()?;
        let mut x = embedding_lookup(token_embd, tokens, c.n_embd)?;

        for (l, gw) in w.layers.iter().enumerate() {
            let (attn_norm, ffn_norm) = self.layer_norms_f32(l)?;
            let attn_in = rmsnorm_and_scale(&x, attn_norm, c.n_embd, c.rms_eps);

            let mut q = gpu.matvec_q8_0(&gw.attn_q, &attn_in, c.n_embd, c.n_embd)?;
            let mut k = gpu.matvec_q8_0(&gw.attn_k, &attn_in, c.n_embd, kv_dim)?;
            let mut v = gpu.matvec_q8_0(&gw.attn_v, &attn_in, c.n_embd, kv_dim)?;

            self.add_layer_biases(l, &mut q, &mut k, &mut v, kv_dim, n_tok)?;

            rope_norm(&mut q, n_tok, c.n_head, c.head_dim, c.rope_dim, &self.freq_table, pos0);
            rope_norm(&mut k, n_tok, c.n_head_kv, c.head_dim, c.rope_dim, &self.freq_table, pos0);

            cache.append(l, &k, &v)?;
            let total_len = pos0 + n_tok;
            let attn = attention(
                &q,
                cache.k_slice(l, total_len),
                cache.v_slice(l, total_len),
                n_tok,
                pos0,
                c.n_head,
                c.n_head_kv,
                c.head_dim,
            );
            let attn_out = gpu.matvec_q8_0(&gw.attn_output, &attn, c.n_embd, c.n_embd)?;
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            let ffn_in = rmsnorm_and_scale(&x, ffn_norm, c.n_embd, c.rms_eps);
            let gate = gpu.matvec_q8_0(&gw.ffn_gate, &ffn_in, c.n_embd, c.n_ff)?;
            let up = gpu.matvec_q8_0(&gw.ffn_up, &ffn_in, c.n_embd, c.n_ff)?;
            let act = swiglu(&gate, &up);
            let ffn_out = gpu.matvec_q8_0(&gw.ffn_down, &act, c.n_ff, c.n_embd)?;
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        let output_norm = self.output_norm_f32()?;
        let final_x = rmsnorm_and_scale(&x, output_norm, c.n_embd, c.rms_eps);
        let logits = gpu.matvec_q8_0(&w.output, &final_x, c.n_embd, c.vocab)?;
        Ok(logits)
    }
}
```

Nota: confirmar que `embedding_lookup`, `rmsnorm_and_scale`, `rope_norm`, `swiglu`, `attention`, `KvCache` são acessíveis como `crate::ops::*` / `crate::attention::*` (são — `model.rs` os importa dos mesmos caminhos).

- [ ] **Step 6: Rodar o teste do mock**

Run: `cargo test -p llama-model --features gpu forward_gpu_mock -- --nocapture 2>&1 | tail -15`
Expected: PASS "forward_gpu(mock) == forward CPU — 151936 logits".

- [ ] **Step 7: Commit**

```bash
git add crates/llama-model/src/gpu.rs crates/llama-model/src/model.rs crates/llama-model/src/config.rs
git commit -m "feat(model): forward_gpu (decode) validado bit-a-bit vs CPU via mock"
```

---

## Task 4: `generate_streaming_gpu` (prefill CPU + decode GPU)

**Files:**
- Modify: `crates/llama-model/src/gpu.rs`

- [ ] **Step 1: Escrever o teste com mock (RED)**

Adicionar em `mod tests` de `gpu.rs`:
```rust
    #[test]
    fn generate_streaming_gpu_mock_igual_a_cpu() {
        use llama_sampling::Sampler;
        use llama_tokenizer::Tokenizer;
        use rand::{SeedableRng, rngs::SmallRng};

        let Some((bytes, f, cfg)) = load_qwen() else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let model = crate::model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
        let tok = Tokenizer::from_gguf(&f).unwrap();
        let w = GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
        let sampler = make_greedy_sampler();

        let mut cpu_out = String::new();
        let mut r1 = SmallRng::seed_from_u64(0);
        model
            .generate_streaming(&tok, "Hello", 8, &sampler, &mut r1, &mut |p| cpu_out.push_str(p))
            .unwrap();

        let mut gpu_out = String::new();
        let mut r2 = SmallRng::seed_from_u64(0);
        model
            .generate_streaming_gpu(
                &tok, "Hello", 8, &sampler, &mut r2, &CpuMockMatmul, &w,
                &mut |p| gpu_out.push_str(p),
            )
            .unwrap();

        assert_eq!(cpu_out, gpu_out, "saída GPU(mock) deve igualar CPU");
        eprintln!("generate_streaming_gpu(mock) == CPU: {gpu_out:?}");
    }
```

Nota: `make_greedy_sampler()` é um helper a adicionar no `mod tests` que constrói um `Sampler` greedy usando a API real de `llama-sampling` (espelhar `choose_sampler` de `crates/llama-cli/src/runner.rs` com temperatura 0 / top-k 1). Definir junto:
```rust
    fn make_greedy_sampler() -> llama_sampling::Sampler {
        // Ajustar à API real de llama-sampling usada em runner.rs::choose_sampler.
        llama_sampling::Sampler::greedy()
    }
```

- [ ] **Step 2: Rodar — deve falhar**

Run: `cargo test -p llama-model --features gpu generate_streaming_gpu_mock -- --nocapture 2>&1 | head -10`
Expected: FAIL — método não existe.

- [ ] **Step 3: Implementar `generate_streaming_gpu`**

Adicionar em `gpu.rs`, dentro do `impl Model` (mesmo bloco do `forward_gpu`):
```rust
    /// Igual a `generate_streaming`, mas o **decode** roda na GPU via `gpu`.
    /// O **prefill** do prompt (n_tok>1) permanece na CPU (shader é matvec).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_gpu(
        &self,
        tokenizer: &llama_tokenizer::Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &llama_sampling::Sampler,
        rng: &mut impl rand::Rng,
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
        on_token: &mut impl FnMut(&str),
    ) -> Result<(), ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        let mut cache = self.new_cache();

        // Prefill na CPU.
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
            // Decode na GPU.
            let logits = self.forward_gpu(&[next], &mut cache, gpu, w)?;
            let idx = sampler.sample(&logits, rng);
            next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
        }
        Ok(())
    }
```

- [ ] **Step 4: Confirmar deps**

`gpu.rs` usa `llama_sampling`, `llama_tokenizer`, `rand` — todos já deps de `llama-model` (usados em `generate.rs`). Nenhum ajuste esperado.

- [ ] **Step 5: Rodar o teste**

Run: `cargo test -p llama-model --features gpu generate_streaming_gpu_mock -- --nocapture 2>&1 | tail -10`
Expected: PASS — saída GPU(mock) idêntica à CPU.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-model/src/gpu.rs
git commit -m "feat(model): generate_streaming_gpu (prefill CPU + decode GPU)"
```

---

## Task 5: `DualGpuBackend` em llama-vulkan + validação no hardware real

**Files:**
- Modify: `crates/llama-vulkan/Cargo.toml`
- Create: `crates/llama-vulkan/src/backend.rs`
- Modify: `crates/llama-vulkan/src/lib.rs`
- Modify: `crates/llama-model/src/gpu.rs` (helpers públicos opacos)
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Adicionar dep llama-model (com feature gpu)**

Editar `crates/llama-vulkan/Cargo.toml`, em `[dependencies]`:
```toml
llama-model = { workspace = true, features = ["gpu"] }
```

- [ ] **Step 2: Adicionar helpers públicos opacos em llama-model**

Para o teste de integração externo validar sem expor `KvCache`/`forward_gpu` (ambos `pub(crate)`), adicionar em `gpu.rs`, `impl Model`:
```rust
    /// Decode de 1 passo na CPU (prefill incluso). Cache criado internamente.
    pub fn decode_one_cpu_owned(&self, prompt: &[u32]) -> Result<u32, ModelError> {
        let mut cache = self.new_cache();
        let logits = self.forward(prompt, &mut cache)?;
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }

    /// Prefill na CPU + 1 decode na GPU. Cache criado internamente.
    pub fn decode_one_gpu_owned(
        &self,
        prompt: &[u32],
        gpu: &dyn GpuMatmul,
        w: &GpuRawWeights,
    ) -> Result<u32, ModelError> {
        let mut cache = self.new_cache();
        let _ = self.forward(prompt, &mut cache)?;
        let last = *prompt.last().ok_or_else(|| ModelError::Gpu("prompt vazio".into()))?;
        let logits = self.forward_gpu(&[last], &mut cache, gpu, w)?;
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }
```

- [ ] **Step 3: Escrever o teste de integração no hardware (RED)**

Adicionar em `crates/llama-vulkan/tests/integration.rs`:
```rust
#[test]
fn forward_gpu_real_token_identico_ao_cpu() {
    use std::path::Path;
    let model_path = Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf");
    let Ok(bytes) = std::fs::read(model_path) else {
        eprintln!("qwen ausente — pulando");
        return;
    };
    let ctx = match llama_vulkan::VulkanContext::new() {
        Ok(c) => c,
        Err(e) => { eprintln!("Vulkan indisponível: {e} — pulando"); return; }
    };
    if ctx.amd_compute_devices().len() < 2 {
        eprintln!("Menos de 2 MI50 — pulando");
        return;
    }

    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let cfg = llama_model::LlamaConfig::from_gguf(&f).unwrap();
    let model = llama_model::Model::load_with_config(&f, &bytes, cfg.clone()).unwrap();
    let w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &cfg).unwrap();
    let backend = llama_vulkan::DualGpuBackend::new(&ctx).expect("backend falhou");

    let prompt = [cfg.bos_id];
    let cpu_tok = model.decode_one_cpu_owned(&prompt).unwrap();
    let gpu_tok = model.decode_one_gpu_owned(&prompt, &backend, &w).unwrap();

    assert_eq!(cpu_tok, gpu_tok, "GPU deve gerar o mesmo token que a CPU");
    eprintln!("Forward GPU real OK — token={gpu_tok}");
}
```

Nota: `gguf` e `llama-model` precisam estar nas `[dev-dependencies]` de `llama-vulkan` (ou normais). Adicionar em `crates/llama-vulkan/Cargo.toml`:
```toml
[dev-dependencies]
gguf = { workspace = true }
```
(`llama-model` já está em `[dependencies]` via Step 1.)

- [ ] **Step 4: Implementar `DualGpuBackend`**

Criar `crates/llama-vulkan/src/backend.rs`:
```rust
//! Adaptador: expõe DualGpuMatmul como `llama_model::GpuMatmul`.

use crate::device::VulkanContext;
use crate::dual_gpu::DualGpuMatmul;
use llama_model::{GpuMatmul, ModelError};

/// Backend dual-MI50 que satisfaz a trait do llama-model.
pub struct DualGpuBackend<'ctx> {
    inner: DualGpuMatmul<'ctx>,
}

impl<'ctx> DualGpuBackend<'ctx> {
    pub fn new(ctx: &'ctx VulkanContext) -> Result<Self, ModelError> {
        let inner = DualGpuMatmul::new(ctx).map_err(|e| ModelError::Gpu(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl GpuMatmul for DualGpuBackend<'_> {
    fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, ModelError> {
        self.inner
            .matvec_q8_0(w_bytes, x, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))
    }
}
```

Registrar em `crates/llama-vulkan/src/lib.rs`:
```rust
mod backend;
pub use backend::DualGpuBackend;
```

- [ ] **Step 5: Build + teste no hardware**

```bash
cargo build -p llama-vulkan 2>&1 | tail -5
cargo test -p llama-vulkan forward_gpu_real -- --nocapture 2>&1 | tail -15
```
Expected: PASS "Forward GPU real OK — token=X" (X igual ao token CPU). Pode aparecer o aviso `GpuTensor::drop: recursos nao liberados` (pré-existente) — não falha o teste.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/ crates/llama-model/src/gpu.rs
git commit -m "feat(vulkan): DualGpuBackend impl GpuMatmul + decode GPU==CPU no hardware"
```

---

## Task 6: Integração CLI `--gpu` + benchmark CPU vs GPU

**Files:**
- Modify: `crates/llama-cli/Cargo.toml`
- Modify: `crates/llama-cli/src/runner.rs`

- [ ] **Step 1: Encadear a feature gpu no CLI**

Editar `crates/llama-cli/Cargo.toml` (`llama-vulkan` já é dep opcional):
```toml
[features]
gpu = ["llama-vulkan", "llama-model/gpu"]
```

- [ ] **Step 2: Substituir o bloco de detecção por execução real**

Em `crates/llama-cli/src/runner.rs`:
1. **Remover** o bloco antigo `if args.gpu { ... }` que só imprime (linhas ~152–182).
2. **Substituir** a chamada `model.generate_streaming(...)` por seleção GPU/CPU. Confirmar que `bytes` e `f` (o `GgufFile`) estão em escopo (lidos logo acima):
```rust
    #[cfg(feature = "gpu")]
    let used_gpu = if args.gpu {
        use llama_vulkan::{DualGpuBackend, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if ctx.amd_compute_devices().len() >= 2 => {
                let devs = ctx.amd_compute_devices();
                eprintln!("[GPU] {} + {} — decode na GPU", devs[0].name(), devs[1].name());
                let gpu_w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let backend = DualGpuBackend::new(&ctx)
                    .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
                model.generate_streaming_gpu(
                    &tokenizer, &args.prompt, args.n_predict, &sampler, &mut rng,
                    &backend, &gpu_w,
                    &mut |piece| {
                        if start.is_none() { start = Some(Instant::now()); }
                        on_token(piece);
                        n_tokens += 1;
                    },
                )?;
                true
            }
            Ok(ctx) => {
                eprintln!("[GPU] {} device(s) AMD (<2) — fallback CPU", ctx.amd_compute_devices().len());
                false
            }
            Err(e) => { eprintln!("[GPU] Vulkan indisponível ({e}) — fallback CPU"); false }
        }
    } else { false };
    #[cfg(not(feature = "gpu"))]
    let used_gpu = false;

    if !used_gpu {
        model.generate_streaming(
            &tokenizer, &args.prompt, args.n_predict, &sampler, &mut rng,
            &mut |piece| {
                if start.is_none() { start = Some(Instant::now()); }
                on_token(piece);
                n_tokens += 1;
            },
        )?;
    }
```
Nota: `ModelError` implementa `std::error::Error` (via thiserror) → `?` propaga `ModelError::Gpu` de `from_gguf`. Ajustar nomes de variáveis (`args.n_predict`, `sampler`, `rng`, `tokenizer`, `start`, `n_tokens`, `on_token`) aos reais no `runner.rs`.

- [ ] **Step 3: Build com feature gpu**

```bash
cargo build -p llama-cli --release --features gpu 2>&1 | tail -8
```
Expected: PASS.

- [ ] **Step 4: Smoke test — gera tokens reais na GPU**

```bash
cargo run -p llama-cli --release --features gpu -- \
  --gpu --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Era uma vez" --n-predict 10 2>&1
```
Expected: "[GPU] AMD Radeon Pro VII ... — decode na GPU" e 10 tokens coerentes (texto, não lixo).

- [ ] **Step 5: Benchmark CPU vs GPU**

```bash
echo "=== CPU ==="
cargo run -p llama-cli --release -- \
  --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Era uma vez" --n-predict 50 2>&1 | grep -iE "tok/s|tokens"
echo "=== GPU (dual MI50, re-upload por matmul) ==="
cargo run -p llama-cli --release --features gpu -- \
  --gpu --model models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "Era uma vez" --n-predict 50 2>&1 | grep -iE "tok/s|tokens"
```
**Atenção:** com re-upload de pesos por matmul, a GPU pode ficar *mais lenta* que a CPU neste marco — o objetivo aqui é **correção**, não velocidade. A aceleração depende da residência de pesos em VRAM (Fase 8).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-cli/
git commit -m "feat(cli): --gpu executa decode real nas 2 MI50 (correcao-primeiro)

Bench (50 tok): CPU=<X> tok/s, GPU=<Y> tok/s (re-upload por matmul)"
```

---

## Self-Review

### Cobertura da Especificação
| Requisito | Task |
|---|---|
| Pesos Q8_0 da GPU lidos do GGUF (Model os descarta) | Task 2 (`GpuRawWeights::from_gguf`) |
| 8 matmuls do decode na GPU, resto na CPU | Task 3 (`forward_gpu`) |
| Validação sem hardware (bit-a-bit) | Task 3 (mock = `quantize_q8_0_split`+`matmul_q8_0_actq`) |
| Loop de geração com decode na GPU | Task 4 (`generate_streaming_gpu`) |
| Row-split dual-MI50 plugado | Task 5 (`DualGpuBackend` → `DualGpuMatmul`) |
| Token-idêntico no hardware real | Task 5 (`forward_gpu_real_token_identico_ao_cpu`) |
| CLI `--gpu` gera tokens reais | Task 6 |
| Benchmark CPU vs GPU | Task 6 Step 5 |

### Consistência de Tipos
- `GpuMatmul::matvec_q8_0(&self, &[u8], &[f32], usize, usize) -> Result<Vec<f32>, ModelError>` — assinatura idêntica na trait (Task 1), mock (Task 3), `DualGpuBackend` (Task 5).
- `GpuRawWeights { layers: Vec<GpuLayerRaw>, output: Vec<u8> }` (Task 2) usado nas Tasks 3–6.
- `forward_gpu(&self, &[u32], &mut KvCache, &dyn GpuMatmul, &GpuRawWeights)` consistente entre Task 3 (def) e Task 5 (uso via `decode_one_gpu_owned`).
- `DualGpuBackend::matvec_q8_0` delega a `DualGpuMatmul::matvec_q8_0` (assinatura real confirmada no código).
- Helpers `decode_one_cpu_owned`/`decode_one_gpu_owned` mantêm `KvCache` privado (criado internamente).

### Gaps Documentados (fora de escopo deste marco)
- **Re-upload de pesos por matmul**: cada `matvec_q8_0` reenvia W à VRAM. Correto, porém lento — residência de pesos em VRAM + descritores persistentes é a Fase 8 (onde a aceleração real aparece).
- **Prefill na CPU**: o shader é matvec (1 vetor). Prefill multi-token continua na CPU; shader matmul (tiled) é Fase 8.
- **`GpuTensor::drop` avisa "recursos nao liberados"**: auditar o cleanup antes de execuções longas (pré-requisito de robustez para Fase 8; não bloqueia este marco).
- **Tesla K80**: não contemplada — vendor NVIDIA filtrado, shader wave64 incompatível com wave32. Avaliação separada.
