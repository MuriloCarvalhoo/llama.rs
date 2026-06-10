//! Backend GPU para o passo de decode. Os matmuls do decode são roteados para a GPU;
//! o forward em si (RMSNorm/RoPE/attention/SwiGLU) permanece em `model.rs`.

use crate::config::LlamaConfig;
use crate::error::ModelError;
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
    pub fn from_gguf(f: &GgufFile, bytes: &[u8], cfg: &LlamaConfig) -> Result<Self, ModelError> {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;

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

#[cfg(feature = "gpu")]
use crate::attention::{KvCache, attention};
#[cfg(feature = "gpu")]
use crate::model::Model;
#[cfg(feature = "gpu")]
use crate::ops::{embedding_lookup, rmsnorm_and_scale, rope_norm, swiglu};

#[cfg(feature = "gpu")]
impl Model {
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

    /// Forward de **decode** (n_tok=1) com os 8 matmuls na GPU.
    /// RMSNorm/RoPE/attention/SwiGLU/bias permanecem na CPU.
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

            rope_norm(
                &mut q,
                n_tok,
                c.n_head,
                c.head_dim,
                c.rope_dim,
                &self.freq_table,
                pos0,
            );
            rope_norm(
                &mut k,
                n_tok,
                c.n_head_kv,
                c.head_dim,
                c.rope_dim,
                &self.freq_table,
                pos0,
            );

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
        let last = &final_x[..c.n_embd];
        let logits = gpu.matvec_q8_0(&w.output, last, c.n_embd, c.vocab)?;
        Ok(logits)
    }
}

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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;
    use crate::config::LlamaConfig;
    use gguf::GgufFile;
    use std::path::Path;

    fn load_qwen() -> Option<(Vec<u8>, GgufFile, LlamaConfig)> {
        let bytes =
            std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let cfg = LlamaConfig::from_gguf(&f).ok()?;
        Some((bytes, f, cfg))
    }

    /// Mock que roda a matemática Q8_0 na CPU (mesmas funções quantize_q8_0_split +
    /// matmul_q8_0_actq). Note: o `forward` da CPU usa pesos *repacked* (block_q8_0x8),
    /// cuja ordem de soma difere ligeiramente — por isso a comparação usa tolerância
    /// (~1e-3), não igualdade exata.
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
        eprintln!(
            "forward_gpu(mock) == forward CPU — {} logits",
            logits_gpu.len()
        );
    }

    fn make_greedy_sampler() -> llama_sampling::Sampler {
        llama_sampling::Sampler::Greedy
    }

    #[test]
    fn generate_streaming_gpu_mock_igual_a_cpu() {
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
            .generate_streaming(&tok, "Hello", 8, &sampler, &mut r1, &mut |p| {
                cpu_out.push_str(p)
            })
            .unwrap();

        let mut gpu_out = String::new();
        let mut r2 = SmallRng::seed_from_u64(0);
        model
            .generate_streaming_gpu(
                &tok,
                "Hello",
                8,
                &sampler,
                &mut r2,
                &CpuMockMatmul,
                &w,
                &mut |p| gpu_out.push_str(p),
            )
            .unwrap();

        assert_eq!(cpu_out, gpu_out, "saída GPU(mock) deve igualar CPU");
        eprintln!("generate_streaming_gpu(mock) == CPU: {gpu_out:?}");
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
