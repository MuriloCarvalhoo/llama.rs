//! Modelo Llama: carrega config+pesos e executa o forward f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgufFile;

use crate::attention::{KvCache, attention};
use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::ops::{
    argmax, embedding_lookup, quantize_q8_0_split, rmsnorm_and_scale, rope_norm, swiglu,
};
use crate::weights::Weights;

/// Soma `bias` (shape `dim`) a cada linha de `x` (shape `n_tok × dim`).
fn add_bias(x: &mut [f32], bias: &[f32], dim: usize, n_tok: usize) {
    for t in 0..n_tok {
        let row = &mut x[t * dim..(t + 1) * dim];
        for (xi, &bi) in row.iter_mut().zip(bias.iter()) {
            *xi += bi;
        }
    }
}

/// Modelo carregado: config + pesos raw (quantizados ou f32).
pub struct Model {
    pub config: LlamaConfig,
    pub(crate) weights: Weights,
    /// Otimização 3 — tabela de frequências RoPE pré-computada.
    /// `freq_table[i] = freq_base^(-2i/rope_dim)` para i em 0..rope_dim/2.
    /// Elimina `powf` no loop interno da RoPE.
    pub(crate) freq_table: Vec<f32>,
}

impl Model {
    /// Carrega de um GGUF já parseado + bytes do arquivo.
    pub fn load(f: &GgufFile, bytes: &[u8]) -> Result<Self, ModelError> {
        let config = LlamaConfig::from_gguf(f)?;
        let weights = Weights::from_gguf(f, bytes, &config)?;
        let freq_table = build_freq_table(config.freq_base, config.rope_dim);
        Ok(Self {
            config,
            weights,
            freq_table,
        })
    }

    /// Carrega com config já validada externamente.
    pub fn load_with_config(
        f: &GgufFile,
        bytes: &[u8],
        config: LlamaConfig,
    ) -> Result<Self, ModelError> {
        let weights = Weights::from_gguf(f, bytes, &config)?;
        let freq_table = build_freq_table(config.freq_base, config.rope_dim);
        Ok(Self {
            config,
            weights,
            freq_table,
        })
    }

    pub(crate) fn new_cache(&self) -> KvCache {
        let c = &self.config;
        let kv_dim = c.n_head_kv * c.head_dim;
        KvCache::new(c.n_layer, c.ctx, kv_dim)
    }

    /// Soma dos bytes raw de todos os pesos (footprint de RAM, sem dequant).
    pub fn memory_bytes(&self) -> usize {
        self.weights.memory_bytes()
    }

    /// Contagem total de elementos em todos os tensores de peso.
    pub fn weight_element_count(&self) -> usize {
        let w = &self.weights;
        let layer_elem: usize = w
            .layers
            .iter()
            .map(|lw| {
                lw.attn_norm.n_elements()
                    + lw.attn_q.n_elements()
                    + lw.attn_k.n_elements()
                    + lw.attn_v.n_elements()
                    + lw.attn_output.n_elements()
                    + lw.ffn_norm.n_elements()
                    + lw.ffn_gate.n_elements()
                    + lw.ffn_up.n_elements()
                    + lw.ffn_down.n_elements()
            })
            .sum();
        w.token_embd.n_elements() + layer_elem + w.output_norm.n_elements() + w.output.n_elements()
    }

    /// Processa `tokens` e devolve logits (tamanho `vocab`) do último token.
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let n_tok = tokens.len();
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let token_embd = self.weights.token_embd.dequant_to_f32()?;
        let mut x = embedding_lookup(token_embd, tokens, c.n_embd)?;

        for (l, lw) in self.weights.layers.iter().enumerate() {
            let attn_norm = lw.attn_norm.dequant_to_f32()?;
            let ffn_norm = lw.ffn_norm.dequant_to_f32()?;

            // Otimização 5 — rmsnorm + mul_rows fundidos em um único passe
            let attn_in = rmsnorm_and_scale(&x, attn_norm, c.n_embd, c.rms_eps);

            // Otimização 3 — pré-quantizar attn_in uma vez; reusar para Q, K, V.
            // Otimização 4 — para modelos grandes (n_embd>=512) e decode (n_tok=1):
            //   Q/K/V são independentes → rayon::join. Para modelos pequenos, o overhead
            //   do rayon supera o ganho (stories260K n_embd=64 → serial).
            let (mut q, mut k, mut v) = if n_tok == 1 && c.n_embd >= 512 {
                let attn_in_q8 = quantize_q8_0_split(&attn_in, c.n_embd, n_tok);
                let ((q_res, k_res), v_res) = rayon::join(
                    || {
                        rayon::join(
                            || {
                                lw.attn_q.matmul_into_with_q8(
                                    &attn_in_q8,
                                    &attn_in,
                                    c.n_embd,
                                    c.n_embd,
                                    n_tok,
                                )
                            },
                            || {
                                lw.attn_k.matmul_into_with_q8(
                                    &attn_in_q8,
                                    &attn_in,
                                    c.n_embd,
                                    kv_dim,
                                    n_tok,
                                )
                            },
                        )
                    },
                    || {
                        lw.attn_v.matmul_into_with_q8(
                            &attn_in_q8,
                            &attn_in,
                            c.n_embd,
                            kv_dim,
                            n_tok,
                        )
                    },
                );
                (q_res?, k_res?, v_res?)
            } else {
                // Prefill: serial com q8 pré-quantizado
                let attn_in_q8 = quantize_q8_0_split(&attn_in, c.n_embd, n_tok);
                let q = lw.attn_q.matmul_into_with_q8(
                    &attn_in_q8,
                    &attn_in,
                    c.n_embd,
                    c.n_embd,
                    n_tok,
                )?;
                let k = lw.attn_k.matmul_into_with_q8(
                    &attn_in_q8,
                    &attn_in,
                    c.n_embd,
                    kv_dim,
                    n_tok,
                )?;
                let v = lw.attn_v.matmul_into_with_q8(
                    &attn_in_q8,
                    &attn_in,
                    c.n_embd,
                    kv_dim,
                    n_tok,
                )?;
                (q, k, v)
            };

            if let Some(b) = &lw.attn_q_bias {
                add_bias(&mut q, b.dequant_to_f32()?, c.n_embd, n_tok);
            }
            if let Some(b) = &lw.attn_k_bias {
                add_bias(&mut k, b.dequant_to_f32()?, kv_dim, n_tok);
            }
            if let Some(b) = &lw.attn_v_bias {
                add_bias(&mut v, b.dequant_to_f32()?, kv_dim, n_tok);
            }

            // Otimização 3 — rope_norm com tabela de frequências pré-computada
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
            let attn_out = lw
                .attn_output
                .matmul_into(&attn, c.n_embd, c.n_embd, n_tok)?;
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            // Otimização 5 — rmsnorm + mul_rows fundidos em um único passe
            let ffn_in = rmsnorm_and_scale(&x, ffn_norm, c.n_embd, c.rms_eps);

            // Pré-quantizar ffn_in uma vez; reusar para gate e up (sequencial para evitar
            // contenção de largura de banda: dois matmuls de 4.6MB em paralelo causam
            // interleaving pelos workers rayon → cache thrashing e pressão dupla de DRAM).
            let ffn_in_q8 = quantize_q8_0_split(&ffn_in, c.n_embd, n_tok);
            let gate = lw
                .ffn_gate
                .matmul_into_with_q8(&ffn_in_q8, &ffn_in, c.n_embd, c.n_ff, n_tok)?;
            let up = lw
                .ffn_up
                .matmul_into_with_q8(&ffn_in_q8, &ffn_in, c.n_embd, c.n_ff, n_tok)?;
            let act = swiglu(&gate, &up);
            let ffn_out = lw.ffn_down.matmul_into(&act, c.n_ff, c.n_embd, n_tok)?;
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        let output_norm = self.weights.output_norm.dequant_to_f32()?;
        // Otimização 5 — output norm + scale fundidos
        let final_x = rmsnorm_and_scale(&x, output_norm, c.n_embd, c.rms_eps);
        let last = &final_x[(n_tok - 1) * c.n_embd..n_tok * c.n_embd];
        let logits = self
            .weights
            .output
            .matmul_into(last, c.n_embd, c.vocab, 1)?;
        Ok(logits)
    }

    pub(crate) fn forward_argmax(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<u32, ModelError> {
        let logits = self.forward(tokens, cache)?;
        u32::try_from(argmax(&logits)).map_err(|_| ModelError::Overflow)
    }

    /// Processa um batch de sequências independentes, cada uma com seu próprio cache.
    /// Retorna um vetor de logits (tamanho `vocab`) por sequência.
    /// `batch` e `caches` devem ter o mesmo comprimento.
    pub(crate) fn forward_batch(
        &self,
        batch: &[&[u32]],
        caches: &mut [KvCache],
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        if batch.len() != caches.len() {
            return Err(ModelError::BatchMismatch(batch.len(), caches.len()));
        }
        batch
            .iter()
            .zip(caches.iter_mut())
            .map(|(tokens, cache)| self.forward(tokens, cache))
            .collect()
    }
}

/// Constrói a tabela de frequências RoPE: `freq_base^(-2i/rope_dim)` para i em 0..rope_dim/2.
fn build_freq_table(freq_base: f32, rope_dim: usize) -> Vec<f32> {
    (0..rope_dim / 2)
        .map(|i| freq_base.powf(-2.0 * i as f32 / rope_dim as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::ops::{embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm};
    use std::path::Path;

    fn load_model() -> Option<Model> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        Model::load(&f, &bytes).ok()
    }

    #[test]
    fn embd_and_qcur_sums_match_oracle() {
        let Some(m) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = &m.config;
        let tokens = [1u32, 403, 407, 261, 378];
        let n_tok = tokens.len();

        let token_embd = m.weights.token_embd.dequant_to_f32().unwrap();
        let x = embedding_lookup(token_embd, &tokens, c.n_embd).unwrap();
        let embd_sum: f32 = x.iter().sum();
        assert!((embd_sum - (-3.354056)).abs() < 1e-2, "embd_sum={embd_sum}");

        let lw = &m.weights.layers[0];
        let attn_norm = lw.attn_norm.dequant_to_f32().unwrap();
        let attn_q_w = lw.attn_q.dequant_to_f32().unwrap();
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let attn_in = mul_rows(&normed, attn_norm, c.n_embd);
        let mut q = matmul(attn_q_w, &attn_in, c.n_embd, c.n_embd, n_tok);
        rope_norm(
            &mut q,
            n_tok,
            c.n_head,
            c.head_dim,
            c.rope_dim,
            &m.freq_table,
            0,
        );
        let q_sum: f32 = q.iter().sum();
        assert!((q_sum - 148.969_82).abs() < 1e-1, "q_sum={q_sum}");
    }

    #[test]
    fn qwen_forward_bos_does_not_panic() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/qwen2.5-0.5b-instruct-q8_0.gguf"))
        else {
            eprintln!("qwen ausente — pulando");
            return;
        };
        let f = GgufFile::parse(&bytes).unwrap();
        let cfg = LlamaConfig::from_gguf(&f).unwrap();
        let m = Model::load_with_config(&f, &bytes, cfg).unwrap();
        let mut cache = m.new_cache();
        let result = m.forward_argmax(&[m.config.bos_id], &mut cache);
        assert!(
            result.is_ok(),
            "forward qwen2 deve retornar Ok: {:?}",
            result.err()
        );
        eprintln!("qwen2 forward BOS → token {}", result.unwrap());
    }

    #[test]
    fn memory_bytes_less_than_file_size() {
        let Ok(bytes) = std::fs::read(Path::new("../../models/stories260K.gguf")) else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let file_size = bytes.len();
        let f = GgufFile::parse(&bytes).unwrap();
        let m = Model::load(&f, &bytes).unwrap();
        assert!(
            m.memory_bytes() <= file_size,
            "memory_bytes={} > file_size={file_size}",
            m.memory_bytes()
        );
    }

    #[test]
    fn forward_batch_matches_separate_forwards() {
        let Some(m) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let tokens_a = [1u32, 403];
        let tokens_b = [1u32, 261];

        // Forward individual
        let mut ca = m.new_cache();
        let mut cb = m.new_cache();
        let logits_a = m.forward(&tokens_a, &mut ca).unwrap();
        let logits_b = m.forward(&tokens_b, &mut cb).unwrap();

        // Forward batch
        let mut ca2 = m.new_cache();
        let mut cb2 = m.new_cache();
        let batch: &[&[u32]] = &[&tokens_a, &tokens_b];
        let results = m.forward_batch(batch, &mut [ca2, cb2]).unwrap();

        for (r, expected) in results[0].iter().zip(logits_a.iter()) {
            assert!(
                (r - expected).abs() < 1e-6,
                "batch[0] diverge: {r} vs {expected}"
            );
        }
        for (r, expected) in results[1].iter().zip(logits_b.iter()) {
            assert!(
                (r - expected).abs() < 1e-6,
                "batch[1] diverge: {r} vs {expected}"
            );
        }
    }

    #[test]
    fn forward_batch_mismatch_returns_error() {
        let Some(m) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let tokens = [1u32];
        let batch: &[&[u32]] = &[&tokens];
        let mut caches = [m.new_cache(), m.new_cache()];
        let err = m.forward_batch(batch, &mut caches);
        assert!(err.is_err());
    }
}
