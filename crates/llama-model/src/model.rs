//! Modelo Llama: carrega config+pesos e executa o forward f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgufFile;

use crate::attention::{KvCache, attention};
use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::ops::{argmax, embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm, swiglu};
use crate::weights::Weights;

/// Modelo carregado: config + pesos raw (quantizados ou f32).
pub struct Model {
    pub config: LlamaConfig,
    pub(crate) weights: Weights,
}

impl Model {
    /// Carrega de um GGUF já parseado + bytes do arquivo.
    pub fn load(f: &GgufFile, bytes: &[u8]) -> Result<Self, ModelError> {
        let config = LlamaConfig::from_gguf(f)?;
        let weights = Weights::from_gguf(f, bytes, &config)?;
        Ok(Self { config, weights })
    }

    /// Carrega com config já validada externamente.
    pub fn load_with_config(
        f: &GgufFile,
        bytes: &[u8],
        config: LlamaConfig,
    ) -> Result<Self, ModelError> {
        let weights = Weights::from_gguf(f, bytes, &config)?;
        Ok(Self { config, weights })
    }

    pub(crate) fn new_cache(&self) -> KvCache {
        KvCache::new(self.config.n_layer)
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
            let attn_q_w = lw.attn_q.dequant_to_f32()?;
            let attn_k_w = lw.attn_k.dequant_to_f32()?;
            let attn_v_w = lw.attn_v.dequant_to_f32()?;
            let attn_out_w = lw.attn_output.dequant_to_f32()?;
            let ffn_norm = lw.ffn_norm.dequant_to_f32()?;
            let ffn_gate_w = lw.ffn_gate.dequant_to_f32()?;
            let ffn_up_w = lw.ffn_up.dequant_to_f32()?;
            let ffn_down_w = lw.ffn_down.dequant_to_f32()?;

            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let attn_in = mul_rows(&normed, attn_norm, c.n_embd);

            let mut q = matmul(attn_q_w, &attn_in, c.n_embd, c.n_embd, n_tok);
            let mut k = matmul(attn_k_w, &attn_in, c.n_embd, kv_dim, n_tok);
            let v = matmul(attn_v_w, &attn_in, c.n_embd, kv_dim, n_tok);

            rope_norm(
                &mut q,
                n_tok,
                c.n_head,
                c.head_dim,
                c.rope_dim,
                c.freq_base,
                pos0,
            );
            rope_norm(
                &mut k,
                n_tok,
                c.n_head_kv,
                c.head_dim,
                c.rope_dim,
                c.freq_base,
                pos0,
            );

            cache.append(l, &k, &v);
            let attn = attention(
                &q,
                &cache.k[l],
                &cache.v[l],
                n_tok,
                pos0,
                c.n_head,
                c.n_head_kv,
                c.head_dim,
            );
            let attn_out = matmul(attn_out_w, &attn, c.n_embd, c.n_embd, n_tok);
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let ffn_in = mul_rows(&normed, ffn_norm, c.n_embd);
            let gate = matmul(ffn_gate_w, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let up = matmul(ffn_up_w, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let act = swiglu(&gate, &up);
            let ffn_out = matmul(ffn_down_w, &act, c.n_ff, c.n_embd, n_tok);
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        let output_norm = self.weights.output_norm.dequant_to_f32()?;
        let output_w = self.weights.output.dequant_to_f32()?;
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let final_x = mul_rows(&normed, output_norm, c.n_embd);
        let last = &final_x[(n_tok - 1) * c.n_embd..n_tok * c.n_embd];
        let logits = matmul(output_w, last, c.n_embd, c.vocab, 1);
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
            c.freq_base,
            0,
        );
        let q_sum: f32 = q.iter().sum();
        assert!((q_sum - 148.969_82).abs() < 1e-1, "q_sum={q_sum}");
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
}
