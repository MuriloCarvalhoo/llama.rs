//! Modelo Llama: carrega config+pesos e executa o forward f32.
#![allow(clippy::indexing_slicing)]

use gguf::GgufFile;

use crate::attention::{KvCache, attention};
use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::ops::{argmax, embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm, swiglu};
use crate::weights::Weights;

/// Modelo carregado: config + pesos f32.
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

    pub(crate) fn new_cache(&self) -> KvCache {
        KvCache::new(self.config.n_layer)
    }

    /// Processa `tokens` (prefill ou 1 token de decode) e devolve os logits
    /// (tamanho `vocab`) do ÚLTIMO token. Atualiza o `cache`.
    pub(crate) fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, ModelError> {
        let c = &self.config;
        let n_tok = tokens.len();
        let pos0 = cache.len();
        let kv_dim = c.n_head_kv * c.head_dim;

        let mut x = embedding_lookup(&self.weights.token_embd, tokens, c.n_embd)?;

        for (l, lw) in self.weights.layers.iter().enumerate() {
            // --- bloco de atenção ---
            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let attn_in = mul_rows(&normed, &lw.attn_norm, c.n_embd);

            let mut q = matmul(&lw.attn_q, &attn_in, c.n_embd, c.n_embd, n_tok);
            let mut k = matmul(&lw.attn_k, &attn_in, c.n_embd, kv_dim, n_tok);
            let v = matmul(&lw.attn_v, &attn_in, c.n_embd, kv_dim, n_tok);

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
            let attn_out = matmul(&lw.attn_output, &attn, c.n_embd, c.n_embd, n_tok);

            // residual 1
            for (xi, &ai) in x.iter_mut().zip(attn_out.iter()) {
                *xi += ai;
            }

            // --- bloco FFN ---
            let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
            let ffn_in = mul_rows(&normed, &lw.ffn_norm, c.n_embd);
            let gate = matmul(&lw.ffn_gate, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let up = matmul(&lw.ffn_up, &ffn_in, c.n_embd, c.n_ff, n_tok);
            let act = swiglu(&gate, &up);
            let ffn_out = matmul(&lw.ffn_down, &act, c.n_ff, c.n_embd, n_tok);

            // residual 2
            for (xi, &fi) in x.iter_mut().zip(ffn_out.iter()) {
                *xi += fi;
            }
        }

        cache.advance(n_tok);

        // norma final + projeção de saída só do último token
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let final_x = mul_rows(&normed, &self.weights.output_norm, c.n_embd);
        let last = &final_x[(n_tok - 1) * c.n_embd..n_tok * c.n_embd];
        let logits = matmul(&self.weights.output, last, c.n_embd, c.vocab, 1);
        Ok(logits)
    }

    /// Atalho: argmax dos logits do último token.
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
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::ops::{embedding_lookup, matmul, mul_rows, rmsnorm, rope_norm};
    use std::path::Path;

    fn load_model() -> Option<(Model, Vec<u8>)> {
        let bytes = std::fs::read(Path::new("../../models/stories260K.gguf")).ok()?;
        let f = GgufFile::parse(&bytes).ok()?;
        let m = Model::load(&f, &bytes).ok()?;
        Some((m, bytes))
    }

    #[test]
    fn embd_and_qcur_sums_match_oracle() {
        let Some((m, _)) = load_model() else {
            eprintln!("modelo ausente — pulando");
            return;
        };
        let c = &m.config;
        let tokens = [1u32, 403, 407, 261, 378]; // "Once upon a time"
        let n_tok = tokens.len();

        // embd sum == -3.354056
        let x = embedding_lookup(&m.weights.token_embd, &tokens, c.n_embd).unwrap();
        let embd_sum: f32 = x.iter().sum();
        assert!((embd_sum - (-3.354056)).abs() < 1e-2, "embd_sum={embd_sum}");

        // Qcur-0 pós-rope sum == 148.969818
        let lw = &m.weights.layers[0];
        let normed = rmsnorm(&x, c.n_embd, c.rms_eps);
        let attn_in = mul_rows(&normed, &lw.attn_norm, c.n_embd);
        let mut q = matmul(&lw.attn_q, &attn_in, c.n_embd, c.n_embd, n_tok);
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
}
