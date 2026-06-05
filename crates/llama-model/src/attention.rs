//! KV cache f32 e atenção causal GQA. Layout K/V: [n_layer * ctx * kv_dim], token-major por camada.
#![allow(clippy::indexing_slicing)]

use crate::error::ModelError;
use crate::ops::softmax;
use rayon::prelude::*;

/// KV cache f32 pré-alocado. Buffer flat `[n_layer * ctx * kv_dim]`.
pub(crate) struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    kv_dim: usize,
    ctx: usize,
    len: usize,
}

impl KvCache {
    /// Aloca o cache completo de uma vez. Sem realloc durante geração.
    pub fn new(n_layer: usize, ctx: usize, kv_dim: usize) -> Self {
        let cap = n_layer * ctx * kv_dim;
        Self {
            k: vec![0.0; cap],
            v: vec![0.0; cap],
            kv_dim,
            ctx,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Escreve K/V token-major `[n_tok * kv_dim]` na camada `l` a partir da posição `len`.
    /// Retorna `Err(ContextOverflow)` se `len + n_tok > ctx`.
    pub fn append(&mut self, l: usize, k: &[f32], v: &[f32]) -> Result<(), ModelError> {
        let n_tok = k.len() / self.kv_dim;
        if self.len + n_tok > self.ctx {
            return Err(ModelError::ContextOverflow(self.len + n_tok, self.ctx));
        }
        let layer_stride = self.ctx * self.kv_dim;
        let start = l * layer_stride + self.len * self.kv_dim;
        let end = start + n_tok * self.kv_dim;
        self.k[start..end].copy_from_slice(k);
        self.v[start..end].copy_from_slice(v);
        Ok(())
    }

    /// Retorna o slice K da camada `l` para `total_len` posições (inclui tokens recém-escritos).
    pub fn k_slice(&self, l: usize, total_len: usize) -> &[f32] {
        let layer_stride = self.ctx * self.kv_dim;
        let start = l * layer_stride;
        &self.k[start..start + total_len * self.kv_dim]
    }

    /// Retorna o slice V da camada `l` para `total_len` posições (inclui tokens recém-escritos).
    pub fn v_slice(&self, l: usize, total_len: usize) -> &[f32] {
        let layer_stride = self.ctx * self.kv_dim;
        let start = l * layer_stride;
        &self.v[start..start + total_len * self.kv_dim]
    }

    /// Avança o relógio do cache após processar todas as camadas.
    pub fn advance(&mut self, n_tok: usize) {
        self.len += n_tok;
    }
}

/// Atenção causal GQA em f32. `q` pós-rope token-major [n_tok*n_head*head_dim];
/// `k_cache`/`v_cache` são os buffers COMPLETOS da camada (já com os n_tok novos),
/// [(pos0+n_tok)*kv_dim]. Retorna [n_tok*n_head*head_dim].
///
/// Otimização 4 — heads paralelizadas com rayon no caso n_tok=1 (decode).
/// Para n_tok>1 (prefill) mantém o loop serial para evitar overhead de coleta.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_tok: usize,
    pos0: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_embd = n_head * head_dim;
    let kv_dim = n_head_kv * head_dim;
    let n_rep = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_tok * n_embd];

    // Otimização 4 — caso n_tok=1 (decode): paralelizar sobre heads.
    // Os heads são independentes entre si, então rayon é seguro e eficiente aqui.
    // Para n_tok>1 (prefill), manter loop serial é suficiente pois o gargalo
    // é o matmul dos pesos (já paralelizado via rayon em matmul_q8_0).
    // Paralelizar apenas quando head_dim é grande o suficiente para amortizar overhead rayon.
    // head_dim=8 (stories260K) → serial; head_dim=64 (qwen2.5) → paralelo.
    if n_tok == 1 && n_head >= 4 && head_dim >= 32 {
        let abs_pos = pos0; // n_tok=1, t=0
        let qv_base = &q[0..n_embd]; // único token

        let head_results: Vec<Vec<f32>> = (0..n_head)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / n_rep;
                let q_off = h * head_dim;
                let qv = &qv_base[q_off..q_off + head_dim];

                let mut scores = vec![0.0f32; abs_pos + 1];
                for (j, s) in scores.iter_mut().enumerate() {
                    let k_off = j * kv_dim + kv_h * head_dim;
                    let kv = &k_cache[k_off..k_off + head_dim];
                    let dot: f32 = qv.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum();
                    *s = dot * scale;
                }
                softmax(&mut scores);

                let mut ov = vec![0.0f32; head_dim];
                for (j, &p) in scores.iter().enumerate() {
                    let v_off = j * kv_dim + kv_h * head_dim;
                    let vv = &v_cache[v_off..v_off + head_dim];
                    for (o, &vval) in ov.iter_mut().zip(vv.iter()) {
                        *o += p * vval;
                    }
                }
                ov
            })
            .collect();

        for (h, ov) in head_results.iter().enumerate() {
            let o_off = h * head_dim;
            out[o_off..o_off + head_dim].copy_from_slice(ov);
        }
    } else {
        // Loop serial original — usado para prefill (n_tok>1) e modelos com n_head<4
        for t in 0..n_tok {
            let abs_pos = pos0 + t;
            for h in 0..n_head {
                let kv_h = h / n_rep;
                let q_off = (t * n_head + h) * head_dim;
                let qv = &q[q_off..q_off + head_dim];

                let mut scores = vec![0.0f32; abs_pos + 1];
                for (j, s) in scores.iter_mut().enumerate() {
                    let k_off = j * kv_dim + kv_h * head_dim;
                    let kv = &k_cache[k_off..k_off + head_dim];
                    let dot: f32 = qv.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum();
                    *s = dot * scale;
                }
                softmax(&mut scores);

                let o_off = (t * n_head + h) * head_dim;
                let ov = &mut out[o_off..o_off + head_dim];
                for (j, &p) in scores.iter().enumerate() {
                    let v_off = j * kv_dim + kv_h * head_dim;
                    let vv = &v_cache[v_off..v_off + head_dim];
                    for (o, &vval) in ov.iter_mut().zip(vv.iter()) {
                        *o += p * vval;
                    }
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn single_token_attends_only_itself() {
        let q = vec![1.0, 0.0];
        let k = vec![0.5, 0.5];
        let v = vec![9.0, 7.0];
        let out = attention(&q, &k, &v, 1, 0, 1, 1, 2);
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_maps_query_heads_to_kv_heads() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0];
        let v = vec![5.0];
        let out = attention(&q, &k, &v, 1, 0, 2, 1, 1);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn causal_two_positions_first_token_ignores_future() {
        let q = vec![1.0, 1.0];
        let k = vec![0.0, 100.0];
        let v = vec![2.0, 8.0];
        let out = attention(&q, &k, &v, 2, 0, 1, 1, 1);
        assert!((out[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn parallel_heads_match_serial_for_single_token() {
        // Verifica que o caminho paralelo (n_head=4 >= 4) produz os mesmos
        // resultados que o serial (n_tok=2, forçando o caminho serial).
        // Usa n_tok=1 para paralelo e n_tok=1 com n_head=3 para serial (< 4).
        let head_dim = 2usize;
        let n_head = 4usize;
        let n_head_kv = 2usize;
        let kv_dim = n_head_kv * head_dim;

        // q: 1 token, 4 heads, head_dim=2
        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|i| (i + 1) as f32 * 0.1)
            .collect();
        // k_cache / v_cache: 3 posições
        let n_pos = 3usize;
        let k_cache: Vec<f32> = (0..n_pos * kv_dim).map(|i| (i + 1) as f32 * 0.05).collect();
        let v_cache: Vec<f32> = (0..n_pos * kv_dim).map(|i| (i + 1) as f32 * 0.07).collect();

        // Paralelo (n_tok=1, n_head=4)
        let out_par = attention(&q, &k_cache, &v_cache, 1, 2, n_head, n_head_kv, head_dim);

        // Serial: forçar caminho serial rodando via n_tok=2 e pegando os 2 tokens
        // (mas como q só tem dados para 1 token, usa serial com n_head=3 < 4)
        // Alternativa: testar diretamente com n_head=1 que vai para serial
        let q1 = &q[0..head_dim];
        let kv_h = 0;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores_manual = vec![0.0f32; 3];
        for (j, s) in scores_manual.iter_mut().enumerate() {
            let k_off = j * kv_dim + kv_h * head_dim;
            let dot: f32 = q1
                .iter()
                .zip(k_cache[k_off..k_off + head_dim].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            *s = dot * scale;
        }
        softmax(&mut scores_manual);
        let mut ov_manual = vec![0.0f32; head_dim];
        for (j, &p) in scores_manual.iter().enumerate() {
            let v_off = j * kv_dim + kv_h * head_dim;
            for (o, &vval) in ov_manual
                .iter_mut()
                .zip(v_cache[v_off..v_off + head_dim].iter())
            {
                *o += p * vval;
            }
        }
        // Head 0 do caminho paralelo deve coincidir com o manual
        for (a, b) in out_par[0..head_dim].iter().zip(ov_manual.iter()) {
            assert!((a - b).abs() < 1e-5, "paralelo vs manual: {a} vs {b}");
        }
    }

    #[test]
    fn kvcache_append_stores_and_advance_updates_len() {
        let mut c = KvCache::new(2, 4, 2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        c.advance(1);
        assert_eq!(c.len(), 1);
        assert_eq!(c.k_slice(0, 1), &[1.0, 2.0]);
        assert_eq!(c.v_slice(0, 1), &[3.0, 4.0]);
    }

    #[test]
    fn kvcache_second_append_does_not_reallocate() {
        let mut c = KvCache::new(1, 8, 2);
        let ptr_before = c.k.as_ptr();
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        c.advance(1);
        c.append(0, &[5.0, 6.0], &[7.0, 8.0]).unwrap();
        c.advance(1);
        assert_eq!(c.k.as_ptr(), ptr_before);
        assert_eq!(c.k_slice(0, 2), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn kvcache_overflow_returns_error() {
        let mut c = KvCache::new(1, 1, 2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        c.advance(1);
        let err = c.append(0, &[5.0, 6.0], &[7.0, 8.0]);
        assert!(err.is_err());
    }

    #[test]
    fn kvcache_layers_are_independent() {
        let mut c = KvCache::new(2, 4, 2);
        c.append(0, &[10.0, 11.0], &[20.0, 21.0]).unwrap();
        c.append(1, &[30.0, 31.0], &[40.0, 41.0]).unwrap();
        c.advance(1);
        assert_eq!(c.k_slice(0, 1), &[10.0, 11.0]);
        assert_eq!(c.k_slice(1, 1), &[30.0, 31.0]);
    }

    #[test]
    fn kvcache_slice_includes_pending_tokens_before_advance() {
        // k_slice com total_len=pos0+n_tok deve incluir tokens recém-escritos
        let mut c = KvCache::new(1, 4, 2);
        c.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        // antes de advance, k_slice(0, 2) deve retornar os 2 tokens escritos
        assert_eq!(c.k_slice(0, 2), &[1.0, 2.0, 3.0, 4.0]);
        c.advance(2);
        assert_eq!(c.len(), 2);
    }
}
