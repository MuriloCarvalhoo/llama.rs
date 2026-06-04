//! KV cache f32 e atenção causal GQA. Layout K/V: [n_layer * ctx * kv_dim], token-major por camada.
#![allow(clippy::indexing_slicing)]

use crate::error::ModelError;
use crate::ops::softmax;

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

    for t in 0..n_tok {
        let abs_pos = pos0 + t;
        for h in 0..n_head {
            let kv_h = h / n_rep;
            let q_off = (t * n_head + h) * head_dim;
            let qv = &q[q_off..q_off + head_dim];

            // scores causais sobre posições 0..=abs_pos
            let mut scores = vec![0.0f32; abs_pos + 1];
            for (j, s) in scores.iter_mut().enumerate() {
                let k_off = j * kv_dim + kv_h * head_dim;
                let kv = &k_cache[k_off..k_off + head_dim];
                let dot: f32 = qv.iter().zip(kv.iter()).map(|(&a, &b)| a * b).sum();
                *s = dot * scale;
            }
            softmax(&mut scores);

            // saída ponderada por V
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
