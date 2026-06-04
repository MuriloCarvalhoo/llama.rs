//! KV cache f32 e atenção causal GQA. Layout K/V: [pos * (n_head_kv*head_dim)].
#![allow(clippy::indexing_slicing)]

use crate::ops::softmax;

/// KV cache f32, uma entrada por camada. `len` = posições já armazenadas.
pub(crate) struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Anexa K/V (token-major [n_tok*kv_dim]) à camada `l`.
    pub fn append(&mut self, l: usize, k: &[f32], v: &[f32]) {
        self.k[l].extend_from_slice(k);
        self.v[l].extend_from_slice(v);
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
    use super::*;

    #[test]
    fn single_token_attends_only_itself() {
        // n_head=1, n_head_kv=1, head_dim=2, 1 token, pos0=0.
        // Só uma posição → softmax trivial → out == V.
        let q = vec![1.0, 0.0];
        let k = vec![0.5, 0.5];
        let v = vec![9.0, 7.0];
        let out = attention(&q, &k, &v, 1, 0, 1, 1, 2);
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn gqa_maps_query_heads_to_kv_heads() {
        // n_head=2, n_head_kv=1 → ambos query heads usam kv head 0. head_dim=1.
        // 1 token, 1 posição → out de cada head == V[0].
        let q = vec![1.0, 2.0]; // head0=1, head1=2
        let k = vec![3.0];
        let v = vec![5.0];
        let out = attention(&q, &k, &v, 1, 0, 2, 1, 1);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn causal_two_positions_first_token_ignores_future() {
        // n_head=1,n_head_kv=1,head_dim=1, 2 tokens prefill, pos0=0.
        // K=[k0,k1], V=[v0,v1]. token0 (pos0) só vê pos0 → out0=v0.
        let q = vec![1.0, 1.0];
        let k = vec![0.0, 100.0]; // k1 grande não afeta token0
        let v = vec![2.0, 8.0];
        let out = attention(&q, &k, &v, 2, 0, 1, 1, 1);
        assert!((out[0] - 2.0).abs() < 1e-6); // token0 → v0
    }

    #[test]
    fn kvcache_append_and_advance() {
        let mut c = KvCache::new(2);
        c.append(0, &[1.0, 2.0], &[3.0, 4.0]);
        c.advance(1);
        assert_eq!(c.len(), 1);
        assert_eq!(c.k[0], vec![1.0, 2.0]);
    }
}
