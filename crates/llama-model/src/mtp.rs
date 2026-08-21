//! Cabeça de multi-token prediction (MTP) executada na CPU.
//!
//! Serve para **medir a taxa de aceitação** — que é o teto do ganho do speculative
//! decoding, `tok/s = base × (1 + aceitação)` — sem depender de um plano Vulkan novo. A
//! proposta que esta rotina produz é a mesma que a GPU produziria; só o tempo difere, e o
//! tempo não entra na taxa. Ver `docs/mtp-implementacao.md`.
//!
//! Espelha, passo a passo, o que `build_plan` monta para uma camada de atenção do qwen35:
//! norma → q|k|v → QK-norm → rope → atenção → portão → saída → norma → FFN. Qualquer
//! divergência aqui aparece como aceitação artificialmente baixa, então o teste
//! `mtp_aceitacao` compara a proposta contra o token que o modelo de fato amostra.

use rayon::prelude::*;

use crate::config::LlamaConfig;
use crate::error::ModelError;
use crate::gpu::{MixerRaw, MtpAux, MtpRaw, QTensor, TokenEmbd};
use crate::ops;

/// Bytes de uma linha de `n_in` elementos, no tipo dado.
fn bytes_por_linha(ty: gguf::GgmlType, n_in: usize) -> Option<usize> {
    let (elems, bytes) = match ty {
        gguf::GgmlType::Q8_0 => (32usize, 34usize),
        gguf::GgmlType::Q4_K => (256, 144),
        gguf::GgmlType::Q5_K => (256, 176),
        gguf::GgmlType::Q6_K => (256, 210),
        _ => return None,
    };
    n_in.is_multiple_of(elems).then(|| n_in / elems * bytes)
}

/// `y[r] = <linha r de w, x>`, desquantizando **uma linha por vez**.
///
/// Não materializa o tensor em f32 de propósito: o `output.weight` do qwen35 tem 248 320
/// linhas e custaria 5 GB de RAM só para o vocabulário. Cada thread desquantiza a sua
/// linha e descarta.
fn matvec_q(w: &QTensor<'_>, n_in: usize, n_out: usize, x: &[f32]) -> Result<Vec<f32>, ModelError> {
    let lin = bytes_por_linha(w.ty, n_in)
        .ok_or_else(|| ModelError::Gpu(format!("{:?} não tem dequant por linha", w.ty)))?;
    if w.bytes.len() < n_out * lin {
        return Err(ModelError::Gpu(format!(
            "peso curto: {} bytes para {n_out} linhas de {lin}",
            w.bytes.len()
        )));
    }
    (0..n_out)
        .into_par_iter()
        .map(|r| {
            let linha = w
                .bytes
                .get(r * lin..(r + 1) * lin)
                .ok_or_else(|| ModelError::Gpu("linha fora do peso".into()))?;
            let dq = ggml_cpu::dequant_to_f32(linha, w.ty)
                .map_err(|e| ModelError::Gpu(e.to_string()))?;
            Ok(dq.iter().zip(x).map(|(&a, &b)| a * b).sum())
        })
        .collect()
}

/// RMSNorm por cabeça, in-place, com as cabeças espaçadas de `stride`.
///
/// No qwen35 as queries têm `stride = 2 * head_dim`, porque o portão mora ao lado de cada
/// query dentro da mesma projeção.
fn norm_por_cabeca(
    x: &mut [f32],
    n_head: usize,
    head_dim: usize,
    stride: usize,
    w: &[f32],
    eps: f32,
) {
    for h in 0..n_head {
        let base = h * stride;
        let Some(cab) = x.get_mut(base..base + head_dim) else {
            continue;
        };
        let ss: f32 = cab.iter().map(|&v| v * v).sum();
        #[allow(clippy::cast_precision_loss)]
        let escala = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for (v, &g) in cab.iter_mut().zip(w) {
            *v = *v * escala * g;
        }
    }
}

/// RoPE por cabeça com `stride` explícito — mesma razão do `norm_por_cabeca`.
fn rope_stride(
    x: &mut [f32],
    n_head: usize,
    stride: usize,
    rope_dim: usize,
    freq: &[f32],
    pos: usize,
) {
    #[allow(clippy::cast_precision_loss)]
    let p = pos as f32;
    for h in 0..n_head {
        let base = h * stride;
        for i in 0..rope_dim / 2 {
            let Some(&fr) = freq.get(i) else { continue };
            let (s, c) = (p * fr).sin_cos();
            let (ia, ib) = (base + 2 * i, base + 2 * i + 1);
            let (Some(&a), Some(&b)) = (x.get(ia), x.get(ib)) else {
                continue;
            };
            if let Some(v) = x.get_mut(ia) {
                *v = a * c - b * s;
            }
            if let Some(v) = x.get_mut(ib) {
                *v = a * s + b * c;
            }
        }
    }
}

/// Estado da cabeça MTP entre tokens: ela tem KV-cache próprio, separado do modelo.
pub struct MtpHead {
    kcache: Vec<f32>,
    vcache: Vec<f32>,
    /// Quantas posições já foram escritas no cache.
    len: usize,
    ctx: usize,
}

impl MtpHead {
    /// Aloca o KV-cache do bloco, `ctx × kv_dim` para K e para V.
    #[must_use]
    pub fn new(cfg: &LlamaConfig) -> Self {
        let kv_dim = cfg.n_head_kv * cfg.head_dim;
        Self {
            kcache: vec![0.0; cfg.ctx * kv_dim],
            vcache: vec![0.0; cfg.ctx * kv_dim],
            len: 0,
            ctx: cfg.ctx,
        }
    }

    /// Esquece o contexto — usar entre gerações independentes.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Propõe o token **seguinte** a `token`, a partir do hidden state `h` do último layer.
    ///
    /// `h` é o hidden do passo que produziu `token`; `token_embd` é a tabela de embedding,
    /// da qual só a linha do token é dequantizada; `output` é a projeção de vocabulário
    /// compartilhada com o modelo.
    ///
    /// # Errors
    /// Se algum peso tiver forma incompatível com a config.
    // Os pesos já vêm agrupados em `MtpRaw`/`MtpAux`/`QTensor`; o que sobra são slices
    // distintos do passo. Empacotá-los daria uma struct de uso único.
    #[allow(clippy::too_many_arguments)]
    pub fn propor(
        &mut self,
        cfg: &LlamaConfig,
        raw: &MtpRaw<'_>,
        aux: &MtpAux,
        output: &QTensor<'_>,
        token_embd: &TokenEmbd<'_>,
        freq: &[f32],
        h: &[f32],
        token: u32,
    ) -> Result<u32, ModelError> {
        let n_embd = cfg.n_embd;
        let head_dim = cfg.head_dim;
        let n_head = cfg.n_head;
        let kv_dim = cfg.n_head_kv * head_dim;
        let attn_dim = head_dim * n_head;
        let eps = cfg.rms_eps;
        let pos = self.len;
        if pos >= self.ctx {
            return Err(ModelError::Gpu("KV-cache do MTP cheio".into()));
        }

        // 1. eh_proj([enorm(embedding do token) ; hnorm(hidden)]).
        let emb = token_embd.linha(token)?;
        let mut cat = ops::rmsnorm_and_scale(&emb, &aux.enorm, n_embd, eps);
        cat.extend(ops::rmsnorm_and_scale(h, &aux.hnorm, n_embd, eps));
        let mut x = matvec_q(&raw.eh_proj, n_embd * 2, n_embd, &cat)?;

        // 2. A camada de decoder do bloco. O mixer é atenção — ver `MtpRaw`.
        let MixerRaw::Attn {
            attn_q,
            attn_k,
            attn_v,
            attn_output,
        } = &raw.layer.mixer
        else {
            return Err(ModelError::Gpu("bloco MTP devia ser de atenção".into()));
        };
        let normed = ops::rmsnorm_and_scale(&x, &aux.layer.attn_norm, n_embd, eps);
        // No qwen35 a projeção de Q sai com query e portão lado a lado por cabeça.
        let mut q = matvec_q(attn_q, n_embd, attn_dim * 2, &normed)?;
        let mut k = matvec_q(attn_k, n_embd, kv_dim, &normed)?;
        let v = matvec_q(attn_v, n_embd, kv_dim, &normed)?;

        if let (Some(qn), Some(kn)) = (&aux.layer.q_norm, &aux.layer.k_norm) {
            norm_por_cabeca(&mut q, n_head, head_dim, head_dim * 2, qn, eps);
            norm_por_cabeca(&mut k, cfg.n_head_kv, head_dim, head_dim, kn, eps);
        }
        rope_stride(&mut q, n_head, head_dim * 2, cfg.rope_dim, freq, pos);
        rope_stride(&mut k, cfg.n_head_kv, head_dim, cfg.rope_dim, freq, pos);

        // 3. Escreve no cache próprio e faz a atenção causal sobre 0..=pos.
        let off = pos * kv_dim;
        if let Some(dst) = self.kcache.get_mut(off..off + kv_dim) {
            dst.copy_from_slice(&k);
        }
        if let Some(dst) = self.vcache.get_mut(off..off + kv_dim) {
            dst.copy_from_slice(&v);
        }
        self.len += 1;

        let n_rep = n_head / cfg.n_head_kv;
        #[allow(clippy::cast_precision_loss)]
        let escala = 1.0 / (head_dim as f32).sqrt();
        let mut attn = vec![0f32; attn_dim];
        for hh in 0..n_head {
            let kv_h = hh / n_rep;
            let qbase = hh * head_dim * 2;
            let mut scores = Vec::with_capacity(self.len);
            for j in 0..self.len {
                let kb = j * kv_dim + kv_h * head_dim;
                let s: f32 = (0..head_dim)
                    .map(|d| {
                        q.get(qbase + d).copied().unwrap_or(0.0)
                            * self.kcache.get(kb + d).copied().unwrap_or(0.0)
                    })
                    .sum();
                scores.push(s * escala);
            }
            ops::softmax(&mut scores);
            for (j, &p) in scores.iter().enumerate() {
                let vb = j * kv_dim + kv_h * head_dim;
                for d in 0..head_dim {
                    if let Some(a) = attn.get_mut(hh * head_dim + d) {
                        *a += p * self.vcache.get(vb + d).copied().unwrap_or(0.0);
                    }
                }
            }
        }

        // 4. Portão do qwen35: a saída da atenção passa por sigmoid do portão, que veio na
        //    segunda metade de cada cabeça da projeção de Q.
        for hh in 0..n_head {
            for d in 0..head_dim {
                let g = q
                    .get(hh * head_dim * 2 + head_dim + d)
                    .copied()
                    .unwrap_or(0.0);
                if let Some(a) = attn.get_mut(hh * head_dim + d) {
                    *a *= 1.0 / (1.0 + (-g).exp());
                }
            }
        }

        // 5. Projeção de saída, com o residual do mixer.
        let proj = matvec_q(attn_output, attn_dim, n_embd, &attn)?;
        for (xi, p) in x.iter_mut().zip(&proj) {
            *xi += p;
        }

        // 6. FFN com SwiGLU, também com residual.
        let normed2 = ops::rmsnorm_and_scale(&x, &aux.layer.ffn_norm, n_embd, eps);
        let gate = matvec_q(&raw.layer.ffn_gate, n_embd, cfg.n_ff, &normed2)?;
        let up = matvec_q(&raw.layer.ffn_up, n_embd, cfg.n_ff, &normed2)?;
        let ffn = matvec_q(
            &raw.layer.ffn_down,
            cfg.n_ff,
            n_embd,
            &ops::swiglu(&gate, &up),
        )?;
        for (xi, fv) in x.iter_mut().zip(&ffn) {
            *xi += fv;
        }

        // 7. Norma final do bloco (faz o papel do `output_norm`) e projeção de vocabulário.
        let final_h = ops::rmsnorm_and_scale(&x, &aux.shared_head_norm, n_embd, eps);
        let logits = matvec_q(output, n_embd, cfg.vocab, &final_h)?;
        u32::try_from(ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }
}
