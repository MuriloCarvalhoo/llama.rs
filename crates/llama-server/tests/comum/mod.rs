//! Peças compartilhadas dos testes: vocabulário sintético e backend roteirizado.
//!
//! Substituem a GPU e o modelo real para exercitar a costura do servidor — template,
//! tokens especiais, prefill com reuso, amostragem e formatação da resposta.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation
)]

use std::cell::RefCell;

use llama_chat::{Esforco, Mensagem};
use llama_model::{GpuResidentDecode, ModelError};
use llama_server::api::Pedido;
use llama_tokenizer::{Tokenizer, Vocab};
use serde_json::Value;

/// Id de `<|im_end|>` no vocab sintético (a ordem de `ESPECIAIS` fixa os ids).
pub const EOS: u32 = 3;

const ESPECIAIS: [&str; 7] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
];

/// Vocab BPE mínimo: os marcadores do template, um token por byte imprimível, e um
/// merge para o tokenizer seguir o caminho BPE (com `merges` vazio ele vira SPM).
pub fn tokenizer_de_teste() -> Tokenizer {
    let mut tokens: Vec<String> = vec!["<unk>".to_owned()];
    let mut tipos: Vec<i32> = vec![2];
    for e in ESPECIAIS {
        tokens.push(e.to_owned());
        tipos.push(3);
    }
    // Byte-level GPT-2: imprimíveis são identidade, espaço vira 'Ġ' e '\n' vira 'Ċ'.
    for c in (33u8..=126).map(char::from).chain(['\u{0120}', '\u{010a}']) {
        tokens.push(c.to_string());
        tipos.push(1);
    }
    let o = tokens.iter().position(|t| t == "o").unwrap() as u32;
    let i = tokens.iter().position(|t| t == "i").unwrap() as u32;
    tokens.push("oi".to_owned());
    tipos.push(1);

    let n = tokens.len();
    let eos = tokens.iter().position(|t| t == "<|im_end|>").unwrap() as u32;
    let bos = tokens.iter().position(|t| t == "<|endoftext|>").unwrap() as u32;
    Tokenizer::new(Vocab::new(
        tokens,
        vec![0.0; n],
        tipos,
        bos,
        eos,
        0,
        vec![(o, i)],
    ))
}

/// Backend que ignora a matemática e devolve, a cada passo, o token que o roteiro manda.
pub struct Roteirizado {
    /// Chamadas de decode antes do primeiro token gerado (o prompt).
    offset: usize,
    roteiro: Vec<u32>,
    vocab: usize,
    chamadas: RefCell<usize>,
    resets: RefCell<usize>,
}

impl Roteirizado {
    pub fn novo(offset: usize, roteiro: Vec<u32>, vocab: usize) -> Roteirizado {
        Roteirizado {
            offset,
            roteiro,
            vocab,
            chamadas: RefCell::new(0),
            resets: RefCell::new(0),
        }
    }
    pub fn decodes(&self) -> usize {
        *self.chamadas.borrow()
    }
}

impl GpuResidentDecode for Roteirizado {
    fn decode(&self, _token: u32, _pos: usize) -> Result<Vec<f32>, ModelError> {
        let n = *self.chamadas.borrow();
        *self.chamadas.borrow_mut() = n + 1;
        let mut logits = vec![0.0f32; self.vocab];
        // Os logits que decidem o primeiro token gerado são os da última chamada do
        // prefill — daí o `offset` ser `n_prompt - 1`.
        // Roteiro esgotado: encerra o turno em vez de gerar lixo até o teto do ctx.
        let alvo = self
            .roteiro
            .get(n.saturating_sub(self.offset))
            .copied()
            .unwrap_or(EOS);
        logits[alvo as usize] = 10.0;
        Ok(logits)
    }
    fn reset(&self) {
        *self.resets.borrow_mut() += 1;
    }
}

/// Backend roteirizado **por posição**, com cabeça MTP e os limites de contexto do Vulkan:
/// `decode` recusa `pos >= ctx` e `decode_verify` recusa `pos0 + VERIFY_TOK > ctx`, como
/// `verify_shard`. O token que o "modelo" dá depois da posição `pos` é `roteiro[pos - offset]`.
pub struct RoteirizadoMtp {
    offset: usize,
    roteiro: Vec<u32>,
    vocab: usize,
    ctx: usize,
    mtp: bool,
    /// Se a cabeça propõe o token certo (verify aceita tudo) ou um errado (rejeita o 1º).
    acerta: bool,
    /// Comprimento do KV: onde a próxima proposta começa.
    len: RefCell<usize>,
    /// Propostas feitas desde o último verify — a segunda é a encadeada.
    propostas: RefCell<usize>,
}

impl RoteirizadoMtp {
    fn alvo(&self, pos: usize) -> u32 {
        pos.checked_sub(self.offset)
            .and_then(|i| self.roteiro.get(i))
            .copied()
            .unwrap_or(EOS)
    }
    fn logits(&self, pos: usize) -> Vec<f32> {
        let mut l = vec![0.0f32; self.vocab];
        l[self.alvo(pos) as usize] = 10.0;
        l
    }
}

impl GpuResidentDecode for RoteirizadoMtp {
    fn decode(&self, _token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        if pos >= self.ctx {
            return Err(ModelError::Gpu(format!(
                "decode em {pos} com ctx {}",
                self.ctx
            )));
        }
        *self.len.borrow_mut() = pos + 1;
        Ok(self.logits(pos))
    }
    fn reset(&self) {
        *self.len.borrow_mut() = 0;
    }
    fn tem_mtp(&self) -> bool {
        self.mtp
    }
    fn propor_mtp(&self, _token: u32, _hidden_idx: usize) -> Result<u32, ModelError> {
        let i = *self.propostas.borrow();
        *self.propostas.borrow_mut() = i + 1;
        // A 1ª proposta é o token depois da posição `len`; a encadeada, o seguinte a ele.
        let certo = self.alvo(*self.len.borrow() + i);
        // 0 é `<unk>`, que nenhum roteiro gera.
        Ok(if self.acerta { certo } else { 0 })
    }
    fn decode_verify(
        &self,
        _tokens: &[u32; llama_model::VERIFY_TOK],
        pos0: usize,
    ) -> Result<Vec<f32>, ModelError> {
        if pos0 + llama_model::VERIFY_TOK > self.ctx {
            return Err(ModelError::Gpu(format!(
                "verify em {pos0} com ctx {}",
                self.ctx
            )));
        }
        *self.propostas.borrow_mut() = 0;
        *self.len.borrow_mut() = pos0 + llama_model::VERIFY_TOK;
        Ok((pos0..pos0 + llama_model::VERIFY_TOK)
            .flat_map(|p| self.logits(p))
            .collect())
    }
    fn rollback_verify(&self, manter: usize) -> Result<(), ModelError> {
        *self.len.borrow_mut() -= llama_model::VERIFY_TOK - manter;
        Ok(())
    }
}

/// Quantos tokens o prompt de `p` tem depois do template.
pub fn tokens_do_prompt(p: &Pedido, tok: &Tokenizer) -> usize {
    let prompt = llama_chat::render(
        &p.mensagens,
        &llama_chat::Opcoes {
            ferramentas: p.ferramentas.clone(),
            add_generation_prompt: true,
            enable_thinking: p.pensar,
            esforco: p.esforco,
        },
    )
    .unwrap();
    tok.encode_special(&prompt).len()
}

/// Como [`cenario`], mas com o backend de MTP e o contexto fixado em `ctx`.
pub fn cenario_mtp(
    p: &Pedido,
    gerado: &str,
    tok: &Tokenizer,
    ctx: usize,
    mtp: bool,
    acerta: bool,
) -> RoteirizadoMtp {
    RoteirizadoMtp {
        offset: tokens_do_prompt(p, tok) - 1,
        roteiro: tok.encode_special(gerado),
        vocab: 200,
        ctx,
        mtp,
        acerta,
        len: RefCell::new(0),
        propostas: RefCell::new(0),
    }
}

pub fn pedido(mensagens: Vec<Value>, ferramentas: Vec<Value>) -> Pedido {
    Pedido {
        modelo: "teste".to_owned(),
        mensagens: mensagens
            .iter()
            .map(|m| Mensagem::de_json(m).unwrap())
            .collect(),
        ferramentas,
        stream: false,
        max_tokens: None,
        temperatura: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(1),
        stop: Vec::new(),
        esforco: Esforco::Medium,
        pensar: true,
    }
}

/// Monta o cenário: quantos tokens o prompt tem e qual roteiro o modelo "gera".
pub fn cenario(p: &Pedido, gerado: &str, tok: &Tokenizer) -> Roteirizado {
    let roteiro = tok.encode_special(gerado);
    Roteirizado::novo(tokens_do_prompt(p, tok) - 1, roteiro, 200)
}
