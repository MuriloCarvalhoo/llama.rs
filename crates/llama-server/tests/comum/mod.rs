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
    let n_prompt = tok.encode_special(&prompt).len();
    let roteiro = tok.encode_special(gerado);
    Roteirizado::novo(n_prompt - 1, roteiro, 200)
}
