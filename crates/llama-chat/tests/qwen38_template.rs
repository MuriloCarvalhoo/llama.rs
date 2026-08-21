//! Cada caso de `refs/chat_qwen38.json` é a saída do Jinja **real** do GGUF, gerada
//! por `scripts/gen-chat-refs.py`. A reimplementação em Rust tem de bater byte a byte:
//! um `\n` a mais no lugar errado já tira o prompt da distribuição do treino.

#![allow(clippy::indexing_slicing)]

use llama_chat::{Esforco, Mensagem, Opcoes, render};
use serde_json::Value;

fn opcoes_do_caso(caso: &Value) -> Opcoes {
    Opcoes {
        ferramentas: caso["tools"].as_array().cloned().unwrap_or_default(),
        add_generation_prompt: caso["add_generation_prompt"].as_bool().unwrap_or(true),
        enable_thinking: caso["enable_thinking"].as_bool().unwrap_or(true),
        esforco: match caso["reasoning_effort"].as_str() {
            Some("low") => Esforco::Low,
            Some("medium") => Esforco::Medium,
            _ => Esforco::Xhigh,
        },
    }
}

#[test]
fn render_bate_com_o_jinja_do_gguf_em_todos_os_casos() {
    let bruto = std::fs::read_to_string("../../refs/chat_qwen38.json").unwrap();
    let refs: Value = serde_json::from_str(&bruto).unwrap();
    let casos = refs["casos"].as_array().unwrap();
    assert!(!casos.is_empty(), "referência vazia");

    let mut falhas = Vec::new();
    for caso in casos {
        let nome = caso["nome"].as_str().unwrap();
        let mensagens: Vec<Mensagem> = caso["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Mensagem::de_json(m).unwrap())
            .collect();

        let obtido = render(&mensagens, &opcoes_do_caso(caso)).unwrap();
        let esperado = caso["esperado"].as_str().unwrap();
        if obtido != esperado {
            falhas.push(format!(
                "caso {nome}:\n  esperado={esperado:?}\n  obtido  ={obtido:?}"
            ));
        }
    }
    assert!(falhas.is_empty(), "divergências:\n{}", falhas.join("\n\n"));
}
