//! Geração de ponta a ponta sem GPU, com o backend roteirizado de `comum`.
//!
//! O que se prova aqui é a costura — template → tokens especiais → prefill com reuso →
//! amostragem → separação de raciocínio/tool call → campos da resposta. A matemática do
//! decode é do backend real e tem os seus próprios testes contra a referência de CPU.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod comum;

use comum::{EOS, cenario, pedido, tokenizer_de_teste};
use llama_server::api::Parada;
use llama_server::motor::Motor;
use serde_json::json;

#[test]
fn separa_raciocinio_de_conteudo_e_para_no_im_end() {
    let tok = tokenizer_de_teste();
    let p = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    let gpu = cenario(&p, "penso</think>resposta<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let r = motor.responder(&p, |_| true).unwrap();

    assert_eq!(r.reasoning, "penso");
    assert_eq!(r.conteudo, "resposta");
    assert_eq!(r.parada, Some(Parada::Fim));
    assert!(r.tokens_saida > 0);
}

#[test]
fn tool_call_sai_como_chamada_com_finish_reason_proprio() {
    let tok = tokenizer_de_teste();
    let ferramenta = json!({
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }
    });
    let p = pedido(
        vec![json!({"role": "user", "content": "oi"})],
        vec![ferramenta],
    );
    let gerado = "vou ler</think><tool_call>\n<function=read>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call><|im_end|>";
    let gpu = cenario(&p, gerado, &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let r = motor.responder(&p, |_| true).unwrap();

    assert_eq!(r.chamadas.len(), 1, "esperava uma chamada");
    assert_eq!(r.chamadas[0].nome, "read");
    assert_eq!(r.chamadas[0].argumentos, r#"{"path":"a.rs"}"#);
    assert_eq!(r.parada, Some(Parada::Ferramenta));
    assert!(
        r.conteudo.is_empty(),
        "o XML da chamada não pode vazar no conteúdo: {:?}",
        r.conteudo
    );
}

#[test]
fn max_tokens_corta_a_geracao_e_marca_length() {
    let tok = tokenizer_de_teste();
    let mut p = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    p.max_tokens = Some(3);
    let gpu = cenario(&p, "</think>abcdefgh<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let r = motor.responder(&p, |_| true).unwrap();

    assert_eq!(r.tokens_saida, 3);
    assert_eq!(r.parada, Some(Parada::Limite));
}

#[test]
fn stop_do_cliente_interrompe_a_geracao() {
    let tok = tokenizer_de_teste();
    let mut p = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    p.stop = vec!["PARE".to_owned()];
    let gpu = cenario(&p, "</think>xxPAREyy<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let r = motor.responder(&p, |_| true).unwrap();

    assert!(r.conteudo.contains("PARE"));
    assert!(!r.conteudo.contains("yy"), "parou tarde: {:?}", r.conteudo);
    assert_eq!(r.parada, Some(Parada::Fim));
}

/// O ganho do item de reuso: o segundo turno só processa o que é novo.
#[test]
fn segundo_turno_so_processa_o_que_cresceu() {
    let tok = tokenizer_de_teste();
    let p1 = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    // Formato que o modelo de fato emite: raciocínio, `\n</think>\n\n`, conteúdo. É o
    // que o template reconstrói byte a byte no turno seguinte — e é isso que faz o
    // prefixo casar.
    let gpu = cenario(&p1, "penso\n</think>\n\nok<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);
    let r1 = motor.responder(&p1, |_| true).unwrap();
    let decodes_do_primeiro = gpu.decodes();

    // Mesmo histórico + a resposta que o modelo deu + uma pergunta nova.
    let p2 = pedido(
        vec![
            json!({"role": "user", "content": "oi"}),
            json!({"role": "assistant", "content": r1.conteudo, "reasoning_content": r1.reasoning}),
            json!({"role": "user", "content": "e ai"}),
        ],
        Vec::new(),
    );
    let novos = motor.responder(&p2, |_| true).unwrap();

    let decodes_do_segundo = gpu.decodes() - decodes_do_primeiro;
    let tokens_do_segundo_prompt = novos.tokens_prompt;
    assert!(
        decodes_do_segundo < tokens_do_segundo_prompt,
        "o segundo turno reprocessou tudo: {decodes_do_segundo} decodes para \
         {tokens_do_segundo_prompt} tokens de prompt"
    );
}

#[test]
fn prompt_maior_que_o_contexto_e_erro() {
    let tok = tokenizer_de_teste();
    let p = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    let gpu = cenario(&p, "</think>ok<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4, EOS);

    assert!(motor.responder(&p, |_| true).is_err());
}

/// Limitação conhecida, documentada aqui como comportamento: se o cliente devolve o
/// histórico **sem** o `reasoning_content`, o template reconstrói o turno do assistant
/// diferente do que foi gerado, o prefixo diverge no meio e — como o estado recorrente
/// do delta-net não volta atrás — o cache inteiro é descartado.
#[test]
fn cliente_que_descarta_o_reasoning_perde_o_cache() {
    let tok = tokenizer_de_teste();
    let p1 = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    let gpu = cenario(&p1, "penso\n</think>\n\nok<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);
    let r1 = motor.responder(&p1, |_| true).unwrap();
    let antes = gpu.decodes();

    // Mesmo histórico, mas sem devolver o raciocínio.
    let p2 = pedido(
        vec![
            json!({"role": "user", "content": "oi"}),
            json!({"role": "assistant", "content": r1.conteudo}),
            json!({"role": "user", "content": "e ai"}),
        ],
        Vec::new(),
    );
    let r2 = motor.responder(&p2, |_| true).unwrap();

    assert_eq!(
        gpu.decodes() - antes,
        r2.tokens_prompt,
        "sem o reasoning de volta, o prompt inteiro é reprocessado"
    );
}

/// Cliente que desiste no meio: o motor tem de parar de gerar. Sem isso a GPU fica
/// presa produzindo tokens que ninguém vai ler — e a próxima requisição espera na fila.
#[test]
fn emissor_que_desiste_interrompe_a_geracao() {
    let tok = tokenizer_de_teste();
    let mut p = pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new());
    p.max_tokens = Some(200);
    let gpu = cenario(&p, "</think>abcdefghij<|im_end|>", &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let mut vistos = 0;
    let r = motor
        .responder(&p, |_| {
            vistos += 1;
            vistos < 3
        })
        .unwrap();

    assert!(
        r.tokens_saida < 10,
        "devia ter parado logo, gerou {}",
        r.tokens_saida
    );
}
