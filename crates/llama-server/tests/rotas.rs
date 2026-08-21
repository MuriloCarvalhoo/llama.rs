//! As rotas HTTP com o motor inteiro por trás — só o backend é falso.
//!
//! É o teste que garante que o que sai no socket é o que um cliente compatível com a
//! OpenAI espera ler: JSON de uma vez, ou SSE terminado em `[DONE]`.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod comum;

use comum::{EOS, cenario, pedido, tokenizer_de_teste};
use llama_server::http::Requisicao;
use llama_server::motor::Motor;
use llama_server::servidor::rotear;
use serde_json::{Value, json};

fn req(metodo: &str, caminho: &str, corpo: &str) -> Requisicao {
    Requisicao {
        metodo: metodo.to_owned(),
        caminho: caminho.to_owned(),
        corpo: corpo.as_bytes().to_vec(),
    }
}

/// Roda uma requisição contra um motor cujo modelo "gera" `gerado`.
///
/// O cenário é montado a partir do **pedido já parseado**: o tamanho do prompt depende
/// das opções que a rota extrai do JSON (esforço de raciocínio, tools), e é ele que diz
/// em que ponto o roteiro do backend falso começa.
fn responder(req: &Requisicao, gerado: &str) -> String {
    let tok = tokenizer_de_teste();
    let p = llama_server::api::parse_pedido(&req.corpo)
        .unwrap_or_else(|_| pedido(vec![json!({"role": "user", "content": "oi"})], Vec::new()));
    let gpu = cenario(&p, gerado, &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);
    let mut saida: Vec<u8> = Vec::new();
    rotear(req, "modelo-teste", &mut motor, &mut saida).unwrap();
    String::from_utf8(saida).unwrap()
}

/// Corpo JSON de uma resposta HTTP completa.
fn corpo_json(resposta: &str) -> Value {
    let (_, corpo) = resposta.split_once("\r\n\r\n").unwrap();
    serde_json::from_str(corpo).unwrap()
}

#[test]
fn get_models_lista_o_modelo_carregado() {
    let r = responder(&req("GET", "/v1/models", ""), "</think>ok<|im_end|>");

    assert!(r.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(corpo_json(&r)["data"][0]["id"], "modelo-teste");
}

#[test]
fn rota_desconhecida_responde_404() {
    let r = responder(&req("GET", "/nada", ""), "</think>ok<|im_end|>");
    assert!(r.starts_with("HTTP/1.1 404"));
}

#[test]
fn corpo_invalido_responde_400_com_envelope_de_erro() {
    let r = responder(
        &req("POST", "/v1/chat/completions", "{quebrado"),
        "</think>ok<|im_end|>",
    );

    assert!(r.starts_with("HTTP/1.1 400"));
    assert_eq!(corpo_json(&r)["error"]["type"], "invalid_request_error");
}

#[test]
fn chat_sem_stream_responde_json_com_o_conteudo() {
    let pedido = r#"{"model":"m","messages":[{"role":"user","content":"oi"}],"temperature":0}"#;

    let r = responder(
        &req("POST", "/v1/chat/completions", pedido),
        "penso</think>ok<|im_end|>",
    );

    assert!(r.starts_with("HTTP/1.1 200 OK"));
    let v = corpo_json(&r);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "ok");
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "penso");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn chat_com_stream_manda_sse_ate_o_done() {
    let pedido = r#"{"model":"m","messages":[{"role":"user","content":"oi"}],"stream":true,"temperature":0}"#;

    let r = responder(
        &req("POST", "/v1/chat/completions", pedido),
        "penso</think>ok<|im_end|>",
    );

    assert!(r.contains("Content-Type: text/event-stream"));
    assert!(r.trim_end().ends_with("data: [DONE]"));

    let eventos: Vec<Value> = r
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();

    assert!(
        eventos
            .iter()
            .all(|e| e["object"] == "chat.completion.chunk")
    );
    let texto: String = eventos
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    let raciocinio: String = eventos
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    assert_eq!(texto, "ok");
    assert_eq!(raciocinio, "penso");
    assert_eq!(
        eventos.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[test]
fn stream_com_tool_call_manda_o_delta_de_tool_calls() {
    let ferramenta = json!({
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }
    });
    let pedido = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "oi"}],
        "tools": [ferramenta],
        "stream": true,
        "temperature": 0
    });
    let gerado = "</think><tool_call>\n<function=read>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call><|im_end|>";

    let tok = tokenizer_de_teste();
    let p = llama_server::api::parse_pedido(pedido.to_string().as_bytes()).unwrap();
    let gpu = cenario(&p, gerado, &tok);
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);
    let mut saida: Vec<u8> = Vec::new();
    rotear(
        &req("POST", "/v1/chat/completions", &pedido.to_string()),
        "modelo-teste",
        &mut motor,
        &mut saida,
    )
    .unwrap();
    let r = String::from_utf8(saida).unwrap();

    let eventos: Vec<Value> = r
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let chamada = eventos
        .iter()
        .find(|e| !e["choices"][0]["delta"]["tool_calls"].is_null())
        .expect("nenhum delta de tool_calls");

    let tc = &chamada["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tc["index"], 0);
    assert_eq!(tc["function"]["name"], "read");
    assert_eq!(tc["function"]["arguments"], r#"{"path":"a.rs"}"#);
    assert_eq!(
        eventos.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
}
