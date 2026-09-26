//! As rotas HTTP com o motor inteiro por trás — só o backend é falso.
//!
//! É o teste que garante que o que sai no socket é o que um cliente compatível com a
//! OpenAI espera ler: JSON de uma vez, ou SSE terminado em `[DONE]`.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

mod comum;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use comum::{EOS, Roteirizado, cenario, pedido, tokenizer_de_teste};
use llama_model::{GpuResidentDecode, ModelError};
use llama_server::http::Requisicao;
use llama_server::motor::Motor;
use llama_server::servidor::{Prazos, aceitar, rotear, trabalhar};
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
    let pedido =
        r#"{"model":"modelo-teste","messages":[{"role":"user","content":"oi"}],"temperature":0}"#;

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
    let pedido = r#"{"model":"modelo-teste","messages":[{"role":"user","content":"oi"}],"stream":true,"temperature":0}"#;

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

/// Prompt que não cabe no contexto é erro do pedido: 400, e no streaming **antes** de abrir
/// o SSE — um 200 seguido de evento de erro faz o cliente achar que a geração começou.
/// A requisição seguinte, que cabe, continua sendo atendida.
#[test]
fn prompt_maior_que_o_contexto_responde_400_antes_do_stream() {
    let tok = tokenizer_de_teste();
    let normal =
        r#"{"model":"modelo-teste","messages":[{"role":"user","content":"oi"}],"temperature":0}"#;
    let curto = llama_server::api::parse_pedido(normal.as_bytes()).unwrap();
    let gpu = cenario(&curto, "</think>ok<|im_end|>", &tok);
    // Cabe o prompt curto e a resposta; não cabe o longo.
    let ctx = comum::tokens_do_prompt(&curto, &tok) + 16;
    let mut motor = Motor::novo(&tok, &gpu, ctx, EOS);
    let longo = "palavra ".repeat(200);

    for stream in [false, true] {
        let corpo = json!({
            "model": "modelo-teste",
            "messages": [{"role": "user", "content": longo}],
            "stream": stream,
            "temperature": 0
        });
        let mut saida: Vec<u8> = Vec::new();
        rotear(
            &req("POST", "/v1/chat/completions", &corpo.to_string()),
            "modelo-teste",
            &mut motor,
            &mut saida,
        )
        .unwrap();
        let r = String::from_utf8(saida).unwrap();

        assert!(r.starts_with("HTTP/1.1 400"), "stream={stream}: {r}");
        assert!(!r.contains("text/event-stream"), "stream={stream}: {r}");
        assert_eq!(corpo_json(&r)["error"]["type"], "invalid_request_error");
    }

    let mut saida: Vec<u8> = Vec::new();
    rotear(
        &req("POST", "/v1/chat/completions", normal),
        "modelo-teste",
        &mut motor,
        &mut saida,
    )
    .unwrap();
    let r = String::from_utf8(saida).unwrap();
    assert!(r.starts_with("HTTP/1.1 200 OK"), "{r}");
    assert_eq!(corpo_json(&r)["choices"][0]["message"]["content"], "ok");
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
        "model": "modelo-teste",
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

/// `model` diferente do servido é 404 no formato da OpenAI, e no streaming antes do SSE.
#[test]
fn modelo_diferente_do_servido_responde_404_model_not_found() {
    for stream in [false, true] {
        let corpo = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "oi"}],
            "stream": stream
        });
        let r = responder(
            &req("POST", "/v1/chat/completions", &corpo.to_string()),
            "</think>ok<|im_end|>",
        );
        assert!(r.starts_with("HTTP/1.1 404"), "stream={stream}: {r}");
        assert!(!r.contains("text/event-stream"), "stream={stream}: {r}");
        assert_eq!(corpo_json(&r)["error"]["code"], "model_not_found");
    }
}

/// Sem `model` o pedido vale: cliente local costuma omitir.
#[test]
fn pedido_sem_model_e_atendido() {
    let corpo = r#"{"messages":[{"role":"user","content":"oi"}],"temperature":0}"#;
    let r = responder(
        &req("POST", "/v1/chat/completions", corpo),
        "</think>ok<|im_end|>",
    );
    assert!(r.starts_with("HTTP/1.1 200 OK"), "{r}");
}

#[test]
fn parametro_invalido_no_stream_responde_400_antes_do_sse() {
    let corpo = r#"{"model":"modelo-teste","messages":[{"role":"user","content":"oi"}],"stream":true,"temperature":5}"#;
    let r = responder(
        &req("POST", "/v1/chat/completions", corpo),
        "</think>ok<|im_end|>",
    );
    assert!(r.starts_with("HTTP/1.1 400"), "{r}");
    assert!(!r.contains("text/event-stream"), "{r}");
    assert_eq!(corpo_json(&r)["error"]["type"], "invalid_request_error");
}

/// Backend que demora em cada passo, para uma geração durar o bastante de observar o
/// servidor enquanto ela roda.
struct Lento(Roteirizado);

impl GpuResidentDecode for Lento {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
        std::thread::sleep(Duration::from_millis(20));
        self.0.decode(token, pos)
    }
    fn reset(&self) {
        self.0.reset();
    }
}

/// Manda `bruto` numa conexão nova e lê a resposta até o servidor fechar.
fn http_cru(porta: u16, bruto: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", porta)).unwrap();
    s.write_all(bruto.as_bytes()).unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    r
}

fn post_chat(corpo: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{corpo}",
        corpo.len()
    )
}

const PRAZOS: Prazos = Prazos {
    leitura: Duration::from_secs(5),
    escrita: Duration::from_secs(5),
};

/// O E03: com uma geração rodando, `/health` responde na hora em vez de esperar a fila —
/// a thread do HTTP responde sozinha o que não precisa do motor.
#[test]
fn health_responde_durante_uma_geracao() {
    let escuta = TcpListener::bind("127.0.0.1:0").unwrap();
    let porta = escuta.local_addr().unwrap().port();
    let (fila, trabalhos) = mpsc::sync_channel(4);
    std::thread::spawn(move || aceitar(&escuta, "modelo-teste", &fila, PRAZOS));

    let corpo =
        r#"{"model":"modelo-teste","messages":[{"role":"user","content":"oi"}],"temperature":0}"#;
    let tok = tokenizer_de_teste();
    let p = llama_server::api::parse_pedido(corpo.as_bytes()).unwrap();
    let gpu = Lento(cenario(&p, "</think>ok<|im_end|>", &tok));
    let mut motor = Motor::novo(&tok, &gpu, 4096, EOS);

    let geracao = std::thread::spawn(move || {
        let r = http_cru(porta, &post_chat(corpo));
        (r, Instant::now())
    });
    let saude = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let r = http_cru(porta, "GET /health HTTP/1.1\r\n\r\n");
        (r, Instant::now())
    });
    // O worker roda aqui, dono do motor, e atende um pedido só.
    trabalhar(trabalhos.iter().take(1), "modelo-teste", &mut motor);

    let (r_gen, fim_gen) = geracao.join().unwrap();
    let (r_saude, fim_saude) = saude.join().unwrap();
    assert!(r_saude.starts_with("HTTP/1.1 200 OK"), "{r_saude}");
    assert!(r_gen.starts_with("HTTP/1.1 200 OK"), "{r_gen}");
    assert_eq!(corpo_json(&r_gen)["choices"][0]["message"]["content"], "ok");
    assert!(fim_saude < fim_gen, "o /health esperou a geração terminar");
}

/// Fila cheia é 503 com `Retry-After`, não uma conexão que espera sem prazo.
#[test]
fn fila_cheia_responde_503_com_retry_after() {
    let escuta = TcpListener::bind("127.0.0.1:0").unwrap();
    let porta = escuta.local_addr().unwrap().port();
    // Fila de 1 e nenhum worker: o primeiro pedido ocupa a vaga, o segundo não cabe.
    let (fila, _trabalhos) = mpsc::sync_channel(1);
    std::thread::spawn(move || aceitar(&escuta, "modelo-teste", &fila, PRAZOS));

    let corpo = r#"{"messages":[{"role":"user","content":"oi"}]}"#;
    let mut primeiro = TcpStream::connect(("127.0.0.1", porta)).unwrap();
    primeiro.write_all(post_chat(corpo).as_bytes()).unwrap();
    std::thread::sleep(Duration::from_millis(200));

    let r = http_cru(porta, &post_chat(corpo));
    assert!(r.starts_with("HTTP/1.1 503"), "{r}");
    assert!(r.contains("Retry-After: "), "{r}");
    drop(primeiro);
}

/// Cliente que abre a conexão e não termina os cabeçalhos é encerrado no prazo com 408, e
/// o servidor segue atendendo.
#[test]
fn cliente_parado_e_encerrado_no_prazo() {
    let escuta = TcpListener::bind("127.0.0.1:0").unwrap();
    let porta = escuta.local_addr().unwrap().port();
    let (fila, _trabalhos) = mpsc::sync_channel(1);
    let prazos = Prazos {
        leitura: Duration::from_millis(200),
        escrita: Duration::from_secs(5),
    };
    std::thread::spawn(move || aceitar(&escuta, "modelo-teste", &fila, prazos));

    let t0 = Instant::now();
    let r = http_cru(porta, "GET /health HTTP/1.1\r\n");
    assert!(r.starts_with("HTTP/1.1 408"), "{r}");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());

    let r = http_cru(porta, "GET /health HTTP/1.1\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 200 OK"), "{r}");
}
