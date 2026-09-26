//! Protocolo: o pedaço da API de chat da OpenAI que um agente de código usa.

use llama_chat::{Esforco, Mensagem};
use serde_json::{Value, json};

/// Defaults de amostragem do próprio modelo (`general.sampling.*` no GGUF do
/// Qwen3.8-27B). Valem quando o cliente não manda os seus.
pub const TEMP_PADRAO: f32 = 1.0;
pub const TOP_P_PADRAO: f32 = 0.95;
pub const TOP_K_PADRAO: usize = 20;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ApiError {
    #[error("corpo não é JSON válido")]
    JsonInvalido,
    #[error("campo `messages` ausente ou vazio")]
    SemMensagens,
    #[error("{0}")]
    Chat(String),
    /// Parâmetro que muda a resposta e chegou fora do que o servidor atende.
    #[error("{0}")]
    Parametro(String),
}

#[derive(Debug, Clone)]
pub struct Pedido {
    pub modelo: String,
    pub mensagens: Vec<Mensagem>,
    pub ferramentas: Vec<Value>,
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperatura: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: Option<u64>,
    /// Sequências que interrompem a geração, além do `<|im_end|>`.
    pub stop: Vec<String>,
    pub esforco: Esforco,
    pub pensar: bool,
}

/// Por que a geração parou — o `finish_reason` da resposta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parada {
    Fim,
    Limite,
    Ferramenta,
}

impl Parada {
    fn texto(self) -> &'static str {
        match self {
            Parada::Fim => "stop",
            Parada::Limite => "length",
            Parada::Ferramenta => "tool_calls",
        }
    }
}

/// Uma chamada de ferramenta pronta para a resposta.
#[derive(Debug, Clone, PartialEq)]
pub struct ChamadaPronta {
    pub id: String,
    pub nome: String,
    /// Argumentos serializados — é assim que a API os transporta.
    pub argumentos: String,
}

/// Lê o corpo de `POST /v1/chat/completions`.
pub fn parse_pedido(corpo: &[u8]) -> Result<Pedido, ApiError> {
    let v: Value = serde_json::from_slice(corpo).map_err(|_| ApiError::JsonInvalido)?;
    let msgs = v
        .get("messages")
        .and_then(Value::as_array)
        .filter(|m| !m.is_empty())
        .ok_or(ApiError::SemMensagens)?;
    let mensagens = msgs
        .iter()
        .map(Mensagem::de_json)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Chat(e.to_string()))?;

    // Parâmetro que muda a resposta e chega inválido é erro, não default silencioso: antes
    // `temperature: 5` passava direto. `null` vale como ausente — é como vários clientes
    // mandam os campos opcionais — e `max_tokens: -1` também: é como o llama.cpp diz "sem
    // limite", e clientes feitos para ele mandam assim.
    let sem_limite = |nome: &str| v.get(nome).and_then(Value::as_i64) == Some(-1);
    let campo = |nome: &str| v.get(nome).filter(|x| !x.is_null());
    let numero =
        |nome: &str, faixa: &str, ok: fn(f64) -> bool| match campo(nome) {
            None => Ok(None),
            Some(x) => x.as_f64().filter(|&n| ok(n)).map(Some).ok_or_else(|| {
                ApiError::Parametro(format!("`{nome}` tem de ser um número {faixa}"))
            }),
        };
    let inteiro = |nome: &str, minimo: usize| match campo(nome) {
        None => Ok(None),
        Some(x) => x
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n >= minimo)
            .map(Some)
            .ok_or_else(|| {
                ApiError::Parametro(format!("`{nome}` tem de ser um inteiro >= {minimo}"))
            }),
    };
    let max_tokens = if sem_limite("max_tokens") {
        None
    } else {
        inteiro("max_tokens", 1)?
    };
    let max_completion_tokens = if sem_limite("max_completion_tokens") {
        None
    } else {
        inteiro("max_completion_tokens", 1)?
    };
    let temperatura = numero("temperature", "entre 0 e 2", |t| (0.0..=2.0).contains(&t))?;
    let top_p = numero("top_p", "em (0, 1]", |p| p > 0.0 && p <= 1.0)?;
    let top_k = inteiro("top_k", 0)?;
    // Uma escolha por pedido e texto livre: aceitar `n: 2` ou `json_object` e devolver uma
    // escolha em texto livre parece atendido e não é.
    if campo("n").is_some_and(|n| n.as_u64() != Some(1)) {
        return Err(ApiError::Parametro(
            "`n` só aceita 1: o servidor gera uma escolha por pedido".to_owned(),
        ));
    }
    if let Some(formato) = campo("response_format") {
        let tipo = formato.get("type").and_then(Value::as_str).unwrap_or("?");
        if tipo != "text" {
            return Err(ApiError::Parametro(format!(
                "`response_format` `{tipo}` não é suportado; só `text`"
            )));
        }
    }
    Ok(Pedido {
        modelo: v
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        mensagens,
        ferramentas: v
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        stream: v.get("stream").and_then(Value::as_bool).unwrap_or(false),
        max_tokens: max_tokens.or(max_completion_tokens),
        #[allow(clippy::cast_possible_truncation)]
        temperatura: temperatura.map_or(TEMP_PADRAO, |t| t as f32),
        #[allow(clippy::cast_possible_truncation)]
        top_p: top_p.map_or(TOP_P_PADRAO, |t| t as f32),
        top_k: top_k.unwrap_or(TOP_K_PADRAO),
        seed: v.get("seed").and_then(Value::as_u64),
        stop: paradas(v.get("stop")),
        esforco: match v.get("reasoning_effort").and_then(Value::as_str) {
            Some("low") => Esforco::Low,
            Some("medium") => Esforco::Medium,
            _ => Esforco::Xhigh,
        },
        // `enable_thinking` não é da API da OpenAI, mas é como o llama.cpp e o vLLM
        // expõem o switch do template; o padrão do modelo é pensar.
        pensar: v
            .get("enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

/// `stop` aceita string ou lista de strings.
fn paradas(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(itens)) => itens
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

fn uso(tokens: (usize, usize)) -> Value {
    let (prompt, saida) = tokens;
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": saida,
        "total_tokens": prompt + saida,
    })
}

fn chamadas_json(chamadas: &[ChamadaPronta]) -> Vec<Value> {
    chamadas
        .iter()
        .map(|c| {
            json!({
                "id": c.id,
                "type": "function",
                "function": {"name": c.nome, "arguments": c.argumentos},
            })
        })
        .collect()
}

/// Resposta de uma vez (`stream: false`).
#[allow(clippy::too_many_arguments)]
pub fn resposta_completa(
    id: &str,
    criado: u64,
    modelo: &str,
    conteudo: &str,
    reasoning: &str,
    chamadas: &[ChamadaPronta],
    parada: Parada,
    tokens: (usize, usize),
) -> Value {
    let mut mensagem = json!({"role": "assistant", "content": conteudo});
    if let Some(campos) = mensagem.as_object_mut() {
        if !reasoning.is_empty() {
            campos.insert("reasoning_content".to_owned(), json!(reasoning));
        }
        if !chamadas.is_empty() {
            campos.insert("tool_calls".to_owned(), json!(chamadas_json(chamadas)));
        }
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": criado,
        "model": modelo,
        "choices": [{"index": 0, "message": mensagem, "finish_reason": parada.texto()}],
        "usage": uso(tokens),
    })
}

/// Envelope de um evento de streaming.
pub fn chunk(id: &str, criado: u64, modelo: &str, delta: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": criado,
        "model": modelo,
        "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}],
    })
}

pub fn delta_texto(texto: &str) -> Value {
    json!({"role": "assistant", "content": texto})
}

/// `reasoning_content` é o campo que Qwen e DeepSeek usam para o raciocínio, e o que
/// os clientes compatíveis com a OpenAI leem.
pub fn delta_reasoning(texto: &str) -> Value {
    json!({"role": "assistant", "reasoning_content": texto})
}

/// A chamada só é conhecida inteira, então nome e argumentos vão num delta só —
/// o cliente acumula por `index`.
pub fn delta_chamada(indice: usize, chamada: &ChamadaPronta) -> Value {
    json!({
        "role": "assistant",
        "tool_calls": [{
            "index": indice,
            "id": chamada.id,
            "type": "function",
            "function": {"name": chamada.nome, "arguments": chamada.argumentos},
        }],
    })
}

pub fn chunk_final(
    id: &str,
    criado: u64,
    modelo: &str,
    parada: Parada,
    tokens: (usize, usize),
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": criado,
        "model": modelo,
        "choices": [{"index": 0, "delta": {}, "finish_reason": parada.texto()}],
        "usage": uso(tokens),
    })
}

pub fn lista_de_modelos(nome: &str, criado: u64) -> Value {
    json!({
        "object": "list",
        "data": [{"id": nome, "object": "model", "created": criado, "owned_by": "llama-rs"}],
    })
}

/// O `model` do pedido é o servido? Ausente vale (clientes locais costumam omitir), e a caixa
/// não importa. Outro nome não é recusado — o servidor tem um modelo só —, mas vai para o log.
pub fn modelo_confere(pedido: &str, servido: &str) -> bool {
    pedido.is_empty() || pedido.eq_ignore_ascii_case(servido)
}

pub fn erro_json(mensagem: &str) -> Vec<u8> {
    erro_json_de_tipo(mensagem, "invalid_request_error")
}

/// Como [`erro_json`], para falha do servidor: o `type` é o que a OpenAI usa nos 5xx.
pub fn erro_interno_json(mensagem: &str) -> Vec<u8> {
    erro_json_de_tipo(mensagem, "server_error")
}

fn erro_json_de_tipo(mensagem: &str, tipo: &str) -> Vec<u8> {
    let v = json!({"error": {"message": mensagem, "type": tipo}});
    serde_json::to_vec(&v).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing)]
    use super::*;
    use llama_chat::Papel;

    fn pedido_minimo() -> Vec<u8> {
        br#"{"model":"qwen","messages":[{"role":"user","content":"Oi"}]}"#.to_vec()
    }

    #[test]
    fn parse_le_mensagens_e_aplica_os_defaults_do_modelo() {
        let p = parse_pedido(&pedido_minimo()).unwrap();

        assert_eq!(p.modelo, "qwen");
        assert_eq!(p.mensagens.len(), 1);
        assert_eq!(p.mensagens.first().map(|m| m.papel), Some(Papel::User));
        assert!(!p.stream);
        assert_eq!(p.temperatura, TEMP_PADRAO);
        assert_eq!(p.top_p, TOP_P_PADRAO);
        assert_eq!(p.top_k, TOP_K_PADRAO);
        assert!(p.pensar, "o template pensa por padrão");
    }

    #[test]
    fn parse_le_os_parametros_de_amostragem_do_cliente() {
        let corpo = br#"{"messages":[{"role":"user","content":"Oi"}],
            "stream":true,"temperature":0.2,"top_p":0.5,"top_k":7,"seed":99,
            "max_tokens":128,"stop":["FIM"],"reasoning_effort":"low"}"#;

        let p = parse_pedido(corpo).unwrap();

        assert!(p.stream);
        assert_eq!(p.temperatura, 0.2);
        assert_eq!(p.top_p, 0.5);
        assert_eq!(p.top_k, 7);
        assert_eq!(p.seed, Some(99));
        assert_eq!(p.max_tokens, Some(128));
        assert_eq!(p.stop, vec!["FIM".to_owned()]);
        assert_eq!(p.esforco, Esforco::Low);
    }

    /// `max_completion_tokens` é o nome novo do mesmo campo.
    #[test]
    fn parse_aceita_max_completion_tokens() {
        let corpo = br#"{"messages":[{"role":"user","content":"Oi"}],"max_completion_tokens":42}"#;
        assert_eq!(parse_pedido(corpo).unwrap().max_tokens, Some(42));
    }

    #[test]
    fn parse_le_tools_e_stop_como_string_unica() {
        let corpo = br#"{"messages":[{"role":"user","content":"Oi"}],
            "tools":[{"type":"function","function":{"name":"read"}}],"stop":"PARE"}"#;

        let p = parse_pedido(corpo).unwrap();

        assert_eq!(p.ferramentas.len(), 1);
        assert_eq!(p.stop, vec!["PARE".to_owned()]);
    }

    #[test]
    fn json_invalido_e_erro() {
        assert_eq!(
            parse_pedido(b"{nao json").unwrap_err(),
            ApiError::JsonInvalido
        );
    }

    #[test]
    fn sem_mensagens_e_erro() {
        assert_eq!(
            parse_pedido(br#"{"model":"x","messages":[]}"#).unwrap_err(),
            ApiError::SemMensagens
        );
    }

    fn com(extra: &str) -> Result<Pedido, ApiError> {
        let corpo = format!(r#"{{"messages":[{{"role":"user","content":"Oi"}}],{extra}}}"#);
        parse_pedido(corpo.as_bytes())
    }

    /// Parâmetro que muda a resposta e chega inválido é erro do pedido, não default
    /// silencioso. `-1` é a exceção: é o "sem limite" do llama.cpp.
    #[test]
    fn max_tokens_tem_de_ser_inteiro_positivo() {
        assert_eq!(com(r#""max_tokens":-1"#).unwrap().max_tokens, None);
        for ruim in ["0", "-2", "1.5", "\"10\""] {
            assert!(
                matches!(
                    com(&format!(r#""max_tokens":{ruim}"#)),
                    Err(ApiError::Parametro(_))
                ),
                "max_tokens={ruim}"
            );
        }
        assert!(matches!(
            com(r#""max_completion_tokens":0"#),
            Err(ApiError::Parametro(_))
        ));
        assert_eq!(com(r#""max_tokens":null"#).unwrap().max_tokens, None);
    }

    #[test]
    fn temperature_e_top_p_fora_da_faixa_sao_erro() {
        for ruim in [
            r#""temperature":-0.1"#,
            r#""temperature":2.5"#,
            r#""temperature":"alta""#,
            r#""top_p":0"#,
            r#""top_p":1.1"#,
            r#""top_k":-1"#,
        ] {
            assert!(matches!(com(ruim), Err(ApiError::Parametro(_))), "{ruim}");
        }
        let p = com(r#""temperature":2,"top_p":1,"top_k":0"#).unwrap();
        assert_eq!((p.temperatura, p.top_p, p.top_k), (2.0, 1.0, 0));
        assert_eq!(com(r#""temperature":0"#).unwrap().temperatura, 0.0);
    }

    /// O servidor gera uma escolha só e não força formato: pedir `n > 1` ou `json_object`
    /// e receber texto livre parece atendido e não é.
    #[test]
    fn n_maior_que_um_e_response_format_nao_texto_sao_recusados() {
        assert!(matches!(com(r#""n":2"#), Err(ApiError::Parametro(_))));
        assert!(matches!(
            com(r#""response_format":{"type":"json_object"}"#),
            Err(ApiError::Parametro(_))
        ));
        assert!(com(r#""n":1,"response_format":{"type":"text"}"#).is_ok());
    }

    #[test]
    fn modelo_confere_aceita_ausente_e_ignora_caixa() {
        assert!(modelo_confere("", "qwen3.8-27b"));
        assert!(modelo_confere("Qwen3.8-27B", "qwen3.8-27b"));
        assert!(!modelo_confere("gpt-4o", "qwen3.8-27b"));
    }

    #[test]
    fn mensagem_com_papel_invalido_vira_erro_de_chat() {
        let corpo = br#"{"messages":[{"role":"narrador","content":"x"}]}"#;
        assert!(matches!(parse_pedido(corpo), Err(ApiError::Chat(_))));
    }

    #[test]
    fn resposta_completa_tem_o_formato_da_api() {
        let v = resposta_completa(
            "chatcmpl-1",
            1000,
            "qwen",
            "olá",
            "pensei",
            &[],
            Parada::Fim,
            (10, 3),
        );

        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["id"], "chatcmpl-1");
        assert_eq!(v["created"], 1000);
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "olá");
        assert_eq!(v["choices"][0]["message"]["reasoning_content"], "pensei");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 10);
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert_eq!(v["usage"]["total_tokens"], 13);
    }

    #[test]
    fn resposta_com_chamadas_marca_finish_reason_tool_calls() {
        let chamada = ChamadaPronta {
            id: "call_0".to_owned(),
            nome: "read".to_owned(),
            argumentos: r#"{"path":"a.rs"}"#.to_owned(),
        };

        let v = resposta_completa(
            "id",
            1,
            "qwen",
            "",
            "",
            std::slice::from_ref(&chamada),
            Parada::Ferramenta,
            (1, 1),
        );

        let tc = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_0");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "read");
        assert_eq!(tc["function"]["arguments"], r#"{"path":"a.rs"}"#);
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Sem raciocínio o campo não vai: cliente que não conhece `reasoning_content`
    /// não deve receber string vazia no lugar de nada.
    #[test]
    fn resposta_sem_reasoning_omite_o_campo() {
        let v = resposta_completa("id", 1, "m", "oi", "", &[], Parada::Fim, (1, 1));
        assert!(
            v["choices"][0]["message"]
                .get("reasoning_content")
                .is_none()
        );
    }

    #[test]
    fn chunk_de_texto_tem_o_formato_de_delta() {
        let v = chunk("id", 7, "qwen", delta_texto("ol"));

        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["content"], "ol");
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn chunk_de_reasoning_usa_reasoning_content() {
        let v = chunk("id", 7, "qwen", delta_reasoning("hmm"));
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "hmm");
    }

    #[test]
    fn chunk_de_chamada_carrega_indice_id_nome_e_argumentos() {
        let chamada = ChamadaPronta {
            id: "call_1".to_owned(),
            nome: "write".to_owned(),
            argumentos: "{}".to_owned(),
        };

        let v = chunk("id", 7, "qwen", delta_chamada(2, &chamada));

        let tc = &v["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 2);
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["name"], "write");
        assert_eq!(tc["function"]["arguments"], "{}");
    }

    #[test]
    fn chunk_final_fecha_com_motivo_e_uso() {
        let v = chunk_final("id", 7, "qwen", Parada::Limite, (5, 6));

        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["choices"][0]["delta"], json!({}));
        assert_eq!(v["usage"]["total_tokens"], 11);
    }

    #[test]
    fn lista_de_modelos_traz_o_modelo_carregado() {
        let v = lista_de_modelos("qwen3.8-27b", 5);

        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "qwen3.8-27b");
        assert_eq!(v["data"][0]["object"], "model");
    }

    #[test]
    fn erro_json_tem_o_envelope_que_o_cliente_espera() {
        let bytes = erro_json("deu ruim");
        let v: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(v["error"]["message"], "deu ruim");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }
}
