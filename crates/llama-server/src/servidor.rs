//! Roteamento e resposta: o que acontece entre o socket e o motor.

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};

use crate::api::{self, Parada, Pedido};
use crate::http::{self, Requisicao};
use crate::motor::Motor;
use crate::saida::Evento;

/// Aceita conexões em série: o modelo é um só, e uma requisição já ocupa as GPUs.
pub fn laco(
    bind: &str,
    nome: &str,
    mut motor: Motor<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let escuta = TcpListener::bind(bind)?;
    eprintln!("[http] {bind} — modelo `{nome}`");
    for conexao in escuta.incoming() {
        match conexao {
            Ok(fluxo) => {
                if let Err(e) = atender(fluxo, nome, &mut motor) {
                    eprintln!("[http] conexão encerrada: {e}");
                }
            }
            Err(e) => eprintln!("[http] accept falhou: {e}"),
        }
    }
    Ok(())
}

pub fn atender(
    fluxo: TcpStream,
    nome: &str,
    motor: &mut Motor<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut leitor = BufReader::new(fluxo.try_clone()?);
    let mut escritor = BufWriter::new(fluxo);
    let req = match http::ler_requisicao(&mut leitor) {
        Ok(r) => r,
        Err(e) => {
            http::responder(
                &mut escritor,
                400,
                "application/json",
                &api::erro_json(&e.to_string()),
            )?;
            return Ok(());
        }
    };
    rotear(&req, nome, motor, &mut escritor)
}

pub fn rotear<W: Write>(
    req: &Requisicao,
    nome: &str,
    motor: &mut Motor<'_>,
    escritor: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = "application/json";
    match (req.metodo.as_str(), caminho_base(&req.caminho)) {
        ("OPTIONS", _) => http::responder(escritor, 200, json, b"")?,
        ("GET", "/v1/models" | "/models") => {
            let corpo = serde_json::to_vec(&api::lista_de_modelos(nome, agora()))?;
            http::responder(escritor, 200, json, &corpo)?;
        }
        ("GET", "/health") => http::responder(escritor, 200, json, br#"{"status":"ok"}"#)?,
        ("POST", "/v1/chat/completions" | "/chat/completions") => {
            match api::parse_pedido(&req.corpo) {
                Ok(pedido) => responder_chat(&pedido, nome, motor, escritor)?,
                Err(e) => {
                    http::responder(escritor, 400, json, &api::erro_json(&e.to_string()))?;
                }
            }
        }
        _ => http::responder(
            escritor,
            404,
            json,
            &api::erro_json(&format!("sem rota para {} {}", req.metodo, req.caminho)),
        )?,
    }
    Ok(())
}

/// Ignora a query string: `/v1/models?x=1` é a mesma rota.
fn caminho_base(caminho: &str) -> &str {
    caminho.split('?').next().unwrap_or(caminho)
}

pub fn responder_chat<W: Write>(
    pedido: &Pedido,
    nome: &str,
    motor: &mut Motor<'_>,
    escritor: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    let criado = agora();
    let id = format!("chatcmpl-{criado}");

    if !pedido.stream {
        let r = match motor.responder(pedido, |_| true) {
            Ok(r) => r,
            Err(e) => {
                let corpo = api::erro_json(&e.to_string());
                http::responder(escritor, 500, "application/json", &corpo)?;
                return Ok(());
            }
        };
        registrar(&r);
        let corpo = serde_json::to_vec(&api::resposta_completa(
            &id,
            criado,
            nome,
            &r.conteudo,
            &r.reasoning,
            &r.chamadas,
            r.parada.unwrap_or(Parada::Fim),
            (r.tokens_prompt, r.tokens_saida),
        ))?;
        http::responder(escritor, 200, "application/json", &corpo)?;
        return Ok(());
    }

    http::abrir_sse(escritor)?;
    let mut chamadas = 0usize;
    let mut erro_de_envio = None;
    let saida = motor.responder(pedido, |evento| {
        if erro_de_envio.is_some() {
            return false;
        }
        let delta = match evento {
            Evento::Reasoning(t) => api::delta_reasoning(t),
            Evento::Conteudo(t) => api::delta_texto(t),
            Evento::Chamada { nome, argumentos } => {
                let pronta = api::ChamadaPronta {
                    id: format!("call_{chamadas}"),
                    nome: nome.clone(),
                    argumentos: serde_json::to_string(argumentos).unwrap_or_default(),
                };
                chamadas += 1;
                api::delta_chamada(chamadas - 1, &pronta)
            }
        };
        let payload =
            serde_json::to_string(&api::chunk(&id, criado, nome, delta)).unwrap_or_default();
        if let Err(e) = http::evento(escritor, &payload) {
            erro_de_envio = Some(e);
            return false;
        }
        true
    });

    match saida {
        Ok(r) => {
            registrar(&r);
            let fim = api::chunk_final(
                &id,
                criado,
                nome,
                r.parada.unwrap_or(Parada::Fim),
                (r.tokens_prompt, r.tokens_saida),
            );
            http::evento(escritor, &serde_json::to_string(&fim)?)?;
        }
        Err(e) => {
            let payload = String::from_utf8(api::erro_json(&e.to_string()))?;
            http::evento(escritor, &payload)?;
        }
    }
    http::evento(escritor, "[DONE]")?;
    Ok(())
}

/// Log com prefill e decode separados: numa taxa só, um prompt longo faz o decode
/// parecer lento, e é o decode que o usuário sente enquanto lê a resposta.
#[allow(clippy::cast_precision_loss)]
fn registrar(r: &crate::motor::Resultado) {
    let taxa = |n: usize, ms: f64| if ms > 0.0 { n as f64 / (ms / 1e3) } else { 0.0 };
    let reusados = r.tokens_prompt - r.tokens_prefill;
    eprintln!(
        "[gen] prompt {} tok ({} do cache, {} no prefill) {:.2}s ({:.1} tok/s) | \
         decode {} tok {:.2}s ({:.1} tok/s) | amostragem {:.1}ms/tok",
        r.tokens_prompt,
        reusados,
        r.tokens_prefill,
        r.ms_prefill / 1e3,
        taxa(r.tokens_prefill, r.ms_prefill),
        r.tokens_saida,
        r.ms_decode / 1e3,
        taxa(r.tokens_saida, r.ms_decode),
        if r.tokens_saida > 0 {
            r.ms_amostragem / r.tokens_saida as f64
        } else {
            0.0
        },
    );
}

pub fn agora() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
