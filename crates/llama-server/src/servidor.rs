//! Roteamento e resposta: o que acontece entre o socket e o motor.
//!
//! Duas threads. A do HTTP (`aceitar`) lê cada requisição com prazo e limites, responde
//! sozinha o que não precisa do motor — `/health`, `/v1/models`, pedidos inválidos — e põe as
//! gerações numa fila curta. O worker (`trabalhar`) é a thread que criou o backend: o
//! Vulkan e a `Sessao` nunca saem dela (objetos com ponteiros mapeados), e as gerações
//! seguem uma por vez, porque uma já ocupa as duas GPUs.

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use crate::api::{self, Parada, Pedido};
use crate::http::{self, Requisicao};
use crate::motor::{Motor, MotorError};
use crate::saida::Evento;

/// Gerações que esperam o worker. Uma já ocupa as GPUs por segundos a minutos; mais que
/// isso na fila é cliente que desistiria antes de ser atendido, e ouvir 503 com
/// `Retry-After` é melhor que esperar sem prazo.
pub const FILA: usize = 4;

/// Quanto o cliente que achou a fila cheia deve esperar antes de tentar de novo.
const RETRY_AFTER_S: u32 = 5;

/// Prazos do socket. O de leitura corta o cliente que abre a conexão e não termina a
/// requisição; o de escrita, o que para de ler o stream — no worker isso vira erro de envio
/// e cancela a geração em vez de travar a fila.
#[derive(Debug, Clone, Copy)]
pub struct Prazos {
    pub leitura: Duration,
    pub escrita: Duration,
}

/// Um pedido de geração já validado, esperando o worker. O socket vai junto: a resposta sai
/// dele.
pub struct Trabalho {
    pedido: Pedido,
    fluxo: TcpStream,
}

/// Sobe a thread do HTTP e vira o worker na thread atual — a que é dona do motor.
pub fn laco(
    bind: &str,
    nome: &str,
    mut motor: Motor<'_>,
    prazos: Prazos,
) -> Result<(), Box<dyn std::error::Error>> {
    let escuta = TcpListener::bind(bind)?;
    eprintln!("[http] {bind} — modelo `{nome}`");
    let (fila, trabalhos) = sync_channel(FILA);
    let nome_http = nome.to_owned();
    std::thread::Builder::new()
        .name("llama-http".into())
        .spawn(move || aceitar(&escuta, &nome_http, &fila, prazos))?;
    trabalhar(trabalhos.iter(), nome, &mut motor);
    Ok(())
}

/// A thread do HTTP: atende o que não precisa do motor e enfileira as gerações.
pub fn aceitar(escuta: &TcpListener, nome: &str, fila: &SyncSender<Trabalho>, prazos: Prazos) {
    for conexao in escuta.incoming() {
        match conexao {
            Ok(fluxo) => {
                if let Err(e) = triar(fluxo, nome, fila, prazos) {
                    eprintln!("[http] conexão encerrada: {e}");
                }
            }
            Err(e) => eprintln!("[http] accept falhou: {e}"),
        }
    }
}

/// O worker: atende a fila em ordem, com o motor (e o backend por trás dele) desta thread.
pub fn trabalhar(trabalhos: impl Iterator<Item = Trabalho>, nome: &str, motor: &mut Motor<'_>) {
    for t in trabalhos {
        let mut escritor = BufWriter::new(t.fluxo);
        if let Err(e) = responder_chat(&t.pedido, nome, motor, &mut escritor) {
            eprintln!("[http] conexão encerrada: {e}");
        }
    }
}

fn triar(
    fluxo: TcpStream,
    nome: &str,
    fila: &SyncSender<Trabalho>,
    prazos: Prazos,
) -> Result<(), Box<dyn std::error::Error>> {
    fluxo.set_read_timeout(Some(prazos.leitura))?;
    fluxo.set_write_timeout(Some(prazos.escrita))?;
    let mut leitor = BufReader::new(fluxo.try_clone()?);
    let pedido = {
        let mut escritor = &fluxo;
        let req = match http::ler_requisicao(&mut leitor) {
            Ok(r) => r,
            Err(e) => {
                let corpo = api::erro_json(&e.to_string());
                http::responder(&mut escritor, e.status(), "application/json", &corpo)?;
                return Ok(());
            }
        };
        if rota_rapida(&req, nome, &mut escritor)? {
            return Ok(());
        }
        match validar_chat(&req.corpo, nome) {
            Ok(p) => p,
            Err((status, corpo)) => {
                http::responder(&mut escritor, status, "application/json", &corpo)?;
                return Ok(());
            }
        }
    };
    if let Err(TrySendError::Full(t) | TrySendError::Disconnected(t)) =
        fila.try_send(Trabalho { pedido, fluxo })
    {
        let corpo = api::erro_interno_json("fila de geração cheia; tente de novo em instantes");
        http::ocupado(&mut &t.fluxo, RETRY_AFTER_S, &corpo)?;
    }
    Ok(())
}

/// Tudo numa thread só: o caminho dos testes de rota, com as mesmas regras da fila.
pub fn rotear<W: Write>(
    req: &Requisicao,
    nome: &str,
    motor: &mut Motor<'_>,
    escritor: &mut W,
) -> Result<(), Box<dyn std::error::Error>> {
    if rota_rapida(req, nome, escritor)? {
        return Ok(());
    }
    match validar_chat(&req.corpo, nome) {
        Ok(pedido) => responder_chat(&pedido, nome, motor, escritor)?,
        Err((status, corpo)) => http::responder(escritor, status, "application/json", &corpo)?,
    }
    Ok(())
}

/// Responde o que não precisa do motor. `false` é geração — fica para o worker.
fn rota_rapida<W: Write>(
    req: &Requisicao,
    nome: &str,
    escritor: &mut W,
) -> Result<bool, Box<dyn std::error::Error>> {
    let json = "application/json";
    match (req.metodo.as_str(), caminho_base(&req.caminho)) {
        ("POST", "/v1/chat/completions" | "/chat/completions") => return Ok(false),
        ("OPTIONS", _) => http::responder(escritor, 200, json, b"")?,
        ("GET", "/v1/models" | "/models") => {
            let corpo = serde_json::to_vec(&api::lista_de_modelos(nome, agora()))?;
            http::responder(escritor, 200, json, &corpo)?;
        }
        ("GET", "/health") => http::responder(escritor, 200, json, br#"{"status":"ok"}"#)?,
        _ => http::responder(
            escritor,
            404,
            json,
            &api::erro_json(&format!("sem rota para {} {}", req.metodo, req.caminho)),
        )?,
    }
    Ok(true)
}

/// O corpo de um chat vira pedido, ou status e corpo do erro: 400 para parâmetro inválido,
/// 404 para modelo que não é o servido. O que depende do tokenizer (o contexto) fica para o
/// `Motor::preparar`, no worker.
fn validar_chat(corpo: &[u8], nome: &str) -> Result<Pedido, (u16, Vec<u8>)> {
    let pedido = api::parse_pedido(corpo).map_err(|e| (400, api::erro_json(&e.to_string())))?;
    if !api::modelo_confere(&pedido.modelo, nome) {
        return Err((404, api::erro_modelo_json(&pedido.modelo, nome)));
    }
    Ok(pedido)
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
                let (status, corpo) = erro_do_motor(&e);
                http::responder(escritor, status, "application/json", &corpo)?;
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

    // O que o pedido pode fazer falhar é conferido antes do cabeçalho: depois do 200 do SSE
    // o erro só pode sair como evento, e o cliente já achou que a geração começou.
    let preparado = match motor.preparar(pedido) {
        Ok(p) => p,
        Err(e) => {
            let (status, corpo) = erro_do_motor(&e);
            http::responder(escritor, status, "application/json", &corpo)?;
            return Ok(());
        }
    };
    http::abrir_sse(escritor)?;
    let mut chamadas = 0usize;
    let mut erro_de_envio = None;
    let saida = motor.gerar(pedido, preparado, |evento| {
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
            let payload = String::from_utf8(erro_do_motor(&e).1)?;
            http::evento(escritor, &payload)?;
        }
    }
    http::evento(escritor, "[DONE]")?;
    Ok(())
}

/// Status e corpo de um erro do motor: 400 para o que o pedido causou, 500 para o resto.
fn erro_do_motor(e: &MotorError) -> (u16, Vec<u8>) {
    if e.do_pedido() {
        (400, api::erro_json(&e.to_string()))
    } else {
        (500, api::erro_interno_json(&e.to_string()))
    }
}

/// Log com prefill e decode separados: numa taxa só, um prompt longo faz o decode
/// parecer lento, e é o decode que o usuário sente enquanto lê a resposta.
///
/// O TTFT vem junto porque é o número que o usuário de agente reclama: prompt de 27 mil
/// tokens é minuto de espera antes de a resposta começar, e nenhuma taxa de decode
/// compensa isso.
#[allow(clippy::cast_precision_loss)]
fn registrar(r: &crate::motor::Resultado) {
    let taxa = |n: usize, ms: f64| if ms > 0.0 { n as f64 / (ms / 1e3) } else { 0.0 };
    let reusados = r.tokens_prompt - r.tokens_prefill;
    eprintln!(
        "[gen] prompt {} tok ({} do cache, {} no prefill) {:.2}s ({:.1} tok/s) | \
         decode {} tok {:.2}s ({:.1} tok/s) | amostragem {:.1}ms/tok | ttft {}",
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
        r.ms_ttft
            .map_or_else(|| "—".to_owned(), |ms| format!("{:.2}s", ms / 1e3)),
    );
}

pub fn agora() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
