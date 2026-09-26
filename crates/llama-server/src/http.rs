//! HTTP/1.1 do tamanho que este servidor precisa: um método, um caminho, um corpo.
//!
//! O cliente é local (o agente na mesma máquina), então não há TLS, proxy nem
//! keep-alive: cada requisição vive numa conexão, e a resposta em streaming termina
//! quando o socket fecha — que é como o SSE já funciona.

use std::io::{BufRead, Write};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HttpError {
    #[error("requisição vazia")]
    Vazia,
    #[error("linha de requisição malformada: {0}")]
    LinhaInvalida(String),
    #[error("corpo em chunks não é suportado — mande Content-Length")]
    ChunkedNaoSuportado,
    #[error("corpo de {0} bytes acima do limite de {1}")]
    CorpoGrande(usize, usize),
    #[error("linha ou cabeçalhos acima do limite (8 KiB por linha, 64 KiB no total)")]
    CabecalhoGrande,
    #[error("Content-Length inválido: {0}")]
    ContentLengthInvalido(String),
    #[error("o cliente não mandou a requisição dentro do prazo")]
    Prazo,
    #[error("erro de io: {0}")]
    Io(String),
}

impl HttpError {
    /// O status da resposta a uma requisição que falhou assim.
    pub fn status(&self) -> u16 {
        match self {
            HttpError::CabecalhoGrande => 431,
            HttpError::Prazo => 408,
            _ => 400,
        }
    }
}

/// Teto do corpo. Um prompt de agente com histórico e tools chega perto de 1 MB;
/// 32 MB é folga de sobra e ainda protege contra um cliente maluco.
pub const LIMITE_CORPO: usize = 32 * 1024 * 1024;

/// Tetos da linha de requisição e dos cabeçalhos. Sem eles, `read_line` lia até um `\n` que
/// um cliente quebrado nunca manda, e uma linha única crescia em memória além do teto do corpo.
pub const LIMITE_LINHA: usize = 8 * 1024;
pub const LIMITE_CABECALHOS: usize = 64 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub struct Requisicao {
    pub metodo: String,
    pub caminho: String,
    pub corpo: Vec<u8>,
}

/// Lê uma requisição inteira: linha, cabeçalhos e o corpo do `Content-Length`.
pub fn ler_requisicao<R: BufRead>(r: &mut R) -> Result<Requisicao, HttpError> {
    // O teto total vale para a linha de requisição e os cabeçalhos juntos.
    let mut restante = LIMITE_CABECALHOS;
    let mut linha = String::new();
    if ler_linha(r, &mut linha, &mut restante)? == 0 {
        return Err(HttpError::Vazia);
    }
    let mut campos = linha.split_whitespace();
    let (Some(metodo), Some(caminho)) = (campos.next(), campos.next()) else {
        return Err(HttpError::LinhaInvalida(linha.trim().to_owned()));
    };
    let (metodo, caminho) = (metodo.to_owned(), caminho.to_owned());

    let mut tamanho: Option<usize> = None;
    loop {
        let mut cab = String::new();
        if ler_linha(r, &mut cab, &mut restante)? == 0 {
            break;
        }
        let cab = cab.trim_end();
        if cab.is_empty() {
            break;
        }
        let Some((nome, valor)) = cab.split_once(':') else {
            continue;
        };
        let nome = nome.trim().to_ascii_lowercase();
        let valor = valor.trim();
        if nome == "content-length" {
            // RFC 9112 §6.3: comprimento inválido é erro, não zero — virar zero deixava o
            // corpo no socket. Repetido com o mesmo valor é o mesmo tamanho; com outro, não
            // há como saber onde o corpo acaba.
            let n: usize = valor
                .parse()
                .map_err(|_| HttpError::ContentLengthInvalido(valor.to_owned()))?;
            if tamanho.is_some_and(|t| t != n) {
                return Err(HttpError::ContentLengthInvalido(format!(
                    "valores conflitantes ({} e {n})",
                    tamanho.unwrap_or_default()
                )));
            }
            tamanho = Some(n);
        } else if nome == "transfer-encoding" && valor.to_ascii_lowercase().contains("chunked") {
            return Err(HttpError::ChunkedNaoSuportado);
        }
    }
    let tamanho = tamanho.unwrap_or(0);
    if tamanho > LIMITE_CORPO {
        return Err(HttpError::CorpoGrande(tamanho, LIMITE_CORPO));
    }
    let mut corpo = vec![0u8; tamanho];
    if tamanho > 0 {
        r.read_exact(&mut corpo).map_err(io)?;
    }
    Ok(Requisicao {
        metodo,
        caminho,
        corpo,
    })
}

/// Uma linha com teto: no máximo [`LIMITE_LINHA`] e o que sobra de `restante`.
fn ler_linha<R: BufRead>(
    r: &mut R,
    linha: &mut String,
    restante: &mut usize,
) -> Result<usize, HttpError> {
    let teto = LIMITE_LINHA.min(*restante);
    // `take` sobre `&mut R`, não sobre `R`: a leitura continua no mesmo leitor.
    let n = std::io::Read::take(&mut *r, teto as u64 + 1)
        .read_line(linha)
        .map_err(io)?;
    if n > teto {
        return Err(HttpError::CabecalhoGrande);
    }
    *restante -= n;
    Ok(n)
}

fn io(e: std::io::Error) -> HttpError {
    // O prazo de leitura do socket (`set_read_timeout`) chega como `WouldBlock` no Linux.
    match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => HttpError::Prazo,
        _ => HttpError::Io(e.to_string()),
    }
}

/// Resposta completa, de uma vez.
pub fn responder<W: Write>(
    w: &mut W,
    status: u16,
    tipo: &str,
    corpo: &[u8],
) -> std::io::Result<()> {
    escrever(w, status, tipo, "", corpo)
}

/// 503 com `Retry-After`: a fila da inferência está cheia.
pub fn ocupado<W: Write>(w: &mut W, segundos: u32, corpo: &[u8]) -> std::io::Result<()> {
    escrever(
        w,
        503,
        "application/json",
        &format!("Retry-After: {segundos}\r\n"),
        corpo,
    )
}

fn escrever<W: Write>(
    w: &mut W,
    status: u16,
    tipo: &str,
    extra: &str,
    corpo: &[u8],
) -> std::io::Result<()> {
    let cabecalho = format!(
        "HTTP/1.1 {status} {}\r\n\
         Content-Type: {tipo}\r\n\
         Content-Length: {}\r\n\
         {extra}\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        motivo(status),
        corpo.len()
    );
    w.write_all(cabecalho.as_bytes())?;
    w.write_all(corpo)?;
    w.flush()
}

/// Abre uma resposta SSE. Os eventos vão depois, um a um, por `evento`.
pub fn abrir_sse<W: Write>(w: &mut W) -> std::io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Access-Control-Allow-Origin: *\r\n\
          Connection: close\r\n\r\n",
    )?;
    w.flush()
}

/// Um evento SSE. `data: <payload>\n\n`, que é tudo o que a API de chat usa.
pub fn evento<W: Write>(w: &mut W, payload: &str) -> std::io::Result<()> {
    write!(w, "data: {payload}\n\n")?;
    w.flush()
}

fn motivo(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn ler(bruto: &str) -> Result<Requisicao, HttpError> {
        ler_requisicao(&mut Cursor::new(bruto.as_bytes().to_vec()))
    }

    #[test]
    fn le_metodo_caminho_e_corpo() {
        let r = ler("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 7\r\n\r\n{\"a\":1}")
            .unwrap();

        assert_eq!(r.metodo, "POST");
        assert_eq!(r.caminho, "/v1/chat/completions");
        assert_eq!(r.corpo, b"{\"a\":1}");
    }

    #[test]
    fn cabecalho_e_case_insensitive() {
        let r = ler("POST /x HTTP/1.1\r\ncontent-length: 2\r\n\r\noi").unwrap();
        assert_eq!(r.corpo, b"oi");
    }

    #[test]
    fn get_sem_corpo() {
        let r = ler("GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();

        assert_eq!(r.metodo, "GET");
        assert!(r.corpo.is_empty());
    }

    #[test]
    fn requisicao_vazia_e_erro() {
        assert_eq!(ler("").unwrap_err(), HttpError::Vazia);
    }

    #[test]
    fn linha_sem_caminho_e_erro() {
        assert_eq!(
            ler("POST\r\n\r\n").unwrap_err(),
            HttpError::LinhaInvalida("POST".to_owned())
        );
    }

    #[test]
    fn chunked_e_recusado_com_mensagem_propria() {
        let erro = ler("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap_err();
        assert_eq!(erro, HttpError::ChunkedNaoSuportado);
    }

    #[test]
    fn corpo_acima_do_limite_e_recusado_antes_de_alocar() {
        let bruto = format!(
            "POST /x HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            LIMITE_CORPO + 1
        );
        assert_eq!(
            ler(&bruto).unwrap_err(),
            HttpError::CorpoGrande(LIMITE_CORPO + 1, LIMITE_CORPO)
        );
    }

    /// `read_line` sem teto deixava uma linha única crescer em memória além do limite do
    /// corpo — e um cliente que nunca manda `\n` segurava o servidor.
    #[test]
    fn linha_acima_do_limite_e_recusada_com_431() {
        let bruto = format!("GET /{} HTTP/1.1\r\n\r\n", "a".repeat(LIMITE_LINHA));
        let erro = ler(&bruto).unwrap_err();
        assert_eq!(erro, HttpError::CabecalhoGrande);
        assert_eq!(erro.status(), 431);
    }

    #[test]
    fn cabecalhos_acima_do_total_sao_recusados() {
        let um = format!("X-A: {}\r\n", "b".repeat(1000));
        let bruto = format!(
            "GET / HTTP/1.1\r\n{}\r\n",
            um.repeat(LIMITE_CABECALHOS / 1000 + 1)
        );
        assert_eq!(ler(&bruto).unwrap_err(), HttpError::CabecalhoGrande);
    }

    /// RFC 9112 §6.3: `Content-Length` inválido é erro, não zero — virar zero deixava o corpo
    /// no socket para ser lido como a "próxima requisição".
    #[test]
    fn content_length_invalido_ou_conflitante_e_400() {
        for cab in [
            "Content-Length: abc\r\n",
            "Content-Length: -1\r\n",
            "Content-Length: 2\r\nContent-Length: 3\r\n",
        ] {
            let erro = ler(&format!("POST /x HTTP/1.1\r\n{cab}\r\noi!")).unwrap_err();
            assert!(
                matches!(erro, HttpError::ContentLengthInvalido(_)),
                "{cab:?}: {erro:?}"
            );
            assert_eq!(erro.status(), 400);
        }
        // Repetido com o mesmo valor é o mesmo tamanho: aceito.
        let r =
            ler("POST /x HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\noi").unwrap();
        assert_eq!(r.corpo, b"oi");
    }

    #[test]
    fn prazo_estourado_vira_408() {
        let e = io(std::io::Error::from(std::io::ErrorKind::WouldBlock));
        assert_eq!(e, HttpError::Prazo);
        assert_eq!(e.status(), 408);
    }

    #[test]
    fn ocupado_responde_503_com_retry_after() {
        let mut buf = Vec::new();
        ocupado(&mut buf, 5, b"{}").unwrap();
        let texto = String::from_utf8(buf).unwrap();
        assert!(
            texto.starts_with("HTTP/1.1 503 Service Unavailable\r\n"),
            "{texto}"
        );
        assert!(texto.contains("Retry-After: 5\r\n"), "{texto}");
    }

    #[test]
    fn resposta_tem_status_tipo_e_tamanho() {
        let mut buf = Vec::new();
        responder(&mut buf, 200, "application/json", b"{}").unwrap();

        let texto = String::from_utf8(buf).unwrap();
        assert!(texto.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(texto.contains("Content-Type: application/json\r\n"));
        assert!(texto.contains("Content-Length: 2\r\n"));
        assert!(texto.ends_with("\r\n\r\n{}"));
    }

    #[test]
    fn sse_abre_com_event_stream_e_manda_eventos() {
        let mut buf = Vec::new();
        abrir_sse(&mut buf).unwrap();
        evento(&mut buf, "{\"x\":1}").unwrap();
        evento(&mut buf, "[DONE]").unwrap();

        let texto = String::from_utf8(buf).unwrap();
        assert!(texto.contains("Content-Type: text/event-stream\r\n"));
        assert!(texto.ends_with("data: {\"x\":1}\n\ndata: [DONE]\n\n"));
    }
}
