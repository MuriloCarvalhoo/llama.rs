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
    #[error("erro de io: {0}")]
    Io(String),
}

/// Teto do corpo. Um prompt de agente com histórico e tools chega perto de 1 MB;
/// 32 MB é folga de sobra e ainda protege contra um cliente maluco.
pub const LIMITE_CORPO: usize = 32 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub struct Requisicao {
    pub metodo: String,
    pub caminho: String,
    pub corpo: Vec<u8>,
}

/// Lê uma requisição inteira: linha, cabeçalhos e o corpo do `Content-Length`.
pub fn ler_requisicao<R: BufRead>(r: &mut R) -> Result<Requisicao, HttpError> {
    let mut linha = String::new();
    if r.read_line(&mut linha).map_err(io)? == 0 {
        return Err(HttpError::Vazia);
    }
    let mut campos = linha.split_whitespace();
    let (Some(metodo), Some(caminho)) = (campos.next(), campos.next()) else {
        return Err(HttpError::LinhaInvalida(linha.trim().to_owned()));
    };
    let (metodo, caminho) = (metodo.to_owned(), caminho.to_owned());

    let mut tamanho = 0usize;
    loop {
        let mut cab = String::new();
        if r.read_line(&mut cab).map_err(io)? == 0 {
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
            tamanho = valor.parse().unwrap_or(0);
        } else if nome == "transfer-encoding" && valor.to_ascii_lowercase().contains("chunked") {
            return Err(HttpError::ChunkedNaoSuportado);
        }
    }
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

fn io(e: std::io::Error) -> HttpError {
    HttpError::Io(e.to_string())
}

/// Resposta completa, de uma vez.
pub fn responder<W: Write>(
    w: &mut W,
    status: u16,
    tipo: &str,
    corpo: &[u8],
) -> std::io::Result<()> {
    let cabecalho = format!(
        "HTTP/1.1 {status} {}\r\n\
         Content-Type: {tipo}\r\n\
         Content-Length: {}\r\n\
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
        500 => "Internal Server Error",
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
