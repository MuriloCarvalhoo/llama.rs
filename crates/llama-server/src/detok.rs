//! Texto incremental a partir de bytes de token.
//!
//! No BPE byte-level um caractere pode nascer partido entre dois tokens: o primeiro
//! traz `C3`, o seguinte traz `A7`, e só juntos são `ç`. Emitir cada token isolado como
//! texto produz `<27>` no meio de qualquer palavra acentuada — visível em português a
//! cada duas frases. Aqui os bytes ficam retidos até fecharem um caractere.

#[derive(Debug, Default)]
pub struct Detok {
    pendentes: Vec<u8>,
}

impl Detok {
    pub fn novo() -> Detok {
        Detok::default()
    }

    /// Acrescenta bytes e devolve o texto que já está completo.
    pub fn empurrar(&mut self, bytes: &[u8]) -> String {
        self.pendentes.extend_from_slice(bytes);
        let mut saida = String::new();
        loop {
            match std::str::from_utf8(&self.pendentes) {
                Ok(s) => {
                    saida.push_str(s);
                    self.pendentes.clear();
                    return saida;
                }
                Err(e) => {
                    let ate = e.valid_up_to();
                    if let Some(parte) = self.pendentes.get(..ate) {
                        saida.push_str(&String::from_utf8_lossy(parte));
                    }
                    match e.error_len() {
                        // Byte inválido de verdade: marca e segue, senão o stream trava.
                        Some(n) => {
                            saida.push('\u{fffd}');
                            self.pendentes.drain(..ate + n);
                        }
                        // Caractere ainda incompleto: espera o próximo token.
                        None => {
                            self.pendentes.drain(..ate);
                            return saida;
                        }
                    }
                }
            }
        }
    }

    /// Fecha o stream: o que sobrou não vai completar mais.
    pub fn finalizar(&mut self) -> String {
        let resto = String::from_utf8_lossy(&self.pendentes).into_owned();
        self.pendentes.clear();
        resto
    }
}

#[cfg(test)]
mod tests {
    use super::Detok;

    #[test]
    fn ascii_sai_na_hora() {
        let mut d = Detok::novo();
        assert_eq!(d.empurrar(b"ok"), "ok");
    }

    /// O caso que motiva o módulo: 'ç' chega em dois tokens.
    #[test]
    fn caractere_partido_espera_o_byte_que_falta() {
        let mut d = Detok::novo();

        assert_eq!(d.empurrar(&[0xC3]), "", "meio caractere não pode vazar");
        assert_eq!(d.empurrar(&[0xA7]), "ç");
    }

    #[test]
    fn texto_antes_do_caractere_partido_sai_junto() {
        let mut d = Detok::novo();
        let bytes = "coraç".as_bytes();
        // sem o último byte do 'ç': é assim que o pedaço chega do token
        let cortado = bytes.get(..bytes.len() - 1).unwrap();

        assert_eq!(d.empurrar(cortado), "cora");
        assert_eq!(d.empurrar(&[0xA7]), "ç");
    }

    #[test]
    fn emoji_em_quatro_pedacos() {
        let mut d = Detok::novo();
        let bytes = "🦀".as_bytes().to_vec();
        let mut saida = String::new();
        for b in bytes {
            saida.push_str(&d.empurrar(&[b]));
        }
        assert_eq!(saida, "🦀");
    }

    #[test]
    fn byte_invalido_vira_marcador_e_o_stream_continua() {
        let mut d = Detok::novo();
        assert_eq!(d.empurrar(&[0xFF, b'a']), "\u{fffd}a");
    }

    #[test]
    fn finalizar_devolve_o_que_ficou_pendente() {
        let mut d = Detok::novo();
        d.empurrar(&[0xC3]);

        assert_eq!(d.finalizar(), "\u{fffd}");
        assert_eq!(d.finalizar(), "", "não repete o resto");
    }
}
