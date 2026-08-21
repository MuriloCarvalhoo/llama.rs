//! Reuso do KV-cache entre turnos: o que dá para aproveitar do que já foi processado.
//!
//! O ponto delicado é o híbrido: no qwen35, 48 das 65 camadas são delta-net, com
//! **estado recorrente**. O KV-cache de atenção poderia ser truncado no meio (basta
//! recuar o comprimento), mas o estado recorrente não volta atrás sem um snapshot —
//! ele é o produto de todos os tokens processados até aqui, em ordem. Por isso a
//! divergência no meio do prompt custa reprocessar tudo, e só o crescimento por trás
//! (o caso do agente que acrescenta um turno) é barato.
//!
//! Reprocessar um token que já entrou também não é opção: a atenção sobrescreveria o
//! mesmo K/V sem dano, mas a recorrência avançaria duas vezes com o mesmo token.

/// O que fazer com o KV-cache antes de processar um prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuso {
    /// O cache já contém exatamente estes tokens — não há o que processar.
    Completo,
    /// Aproveita o cache inteiro e processa `ids[pos..]`.
    Anexar { pos: usize },
    /// Zera o cache e processa o prompt do começo.
    Reiniciar,
}

/// Decide o reuso comparando o que está no cache com o prompt novo.
pub fn planejar_reuso(cache: &[u32], ids: &[u32]) -> Reuso {
    let comum = cache.iter().zip(ids).take_while(|(a, b)| a == b).count();
    if comum < cache.len() {
        // O cache tem tokens que o prompt novo não confirma — inclusive quando o
        // prompt é um prefixo do cache. Sem snapshot do estado recorrente, é do zero.
        return Reuso::Reiniciar;
    }
    if comum == ids.len() {
        Reuso::Completo
    } else {
        Reuso::Anexar { pos: comum }
    }
}

#[cfg(test)]
mod tests {
    use super::{Reuso, planejar_reuso};

    #[test]
    fn cache_vazio_processa_tudo_sem_reiniciar() {
        assert_eq!(planejar_reuso(&[], &[1, 2, 3]), Reuso::Anexar { pos: 0 });
    }

    /// O caso que paga a conta: o agente manda o histórico inteiro de novo e só o
    /// último turno é novidade.
    #[test]
    fn prompt_que_so_cresce_aproveita_o_cache_inteiro() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3], &[1, 2, 3, 4, 5]),
            Reuso::Anexar { pos: 3 }
        );
    }

    #[test]
    fn prompt_identico_nao_tem_o_que_processar() {
        assert_eq!(planejar_reuso(&[1, 2, 3], &[1, 2, 3]), Reuso::Completo);
    }

    /// Divergência no meio: o estado recorrente do delta-net não volta atrás.
    #[test]
    fn divergencia_no_meio_reinicia() {
        assert_eq!(planejar_reuso(&[1, 2, 3], &[1, 9, 3]), Reuso::Reiniciar);
    }

    /// O cache mais longo que o prompt também reinicia: truncar exigiria snapshot.
    #[test]
    fn prompt_menor_que_o_cache_reinicia() {
        assert_eq!(planejar_reuso(&[1, 2, 3, 4], &[1, 2, 3]), Reuso::Reiniciar);
    }
}

#[cfg(feature = "gpu")]
mod com_backend {
    use super::{Reuso, planejar_reuso};
    use crate::ModelError;
    use crate::gpu::GpuResidentDecode;

    /// Sessão de geração sobre um backend residente.
    ///
    /// Guarda os tokens que estão no KV-cache para não reprocessar o prefixo comum
    /// entre um turno e o seguinte — sem isso, cada requisição de um agente reprocessa
    /// o system prompt e o histórico inteiros.
    pub struct Sessao {
        tokens: Vec<u32>,
        /// Logits do último token processado: respondem de graça quando o prompt
        /// repete exatamente o que já está no cache.
        logits: Vec<f32>,
    }

    impl Sessao {
        /// Zera o cache do backend e começa uma sessão vazia.
        pub fn nova(gpu: &dyn GpuResidentDecode) -> Sessao {
            gpu.reset();
            Sessao {
                tokens: Vec::new(),
                logits: Vec::new(),
            }
        }

        /// Tokens atualmente no KV-cache.
        pub fn tokens(&self) -> &[u32] {
            &self.tokens
        }

        /// Processa `ids` aproveitando o que já estiver no cache e devolve os logits
        /// do último token.
        pub fn prefill(
            &mut self,
            gpu: &dyn GpuResidentDecode,
            ids: &[u32],
        ) -> Result<&[f32], ModelError> {
            if ids.is_empty() {
                return Err(ModelError::Gpu("prompt vazio".into()));
            }
            match planejar_reuso(&self.tokens, ids) {
                Reuso::Completo => {}
                Reuso::Anexar { pos } => self.processar(gpu, ids, pos)?,
                Reuso::Reiniciar => {
                    gpu.reset();
                    self.tokens.clear();
                    self.processar(gpu, ids, 0)?;
                }
            }
            Ok(&self.logits)
        }

        /// Decodifica `token` na posição seguinte à última e devolve os logits.
        pub fn decode(
            &mut self,
            gpu: &dyn GpuResidentDecode,
            token: u32,
        ) -> Result<&[f32], ModelError> {
            self.logits = gpu.decode(token, self.tokens.len())?;
            self.tokens.push(token);
            Ok(&self.logits)
        }

        /// Prefill de `ids[pos0..]`: blocos de `batch_size()` e o resto token a token.
        /// Em batch cada peso do modelo sai da VRAM uma vez para N tokens.
        fn processar(
            &mut self,
            gpu: &dyn GpuResidentDecode,
            ids: &[u32],
            pos0: usize,
        ) -> Result<(), ModelError> {
            let nb = gpu.batch_size();
            let mut pos = pos0;
            while nb > 1 && ids.len() - pos >= nb {
                let Some(bloco) = ids.get(pos..pos + nb) else {
                    break;
                };
                self.logits = gpu.decode_batch(bloco, pos)?;
                pos += nb;
            }
            for &t in ids.get(pos..).unwrap_or(&[]) {
                self.logits = gpu.decode(t, pos)?;
                pos += 1;
            }
            self.tokens
                .extend_from_slice(ids.get(pos0..).unwrap_or(&[]));
            Ok(())
        }
    }
}

#[cfg(feature = "gpu")]
pub use com_backend::Sessao;

#[cfg(all(test, feature = "gpu"))]
mod testes_de_sessao {
    use super::Sessao;
    use crate::ModelError;
    use crate::gpu::GpuResidentDecode;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq, Eq)]
    enum Chamada {
        Reset,
        Decode(u32, usize),
        Batch(Vec<u32>, usize),
    }

    struct BackendFalso {
        nb: usize,
        chamadas: RefCell<Vec<Chamada>>,
    }

    impl BackendFalso {
        fn novo(nb: usize) -> BackendFalso {
            BackendFalso {
                nb,
                chamadas: RefCell::new(Vec::new()),
            }
        }
        fn registradas(&self) -> Vec<String> {
            self.chamadas
                .borrow()
                .iter()
                .map(|c| format!("{c:?}"))
                .collect()
        }
    }

    impl GpuResidentDecode for BackendFalso {
        fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError> {
            self.chamadas.borrow_mut().push(Chamada::Decode(token, pos));
            Ok(vec![f32::from(u16::try_from(token).unwrap_or(0))])
        }
        fn reset(&self) {
            self.chamadas.borrow_mut().push(Chamada::Reset);
        }
        fn batch_size(&self) -> usize {
            self.nb
        }
        fn decode_batch(&self, tokens: &[u32], pos0: usize) -> Result<Vec<f32>, ModelError> {
            self.chamadas
                .borrow_mut()
                .push(Chamada::Batch(tokens.to_vec(), pos0));
            Ok(vec![f32::from(
                u16::try_from(tokens.last().copied().unwrap_or(0)).unwrap_or(0),
            )])
        }
    }

    /// O caso que o agente exercita a cada turno: o prompt cresce por trás.
    #[test]
    fn prompt_que_cresce_processa_so_o_sufixo() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap();
        gpu.chamadas.borrow_mut().clear();

        s.prefill(&gpu, &[1, 2, 3, 4, 5]).unwrap();

        assert_eq!(
            gpu.registradas(),
            vec!["Decode(4, 3)".to_owned(), "Decode(5, 4)".to_owned()]
        );
        assert_eq!(s.tokens(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn prompt_identico_nao_toca_no_backend() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        let antes = s.prefill(&gpu, &[1, 2, 3]).unwrap().to_vec();
        gpu.chamadas.borrow_mut().clear();

        let depois = s.prefill(&gpu, &[1, 2, 3]).unwrap().to_vec();

        assert!(gpu.registradas().is_empty(), "não devia decodificar nada");
        assert_eq!(antes, depois, "os logits do último token continuam valendo");
    }

    #[test]
    fn divergencia_no_meio_reseta_e_reprocessa_tudo() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap();
        gpu.chamadas.borrow_mut().clear();

        s.prefill(&gpu, &[1, 9, 3]).unwrap();

        assert_eq!(
            gpu.registradas(),
            vec![
                "Reset".to_owned(),
                "Decode(1, 0)".to_owned(),
                "Decode(9, 1)".to_owned(),
                "Decode(3, 2)".to_owned()
            ]
        );
        assert_eq!(s.tokens(), &[1, 9, 3]);
    }

    /// O prefill vai em blocos do tamanho do batch; o resto cai token a token.
    #[test]
    fn prefill_usa_blocos_do_tamanho_do_batch() {
        let gpu = BackendFalso::novo(4);
        let mut s = Sessao::nova(&gpu);

        s.prefill(&gpu, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            .unwrap();

        assert_eq!(
            gpu.registradas(),
            vec![
                "Reset".to_owned(),
                "Batch([1, 2, 3, 4], 0)".to_owned(),
                "Batch([5, 6, 7, 8], 4)".to_owned(),
                "Decode(9, 8)".to_owned(),
                "Decode(10, 9)".to_owned(),
                "Decode(11, 10)".to_owned()
            ]
        );
    }

    #[test]
    fn decode_continua_da_posicao_seguinte_ao_prefill() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap();
        gpu.chamadas.borrow_mut().clear();

        s.decode(&gpu, 7).unwrap();

        assert_eq!(gpu.registradas(), vec!["Decode(7, 3)".to_owned()]);
        assert_eq!(s.tokens(), &[1, 2, 3, 7]);
    }

    #[test]
    fn prompt_vazio_e_erro() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        assert!(s.prefill(&gpu, &[]).is_err());
    }
}
