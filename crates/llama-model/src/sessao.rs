//! Reuso do KV-cache entre turnos: o que dá para aproveitar do que já foi processado.
//!
//! O ponto delicado é o híbrido: no qwen35, 48 das 65 camadas são delta-net, com
//! **estado recorrente**. O KV-cache de atenção volta atrás sozinho (basta recuar o
//! comprimento e deixar os tokens novos reescreverem os slots), mas o estado recorrente
//! é o produto de todos os tokens processados até aqui, em ordem, e não desanda.
//!
//! Daí o **snapshot de fronteira de turno**: no fim do prefill de cada requisição a
//! sessão manda o backend copiar o estado recorrente e o comprimento do KV. Uma
//! divergência depois dessa posição passa a custar o recuo até ela, não o prompt inteiro.
//! É exatamente a divergência que aparece na prática — o turno seguinte re-renderiza a
//! **resposta** que o modelo acabou de gerar (bloco de raciocínio removido, chamada de
//! ferramenta reformatada), e ela veio toda depois da fronteira. Guardar o fim da resposta
//! em vez do fim do prompt deixaria essa divergência antes do snapshot, sem cobertura.
//!
//! Um snapshot só, o mais recente. Divergência antes dele continua custando tudo.
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
    /// Restaura o snapshot e processa `ids[pos..]`. `pos` é a posição do snapshot.
    RecuarPara { pos: usize },
    /// Zera o cache e processa o prompt do começo.
    Reiniciar,
}

/// Decide o reuso comparando o que está no cache com o prompt novo.
///
/// `marca` é a posição do snapshot, quando existe um.
pub fn planejar_reuso(cache: &[u32], ids: &[u32], marca: Option<usize>) -> Reuso {
    let comum = cache.iter().zip(ids).take_while(|(a, b)| a == b).count();
    if comum == cache.len() {
        return if comum == ids.len() {
            Reuso::Completo
        } else {
            Reuso::Anexar { pos: comum }
        };
    }
    // O cache tem tokens que o prompt novo não confirma — inclusive quando o prompt é um
    // prefixo do cache. O snapshot salva se ele estiver **dentro** do prefixo comum:
    // `marca <= comum` garante que `ids[..marca] == cache[..marca]`, então o estado
    // guardado é o estado certo para continuar dali.
    match marca {
        Some(m) if m <= comum => Reuso::RecuarPara { pos: m },
        _ => Reuso::Reiniciar,
    }
}

#[cfg(test)]
mod tests {
    use super::{Reuso, planejar_reuso};

    #[test]
    fn cache_vazio_processa_tudo_sem_reiniciar() {
        assert_eq!(
            planejar_reuso(&[], &[1, 2, 3], None),
            Reuso::Anexar { pos: 0 }
        );
    }

    /// O caso que paga a conta: o agente manda o histórico inteiro de novo e só o
    /// último turno é novidade.
    #[test]
    fn prompt_que_so_cresce_aproveita_o_cache_inteiro() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3], &[1, 2, 3, 4, 5], None),
            Reuso::Anexar { pos: 3 }
        );
    }

    #[test]
    fn prompt_identico_nao_tem_o_que_processar() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3], &[1, 2, 3], None),
            Reuso::Completo
        );
    }

    /// Sem snapshot, divergência no meio custa tudo: o estado recorrente não volta atrás.
    #[test]
    fn divergencia_no_meio_sem_snapshot_reinicia() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3], &[1, 9, 3], None),
            Reuso::Reiniciar
        );
    }

    /// Com snapshot antes da divergência, recua até ele em vez de reprocessar tudo.
    #[test]
    fn divergencia_depois_do_snapshot_recua() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3, 4], &[1, 2, 9, 9], Some(2)),
            Reuso::RecuarPara { pos: 2 }
        );
    }

    /// Snapshot **depois** da divergência não serve: o estado guardado já incorporou o
    /// token que o prompt novo desmente.
    #[test]
    fn divergencia_antes_do_snapshot_reinicia() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3, 4], &[1, 9, 3, 4], Some(3)),
            Reuso::Reiniciar
        );
    }

    /// O cache mais longo que o prompt: sem snapshot é do zero, com snapshot dentro do
    /// prefixo é só recuar.
    #[test]
    fn prompt_menor_que_o_cache() {
        assert_eq!(
            planejar_reuso(&[1, 2, 3, 4], &[1, 2, 3], None),
            Reuso::Reiniciar
        );
        assert_eq!(
            planejar_reuso(&[1, 2, 3, 4], &[1, 2, 3], Some(2)),
            Reuso::RecuarPara { pos: 2 }
        );
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
        /// Quantos tokens havia no cache quando o backend guardou o snapshot. `None`
        /// enquanto não houver um (backend sem suporte, ou sessão recém-zerada).
        marca: Option<usize>,
    }

    impl Sessao {
        /// Zera o cache do backend e começa uma sessão vazia.
        pub fn nova(gpu: &dyn GpuResidentDecode) -> Sessao {
            gpu.reset();
            Sessao {
                tokens: Vec::new(),
                logits: Vec::new(),
                marca: None,
            }
        }

        /// Tokens atualmente no KV-cache.
        pub fn tokens(&self) -> &[u32] {
            &self.tokens
        }

        /// Posição do snapshot, quando existe — a fronteira a partir da qual uma
        /// divergência de prompt é barata.
        pub fn marca(&self) -> Option<usize> {
            self.marca
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
            match planejar_reuso(&self.tokens, ids, self.marca) {
                Reuso::Completo => {}
                Reuso::Anexar { pos } => self.processar(gpu, ids, pos)?,
                // O snapshot cobre a divergência: restaura e reprocessa só dali.
                // Se o backend recusar, não sobra alternativa senão o caminho de baixo.
                Reuso::RecuarPara { pos } if gpu.restaurar() => {
                    self.tokens.truncate(pos);
                    self.processar(gpu, ids, pos)?;
                }
                Reuso::RecuarPara { .. } | Reuso::Reiniciar => {
                    self.reiniciar(gpu, ids)?;
                }
            }
            // Fronteira de turno: o que vier depois daqui é resposta gerada, e é ela que o
            // turno seguinte re-renderiza (e pode divergir). Ver o módulo.
            self.marca = gpu.marcar().then_some(self.tokens.len());
            Ok(&self.logits)
        }

        fn reiniciar(
            &mut self,
            gpu: &dyn GpuResidentDecode,
            ids: &[u32],
        ) -> Result<(), ModelError> {
            gpu.reset();
            self.tokens.clear();
            self.marca = None;
            self.processar(gpu, ids, 0)
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
        Marcar,
        Restaurar,
    }

    struct BackendFalso {
        nb: usize,
        /// Se o backend guarda snapshot. `false` reproduz o comportamento de antes desta
        /// frente, que é também o dos backends que não implementam `marcar`.
        snapshot: bool,
        chamadas: RefCell<Vec<Chamada>>,
    }

    impl BackendFalso {
        fn novo(nb: usize) -> BackendFalso {
            BackendFalso {
                nb,
                snapshot: false,
                chamadas: RefCell::new(Vec::new()),
            }
        }
        fn com_snapshot(nb: usize) -> BackendFalso {
            BackendFalso {
                snapshot: true,
                ..BackendFalso::novo(nb)
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
        fn marcar(&self) -> bool {
            if self.snapshot {
                self.chamadas.borrow_mut().push(Chamada::Marcar);
            }
            self.snapshot
        }
        fn restaurar(&self) -> bool {
            if self.snapshot {
                self.chamadas.borrow_mut().push(Chamada::Restaurar);
            }
            self.snapshot
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

    /// O caso real do agente: turno 1 prefila o prompt e gera a resposta; o turno 2 manda o
    /// histórico com a resposta **re-renderizada** e diferente do que foi gerado. Com o
    /// snapshot na fronteira do turno 1, só a resposta e o turno novo são reprocessados —
    /// o histórico inteiro fica no cache.
    #[test]
    fn divergencia_depois_da_fronteira_de_turno_nao_reprocessa_o_historico() {
        let gpu = BackendFalso::com_snapshot(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap(); // prompt do turno 1
        assert_eq!(s.marca(), Some(3), "o snapshot fica no fim do prefill");
        s.decode(&gpu, 7).unwrap(); // resposta gerada
        s.decode(&gpu, 8).unwrap();
        gpu.chamadas.borrow_mut().clear();

        // Turno 2: o 8 virou 9 no re-render, e vem um turno novo atrás.
        s.prefill(&gpu, &[1, 2, 3, 7, 9, 4]).unwrap();

        assert_eq!(
            gpu.registradas(),
            vec![
                "Restaurar".to_owned(),
                "Decode(7, 3)".to_owned(),
                "Decode(9, 4)".to_owned(),
                "Decode(4, 5)".to_owned(),
                "Marcar".to_owned(),
            ],
            "sem Reset: o prefixo [1,2,3] veio do cache"
        );
        assert_eq!(s.tokens(), &[1, 2, 3, 7, 9, 4]);
        assert_eq!(s.marca(), Some(6));
    }

    /// Divergência **antes** da fronteira não tem como ser salva: o estado guardado já
    /// incorporou o token que o prompt novo desmente.
    #[test]
    fn divergencia_antes_da_fronteira_ainda_reinicia() {
        let gpu = BackendFalso::com_snapshot(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap();
        gpu.chamadas.borrow_mut().clear();

        s.prefill(&gpu, &[1, 9, 3, 4]).unwrap();

        assert_eq!(
            gpu.registradas(),
            vec![
                "Reset".to_owned(),
                "Decode(1, 0)".to_owned(),
                "Decode(9, 1)".to_owned(),
                "Decode(3, 2)".to_owned(),
                "Decode(4, 3)".to_owned(),
                "Marcar".to_owned(),
            ]
        );
    }

    /// Backend sem snapshot continua exatamente como antes — nenhuma marca, nenhum recuo.
    #[test]
    fn backend_sem_snapshot_nao_ganha_marca() {
        let gpu = BackendFalso::novo(1);
        let mut s = Sessao::nova(&gpu);
        s.prefill(&gpu, &[1, 2, 3]).unwrap();
        assert_eq!(s.marca(), None);
    }
}
