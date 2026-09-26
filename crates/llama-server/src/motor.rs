//! O laço de geração: prompt renderizado, prefill com reuso, amostragem, parada.

use llama_chat::{Opcoes, render};
use llama_model::{GpuResidentDecode, ModelError, Sessao, VERIFY_TOK};
use llama_sampling::Sampler;
use llama_tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::api::{ChamadaPronta, Parada, Pedido};
use crate::detok::Detok;
use crate::saida::{Evento, Saida, cauda_ambigua};

#[derive(Debug, thiserror::Error)]
pub enum MotorError {
    #[error("prompt inválido: {0}")]
    Chat(String),
    #[error("prompt de {0} tokens não cabe no contexto de {1}")]
    ContextoEstourado(usize, usize),
    #[error("backend: {0}")]
    Backend(String),
}

impl MotorError {
    /// O pedido causou o erro (responde 4xx) ou foi falha interna (5xx)? Um cliente decide
    /// se tenta de novo pelo status: repetir um prompt que não cabe no contexto não adianta.
    pub fn do_pedido(&self) -> bool {
        matches!(
            self,
            MotorError::Chat(_) | MotorError::ContextoEstourado(..)
        )
    }
}

/// O prompt do pedido renderizado, tokenizado e já conferido contra o contexto. Separado da
/// geração para o servidor recusar um pedido que não cabe **antes** de abrir o stream.
pub struct Preparado {
    ids: Vec<u32>,
    t_req: std::time::Instant,
}

/// O que a geração produziu, já separado nos campos da resposta.
#[derive(Debug, Default, Clone)]
pub struct Resultado {
    pub conteudo: String,
    pub reasoning: String,
    pub chamadas: Vec<ChamadaPronta>,
    pub parada: Option<Parada>,
    pub tokens_prompt: usize,
    pub tokens_saida: usize,
    /// Tokens do prompt que **não** vieram do cache — o que o prefill de fato processou.
    pub tokens_prefill: usize,
    /// Tempos, para separar o que é prefill, decode e amostragem. Misturados numa taxa
    /// só, um prompt longo faz o decode parecer lento.
    pub ms_prefill: f64,
    pub ms_decode: f64,
    pub ms_amostragem: f64,
    /// Tempo até o primeiro byte de stream, contado da chegada do pedido — é o que o
    /// usuário sente antes de a resposta começar a aparecer, e o que o prefill decide.
    /// Inclui render do template, tokenização, prefill e os tokens que o detokenizador
    /// segurou até fechar o primeiro caractere. `None` quando nada foi emitido.
    pub ms_ttft: Option<f64>,
}

pub struct Motor<'a> {
    tokenizer: &'a Tokenizer,
    gpu: &'a dyn GpuResidentDecode,
    sessao: Sessao,
    ctx: usize,
    /// Tokens que encerram o turno: `<|im_end|>` e `<|endoftext|>`.
    fim: Vec<u32>,
}

impl<'a> Motor<'a> {
    pub fn novo(
        tokenizer: &'a Tokenizer,
        gpu: &'a dyn GpuResidentDecode,
        ctx: usize,
        eos: u32,
    ) -> Motor<'a> {
        let mut fim = vec![eos];
        for marcador in ["<|im_end|>", "<|endoftext|>"] {
            if let [id] = tokenizer.encode_special(marcador).as_slice()
                && !fim.contains(id)
            {
                fim.push(*id);
            }
        }
        Motor {
            tokenizer,
            gpu,
            sessao: Sessao::nova(gpu),
            ctx,
            fim,
        }
    }

    /// Gera a resposta do pedido, chamando `emitir` a cada evento (para o streaming).
    ///
    /// `emitir` devolve `false` para interromper a geração — é como o servidor desiste
    /// quando o cliente fecha a conexão, em vez de ocupar a GPU com tokens que ninguém
    /// vai ler.
    pub fn responder(
        &mut self,
        pedido: &Pedido,
        emitir: impl FnMut(&Evento) -> bool,
    ) -> Result<Resultado, MotorError> {
        let preparado = self.preparar(pedido)?;
        self.gerar(pedido, preparado, emitir)
    }

    /// A parte de [`Self::responder`] que só depende do pedido: render do template,
    /// tokenização e o limite do contexto. Não toca na GPU.
    pub fn preparar(&self, pedido: &Pedido) -> Result<Preparado, MotorError> {
        // O relógio do TTFT começa aqui: o cliente espera o render e a tokenização junto
        // com o prefill.
        let t_req = std::time::Instant::now();
        let prompt = render(
            &pedido.mensagens,
            &Opcoes {
                ferramentas: pedido.ferramentas.clone(),
                add_generation_prompt: true,
                enable_thinking: pedido.pensar,
                esforco: pedido.esforco,
            },
        )
        .map_err(|e| MotorError::Chat(e.to_string()))?;

        let ids = self.tokenizer.encode_special(&prompt);
        if ids.len() >= self.ctx {
            return Err(MotorError::ContextoEstourado(ids.len(), self.ctx));
        }
        Ok(Preparado { ids, t_req })
    }

    /// Gera a resposta de um pedido já [preparado](Self::preparar) — `preparado` tem de
    /// ter saído do mesmo `pedido`.
    pub fn gerar(
        &mut self,
        pedido: &Pedido,
        preparado: Preparado,
        mut emitir: impl FnMut(&Evento) -> bool,
    ) -> Result<Resultado, MotorError> {
        let Preparado { ids, t_req } = preparado;
        let ja_no_cache = prefixo_comum(self.sessao.tokens(), &ids, self.sessao.marca());
        let t0 = std::time::Instant::now();
        // Snapshot logo depois do último token especial (o `<think>` do prompt de geração):
        // é a última posição que o turno seguinte re-renderiza igual — ver
        // `Sessao::prefill_com_fronteira`.
        let fronteira = ids
            .iter()
            .rposition(|&t| self.tokenizer.e_especial(t))
            .map_or(ids.len(), |i| i + 1);
        let mut logits = self
            .sessao
            .prefill_com_fronteira(self.gpu, &ids, fronteira)
            .map_err(erro_backend)?
            .to_vec();

        let mut res = Resultado {
            tokens_prompt: ids.len(),
            tokens_prefill: ids.len() - ja_no_cache,
            ms_prefill: t0.elapsed().as_secs_f64() * 1e3,
            ..Resultado::default()
        };
        let sampler = sampler_de(pedido);
        // Sem seed no pedido, o relógio serve: o que importa é não repetir a amostra
        // entre requisições, não ter aleatoriedade criptográfica.
        let semente = pedido.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| {
                    d.as_secs().wrapping_mul(1_000_000_000) + u64::from(d.subsec_nanos())
                })
        });
        let mut rng = SmallRng::seed_from_u64(semente);
        let teto = pedido
            .max_tokens
            .unwrap_or(usize::MAX)
            .min(self.ctx - ids.len());

        let mut detok = Detok::novo();
        let mut saida = Saida::nova(pedido.pensar, pedido.ferramentas.clone());
        let mut paradas = FiltroParada::novo(&pedido.stop);
        let mut querem_mais = true;
        let usar_mtp = self.gpu.tem_mtp();
        // Tokens que um passo de MTP já validou (aceitos + o `seguinte` amostrado dentro
        // do passo), esperando emissão. O último da fila é sempre o `seguinte`, que ainda
        // não passou pelo modelo — quando ele sai, é hora do próximo passo.
        let mut pendentes: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let t_decode = std::time::Instant::now();
        while res.tokens_saida < teto && querem_mais {
            let token = if let Some(t) = pendentes.pop_front() {
                t
            } else {
                let t_amostra = std::time::Instant::now();
                let escolhido = sampler.sample(&logits, &mut rng);
                res.ms_amostragem += t_amostra.elapsed().as_secs_f64() * 1e3;
                u32::try_from(escolhido)
                    .map_err(|_| MotorError::Backend("token fora da faixa de u32".to_owned()))?
            };
            if self.fim.contains(&token) {
                res.parada = Some(Parada::Fim);
                break;
            }
            res.tokens_saida += 1;

            let texto = detok.empurrar(&self.tokenizer.decode_bytes(&[token]));
            for evento in saida.empurrar(&texto) {
                for evento in paradas.filtrar(evento) {
                    if res.ms_ttft.is_none() {
                        res.ms_ttft = Some(t_req.elapsed().as_secs_f64() * 1e3);
                    }
                    querem_mais &= registrar(&mut res, &evento, &mut emitir);
                }
            }
            if paradas.parou {
                res.parada = Some(Parada::Fim);
                break;
            }
            // Terminou antes do próximo passo, não depois: o passo seguinte processaria um
            // token que ninguém vai ler — e, perto do fim do contexto, um verify de MTP
            // que o backend recusa.
            if res.tokens_saida >= teto || !querem_mais {
                break;
            }
            if !pendentes.is_empty() {
                // Um aceito do passo anterior: já está no cache, nada a decodificar.
                continue;
            }
            // O verify escreve `VERIFY_TOK` posições a partir da de `token`; sem espaço para
            // todas, o resto da geração segue token a token.
            let pos = self.sessao.tokens().len();
            if usar_mtp && pos + VERIFY_TOK <= self.ctx && llama_model::mtp_compensa(pos) {
                let passo = self
                    .sessao
                    .passo_mtp(self.gpu, &sampler, &mut rng, token)
                    .map_err(erro_backend)?;
                pendentes.extend(passo.aceitos.iter().flatten());
                pendentes.push_back(passo.seguinte);
            } else {
                logits = self
                    .sessao
                    .decode(self.gpu, token)
                    .map_err(erro_backend)?
                    .to_vec();
            }
        }

        res.ms_decode = t_decode.elapsed().as_secs_f64() * 1e3;

        let resto = detok.finalizar();
        let mut finais = if resto.is_empty() {
            Vec::new()
        } else {
            saida.empurrar(&resto)
        };
        finais.extend(saida.finalizar());
        for evento in finais.into_iter().flat_map(|e| paradas.filtrar(e)) {
            registrar(&mut res, &evento, &mut emitir);
        }
        for evento in paradas.descarregar() {
            registrar(&mut res, &evento, &mut emitir);
        }
        if !querem_mais {
            res.parada = Some(Parada::Fim);
        }

        res.parada = Some(match (res.parada, res.chamadas.is_empty()) {
            (_, false) => Parada::Ferramenta,
            (Some(p), true) => p,
            (None, true) => Parada::Limite,
        });
        Ok(res)
    }

    /// Tokens que estão no KV-cache — o que a próxima requisição pode reaproveitar.
    pub fn tokens_no_cache(&self) -> usize {
        self.sessao.tokens().len()
    }
}

/// Quantos tokens iniciais de `ids` já estão no cache — o que o prefill vai poupar.
///
/// Espelha a decisão de `llama_model::planejar_reuso`: divergência no meio descarta o
/// cache, **exceto** o que estiver antes do snapshot de fronteira de turno.
fn prefixo_comum(cache: &[u32], ids: &[u32], marca: Option<usize>) -> usize {
    let comum = cache.iter().zip(ids).take_while(|(a, b)| a == b).count();
    if comum == cache.len() {
        return comum;
    }
    marca.filter(|&m| m <= comum).unwrap_or(0)
}

fn erro_backend(e: ModelError) -> MotorError {
    MotorError::Backend(e.to_string())
}

/// Acumula o evento no resultado e repassa. Devolve `false` quando o consumidor
/// desistiu do stream.
fn registrar(
    res: &mut Resultado,
    evento: &Evento,
    emitir: &mut impl FnMut(&Evento) -> bool,
) -> bool {
    match evento {
        Evento::Reasoning(t) => res.reasoning.push_str(t),
        Evento::Conteudo(t) => res.conteudo.push_str(t),
        Evento::Chamada { nome, argumentos } => res.chamadas.push(ChamadaPronta {
            id: format!("call_{}", res.chamadas.len()),
            nome: nome.clone(),
            argumentos: serde_json::to_string(argumentos).unwrap_or_else(|_| "{}".to_owned()),
        }),
    }
    emitir(evento)
}

/// Aplica o `stop` do cliente **antes** da emissão. Cortar depois não serve: no streaming
/// o que já saiu por SSE não volta, e o delimitador chegaria ao cliente.
///
/// Só o conteúdo passa pelo filtro — é o texto da resposta; raciocínio e chamadas seguem
/// direto. Do conteúdo, fica retido o sufixo que ainda pode ser o começo de uma parada.
struct FiltroParada<'a> {
    stop: &'a [String],
    pendente: String,
    /// Uma parada fechou: nada mais sai, nem o que os buffers soltarem no fim.
    parou: bool,
}

impl<'a> FiltroParada<'a> {
    fn novo(stop: &'a [String]) -> FiltroParada<'a> {
        FiltroParada {
            stop,
            pendente: String::new(),
            parou: false,
        }
    }

    /// Os eventos que já podem sair em troca de `evento`.
    fn filtrar(&mut self, evento: Evento) -> Vec<Evento> {
        if self.parou {
            return Vec::new();
        }
        let Evento::Conteudo(texto) = evento else {
            // Outro evento fecha o trecho de conteúdo: o retido já não tem como continuar
            // numa parada, e sai antes dele para manter a ordem.
            let mut eventos = self.descarregar();
            eventos.push(evento);
            return eventos;
        };
        self.pendente.push_str(&texto);
        // Vale a parada que começa primeiro no texto, não a primeira da lista.
        let corte = self
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| self.pendente.find(s.as_str()))
            .min();
        if let Some(corte) = corte {
            self.pendente.truncate(corte);
            self.parou = true;
            return self.descarregar();
        }
        let reter = self
            .stop
            .iter()
            .map(|s| cauda_ambigua(&self.pendente, s))
            .max()
            .unwrap_or(0);
        let pronto: String = self.pendente.drain(..self.pendente.len() - reter).collect();
        conteudo(pronto)
    }

    /// Fim da geração por outro motivo: o que estava retido era conteúdo.
    fn descarregar(&mut self) -> Vec<Evento> {
        conteudo(std::mem::take(&mut self.pendente))
    }
}

fn conteudo(texto: String) -> Vec<Evento> {
    if texto.is_empty() {
        Vec::new()
    } else {
        vec![Evento::Conteudo(texto)]
    }
}

fn sampler_de(pedido: &Pedido) -> Sampler {
    if pedido.temperatura <= 0.0 {
        return Sampler::Greedy;
    }
    Sampler::TopKP {
        // `top_k: 0` na API significa "sem corte".
        k: if pedido.top_k == 0 {
            usize::MAX
        } else {
            pedido.top_k
        },
        p: pedido.top_p,
        temp: pedido.temperatura,
    }
}

#[cfg(test)]
mod tests {
    use super::FiltroParada;
    use crate::saida::Evento;

    /// Empurra os pedaços como conteúdo e devolve o texto que saiu, com o que o fim soltar.
    fn filtrar(stop: &[&str], pedacos: &[&str]) -> (String, bool) {
        let stop: Vec<String> = stop.iter().map(|&s| s.to_owned()).collect();
        let mut f = FiltroParada::novo(&stop);
        let mut eventos = Vec::new();
        for p in pedacos {
            eventos.extend(f.filtrar(Evento::Conteudo((*p).to_owned())));
            if f.parou {
                break;
            }
        }
        eventos.extend(f.descarregar());
        let texto = eventos
            .iter()
            .map(|e| match e {
                Evento::Conteudo(t) => t.as_str(),
                _ => "",
            })
            .collect();
        (texto, f.parou)
    }

    #[test]
    fn parada_partida_entre_pedacos_corta_no_delimitador() {
        assert_eq!(
            filtrar(&["PARE"], &["xxP", "AR", "Eyy"]),
            ("xx".to_owned(), true)
        );
    }

    #[test]
    fn texto_depois_da_parada_no_mesmo_pedaco_nao_sai() {
        assert_eq!(filtrar(&["PARE"], &["xxPAREyy"]), ("xx".to_owned(), true));
    }

    /// A retenção é por caractere: "ç" tem dois bytes, e reter só um cortaria no meio dele.
    #[test]
    fn retencao_nao_corta_caractere_multibyte() {
        assert_eq!(
            filtrar(&["ção"], &["a", "ç", "ã", "o!"]),
            ("a".to_owned(), true)
        );
        assert_eq!(
            filtrar(&["ção"], &["a", "ç", "ões"]),
            ("ações".to_owned(), false)
        );
    }

    #[test]
    fn sem_parada_tudo_sai_sem_reter() {
        let stop: Vec<String> = Vec::new();
        let mut f = FiltroParada::novo(&stop);
        assert_eq!(
            f.filtrar(Evento::Conteudo("abc".to_owned())),
            vec![Evento::Conteudo("abc".to_owned())]
        );
    }

    /// Um evento que não é conteúdo solta o retido antes dele, na ordem em que veio.
    #[test]
    fn chamada_depois_de_conteudo_retido_preserva_a_ordem() {
        let stop = vec!["PARE".to_owned()];
        let mut f = FiltroParada::novo(&stop);
        assert_eq!(
            f.filtrar(Evento::Conteudo("xP".to_owned())),
            vec![Evento::Conteudo("x".to_owned())]
        );
        let chamada = Evento::Chamada {
            nome: "f".to_owned(),
            argumentos: serde_json::json!({}),
        };
        assert_eq!(
            f.filtrar(chamada.clone()),
            vec![Evento::Conteudo("P".to_owned()), chamada]
        );
    }
}
