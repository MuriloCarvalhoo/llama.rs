//! Lê a saída do modelo em pedaços e separa raciocínio, texto e chamadas de ferramenta.
//!
//! O Qwen3.8 **não** emite tool call em JSON. O formato que o template ensina é XML:
//!
//! ```text
//! <tool_call>
//! <function=read>
//! <parameter=path>
//! src/main.rs
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! O valor de cada parâmetro é texto puro — inclusive multi-linha —, então o tipo vem
//! do schema que o cliente mandou em `tools`. Sem schema, o valor fica string, que é o
//! palpite que não inventa dado.
//!
//! O prompt já abre `<think>`, então a geração **começa** dentro do raciocínio: o que
//! vem antes de `</think>` é `reasoning_content`, não conteúdo.

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Evento {
    Reasoning(String),
    Conteudo(String),
    Chamada { nome: String, argumentos: Value },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Estado {
    Pensando,
    Texto,
    NaChamada,
}

const FIM_THINK: &str = "</think>";
const ABRE_CALL: &str = "<tool_call>";
const FECHA_CALL: &str = "</tool_call>";

pub struct Saida {
    estado: Estado,
    buf: String,
    /// Schemas das ferramentas (formato OpenAI), para tipar os argumentos.
    ferramentas: Vec<Value>,
    /// Se algum conteúdo não-branco já saiu. Até lá, o espaço é formatação do
    /// template (`</think>\n\n`), não resposta.
    conteudo_iniciado: bool,
}

impl Saida {
    /// `pensando` diz se o prompt abriu `<think>` sem fechar (o padrão do template).
    pub fn nova(pensando: bool, ferramentas: Vec<Value>) -> Saida {
        Saida {
            estado: if pensando {
                Estado::Pensando
            } else {
                Estado::Texto
            },
            buf: String::new(),
            ferramentas,
            conteudo_iniciado: false,
        }
    }

    /// Consome mais texto gerado e devolve os eventos que já fecharam.
    pub fn empurrar(&mut self, texto: &str) -> Vec<Evento> {
        self.buf.push_str(texto);
        self.processar(false)
    }

    /// Fim da geração: emite o que sobrou, mesmo sem marcador de fechamento.
    pub fn finalizar(&mut self) -> Vec<Evento> {
        self.processar(true)
    }

    fn processar(&mut self, fim: bool) -> Vec<Evento> {
        let mut eventos = Vec::new();
        loop {
            let (marcador, estado_seguinte) = match self.estado {
                Estado::Pensando => (FIM_THINK, Estado::Texto),
                Estado::Texto => (ABRE_CALL, Estado::NaChamada),
                Estado::NaChamada => (FECHA_CALL, Estado::Texto),
            };
            if let Some(corte) = self.buf.find(marcador) {
                let antes: String = self.buf.drain(..corte).collect();
                self.buf.drain(..marcador.len());
                self.emitir(&antes, &mut eventos);
                self.estado = estado_seguinte;
                continue;
            }
            // Sem marcador: solta o que não pode ser começo de um, e espera o resto.
            // Dentro de uma chamada nada sai antes do `</tool_call>` — ela só faz
            // sentido inteira, e emitir pedaços a perderia.
            let reter = match (fim, self.estado) {
                (true, _) => 0,
                (_, Estado::NaChamada) => self.buf.len(),
                _ => cauda_ambigua(&self.buf, marcador),
            };
            let solta = self.buf.len() - reter;
            let pronto: String = self.buf.drain(..solta).collect();
            self.emitir(&pronto, &mut eventos);
            if fim {
                self.buf.clear();
            }
            return eventos;
        }
    }

    fn emitir(&mut self, texto: &str, eventos: &mut Vec<Evento>) {
        if texto.is_empty() {
            return;
        }
        match self.estado {
            Estado::Pensando => eventos.push(Evento::Reasoning(texto.to_owned())),
            Estado::Texto => {
                let texto = if self.conteudo_iniciado {
                    texto
                } else {
                    texto.trim_start()
                };
                if !texto.is_empty() {
                    self.conteudo_iniciado = true;
                    eventos.push(Evento::Conteudo(texto.to_owned()));
                }
            }
            Estado::NaChamada => {
                if let Some(chamada) = self.parse_chamada(texto) {
                    eventos.push(chamada);
                }
            }
        }
    }

    fn parse_chamada(&self, corpo: &str) -> Option<Evento> {
        let (_, resto) = corpo.split_once("<function=")?;
        let (nome, resto) = resto.split_once('>')?;
        let nome = nome.trim().to_owned();

        let mut argumentos = Map::new();
        let mut restante = resto;
        while let Some((_, apos)) = restante.split_once("<parameter=") {
            let Some((chave, apos)) = apos.split_once('>') else {
                break;
            };
            let Some((valor, apos)) = apos.split_once("</parameter>") else {
                break;
            };
            let chave = chave.trim().to_owned();
            let valor = valor.trim_matches('\n');
            argumentos.insert(chave.clone(), self.tipar(&nome, &chave, valor));
            restante = apos;
        }
        Some(Evento::Chamada {
            nome,
            argumentos: Value::Object(argumentos),
        })
    }

    /// Converte o texto do parâmetro para o tipo que o schema da ferramenta declara.
    fn tipar(&self, ferramenta: &str, parametro: &str, valor: &str) -> Value {
        let tipo = self.tipo_declarado(ferramenta, parametro);
        match tipo.as_deref() {
            Some("integer" | "number") => serde_json::from_str::<serde_json::Number>(valor.trim())
                .map_or_else(|_| Value::String(valor.to_owned()), Value::Number),
            Some("boolean") => match valor.trim() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                outro => Value::String(outro.to_owned()),
            },
            Some("array" | "object") => serde_json::from_str(valor.trim())
                .unwrap_or_else(|_| Value::String(valor.to_owned())),
            // Sem schema: só arrisca o parse quando o texto se anuncia como JSON.
            None if valor.trim_start().starts_with(['[', '{']) => {
                serde_json::from_str(valor.trim())
                    .unwrap_or_else(|_| Value::String(valor.to_owned()))
            }
            _ => Value::String(valor.to_owned()),
        }
    }

    fn tipo_declarado(&self, ferramenta: &str, parametro: &str) -> Option<String> {
        self.ferramentas
            .iter()
            .map(|f| f.get("function").unwrap_or(f))
            .find(|f| f.get("name").and_then(Value::as_str) == Some(ferramenta))?
            .get("parameters")?
            .get("properties")?
            .get(parametro)?
            .get("type")?
            .as_str()
            .map(str::to_owned)
    }
}

/// Quantos bytes do fim de `buf` ainda podem virar `marcador` com o próximo pedaço.
fn cauda_ambigua(buf: &str, marcador: &str) -> usize {
    // `..=max`: com `buf` terminando em "<" e o marcador "</think>", reter 1 byte é o
    // que impede o "<" de vazar como texto e o marcador de nunca ser reconhecido.
    let max = marcador.len().min(buf.len());
    (1..=max)
        .rev()
        .find(|&n| {
            buf.get(buf.len() - n..)
                .is_some_and(|cauda| marcador.starts_with(cauda))
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{Evento, Saida};
    use serde_json::{Value, json};

    fn ferramenta_read() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "recursivo": {"type": "boolean"},
                        "linhas": {"type": "array"}
                    }
                }
            }
        })
    }

    fn tudo(s: &mut Saida, texto: &str) -> Vec<Evento> {
        let mut ev = s.empurrar(texto);
        ev.extend(s.finalizar());
        ev
    }

    #[test]
    fn o_que_vem_antes_do_fim_do_think_e_reasoning() {
        let mut s = Saida::nova(true, Vec::new());

        let ev = tudo(&mut s, "pensando alto\n</think>\n\nresposta");

        assert_eq!(
            ev,
            vec![
                Evento::Reasoning("pensando alto\n".to_owned()),
                Evento::Conteudo("resposta".to_owned())
            ]
        );
    }

    /// O template fecha o raciocínio com `</think>\n\n`: essas duas quebras são
    /// formatação do prompt, não resposta. Deixá-las passar põe uma linha em branco no
    /// começo de toda mensagem do assistente.
    #[test]
    fn quebras_entre_o_think_e_o_conteudo_nao_viram_conteudo() {
        let mut s = Saida::nova(true, Vec::new());

        let ev = tudo(&mut s, "penso\n</think>\n\n  resposta");

        assert_eq!(
            ev,
            vec![
                Evento::Reasoning("penso\n".to_owned()),
                Evento::Conteudo("resposta".to_owned())
            ]
        );
    }

    /// Mas espaço **dentro** do texto é conteúdo e fica.
    #[test]
    fn espaco_depois_do_inicio_do_conteudo_e_preservado() {
        let mut s = Saida::nova(false, Vec::new());
        assert_eq!(
            tudo(&mut s, "a  b\n\nc"),
            vec![Evento::Conteudo("a  b\n\nc".to_owned())]
        );
    }

    #[test]
    fn sem_thinking_tudo_e_conteudo() {
        let mut s = Saida::nova(false, Vec::new());
        assert_eq!(
            tudo(&mut s, "direto ao ponto"),
            vec![Evento::Conteudo("direto ao ponto".to_owned())]
        );
    }

    #[test]
    fn tool_call_vira_chamada_com_argumentos_tipados() {
        let mut s = Saida::nova(false, vec![ferramenta_read()]);

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n\
             <parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>",
        );

        assert_eq!(
            ev,
            vec![Evento::Chamada {
                nome: "read".to_owned(),
                argumentos: json!({"path": "src/main.rs", "limit": 10}),
            }]
        );
    }

    #[test]
    fn booleano_e_array_seguem_o_schema() {
        let mut s = Saida::nova(false, vec![ferramenta_read()]);

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=read>\n<parameter=recursivo>\ntrue\n</parameter>\n\
             <parameter=linhas>\n[1, 2]\n</parameter>\n</function>\n</tool_call>",
        );

        assert_eq!(
            ev,
            vec![Evento::Chamada {
                nome: "read".to_owned(),
                argumentos: json!({"recursivo": true, "linhas": [1, 2]}),
            }]
        );
    }

    /// Sem schema o valor fica string: inventar tipo quebraria uma tool que espera "10".
    #[test]
    fn sem_schema_o_valor_fica_string() {
        let mut s = Saida::nova(false, Vec::new());

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=x>\n<parameter=n>\n10\n</parameter>\n</function>\n</tool_call>",
        );

        assert_eq!(
            ev,
            vec![Evento::Chamada {
                nome: "x".to_owned(),
                argumentos: json!({"n": "10"}),
            }]
        );
    }

    #[test]
    fn valor_multilinha_preserva_as_quebras_internas() {
        let mut s = Saida::nova(false, Vec::new());

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=escrever>\n<parameter=texto>\nlinha 1\nlinha 2\n</parameter>\n</function>\n</tool_call>",
        );

        assert_eq!(
            ev,
            vec![Evento::Chamada {
                nome: "escrever".to_owned(),
                argumentos: json!({"texto": "linha 1\nlinha 2"}),
            }]
        );
    }

    /// O texto chega token a token: nenhum marcador pode escapar partido.
    #[test]
    fn marcador_partido_entre_pedacos_nao_vaza_como_texto() {
        let mut s = Saida::nova(true, Vec::new());
        let inteiro = "raciocínio</think>oi<tool_call>\n<function=f>\n</function>\n</tool_call>fim";

        let mut ev = Vec::new();
        let mut resto = inteiro;
        while !resto.is_empty() {
            let n = resto.chars().next().map_or(0, char::len_utf8);
            let (pedaco, novo) = resto.split_at(n);
            ev.extend(s.empurrar(pedaco));
            resto = novo;
        }
        ev.extend(s.finalizar());

        let textos: String = ev
            .iter()
            .filter_map(|e| match e {
                Evento::Conteudo(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(textos, "oifim", "nenhum pedaço de marcador virou conteúdo");
        assert!(ev.iter().any(|e| matches!(e, Evento::Chamada { .. })));
    }

    #[test]
    fn duas_chamadas_seguidas_saem_separadas() {
        let mut s = Saida::nova(false, Vec::new());

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=a>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=b>\n</function>\n</tool_call>",
        );

        let nomes: Vec<&str> = ev
            .iter()
            .filter_map(|e| match e {
                Evento::Chamada { nome, .. } => Some(nome.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(nomes, vec!["a", "b"]);
    }

    /// Um `<` solto no meio de código não pode travar o stream.
    #[test]
    fn menor_que_no_texto_comum_sai_normalmente() {
        let mut s = Saida::nova(false, Vec::new());
        assert_eq!(
            tudo(&mut s, "if (a < b) return;"),
            vec![Evento::Conteudo("if (a < b) return;".to_owned())]
        );
    }

    #[test]
    fn chamada_sem_fechamento_no_fim_da_geracao_ainda_e_emitida() {
        let mut s = Saida::nova(false, Vec::new());

        let ev = tudo(
            &mut s,
            "<tool_call>\n<function=f>\n<parameter=p>\nv\n</parameter>\n</function>",
        );

        assert_eq!(
            ev,
            vec![Evento::Chamada {
                nome: "f".to_owned(),
                argumentos: json!({"p": "v"}),
            }]
        );
    }
}
