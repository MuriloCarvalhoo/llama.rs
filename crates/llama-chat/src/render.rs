//! Renderização do prompt no formato que o Qwen3.5/3.8 espera.
//!
//! O texto fixo (instruções de esforço, bloco de tools) é copiado literalmente do
//! template do GGUF — qualquer diferença tira o prompt da distribuição do treino.

use serde_json::Value;

use crate::json_py::para_json_estilo_python;

/// Papel de uma mensagem. `developer` entra como `System` — o template os funde.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Papel {
    System,
    User,
    Assistant,
    Tool,
}

/// Quanto o modelo deve pensar. Vira um parágrafo de instrução no bloco system —
/// `Medium` é o único que não injeta nada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Esforco {
    #[default]
    Xhigh,
    Medium,
    Low,
}

/// Uma chamada de ferramenta no histórico. `argumentos` é sempre um objeto JSON.
#[derive(Debug, Clone)]
pub struct Ferramenta {
    pub nome: String,
    pub argumentos: Value,
}

#[derive(Debug, Clone)]
pub struct Mensagem {
    pub papel: Papel,
    pub conteudo: String,
    /// Raciocínio de uma mensagem de assistant (`reasoning_content`), sem as tags.
    pub reasoning: String,
    pub chamadas: Vec<Ferramenta>,
}

/// Opções do render. O padrão é o do template: thinking ligado em esforço `xhigh`.
#[derive(Debug, Clone)]
pub struct Opcoes {
    /// Schemas das ferramentas, no formato OpenAI (`{"type":"function","function":{...}}`).
    pub ferramentas: Vec<Value>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub esforco: Esforco,
}

impl Default for Opcoes {
    fn default() -> Self {
        Self {
            ferramentas: Vec::new(),
            add_generation_prompt: true,
            enable_thinking: true,
            esforco: Esforco::default(),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChatError {
    #[error("nenhuma mensagem")]
    SemMensagens,
    #[error("mensagem de system fora do início da conversa")]
    SystemForaDoInicio,
    #[error("papel desconhecido: {0}")]
    PapelDesconhecido(String),
    #[error("chamada de ferramenta sem nome")]
    ChamadaSemNome,
    #[error("argumentos de {0} não são um objeto JSON")]
    ArgumentosInvalidos(String),
}

const INSTRUCAO_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const INSTRUCAO_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
const CABECALHO_TOOLS: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const FORMATO_TOOLS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

impl Mensagem {
    /// Converte uma mensagem no formato da API OpenAI.
    ///
    /// `arguments` chega como string JSON no protocolo (`"{\"path\":\"a\"}"`) — aqui já
    /// vira objeto, que é o que o template consome. Objeto direto também é aceito.
    pub fn de_json(v: &Value) -> Result<Mensagem, ChatError> {
        let papel = match v.get("role").and_then(Value::as_str).unwrap_or_default() {
            "system" | "developer" => Papel::System,
            "user" => Papel::User,
            "assistant" => Papel::Assistant,
            "tool" => Papel::Tool,
            outro => return Err(ChatError::PapelDesconhecido(outro.to_owned())),
        };
        let conteudo = texto_do_conteudo(v.get("content"));
        let reasoning = v
            .get("reasoning_content")
            .or_else(|| v.get("reasoning"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();

        let mut chamadas = Vec::new();
        for tc in v
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let f = tc.get("function").unwrap_or(tc);
            let nome = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or(ChatError::ChamadaSemNome)?
                .to_owned();
            let argumentos = match f.get("arguments") {
                None | Some(Value::Null) => Value::Object(serde_json::Map::new()),
                Some(Value::String(s)) if s.trim().is_empty() => {
                    Value::Object(serde_json::Map::new())
                }
                Some(Value::String(s)) => serde_json::from_str(s)
                    .map_err(|_| ChatError::ArgumentosInvalidos(nome.clone()))?,
                Some(outro) => outro.clone(),
            };
            if !argumentos.is_object() {
                return Err(ChatError::ArgumentosInvalidos(nome));
            }
            chamadas.push(Ferramenta { nome, argumentos });
        }

        Ok(Mensagem {
            papel,
            conteudo,
            reasoning,
            chamadas,
        })
    }
}

/// `content` é string ou lista de partes; só o texto interessa (não há visão aqui).
fn texto_do_conteudo(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(partes)) => partes
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .concat(),
        _ => String::new(),
    }
}

/// Monta o prompt completo. Espelha `tokenizer.chat_template` do GGUF.
pub fn render(mensagens: &[Mensagem], op: &Opcoes) -> Result<String, ChatError> {
    if mensagens.is_empty() {
        return Err(ChatError::SemMensagens);
    }

    // System/developer consecutivos a partir do índice 0 viram um bloco só.
    let num_sys = mensagens
        .iter()
        .take_while(|m| m.papel == Papel::System)
        .count();
    let merged_system = mensagens
        .iter()
        .take(num_sys)
        .map(|m| m.conteudo.trim())
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    let instrucao = match (op.enable_thinking, op.esforco) {
        (false, _) | (true, Esforco::Medium) => "",
        (true, Esforco::Xhigh) => INSTRUCAO_XHIGH,
        (true, Esforco::Low) => INSTRUCAO_LOW,
    };

    let mut out = String::new();
    if op.ferramentas.is_empty() {
        if !merged_system.is_empty() {
            out.push_str("<|im_start|>system\n");
            if !instrucao.is_empty() {
                out.push_str(instrucao);
                out.push_str("\n\n");
            }
            out.push_str(&merged_system);
            out.push_str("<|im_end|>\n");
        } else if !instrucao.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(instrucao);
            out.push_str("<|im_end|>\n");
        }
    } else {
        out.push_str("<|im_start|>system\n");
        if !instrucao.is_empty() {
            out.push_str(instrucao);
            out.push_str("\n\n");
        }
        out.push_str(CABECALHO_TOOLS);
        for t in &op.ferramentas {
            out.push('\n');
            out.push_str(&para_json_estilo_python(t));
        }
        out.push_str("\n</tools>");
        out.push_str(FORMATO_TOOLS);
        if !merged_system.is_empty() {
            out.push_str("\n\n");
            out.push_str(&merged_system);
        }
        out.push_str("<|im_end|>\n");
    }

    for (i, m) in mensagens.iter().enumerate().skip(num_sys) {
        let conteudo = m.conteudo.trim();
        match m.papel {
            Papel::System => return Err(ChatError::SystemForaDoInicio),
            Papel::User => {
                out.push_str("<|im_start|>user\n");
                out.push_str(conteudo);
                out.push_str("<|im_end|>\n");
            }
            Papel::Assistant => {
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(m.reasoning.trim());
                out.push_str("\n</think>\n\n");
                out.push_str(conteudo);
                for (j, c) in m.chamadas.iter().enumerate() {
                    if j == 0 {
                        if !conteudo.is_empty() {
                            out.push_str("\n\n");
                        }
                    } else {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&c.nome);
                    out.push_str(">\n");
                    for (chave, valor) in c.argumentos.as_object().into_iter().flatten() {
                        out.push_str("<parameter=");
                        out.push_str(chave);
                        out.push_str(">\n");
                        match valor {
                            Value::String(s) => out.push_str(s),
                            outro => out.push_str(&para_json_estilo_python(outro)),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            Papel::Tool => {
                let anterior_era_tool =
                    i > 0 && mensagens.get(i - 1).is_some_and(|p| p.papel == Papel::Tool);
                if !anterior_era_tool {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(conteudo);
                out.push_str("\n</tool_response>");
                let proxima_e_tool = mensagens.get(i + 1).is_some_and(|p| p.papel == Papel::Tool);
                if !proxima_e_tool {
                    out.push_str("<|im_end|>\n");
                }
            }
        }
    }

    if op.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if op.enable_thinking {
            out.push_str("<think>\n");
        } else {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(papel: Papel, conteudo: &str) -> Mensagem {
        Mensagem {
            papel,
            conteudo: conteudo.to_owned(),
            reasoning: String::new(),
            chamadas: Vec::new(),
        }
    }

    #[test]
    fn sem_mensagens_e_erro() {
        assert_eq!(
            render(&[], &Opcoes::default()).unwrap_err(),
            ChatError::SemMensagens
        );
    }

    /// O template levanta exceção; aqui vira erro em vez de prompt silenciosamente torto.
    #[test]
    fn system_depois_do_inicio_e_erro() {
        let msgs = [msg(Papel::User, "oi"), msg(Papel::System, "regra tardia")];
        assert_eq!(
            render(&msgs, &Opcoes::default()).unwrap_err(),
            ChatError::SystemForaDoInicio
        );
    }

    #[test]
    fn papel_desconhecido_e_erro() {
        let v = serde_json::json!({"role": "narrador", "content": "..."});
        assert_eq!(
            Mensagem::de_json(&v).unwrap_err(),
            ChatError::PapelDesconhecido("narrador".to_owned())
        );
    }

    #[test]
    fn chamada_sem_nome_e_erro() {
        let v = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"arguments": "{}"}}]
        });
        assert_eq!(
            Mensagem::de_json(&v).unwrap_err(),
            ChatError::ChamadaSemNome
        );
    }

    #[test]
    fn argumentos_que_nao_viram_objeto_sao_erro() {
        let quebrado = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read", "arguments": "{não é json"}}]
        });
        assert_eq!(
            Mensagem::de_json(&quebrado).unwrap_err(),
            ChatError::ArgumentosInvalidos("read".to_owned())
        );

        let lista = serde_json::json!({
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read", "arguments": [1, 2]}}]
        });
        assert_eq!(
            Mensagem::de_json(&lista).unwrap_err(),
            ChatError::ArgumentosInvalidos("read".to_owned())
        );
    }

    /// No protocolo OpenAI `arguments` é uma **string** JSON; o template quer objeto.
    #[test]
    fn de_json_parseia_arguments_em_string() {
        let v = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": "{\"path\": \"a.rs\"}"}
            }]
        });
        let m = Mensagem::de_json(&v).unwrap();
        let [chamada] = m.chamadas.as_slice() else {
            panic!("esperava exatamente uma chamada");
        };
        assert_eq!(chamada.nome, "read");
        assert_eq!(
            chamada.argumentos.get("path").and_then(Value::as_str),
            Some("a.rs")
        );
    }

    #[test]
    fn arguments_ausente_ou_vazio_vira_objeto_vazio() {
        let v = serde_json::json!({
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "hora"}},
                {"function": {"name": "hora", "arguments": "  "}}
            ]
        });
        let m = Mensagem::de_json(&v).unwrap();
        assert!(m.chamadas.iter().all(|c| {
            c.argumentos
                .as_object()
                .is_some_and(serde_json::Map::is_empty)
        }));
        assert_eq!(m.chamadas.len(), 2);
    }

    #[test]
    fn developer_entra_como_system() {
        let v = serde_json::json!({"role": "developer", "content": "regra"});
        assert_eq!(Mensagem::de_json(&v).unwrap().papel, Papel::System);
    }

    #[test]
    fn conteudo_nulo_vira_string_vazia() {
        let v = serde_json::json!({"role": "assistant", "content": null});
        assert_eq!(Mensagem::de_json(&v).unwrap().conteudo, "");
    }
}
