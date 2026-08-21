#![forbid(unsafe_code)]
//! Chat template do Qwen3.5/3.8 (ChatML com tools em XML), escrito à mão.
//!
//! O template vive nos metadados do GGUF em Jinja — com macros, `namespace`, slicing
//! reverso e `raise_exception`. Em vez de embutir um motor Jinja no runtime, o formato
//! está reimplementado aqui e provado caso a caso contra a saída do Jinja real:
//! `refs/chat_qwen38.json`, gerado por `scripts/gen-chat-refs.py`.
//!
//! O que **não** está implementado, por não ser usado: imagens/vídeo (o runtime não
//! tem visão) e `preserve_thinking=false` (o padrão do template é preservar).

mod json_py;
mod render;

pub use render::{ChatError, Esforco, Ferramenta, Mensagem, Opcoes, Papel, render};
