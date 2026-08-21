#![forbid(unsafe_code)]
//! Servidor HTTP compatível com a API de chat da OpenAI para o backend residente.

pub mod api;
pub mod detok;
pub mod http;
pub mod motor;
pub mod saida;
pub mod servidor;
