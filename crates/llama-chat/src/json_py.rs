//! JSON com os separadores do Python — `{"a": 1, "b": 2}`, não `{"a":1,"b":2}`.
//!
//! O bloco `<tools>` do prompt é gerado no treino pelo `tojson` do `transformers`, que
//! é `json.dumps` puro: espaço depois de `:` e de `,`, e sem escapar não-ASCII. O
//! `serde_json::to_string` é compacto, então a diferença apareceria em todo schema de
//! ferramenta que o modelo lê.

use serde_json::Value;

pub(crate) fn para_json_estilo_python(v: &Value) -> String {
    match v {
        Value::Object(campos) => {
            let itens: Vec<String> = campos
                .iter()
                .map(|(chave, valor)| {
                    let chave = serde_json::to_string(chave).unwrap_or_default();
                    format!("{chave}: {}", para_json_estilo_python(valor))
                })
                .collect();
            format!("{{{}}}", itens.join(", "))
        }
        Value::Array(itens) => {
            let itens: Vec<String> = itens.iter().map(para_json_estilo_python).collect();
            format!("[{}]", itens.join(", "))
        }
        // Escalares: o próprio serde_json já escapa como o Python com ensure_ascii=False.
        escalar => serde_json::to_string(escalar).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objeto_sai_com_espaco_depois_dos_separadores() {
        let v: Value = serde_json::from_str(r#"{"a":1,"b":[1,2]}"#).unwrap();
        assert_eq!(para_json_estilo_python(&v), r#"{"a": 1, "b": [1, 2]}"#);
    }

    #[test]
    fn nao_ascii_fica_literal() {
        let v: Value = serde_json::from_str(r#"{"k":"ç"}"#).unwrap();
        assert_eq!(para_json_estilo_python(&v), r#"{"k": "ç"}"#);
    }
}
