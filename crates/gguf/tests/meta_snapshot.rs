//! Confere a saída parseada contra o snapshot revisado refs/stories260k-meta.json.
#![allow(clippy::indexing_slicing, clippy::cast_possible_truncation)]
use serde_json::Value;

#[test]
fn matches_reviewed_snapshot() {
    let Ok(model) = std::fs::read("../../models/stories260K.gguf") else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&model).unwrap();
    let snap: Value =
        serde_json::from_str(&std::fs::read_to_string("../../refs/stories260k-meta.json").unwrap())
            .unwrap();

    assert_eq!(f.version as u64, snap["version"].as_u64().unwrap());

    for (key, expected) in snap["scalars"].as_object().unwrap() {
        let got = f.get(key).unwrap();
        match expected {
            Value::String(s) => assert_eq!(got.as_str(key).unwrap(), s, "{key}"),
            Value::Number(n) if n.is_f64() && n.as_u64().is_none() => {
                let e = n.as_f64().unwrap() as f32;
                assert!((got.as_f32(key).unwrap() - e).abs() < 1e-9, "{key}");
            }
            Value::Number(n) => {
                assert_eq!(
                    got.as_u32(key).unwrap() as u64,
                    n.as_u64().unwrap(),
                    "{key}"
                )
            }
            _ => panic!("tipo inesperado no snapshot para {key}"),
        }
    }

    for (key, len) in snap["array_lengths"].as_object().unwrap() {
        assert_eq!(
            f.get(key).unwrap().array_len(),
            Some(len.as_u64().unwrap() as usize),
            "{key}"
        );
    }

    assert_eq!(
        f.tensors.len() as u64,
        snap["tensor_count"].as_u64().unwrap()
    );
}
