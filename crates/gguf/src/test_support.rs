//! Builder de bytes GGUF para testes. Compilado apenas com `cfg(test)`.
#![cfg(test)]

/// Acumula bytes de um arquivo GGUF v3 little-endian.
pub(crate) struct GgufBuilder {
    kv: Vec<u8>,
    kv_count: u64,
    tensors: Vec<u8>,
    tensor_count: u64,
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self { kv: Vec::new(), kv_count: 0, tensors: Vec::new(), tensor_count: 0 }
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Adiciona KV escalar u32 (value_type = 4).
    pub fn kv_u32(mut self, key: &str, val: u32) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&4u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// Adiciona KV escalar f32 (value_type = 6).
    pub fn kv_f32(mut self, key: &str, val: f32) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// Adiciona KV string (value_type = 8).
    pub fn kv_string(mut self, key: &str, val: &str) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        Self::push_string(&mut self.kv, val);
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de strings (value_type = 9, elem_type = 8).
    pub fn kv_str_array(mut self, key: &str, vals: &[&str]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            Self::push_string(&mut self.kv, v);
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de f32 (value_type = 9, elem_type = 6).
    pub fn kv_f32_array(mut self, key: &str, vals: &[f32]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&6u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona KV array de i32 (value_type = 9, elem_type = 5).
    pub fn kv_i32_array(mut self, key: &str, vals: &[i32]) -> Self {
        Self::push_string(&mut self.kv, key);
        self.kv.extend_from_slice(&9u32.to_le_bytes());
        self.kv.extend_from_slice(&5u32.to_le_bytes());
        self.kv.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.kv.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    /// Adiciona um tensor info (sem dados). `ggml_type` é o id u32.
    pub fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
        Self::push_string(&mut self.tensors, name);
        self.tensors.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.tensors.extend_from_slice(&d.to_le_bytes());
        }
        self.tensors.extend_from_slice(&ggml_type.to_le_bytes());
        self.tensors.extend_from_slice(&offset.to_le_bytes());
        self.tensor_count += 1;
        self
    }

    /// Serializa header + KV + tensor infos. NÃO inclui padding nem dados.
    pub fn build_meta_only(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kv);
        out.extend_from_slice(&self.tensors);
        out
    }

    /// Serializa tudo + padding até `alignment` + `data`.
    pub fn build_with_data(&self, alignment: usize, data: &[u8]) -> Vec<u8> {
        let mut out = self.build_meta_only();
        while out.len() % alignment != 0 {
            out.push(0);
        }
        out.extend_from_slice(data);
        out
    }
}
