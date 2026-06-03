# Design: Fase 1 — Parser GGUF + Tokenizer SPM (`gguf`, `llama-tokenizer`)

**Data:** 2026-06-03
**Status:** Aprovado para planejamento de implementação
**Spec-pai:** [docs/superpowers/specs/2026-06-03-llama-rs-rewrite-design.md](2026-06-03-llama-rs-rewrite-design.md)

## Objetivo

Entregar a primeira fatia vertical de leitura do modelo: um parser do formato
GGUF e o tokenizer SPM (Llama), validados diferencialmente contra o oráculo
C++. Critério de aceite da fase (do spec-pai): **metadados idênticos ao oráculo**
e **tokens bit-exact** num corpus de teste.

## Escopo

| Incluído | Excluído (fase posterior) |
|---|---|
| Parser GGUF v3, little-endian, sobre `&[u8]` | mmap (Fase 2), GGUF v1/v2, big-endian |
| Acesso raw aos bytes de cada tensor (nome, shape, dtype, offset) | Dequantização / views tipadas (Fase 3) |
| Tokenizer **SPM / Llama** (merge-by-score + byte-fallback) | BPE (Qwen, Fase 7), UGM |
| `encode` (critério) + `decode` (trivial, usado na Fase 4) | Sampling, geração (Fase 4) |

## Decisões de design (aprovadas)

### (a) Zero-unsafe vs mmap — parser opera sobre `&[u8]`

O spec-pai exige `gguf` como `#![forbid(unsafe_code)]` *e* menciona memmap2.
`memmap2::Mmap::map` é `unsafe fn`, o que conflitaria com `forbid`. **Resolução:**
o parser `gguf` parseia um slice `&[u8]` emprestado — parsing puro, 100% safe,
`forbid` honrado. Como os bytes são obtidos (mmap vs `std::fs::read`) é
responsabilidade da camada de carregamento da Fase 2, onde o único
`unsafe { Mmap::map(..) }` pode morar com comentário `// SAFETY:` e invariante
documentada. Benefício colateral: um parser sobre slice é o formato ideal para
proptest com bytes arbitrários e malformados. Na Fase 1 os testes carregam o
arquivo com `std::fs::read` (os modelos de teste são pequenos).

### (b) `GgmlType` mora em `gguf` na Fase 1

O spec-pai aloca o sistema de tipos em `ggml-core`, que só nasce na Fase 2. O
parser precisa do enum de tipos ggml (com `block_size`/`type_size`) para
localizar e dimensionar os dados de cada tensor. Definir `GgmlType` em `gguf`
agora evita desenhar uma interface cross-crate especulativa (YAGNI). Quando
`ggml-core` for criado na Fase 2, mover/re-exportar é um refactor mecânico.
*Seam conhecido, documentado.*

### (c) Corpus de tokens expandido via oráculo

`refs/tokens.json` (4 casos hoje) será expandido com casos gerados pelo
`llama-tokenize` do oráculo (ground-truth) que estressam: byte-fallback
(caracteres fora do vocab), espaços líderes/repetidos, unicode multibyte e
pontuação. Mais confiança na garantia "bit-exact".

## Arquitetura

```
crates/
├── gguf/                       # #![forbid(unsafe_code)] — dep: thiserror
│   ├── src/lib.rs              # API pública, re-exports
│   ├── src/error.rs            # GgufError (thiserror)
│   ├── src/reader.rs           # Cursor com bounds-check sobre &[u8]
│   ├── src/types.rs            # MetadataValue, GgmlType + tabela de blocos
│   ├── src/parse.rs            # header → KVs → tensor infos → alinhamento
│   └── src/file.rs             # GgufFile + tensor_data()
└── llama-tokenizer/            # #![forbid(unsafe_code)] — dep: gguf, thiserror
    ├── src/lib.rs              # API: encode/decode
    ├── src/error.rs            # TokenizerError (thiserror)
    ├── src/vocab.rs            # Vocab + Vocab::from_gguf()
    └── src/spm.rs              # Algoritmo SPM (réplica do llm_tokenizer_spm)
```

`oracle/` permanece na raiz do workspace. `members += ["crates/gguf",
"crates/llama-tokenizer"]`.

## Crate `gguf`

### Formato (GGUF v3, little-endian)

1. **Header:** magic `GGUF` (`0x46554747` LE), `version: u32` (=3),
   `tensor_count: u64`, `metadata_kv_count: u64`.
2. **Metadata KV** (×`metadata_kv_count`): `key` (string GGUF: `u64` len + bytes
   UTF-8), `value_type: u32`, `value` (conforme tipo).
3. **Tensor infos** (×`tensor_count`): `name` (string GGUF), `n_dims: u32`,
   `dims: [u64; n_dims]`, `type: u32` (id ggml), `offset: u64` (relativo ao
   início da seção de dados).
4. **Padding** até `general.alignment` (default 32), depois a seção de dados.

### Tipos GGUF de metadados (`MetadataValue`)

Enum cobrindo os 13 tipos: `Uint8`, `Int8`, `Uint16`, `Int16`, `Uint32`,
`Int32`, `Float32`, `Bool`, `String`, `Array(Box<...>)`, `Uint64`, `Int64`,
`Float64`. `Array` carrega o tipo do elemento + os elementos. Acessores
ergonômicos com erro tipado quando o tipo não casa (ex.: `as_u32()`,
`as_str()`, `as_f32_array()`).

### `GgmlType`

Enum dos type-ids ggml (F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K …
Q6_K, Q8_K, IQ*, BF16, etc.), cada um com `block_size` (elementos por bloco) e
`type_size` (bytes por bloco). Usado por `tensor_data` para calcular o tamanho
em bytes de um tensor: `n_bytes = (n_elements / block_size) * type_size`.
`TryFrom<u32>` para o id; id desconhecido → erro.

### API pública

```rust
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    // privado: alignment, offset da seção de dados
}

pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl GgufFile {
    /// Parseia metadados + tensor infos a partir do conteúdo completo do arquivo.
    pub fn parse(bytes: &[u8]) -> Result<GgufFile, GgufError>;
    /// Bytes raw de um tensor (sem dequant). Slice sobre a seção de dados.
    pub fn tensor_data<'a>(&self, bytes: &'a [u8], t: &TensorInfo)
        -> Result<&'a [u8], GgufError>;
}
```

### Robustez (entrada não-confiável)

GGUF é entrada não-confiável: **todo** acesso a bytes passa por bounds-check no
`reader.rs` e retorna `Result`. Nenhum `panic!`/`unwrap`/index-slicing. Strings
inválidas (não-UTF-8), counts absurdos (overflow ao multiplicar), offsets fora
da seção de dados, `n_dims` excessivo → erro tipado, nunca abort.

## Crate `llama-tokenizer`

### `Vocab`

```rust
pub struct Vocab {
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    bos_id: u32,
    eos_id: u32,
    unk_id: u32,
}
impl Vocab {
    pub fn from_gguf(f: &GgufFile) -> Result<Vocab, TokenizerError>;
}
```

Lê `tokenizer.ggml.{tokens,scores,token_type}` e os `*_token_id` dos metadados.
Valida que `tokenizer.ggml.model == "llama"` (SPM); outro valor → erro claro
(BPE fica para fase posterior).

### Algoritmo SPM (`spm.rs`) — réplica do `llm_tokenizer_spm` do llama.cpp

1. **Normalização:** prefixar espaço (`add_space_prefix`, default true) e
   substituir `' '` por `▁` (U+2581).
2. **Símbolos:** dividir em símbolos UTF-8 (um char por símbolo inicial),
   ligados em lista duplamente encadeada (índices `prev`/`next`).
3. **Merge por bigrama:** fila de prioridade de pares adjacentes; o score do par
   é o `score` do token resultante da concatenação (lookup no vocab). Funde
   repetidamente o par de maior score (desempate pela posição, como no C++),
   atualizando os vizinhos.
4. **Resegmentação / byte-fallback:** para cada símbolo final, se está no vocab
   usa o id; senão emite os tokens-byte `<0xXX>` de cada byte do símbolo.
5. **BOS:** prefixa `bos_id` quando `add_bos` (os casos de referência começam
   todos com token 1 = BOS).

`decode`: concatena as strings dos tokens, reverte `▁`→`' '` e remove o espaço
de prefixo; tokens-byte `<0xXX>` viram os bytes correspondentes.

### API pública

```rust
pub struct Tokenizer { vocab: Vocab }
impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self;
    pub fn from_gguf(f: &GgufFile) -> Result<Self, TokenizerError>;
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32>;
    pub fn decode(&self, ids: &[u32]) -> String;
}
```

O algoritmo SPM opera sobre os structs simples de `Vocab` → unit-testável sem
depender de `gguf`.

## Estratégia de validação

### Tokenizer (critério primário da fase)

`encode(text, add_bos=true)` deve ser **bit-exact** contra cada caso de
`refs/tokens.json`. Corpus expandido pelo oráculo (`llama-tokenize`) cobrindo
byte-fallback, espaços, unicode e pontuação.

### Metadados ("idênticos ao oráculo")

Sem binário `gguf-dump` no oráculo. O loader do llama.cpp imprime todos os KVs
no stderr (`llama_model_loader: - kv N: <chave> <tipo> = <valor>`), valores
completos para escalares. Estratégia:

- **Escalares:** snapshot revisado em `refs/stories260k-meta.json` (conferido
  contra o dump do loader), comparado à saída parseada pelo `gguf`.
- **Arrays** (`tokens`/`scores`/`token_type`): validados por tamanho
  (`arr[str,512]` ⇒ len 512) e **transitivamente** pelo tokenizer bit-exact
  (que só funciona se tokens/scores/token_types foram lidos corretamente).

### Robustez do parser

proptest: bytes arbitrários e arquivos truncados em qualquer offset →
`GgufFile::parse` sempre retorna `Result`, nunca panica.

## Gate de validação por tarefa (do spec-pai)

1. TDD — teste existia e falhava antes da implementação
2. `cargo fmt --check` + `cargo clippy --all-targets -- -D warnings` limpos
3. `cargo test` do workspace verde
4. Teste diferencial vs oráculo dentro do critério da fase
5. Cobertura ≥ 80% no crate tocado (`cargo llvm-cov`)
6. proptest no parser (entrada não-confiável: round-trip + malformados)
7. `rust-review` sem CRITICAL/HIGH

Sem `unsafe` em ambos os crates ⇒ miri não se aplica nesta fase.

## Dependências novas

| Crate | Dependências | Justificativa |
|---|---|---|
| `gguf` | `thiserror` | erros tipados; parsing puro com `from_le_bytes` (sem bytemuck nesta fase) |
| `llama-tokenizer` | `gguf`, `thiserror` | vocab vem do GGUF; algoritmo isolável dos dados |
| dev (workspace) | `proptest` | fuzz do parser |

`cargo deny check` quando as dependências mudarem.

## Riscos

- **Fidelidade do SPM:** desempates e ordem de merge precisam casar com o C++
  para bit-exact. Mitigação: corpus diferencial amplo + bissecção camada por
  camada (`systematic-debugging`) se divergir; ler o `llm_tokenizer_spm.cpp` do
  upstream como referência (somente leitura).
- **Cobertura de `GgmlType`:** stories260K provavelmente é F32; muitos type-ids
  não terão tensor real para exercitar `tensor_data`. Mitigação: testar o cálculo
  de tamanho por bloco com vetores sintéticos por tipo; o qwen2.5-q8_0 exercita
  Q8_0 de fato.
- **Seam `GgmlType` (b):** mover para `ggml-core` na Fase 2 é trabalho extra,
  porém mecânico e previsto.
