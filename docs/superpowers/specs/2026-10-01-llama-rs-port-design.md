# llama.rs — porte do llama.cpp para Rust, sincronizado com a master

**Data:** 2026-10-01 · **Upstream base:** `ggml-org/llama.cpp` @ `ec7630a6` · **Toolchain:** Rust 1.99.0

## 1. Objetivo

Uma cópia do llama.cpp em Rust, restrita ao hardware desta máquina (2× Xeon E5-2680 v4 com
AVX2, 2× AMD MI50 gfx906), que **acompanha a master do upstream**: cada commit relevante do
llama.cpp é portado para o módulo Rust equivalente.

O `~/llama.rs-MI50` não é a base. Ele serve só de referência (GGUF e tokenizer já bit-exact,
lições de Vulkan na gfx906), porque a estrutura dele não espelha o upstream e o sync depende
desse espelhamento.

## 2. Decisões

| # | Decisão | Escolha |
|---|---|---|
| D1 | Escopo | Subset do hardware: ggml core + CPU (x86) + Vulkan + HIP |
| D2 | Kernels de GPU | Reusados **verbatim** (`.comp`, `.cu`), compilados no `build.rs`; só o código host vira Rust |
| D3 | Projeto MI50 | Começar do zero; MI50 apenas como consulta |
| D4 | Representação do ggml | **Espelho idiomático**: mesma estrutura de arquivos e funções, dados idiomáticos (arena com `TensorId(u32)`, `buffer + offset` em vez de ponteiro cru) |
| D5 | Arquiteturas | `llama`, `qwen2`, `qwen3`, `qwen35` (híbrido atenção + gated delta-net) + MTP, multimodal (`mtmd`, `qwen3vl`) |
| D6 | Tools | `llama-cli`, `llama-completion`, `llama-server`, `llama-bench`, `llama-perplexity` |
| D7 | Modelos de teste | Em cada teste, o modelo mais rápido que cobre o caso (§5.2) |

### Fora do escopo

Backend CUDA para NVIDIA (os fontes de `ggml-cuda` entram só como `vendor` da ilha HIP), Metal, SYCL, OpenCL, WebGPU, Hexagon, CANN, MUSA, OpenVINO, zDNN,
RPC; caminhos de CPU ARM/RISC-V/PowerPC/LoongArch/s390/WASM, AMX e KleidiAI; as demais
arquiteturas de `src/models/`; `quantize`, `gguf-split`, `imatrix`, `tts` e demais tools; os
scripts Python de conversão. Tudo isso fica marcado como `ignore` em `sync/map.toml` (§4.1).

## 3. Arquitetura

### 3.1 Workspace

```
llama.rs/
├─ rust-toolchain.toml     # channel = "1.99.0"
├─ Cargo.toml              # workspace, edition 2024, resolver 3, lints
├─ UPSTREAM.toml           # SHA base, SHA sincronizado, status por crate
├─ sync/map.toml           # caminho upstream → port | vendor | ignore
├─ sync/pending/           # tarefas de sync geradas pelo xtask
├─ vendor/upstream/        # cópias verbatim, com o caminho original preservado
├─ crates/
│  ├─ ggml/                ← ggml/include, ggml.c, ggml-alloc.c, ggml-backend*.cpp, ggml-quants.c, gguf.cpp
│  ├─ ggml-cpu/            ← ggml/src/ggml-cpu (genérico + arch/x86)
│  ├─ ggml-vulkan/         ← ggml-vulkan.cpp (host em Rust); build.rs ← vulkan-shaders-gen.cpp
│  ├─ ggml-hip/            ← ilha C++: ggml-cuda + ggml-base do upstream, verbatim, atrás de ponte C
│  ├─ llama/               ← src/ (src/models/<arch>.cpp → src/models/<arch>.rs)
│  ├─ common/              ← common/
│  ├─ mtmd/                ← tools/mtmd (biblioteca)
│  ├─ llama-cli/ llama-completion/ llama-server/ llama-bench/ llama-perplexity/  ← tools/*
│  └─ oracle/              ← bindings (bindgen) para libggml/libllama do upstream; só dev-dependency
└─ xtask/                  ← cargo xtask {sync, check-map, oracle-build, ci}
```

### 3.2 Regras de espelhamento

1. Um arquivo C/C++ do upstream vira um módulo Rust com o nome em snake_case
   (`src/llama-vocab.cpp` → `crates/llama/src/llama_vocab.rs`). Um header que só declara
   tipos se funde ao módulo da implementação.
2. Cada módulo abre com `//! upstream: <caminho>`; cada função portada traz
   `/// upstream: <símbolo C++>`. O `xtask sync` usa esses marcadores para achar o destino de
   um diff.
3. As funções ficam na mesma ordem do arquivo original. Não há reorganização estrutural:
   o diff do upstream precisa cair no mesmo lugar do Rust.
4. Nomes: o prefixo vira módulo (`ggml_mul_mat` → `ggml::Context::mul_mat`;
   `llama_vocab::tokenize` → `llama_vocab::LlamaVocab::tokenize`).
5. Componentes que o upstream implementa à mão são **portados**, não trocados por crates
   equivalentes: motor jinja (`common/jinja`), regex de pré-tokenização (`src/unicode.cpp`),
   parsers de chat e PEG, json-schema→grammar. Isso preserva comportamento idêntico e mantém
   o diff do upstream aplicável.
6. Exceção documentada: o transporte HTTP do server. O upstream usa cpp-httplib; aqui o
   `server_http.rs` usa axum e preserva rotas, payloads e códigos de status. Esse módulo é o
   único sem espelhamento linha a linha.

### 3.3 Representação do ggml (D4)

- `Context` é uma arena: `tensors: Vec<Tensor>`, referenciados por `TensorId(u32)`.
- `Tensor { ty, ne: [i64; 4], nb: [usize; 4], op: Op, op_params, src: [Option<TensorId>; 10],
  view_src, view_offs, data: Option<BufferSlice>, flags, name }` — os mesmos campos de
  `struct ggml_tensor`, com `data` como `buffer + offset`.
- `Op` é um enum com as 105 variantes de `enum ggml_op`; `GgmlType` cobre os 36 tipos.
- Backends implementam traits que espelham `ggml-backend-impl.h` (`BackendBuffer`,
  `BackendBufferType`, `Backend`, `BackendDevice`, `BackendReg`).

### 3.4 Toolchain e dependências

- `rust-toolchain.toml` fixa a última estável (1.99.0 em 2026-10-01). O job semanal roda
  `rustup check` e sobe a versão quando há release.
- Dependências sempre na última versão: o job semanal roda `cargo upgrade --incompatible` e
  `cargo update`, e só faz merge com `cargo xtask ci` verde.
- Versões no início: `ash` 0.38.0, `memmap2` 0.9.11, `half` 2.7.1, `rayon` 1.12.0,
  `thiserror` 2.0.21, `serde_json` 1.0.151, `clap` 4.6.7, `axum` 0.8.9, `tokio` 1.53.1,
  `cc` 1.5.1, `bindgen` 0.73.2, `arrayvec` 0.7.8, `proptest` 1.11.0, `criterion` 0.8.2.
- Práticas de `docs/rust-praticas-da-documentacao.md` valem em todo o código: pré-alocar,
  não clonar o que pode ser emprestado, não fazer `collect` intermediário, `vec![0; n]`,
  `sort_unstable`, `BufWriter`/`BufReader`, perfil release com `lto = "thin"` e
  `codegen-units = 1`, `target-cpu=native` em `.cargo/config.toml`.

### 3.5 unsafe e lints

- Workspace: `unsafe_code = "deny"`.
- Liberação, com `clippy::undocumented_unsafe_blocks = "deny"` e um comentário
  `// SAFETY:` por bloco, só em: `ggml-cpu` (intrínsecos SIMD), `ggml-vulkan` (ash),
  `ggml-hip` (FFI), `oracle` (FFI) e o módulo de mmap do loader (`llama_mmap.rs`).
- Clippy: `unwrap_used`, `expect_used`, `panic` = deny; `cast_possible_truncation`,
  `cast_sign_loss`, `cast_possible_wrap` = warn (porte de C usa casts por toda parte);
  `indexing_slicing` = warn, permitido nos kernels de `ggml-cpu`.

## 4. Sync com a master

### 4.1 Estado

```toml
# UPSTREAM.toml
[upstream]
repo   = "https://github.com/ggml-org/llama.cpp"
cursor = "ec7630a640789c393694fb194f1bbbf0369fc62d"   # último commit com tarefas geradas
synced = "ec7630a640789c393694fb194f1bbbf0369fc62d"   # todas as tarefas até aqui concluídas

[crates.ggml]
baseline = "ec7630a640789c393694fb194f1bbbf0369fc62d"   # alvo do porte inicial desta crate
status   = "porting"                                    # porting | synced
```

Só crates registradas em `[crates]` recebem tarefas. Commits anteriores ao registro de uma
crate não geram tarefa para ela (§4.5).

`sync/map.toml` é uma lista ordenada de regras; vale a primeira que casar:

```toml
[[rule]]
path = "ggml/src/ggml-vulkan/vulkan-shaders/**"
action = "vendor"
crate = "ggml-vulkan"

[[rule]]
path = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
action = "port"
crate = "ggml-vulkan"

[[rule]]
path = "src/models/qwen35.cpp"
action = "port"
crate = "llama"

[[rule]]
path = "src/models/*.cpp"
action = "ignore"

[[rule]]
path = "ggml/src/ggml-metal/**"
action = "ignore"
```

### 4.2 `cargo xtask check-map`

Todo arquivo da árvore do upstream precisa casar exatamente uma regra. Arquivo sem regra faz
o comando falhar. Assim nada novo do upstream passa sem classificação.

### 4.3 `cargo xtask sync`

1. `git fetch` num clone dedicado em `.upstream/llama.cpp` (ignorado pelo git). O
   `~/llama.cpp` local não é tocado. O clone é **completo** (sem `--filter`): um clone
   parcial busca objeto por objeto sob demanda, e um clone parcial interrompido vira um laço
   de buscas que não termina. O clone é feito em `llama.cpp.tmp` e renomeado no fim.
2. Lista os commits `cursor..origin/master`, do mais antigo para o mais novo.
3. Classifica os arquivos tocados por cada commit. Um caminho sem regra **para** o sync
   nesse commit; o `cursor` guarda o progresso até o commit anterior.
   - `vendor` e `port` → uma tarefa `sync/pending/<seq>-<sha7>-<crate>.md` por crate
     registrada, com a mensagem do commit, as listas de arquivos, o diff dos arquivos a
     portar e os módulos Rust de destino (pelos marcadores `//! upstream:`);
   - `ignore` → pula.
4. Para cada tarefa, na ordem de `seq` dentro da crate:
   1. `cargo xtask task apply <tarefa>` copia os arquivos vendor **na versão daquele commit**.
      Copiar só na hora da tarefa mantém shader e código host da mesma versão.
   2. Portar o diff para os módulos Rust.
   3. `cargo xtask ci` verde.
   4. `cargo xtask task done <tarefa>` remove a tarefa (recusa se houver tarefa anterior da
      mesma crate pendente) e recalcula `synced`: o pai do commit da tarefa pendente mais
      antiga, ou o `cursor` quando não sobra nenhuma.
   5. Commit `sync(<crate>): <assunto do upstream> (upstream <sha7>)`.
5. `cargo xtask sync --dry-run` mostra a classificação por commit e por crate, sem gravar.

### 4.4 Ordem e paralelismo no sync

- Dentro de um crate, os commits entram em sequência.
- Crates diferentes são portados em paralelo (um agente por crate).
- Um commit que cruza crates segue a ordem
  `ggml → ggml-cpu | ggml-vulkan | ggml-hip → llama → common → mtmd → tools`. Um crate não
  passa de um commit que um crate abaixo dele ainda não absorveu.

### 4.5 Porte inicial vs. sync contínuo

- Uma crate nasce registrada em `UPSTREAM.toml` com `baseline` = o `cursor` daquele dia.
  Exceção: se uma crate de que ela depende ainda está em `porting`, herda o `baseline` dela.
- Uma crate em `porting` mira o próprio `baseline` fixo, nunca um alvo móvel. O oráculo é
  compilado nesse SHA.
- Enquanto ela porta, o `xtask sync` acumula as tarefas dela em `sync/pending/`. Ao terminar
  o porte inicial, ela processa essas tarefas (catch-up) e passa para `synced`.

### 4.6 Cadência e volume

- Medido nos 30 dias até 2026-10-01: 595 commits no upstream. **240 tocam o escopo** (~8 por
  dia) e 84 só mexem em shaders/kernels, que são cópia automática.
- No início o sync é manual. Quando o primeiro crate chegar a `synced`, um routine diário
  (`/schedule`) passa a rodar `xtask sync` e abrir um PR por crate.
- Toolchain e dependências sobem semanalmente no mesmo job (§3.4).

## 5. Testes e oráculo

### 5.1 Oráculo

- `cargo xtask oracle-build <sha>` compila o upstream nesse SHA (CPU + Vulkan) em
  `.upstream/build-<sha7>/`, com cache por SHA.
- A crate `oracle` expõe `libggml`/`libllama` via bindgen. Só entra como dev-dependency.
- A referência é sempre o upstream **no mesmo SHA** do crate testado.

### 5.2 Camadas

| # | Camada | Comparação | Critério | Insumo |
|---|---|---|---|---|
| 1 | Unidade | os `tests/test-*.cpp` do upstream que cobrem o escopo (de 51 no total), portados para o crate dono | o mesmo do teste C++ | testes do upstream (também sincronizados) |
| 2 | Tokenizer | `models/ggml-vocab-{llama-spm,llama-bpe,qwen2,qwen35}.gguf.inp/.out` | bit-exact | fixtures do upstream |
| 3 | Ops ggml | `test-backend-ops` portado: op × tipo × shape; Rust-CPU e Rust-Vulkan contra upstream-CPU | NMSE com os limiares do upstream; quantize/dequantize bit-exact | 105 ops, 36 tipos |
| 4 | Arquitetura | logits por posição com teacher forcing | top-1 igual em todas as posições + NMSE | modelos aleatórios de 2–6 camadas, gerados como em `tests/test-llama-archs.cpp` |
| 5 | Modelo real | logits com teacher forcing + 64 tokens greedy | top-1 igual em ≥ 99% das posições | menor modelo real por arch (abaixo) |
| 6 | Perplexity | `llama-perplexity` Rust vs. upstream, wikitext-2, 20 primeiros chunks | diferença de PPL < 0,1% | — |
| 7 | Server | suíte pytest do upstream (`tools/server/tests`) contra o binário Rust (`server_path`) | os testes do escopo passam | suíte do upstream, vendor |
| 8 | Desempenho | `llama-bench` pp512/tg128, Vulkan nas 2× MI50 | ≥ 95% do upstream | modelos locais de 27B/32B |

Modelos da camada 5:

| Arch | Modelo | Origem |
|---|---|---|
| llama | `stories260K.gguf` | local |
| qwen2 | `qwen2.5-0.5b-instruct-q8_0.gguf` | local |
| qwen3 | `Qwen3-0.6B` Q8_0 | download (~600 MB) |
| qwen35 + MTP | o menor Qwen3.5/3.8 com MTP no Hugging Face; sem um pequeno, o 27B local só no noturno | a verificar no M4 |
| mtmd | mmproj do menor Qwen-VL disponível | a verificar no M5 |

### 5.3 Gatilhos

- `cargo test`: camadas 1–4 (segundos, sem download).
- `cargo xtask ci`: camadas 1–7. É o portão de toda tarefa de porte e de sync.
- Noturno local: camada 8 e modelos grandes (precisa das GPUs).

## 6. Paralelismo e marcos

### 6.1 Contratos primeiro

Antes do fan-out, os headers viram tipos e traits Rust **completos**: `ggml.h` (o enum `Op`
com 105 variantes, `GgmlType` com 36 tipos, `Tensor`), `ggml-common.h` (structs de bloco),
`ggml-backend-impl.h` (traits de backend) e `llama.h` (API pública). As trilhas paralelas
implementam contra esses contratos. Mudar um contrato depois do fan-out exige parar as
trilhas afetadas.

### 6.2 Caminho crítico (sequencial)

```
M0 scaffold + xtask + oráculo
 → contratos (headers)
 → ggml.c núcleo (contexto, views, construtores de op, grafo)
 → ggml-backend + ggml-alloc + scheduler
 → esqueleto do backend CPU + ops mínimas do llama
   (get_rows, mul_mat, rms_norm, rope, soft_max, add, mul, silu, cpy, cont, view/permute)
 → llama core (loader, model, graph, kv-cache, context)
 → models/llama.rs
 → M1
```

### 6.3 Trilhas paralelas

Cada trilha tem um agente e um worktree próprios, e é dona de crates/módulos disjuntos.

| Começa | Trilha | Conteúdo |
|---|---|---|
| após M0 | B — quants | quantize/dequantize de referência e `vec_dot` AVX2, por família (legacy, K, IQ, MXFP4) |
| após M0 | C — gguf/tokenizer | `gguf.cpp`, `unicode.cpp`, `llama-vocab.cpp` (SPM, BPE, pré-tokenizers qwen2/qwen35) |
| após M0 | D — texto | jinja, json-schema→grammar, peg-parser, `llama-grammar`, lógica do `llama-sampler` |
| após M0 | E1 — shaders Vulkan | `build.rs` que porta `vulkan-shaders-gen.cpp` e compila os 188 `.comp` com `glslc` |
| após contratos | F — ops CPU | as 105 ops em largura, por família (2–3 agentes) |
| após contratos | E2 — Vulkan host | device, memória, pipelines, depois despacho de ops por família |
| após M1 | G — archs | qwen2 → qwen3 → qwen35 (memória híbrida + recorrente, gated delta-net) → MTP |
| após M1 | H — tools | cli/completion → perplexity + bench → server |
| por último | I — mtmd e HIP | `mtmd` + `qwen3vl`; depois a ilha HIP (spec próprio antes de começar) |

Regras:

1. Arquivos compartilhados (`Cargo.toml` do workspace, `UPSTREAM.toml`, `sync/map.toml`) só
   mudam na etapa de integração, um agente por vez.
2. Uma trilha só faz merge com `cargo xtask ci` verde.
3. A partir do M1, cada crate concluído passa para `synced` e entra no sync diário.

### 6.4 Marcos

Estimativas com 3–5 agentes em paralelo. A incerteza é alta; reavaliar depois do M1.

| Marco | Entrega | Estimativa |
|---|---|---|
| M0 | workspace, Rust 1.99, `xtask sync`/`check-map`/`oracle-build`, `map.toml` cobrindo toda a árvore | 1–2 dias |
| M1 | `stories260K` gera saída idêntica ao upstream na CPU | 1–2 semanas |
| M2 | qwen2.5-0.5b e Qwen3-0.6B na CPU (Q8_0, Q4_K, Q5_K, Q6_K); cli/completion | 3–4 semanas |
| M3 | Vulkan nas MI50 com todos os modelos do escopo, ≥ 95% do upstream | 5–7 semanas |
| M4 | qwen35 + MTP; server com a suíte pytest passando; perplexity; bench | 8–10 semanas |
| M5 | mtmd (multimodal) | 11–12 semanas |
| M6 | ilha HIP | 13+ semanas |

Custo do sync com tudo em `synced`: ~8 commits/dia no escopo, estimado em 1–2 h de agente
por dia.

## 7. Riscos

| Risco | Mitigação |
|---|---|
| Volume: ~220 mil linhas de C/C++ no escopo | Marcos pequenos; reavaliar as estimativas após o M1 |
| O upstream muda contratos (`ggml/include`: 10 commits em 30 dias) durante o porte inicial | O porte inicial mira o `baseline` fixo; mudanças de contrato entram no catch-up, na ordem de §4.4 |
| Divergência de ponto flutuante (ordem de SIMD/threads) | Critérios por NMSE e top-1, como o próprio upstream faz; bit-exact só onde o upstream é determinístico (quantização, tokenizer) |
| `ash` 0.38 cobre Vulkan 1.3.281; o `ggml-vulkan` pode usar extensões mais novas | Verificar no início da trilha E2; se faltar extensão, carregar os ponteiros de função à mão ou usar `ash` do git |
| A ilha HIP exige structs C compatíveis com o upstream | Isolada no M6, com spec próprio; não afeta o núcleo idiomático |
| ROCm atual com suporte limitado à gfx906 | O Vulkan cobre as MI50 desde o M3; o HIP é incremental |

## 8. Decisões pendentes para o plano do M1

Levantadas na revisão final do M0 (2026-10-01). O xtask do M0 não as trata; o plano do M1
precisa resolvê-las **antes** de registrar a primeira crate em `UPSTREAM.toml`.

1. **`baseline` e `status` não são lidos pelo `sync`.** O sync só checa se a crate está
   registrada. Um `baseline` herdado mais antigo que o `cursor` (§4.5) perde as tarefas do
   intervalo; um mais novo gera tarefas já cobertas pelo porte inicial. Opções: um comando
   `cargo xtask register <crate>` que garante `baseline == cursor` (ou gera as tarefas que
   faltam), ou um cursor por crate.
2. **A ordem entre crates de §4.4 não é imposta.** `task done` só exige a ordem dentro da
   crate, e o `seq` de um mesmo commit segue a ordem alfabética das crates, não a de
   dependência. Precisa existir antes do primeiro catch-up.
3. **Oráculo, `synced` e trabalho em paralelo.**
   - `oracle-build` usa o `synced` global por padrão, mas uma crate em `porting` é testada no
     próprio `baseline` (§4.5, §5.1).
   - Todo `task done` reescreve a linha única `synced`, e PRs paralelos por crate (§4.4, §6.3)
     conflitam nela.
   - Cada worktree clona o upstream de novo em `.upstream/`. Direção sugerida: `synced` por
     crate calculado sob demanda, um clone compartilhado entre worktrees e trava no
     `oracle-build`.

Também em aberto, de menor prioridade:

- A regra `tools/**` → `ignore` do `map.toml` engole um arquivo novo dentro de um diretório
  de tool do escopo. O ideal é trocar por listas explícitas por diretório.
- A cópia vendor inicial de uma crate no próprio `baseline` (por exemplo os
  `models/ggml-vocab-*` da crate `llama`) não tem comando.
- `UPSTREAM.toml` e as tarefas são gravados com `fs::write`. Gravar num temporário e renomear
  tornaria a escrita atômica.
