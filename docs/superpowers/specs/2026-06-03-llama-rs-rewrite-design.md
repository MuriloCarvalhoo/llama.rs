# Design: Reescrita do llama.cpp em Rust (llama-rs)

**Data:** 2026-06-03
**Status:** Aprovado para planejamento de implementação (Fase 0)

## Objetivo

Reescrever o llama.cpp em Rust com paridade funcional de features, **restrita ao hardware desta máquina**. Execução por fatias verticais com validação diferencial contínua contra o llama.cpp C++ original (oráculo).

## Escopo de hardware (decisão deliberada para poupar esforço)

| Alvo | Suportado | Excluído |
|---|---|---|
| CPU x86_64 | Kernels SIMD **AVX2 + FMA + F16C** + fallback portátil safe (referência para testes/miri) | NEON/ARM, AVX-512, AMX, RISC-V, WASM, demais ISAs |
| GPU | **Vulkan via wgpu**, alvo 2× AMD Vega 20 (Radeon Pro VII/MI50, gfx906, RADV) | CUDA (2× Tesla K80 ignoradas — Kepler legado, mais lentas que as MI50), Metal, SYCL, HIP, OpenCL, CANN e demais backends |

O design de backend via trait deixa a porta aberta para backends futuros, mas nenhum além de CPU e Vulkan está no roadmap.

## Políticas

- **`unsafe` pragmático:** permitido apenas em `ggml-cpu` (e `ggml-vulkan` se necessário). Todo bloco `unsafe` exige comentário `// SAFETY:` com a invariante, teste sob miri e teste de equivalência contra a versão safe de referência. Demais crates: `#![forbid(unsafe_code)]`. API pública 100% safe.
- **Dependências:** crates maduros permitidos (`memmap2`, `rayon`, `half`, `zerocopy`/`bytemuck`, `thiserror`, `wgpu`, `proptest`, `criterion`). Verificados com `cargo deny`.
- **Oráculo somente-leitura:** o diretório `llama.cpp/` upstream nunca é modificado; serve para gerar saídas de referência.

## Arquitetura (workspace Cargo)

```
llama-rs/
├── crates/
│   ├── gguf/            # Parser GGUF (zero unsafe, zerocopy/memmap2)
│   ├── ggml-core/       # Tensor, Shape, DType, formatos quant, grafo de computação
│   ├── ggml-cpu/        # Kernels CPU — ÚNICO crate com unsafe SIMD (AVX2+FMA+F16C, rayon)
│   ├── ggml-vulkan/     # Backend wgpu (Fase 6)
│   ├── llama-tokenizer/ # BPE / SPM / UGM
│   ├── llama-model/     # Arquiteturas, hparams, carregamento de pesos
│   ├── llama-context/   # KV-cache, batch, execução do grafo
│   ├── llama-sampling/  # Greedy, top-k/p, temperatura, penalties, grammar (fase tardia)
│   ├── llama-cli/       # Binário equivalente ao llama-cli
│   └── llama-server/    # Fase tardia
├── oracle/              # Harness diferencial: roda o C++ e compara tokens/tensors/logits
└── llama.cpp/           # Upstream — SOMENTE LEITURA (oráculo)
```

Princípios estruturais:

- Backends implementam um trait comum (`Backend`); ops do grafo são **enum fechado**, não `dyn Trait`.
- Ownership real: pesos imutáveis (`Arc<[u8]>` sobre mmap); estado mutável (KV-cache) com dono único no `Context`. Sem `Rc<RefCell>` simulando ponteiros C++.
- A estrutura Rust espelha **responsabilidades**, não os arquivos C++.

## Fases (cada fase = ciclo próprio spec → plano → implementação)

| Fase | Entrega | Critério de aceite (vs oráculo) |
|---|---|---|
| 0 | Infra: workspace, CI, lints, harness oráculo, modelo de teste (tiny Llama/Qwen ~0.5B) | Harness roda o C++ e captura tokens/logits/tensors de referência |
| 1 | Parser GGUF + tokenizer | Metadados idênticos; tokens bit-exact num corpus de teste |
| 2 | Forward pass CPU f32, 1 arquitetura (Llama) | Logits ≤ 1e-4 de tolerância, camada por camada |
| 3 | Quantização: Q8_0, Q4_0, Q4_K, Q6_K, ... | Dequant bit-exact; perplexity igual à do C++ |
| 4 | Sampling + geração ponta-a-ponta (`llama-cli`) | Greedy: sequência de tokens idêntica ao C++ |
| 5 | KV-cache completo, batching, multi-thread | Mesmos resultados com batch>1; throughput ≥ 70% do C++ CPU |
| 6 | Backend Vulkan/wgpu (alvo MI50/gfx906) | Logits CPU ≡ GPU; speedup mensurável |
| 7 | Mais arquiteturas (Qwen, Mistral, Gemma, Phi, ...) | Cada arch validada contra oráculo |
| 8+ | Server, grammar, LoRA, embeddings, multimodal | Paridade incremental de features |

A partir da Fase 4 existe um `llama-cli` Rust gerando texto real; o restante é alargamento com a mesma máquina de validação.

## Skills por tipo de tarefa

| Momento | Skill/Ferramenta |
|---|---|
| Início de cada fase | `superpowers:brainstorming` → `superpowers:writing-plans` |
| Execução do plano | `superpowers:subagent-driven-development` + Workflow (ultracode): fan-out de agentes por op/kernel/módulo, verificação adversarial |
| Isolamento | `superpowers:using-git-worktrees`; agentes paralelos com `isolation: worktree` |
| Cada tarefa de código | `superpowers:test-driven-development` + `rust-test` (TDD obrigatório) |
| Erro de build/borrow checker | `rust-build` |
| Bug / divergência do oráculo | `superpowers:systematic-debugging` (bissecção camada por camada) |
| Consulta de idioma Rust | `ecc:rust-patterns`, `ecc:rust-testing` |
| Fim de cada tarefa | `rust-review` + `/code-review` (bloqueia CRITICAL/HIGH) |
| Antes de declarar concluído | `superpowers:verification-before-completion` |
| Performance (Fases 5+) | `ecc:benchmark` / `ecc:latency-critical-systems` + criterion |
| Fim de fase | `superpowers:finishing-a-development-branch` + `/pr` |
| Automação | `/hookify`: `cargo fmt` + `clippy` PostToolUse em edits de `.rs` |

**Não usar:** `cpp-build`/`cpp-test`/`cpp-review` (oráculo é somente-leitura); skills web/frontend/vercel/supabase; skills `multi-*` (Workflow já cobre orquestração).

## Gate de validação por tarefa (checklist fixo)

Uma tarefa só fecha quando tudo passa, nesta ordem:

1. TDD respeitado — teste existia e falhava antes da implementação
2. `cargo fmt --check` e `cargo clippy --all-targets -- -D warnings` limpos
3. `cargo test` do workspace verde
4. Teste diferencial vs oráculo dentro do critério da fase
5. Cobertura ≥ 80% no crate tocado (`cargo llvm-cov`)
6. Se tocou `unsafe`: `cargo +nightly miri test` + equivalência kernel SIMD vs versão safe
7. Se é parser/quant: proptest (round-trip + entradas malformadas — GGUF é entrada não-confiável)
8. Se é hot path: criterion sem regressão > 5% vs baseline
9. `rust-review` sem issues CRITICAL/HIGH
10. `cargo deny check` quando dependências mudarem

Itens 2, 3 e 5 são automatizados em CI + hooks. No ultracode, cada agente implementador recebe o gate no prompt; um verificador adversarial reexecuta os passos 2–4 independentemente.

## Más práticas banidas

### Por lint (automático, `[workspace.lints]`)

- `unsafe_code = "forbid"` em todos os crates exceto `ggml-cpu`/`ggml-vulkan`
- `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic` = deny em código de biblioteca (ok em testes/exemplos) → erros via `thiserror` + `Result`
- `clippy::cast_possible_truncation` (e família) = deny → `try_from`, nunca `as` para estreitar
- `clippy::indexing_slicing` = warn em código de lib → preferir `get()`/iterators

### Por convenção (rust-review verifica)

- Transliterar C++: `Rc<RefCell<T>>`/`Arc<Mutex<T>>` simulando ponteiros nus → redesenhar ownership
- `static mut` / estado global mutável
- `unsafe` sem `// SAFETY:` → reprovação automática
- `transmute` para reinterpretar bytes → `bytemuck`/`zerocopy`
- `.clone()` para calar o borrow checker em hot path
- `let _ =` ignorando erro silenciosamente
- `Box<dyn Trait>` onde enum fecha o conjunto (ops = enum; backends = trait)
- Generics/lifetimes especulativos (YAGNI)
- Alocação dentro do loop de inferência → scratch buffers pré-alocados no `Context`
- `#[allow(...)]` para silenciar lint sem justificativa no código
- `panic!`/`unwrap` em caminho de entrada não-confiável (GGUF malformado retorna erro, nunca aborta)

## Riscos

- **Escala:** mesmo com escopo de hardware reduzido, é um projeto de meses. Mitigação: fatias verticais — sempre há um artefato executável e validado.
- **Divergência numérica:** ordem de operações em float difere entre implementações. Mitigação: tolerâncias definidas por fase; comparação camada por camada para localizar divergências.
- **Upstream móvel:** o llama.cpp evolui. Mitigação: pinar o commit do oráculo; paridade é contra esse pin.
- **Vulkan em gfx906:** cobertura de drivers RADV para compute é boa, mas wgpu impõe limites (ex.: sem subgroup ops em algumas versões). Mitigação: spike técnico no início da Fase 6; fallback para `ash` (Vulkan puro) se wgpu limitar demais.
