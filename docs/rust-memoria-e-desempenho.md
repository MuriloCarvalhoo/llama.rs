# Rust: memória e desempenho para runtime de inferência

**Data:** 2026-08-13 · **Escopo:** llama-rs (Vulkan/`ash`, 2× MI50, modelos GGUF de 0.5B a 30B)

Documento derivado de pesquisa com fontes. Onde não há medição publicada, está dito explicitamente
— nenhum número aqui é estimado sem rótulo.

---

## Resumo executivo

| Pergunta | Resposta curta |
|---|---|
| async/await ajuda? | **Não.** A espera real é `vkWaitForFences`, que é bloqueante de driver. |
| Trocar o alocador global (jemalloc/mimalloc)? | **Não muda nada** no nosso padrão (alocações grandes e longevas). |
| `parking_lot` em vez de `std::sync::Mutex`? | **Não.** `std` é futex-based desde 1.62 e nossa contenção é ~zero. |
| `crossbeam` para canais? | **Já temos.** `std::sync::mpsc` **é** crossbeam vendorizado desde Rust 1.67. |
| `std::simd`? | **Indisponível** (nightly). Usar `std::arch` (safe desde 1.87) se o profiler pedir. |
| O que dá mais desempenho por esforço? | Eliminar `queue_wait_idle`; `[profile.release]` com LTO. |
| O que dá mais economia de memória? | Não copiar os pesos — emprestar do mmap. |

---

## 1. Memória: o que realmente custa

### 1.1 A hierarquia de custos (medida neste projeto)

Carregando Qwen2.5-14B Q8_0 (14.6 GiB) com `--gpu-resident`, **antes** das correções desta sessão:

| Fonte | Custo | Necessário no caminho GPU? |
|---|---|---|
| `fs::read` do `.gguf` | 15 GB anônimo | Não — mmap resolve |
| `Weights::from_gguf` → `raw.to_vec()` por tensor | ~14.6 GB anônimo | **Não** — o forward roda na GPU |
| `repack_q8_0_8rows` (segunda cópia antes de descartar a primeira) | +1 tensor de pico | **Não** |
| `token_embd.dequant_to_f32()` (cache f32 = 4× o Q8_0) | 3.11 GB | Só 1 linha por token |
| `GpuRawWeights` → `raw.to_vec()` | 14.6 GB | Não — a GPU lê do mmap |

Resultado: OOM kill em máquina de 62 GB **sem swap**. Todas as fontes acima foram corrigidas
(ver `PROGRESS.md`), e o 14B passou a carregar e gerar texto.

**Lição transferível:** num pipeline de inferência, o inimigo não é o tamanho do modelo — é o
**número de imagens do modelo** que existem simultaneamente na RAM.

### 1.2 mmap: como usar direito

`memmap2` (atual **0.9.11**, MSRV 1.65) é a escolha certa, mas a API que usamos (`Mmap::map`) é a
mais pobre. O que está na mesa:

| Recurso | Para que serve aqui |
|---|---|
| `MmapOptions::populate()` (`MAP_POPULATE`) | Prefault das page tables; evita tempestade de page fault no load |
| `Mmap::advise(Advice::Sequential)` | O kernel lê à frente **e** libera as páginas logo após o acesso — exatamente o padrão "leio tudo uma vez para subir à VRAM" |
| `Advice::WillNeed` | Prefetch assíncrono |
| `posix_fadvise(POSIX_FADV_DONTNEED)` | **A única forma** de devolver page cache de arquivo |

Armadilha documentada: **`MADV_DONTNEED` não libera page cache de arquivo** — em mapeamentos
compartilhados ele só derruba as PTEs do processo ([madvise(2)](https://man7.org/linux/man-pages/man2/madvise.2.html)).
Para liberar de verdade é `posix_fadvise`, com offset/len alinhados a página
([posix_fadvise(2)](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)).

Números publicados de `MAP_POPULATE` (leitura sequencial de 4 GB, discussão de kernel — ordem de
grandeza, não benchmark moderno): `read()` 1.06 s · mmap 1.42 s · mmap+POPULATE 1.02 s
([LKML](https://lkml.iu.edu/hypermail/linux/kernel/1505.1/04126.html)). Ele ajuda quando se toca
tudo, uma vez — nosso caso. **Não há medição publicada de mmap vs read para GGUF de 10-30 GB.**

**Risco de UB:** todo construtor file-backed do memmap2 é `unsafe` porque modificação externa do
arquivo enquanto mapeado é UB. Não é teórico: **RUSTSEC-2025-0132** foi emitido contra `maxminddb`
por expor `open_mmap` como safe. Nossa premissa (modelo imutável durante a execução) é a mesma do
llama.cpp e está documentada no `SAFETY` de `runner.rs`.

### 1.3 Ownership sem cópia: o padrão idiomático em 2026

O problema clássico: um struct quer guardar o `Mmap` **e** slices apontando para dentro dele.

**O que está morto:**
- `owning_ref` — **RUSTSEC-2022-0040**, unsound, sem correção, mantenedor ausente.
- `ouroboros` ≤ 0.15 — **RUSTSEC-2023-0042** (corrigido em 0.16/0.18, mas é proc-macro pesada).

**O que usar:**

| Opção | Quando |
|---|---|
| **Lifetime explícito** (`Struct<'a>` com `&'a [u8]`) | Quando o dono do buffer vive no mesmo escopo. **É o que fizemos** — `RawTensor<'a>` → `Weights<'a>` → `Model<'a>` |
| **`bytes::Bytes`** (1.12.1) | Quando o lifetime "contamina" API demais. `Bytes::from_owner(mmap)` (desde 1.9.0) + `slice_ref()` dá slices **owned** em O(1), sem cópia e sem lifetime |
| `yoke` (0.8.3, ICU4X) / `self_cell` (1.3.0) | Estruturas desserializadas complexas sobre buffer emprestado |

**Armadilha crítica: `Arc<[u8]>::from(vec)` COPIA.** O refcount de um `Arc<[T]>` fica *antes* dos
dados na mesma alocação; um `Vec` não tem esse espaço, então a conversão realoca e move tudo. Para
15 GB é fatal. E `Arc<[u8]>` não suporta subslicing — `bytes::Bytes` existe exatamente para isso.

> **Decisão neste projeto:** lifetime explícito. `Bytes::from_owner` seria a alternativa se o `'a`
> começasse a contaminar demais a API — vale reavaliar se o refactor de `Model<'a>` incomodar.

### 1.4 Zero-copy: `zerocopy` vs `bytemuck`

| | `zerocopy` 0.8.56 (Google) | `bytemuck` 1.25.2 |
|---|---|---|
| Traits | Granulares (`FromBytes`, `TryFromBytes`, `IntoBytes`, `Unaligned`) | Um só (`Pod`) |
| Desalinhamento | `Err` explícito; `Unaligned` permite modelar wire-format | panic/erro em runtime |

**Recomendação para o crate `gguf`:** `zerocopy`. Blocos Q8_0 têm 34 bytes (`f16` + 32×`i8`) em
offsets arbitrários — alinhamento não garantido. `#[derive(FromBytes, KnownLayout, Immutable,
Unaligned)]` + `ref_from_bytes` elimina `unsafe`, o que casa com o `#![forbid(unsafe_code)]` do
crate.

### 1.5 Alocadores: por que não importam aqui

A medição mais confiável ([*Battle of the Mallocators*, 2025](http://smalldatum.blogspot.com/2025/04/battle-of-mallocators.html)):

| Workload | glibc | tcmalloc | jemalloc |
|---|---|---|---|
| MyRocks (churn de tamanhos variados), pool 10 GB | 36.2 GB RSS | 13.1 GB | 12.2 GB |
| **InnoDB (bloco gigante, reusado), pool 80 GB** | **86.5 GB** | **85.3 GB** | **87.0 GB** |

**A linha do InnoDB é a nossa**: alocar grande uma vez e reusar. Diferença entre alocadores:
ruído. Alocações > 128 KiB vão direto para `mmap` anônimo no glibc e voltam ao kernel no `free`,
sem passar pelo heap.

Onde *poderia* importar: churn de buffers temporários por token (ex.: `quantize_q8_0_split` aloca
um `Vec<u8>` por chamada). Aí o knob real é o **decay do jemalloc**
(`dirty_decay_ms`/`muzzy_decay_ms`), não a escolha do alocador. Melhor ainda: buffers pré-alocados
no contexto — que o `ResidentForward` já faz.

**Não há medição publicada de alocador em inferência LLM em Rust.**

### 1.6 Sobreviver sem swap

Fatos que mudam o desenho:

1. **Não existe "alocar defensivamente e checar erro" no Linux.** Com overcommit, o kernel não sabe
   no `mmap()` se haverá memória; ele só descobre no page fault. `Vec::try_reserve` retornando
   `Err` sob pressão é ilusão para alocações grandes.
2. **`MAP_NORESERVE` é ignorado** em `vm.overcommit_memory=2`, e irrelevante para mapa read-only de
   arquivo ([kernel.org](https://www.kernel.org/doc/html/v5.1/vm/overcommit-accounting.html)).
3. **Sem swap, páginas anônimas não têm fallback.** File-backed limpas são descartáveis; anônimas
   não. Por isso mmap > `Vec` do mesmo tamanho.

Defesas que funcionam, em ordem:
1. Não alocar anônimo em GB (§1.1).
2. Preferir memória file-backed.
3. Upload em chunks + `posix_fadvise(DONTNEED)` após cada fence.
4. **Checagem pré-voo**: ler `MemAvailable` de `/proc/meminfo` e falhar com mensagem clara em vez de
   deixar o OOM killer decidir. É a única forma de o processo controlar o desfecho.

### 1.7 NUMA

- **`MPOL_BIND`** restringe ao nodemask: se o nó enche, **OOM mesmo com RAM livre nos outros**.
  Foi exatamente o que matou o 14B aqui.
- **`MPOL_PREFERRED`** tenta o nó e cai para os outros — a política correta quando o buffer é
  comparável à RAM de um nó.
- **`MPOL_INTERLEAVE`** maximiza banda agregada quando threads de vários nós leem tudo.

| Cenário | Política |
|---|---|
| Caminho GPU (pesos vão para VRAM) | **Nenhuma** ← corrigido nesta sessão |
| CPU, modelo cabe folgado num nó | `MPOL_BIND` |
| CPU, modelo ≈ RAM de um nó | `MPOL_PREFERRED` |
| CPU, threads em todos os sockets | `MPOL_INTERLEAVE` |

Detalhe não considerado hoje: no caminho GPU o nó que importa é o **mais próximo do PCIe root
complex da GPU** (`/sys/class/drm/cardN/device/numa_node`), não o nó 0.

Crate mantida: **`hwlocality`** (as bindings `hwloc2`/`hwloc-rs` estão mortas). Mas 3 syscalls via
`libc` bastam — e são mais fáceis de manter que o `asm!` inline com números de syscall hardcoded
que temos hoje em `runner.rs`.

---

## 2. Concorrência: async, threads e sincronização

### 2.1 async/await: não

O overhead do runtime **não** é o argumento — medição de 10M iterações
([alamb/rust_tokio_overhead](https://github.com/alamb/rust_tokio_overhead)): 401 ns não-async ·
412 ns async single-thread · 449 ns async multi-thread. Overhead por task: dezenas de ns.

Os argumentos reais:

1. **A espera é bloqueante por natureza.** `vkWaitForFences`/`vkWaitSemaphores` são chamadas de
   driver. Para usá-las em async você precisa de `spawn_blocking` (criou uma thread, com um hop de
   scheduler a mais) ou polling com `vkGetFenceStatus` (reimplementou spin-wait, pior).
2. **O caso do wgpu prova o oposto do que parece.** O PR
   [gfx-rs/wgpu#2698](https://github.com/gfx-rs/wgpu/pull/2698) **converteu `map_async` de async
   para callback**, com a justificativa: *"async functions imply a reactor exists to make futures
   resolve, which is not the case in wgpu"*. E `PollType::Wait` **bloqueia** no backend nativo —
   *"On WebGPU, the `Wait` variant has no effect; callbacks are invoked from the window event
   loop"*. **async no wgpu é imposto do navegador, não otimização de GPU.**
3. **Starvation.** O próprio InfluxData documenta que usar Tokio para CPU-bound aumenta latência de
   cauda a ponto de health checks matarem o processo.

**Não há nenhum benchmark publicado comparando async vs threads para submissão Vulkan.**

### 2.2 rayon: o que ele faz quando ocioso

De `rayon-core/src/sleep/mod.rs`: `ROUNDS_UNTIL_SLEEPY = 32`. Um worker ocioso faz 32 rodadas de
busca (varrendo todas as deques + injector queue) com `thread::yield_now()` no meio, antes de
dormir em futex. Não é spin puro nem park puro.

Consequência para `dual_gpu.rs`: **`rayon::join` chamado de fora do pool bloqueia a thread
chamadora** e passa pela injector queue, possivelmente acordando um worker adormecido — por camada,
por token.

Medições reais ([Endignoux, 2024](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html)):
- `strace`: futex dominando **98.8%** do tempo de syscall; threads ociosas em `sched_yield()`.
- **Pinning de CPU rendeu 10–20% com 8 threads.**
- Work-stealing não desce à granularidade de item — com input desbalanceado, uma thread fica para
  trás enquanto as outras giram fazendo syscalls.

> **Correção de uma premissa herdada dos docs deste projeto:** a ideia de que o wake-up de worker
> custa ~1 ms **não se sustenta**. A ordem de grandeza indicada pelas fontes é **dezenas de µs**
> (`pthread_cond_timedwait(5 µs)` → ~60 µs reais). Num orçamento de 25 ms/token isso é 0.24%.
> O custo que importa é o `queue_wait_idle`, não o wake de thread.

Para 2 GPUs, o padrão certo é **uma thread persistente por GPU** (`std::thread::scope`, estável
desde 1.63) — o que também satisfaz a regra Vulkan de "uma command pool por thread".

### 2.3 Sincronização: o que mudou no `std`

Dois fatos que invalidam muito conselho antigo da web:

1. **`std::sync::Mutex` é futex-based desde [rust#95035](https://github.com/rust-lang/rust/pull/95035)**
   (3 estados, spin de ~100 iterações antes do syscall). As claims históricas de "parking_lot é 1.5–5×
   mais rápido" são **pré-futex**. O diferencial remanescente do `parking_lot` é *fairness* sob
   contenção alta com holds longos — não é o nosso caso.
2. **`std::sync::mpsc` É o crossbeam-channel vendorizado desde Rust 1.67**
   ([rust#93563](https://github.com/rust-lang/rust/pull/93563)). Qualquer texto dizendo "crossbeam é
   2–10× mais rápido que std mpsc" é obsoleto.

Ainda justificam `crossbeam`: MPMC, `select!`, `crossbeam-deque`, `Backoff`, `CachePadded`.

`CachePadded` usa **128 bytes** em x86-64 (não 64) porque desde Sandy Bridge o spatial prefetcher
carrega cache lines em pares. A própria doc avisa que são palpites, não garantia.

### 2.4 Vulkan: onde está o desperdício

Regras com número medido:

- **`vkDeviceWaitIdle` → fences: 72 ms → 56 ms de frame time (22%)** no sample oficial da Khronos
  ([wait_idle sample](https://docs.vulkan.org/samples/latest/samples/performance/wait_idle/README.html)).
  Motivo: WaitIdle drena o pipeline inteiro, criando bolhas.
- **Submissões são caras**: *"queue submission is one of the most CPU overhead-incurring actions in
  a Vulkan driver"*. llama.cpp faz batch de até 100 nós por submit.
- **`vkQueueSubmit` não é thread-safe por queue**, mas queues de **devices diferentes são
  independentes** — 2 threads submetendo para 2 GPUs não colidem.
- **Timeline semaphores** (core em 1.2, suportado em AMD) são superset de fence+semaphore, com
  granularidade mais fina e sem exigir sincronização externa do objeto.

> **Aplicação direta:** este projeto tem `queue_wait_idle` em quase todos os caminhos de submissão
> (`matmul.rs`, `resident.rs`, `resident_forward.rs`). É o anti-padrão exato que a Khronos mede
> como 22% pior. Só o laço de token já usa `wait_for_fences`.

### 2.5 SIMD

**`std::simd` continua nightly em 2026** — a meta de estabilização não foi renovada para 2026
([project goals, abr/2026](https://blog.rust-lang.org/2026/05/18/project-goals-2026-04/)). O
toolchain aqui está pinado em **stable 1.96.0**, então está fora.

Novidade relevante: **desde Rust 1.87 a maioria dos intrinsics de `std::arch` é chamável de código
safe** quando as target features estão habilitadas em tempo de compilação. Como o projeto já compila
com `target-cpu=native` e tem `unsafe_code = "deny"`, isso dá SIMD explícito sem furar a política.

Alternativas estáveis: `wide` (maduro, sem multiversioning), `pulp` (multiversioning + dispatch
runtime, base do `faer`). **Não há benchmark publicado comparando pulp/wide/autovetorização.**

**Prioridade: baixa** — o trabalho pesado está na GPU.

### 2.6 Flags de build

Estado atual: `target-cpu=native` ✅, mas **nenhum `[profile.release]`** → sem LTO,
`codegen-units=16`, `panic=unwind`.

| Flag | Ganho medido | Fonte |
|---|---|---|
| LTO | 10–20% (Perf Book); **-3.74% wall-time médio, até 10%** em crates reais | [rust#159149](https://github.com/rust-lang/rust/pull/159149) |
| `codegen-units=1` + `lto="fat"` | 5–10% | agregado |
| PGO | 10%+ (Perf Book); **~15% em código não-otimizado, ZERO após corrigir hotspots** | [alphakhaw](https://alphakhaw.com/blog/seqpacker-profiling-rust-flamegraph-pgo-bolt) |
| BOLT | ~2–5%; **nenhuma melhora além de PGO** num caso real | idem |

**Leitura honesta:** LTO + `codegen-units=1` é ganho barato e confiável. **PGO/BOLT não são** — o
caso do alphakhaw é instrutivo: PGO deu 15% em código com hotspots não corrigidos e **zero** depois
que os hotspots foram corrigidos no código-fonte. PGO recupera desempenho deixado na mesa; não cria
desempenho novo.

Atenção: há bug histórico de hang com `lto=true` + `target-cpu=native`
([rust#49766](https://github.com/rust-lang/rust/issues/49766), 2018) — usamos ambos, vale validar.

---

## 3. Ações priorizadas para este projeto

| # | Ação | Ganho | Esforço | Status |
|---|---|---|---|---|
| 1 | Não construir `Weights` de CPU no caminho GPU | ~15 GB RSS | Baixo | ✅ feito (repack preguiçoso + `RawTensor<'a>`) |
| 2 | `GpuRawWeights` emprestar do mmap | 14.6 GB | Baixo | ✅ feito |
| 3 | `token_embd` fora da VRAM | 3.1 GB VRAM | Baixo | ✅ feito |
| 4 | **Eliminar `queue_wait_idle`** (usar fences/timeline) | **~22% (medido pela Khronos)** | Médio | ⬜ pendente |
| 5 | `[profile.release]`: `lto="thin"`, `codegen-units=1` | 3–20% | Trivial | ⬜ pendente |
| 6 | `MmapOptions::populate()` + `Advice::Sequential` | Load mais rápido | Trivial | ⬜ pendente |
| 7 | 2 threads persistentes (1/GPU) em vez de `rayon::join` | Latência determinística | Médio | ⬜ pendente |
| 8 | `token_embd` sem materializar f32 (dequant de 1 linha) | 6.2 GB RAM | Médio | ⬜ pendente |
| 9 | Upload em chunks + `posix_fadvise(DONTNEED)` | Pico de page cache | Médio | ⬜ pendente |
| 10 | `zerocopy` no crate `gguf` | Remove `unsafe` | Médio | ⬜ pendente |
| — | Trocar alocador global | **~0** | Trivial | ❌ não fazer |
| — | async/await | negativo | — | ❌ não fazer |
| — | `parking_lot`, `crossbeam` para canais | ~0 | — | ❌ não fazer |
| — | `Arc<[u8]>::from(vec)` | **copia 15 GB** | — | ❌ nunca |

---

## 4. O que a pesquisa NÃO encontrou

Declarado para evitar que alguém trate estimativa como medição:

- Benchmark de async vs threads para submissão Vulkan/GPU.
- Custo em µs de `vkQueueSubmit` (só a afirmação qualitativa).
- Latência de park/unpark medida em Rust. **A premissa de "wake-up de 1 ms" dos docs antigos deste
  projeto não se sustenta** — as fontes indicam dezenas de µs.
- Medição de mmap vs read para GGUF de 10–30 GB.
- Comparação de alocador em inferência LLM em Rust.
- Trade-off medido `MPOL_BIND` vs `MPOL_INTERLEAVE` para inferência quantizada.
- Comparação numérica pulp / wide / autovetorização.
- Modelo de threading multi-GPU do `burn` ou do `candle` (sem documentação autoritativa).

Os benchmarks acima só podem ser feitos aqui — o repo já tem `bench-results/` para isso.
