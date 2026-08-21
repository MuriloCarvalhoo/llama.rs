# Frente 4 — Carga do modelo: ~35 s → ≤6 s (cache quente)

> **Para execução:** independente das outras frentes. Cada tarefa fecha com a medição
> do tempo de carga separado por fase e registro aqui.

**Meta:** subir os 16,3 GB do Qwen3.8-27B para as duas MI50 em ≤6 s com page cache
quente; com cache frio, ficar limitado pela leitura do disco (+20% no máximo). Além do
tempo, devolver ~5 GB de RAM que a carga desperdiça hoje.

**Onde o tempo vai hoje** (mapeado no código; medir na tarefa 1):

| custo | onde | natureza |
|---|---|---|
| upload serial, 1 fence por tensor | `tensor.rs:309` `one_shot_copy`: aloca CB, cria fence, submete, **espera `u64::MAX`**, destrói staging — por tensor, ~600 tensores | latência × N |
| 1 `vkAllocateMemory`+`vkCreateBuffer` por tensor | `GpuTensor` não usa o `GpuAllocator` de chunks de 1,5 GB que já existe em `alloc.rs:6` | latência × N |
| repack Q8_0 (34→36 B) e pad Q6_K (210→212 B) | `tensor.rs:107-137` e `:48-53`, laços escalares single-thread sobre GB | CPU serial |
| dequant do `token_embd` inteiro para f32 | `gpu.rs:422`: 248320×5120×4 B = **5,1 GB de RAM** e segundos de CPU | CPU serial + RAM |
| page-in do mmap sob demanda | `runner.rs:17-23` sem `populate`/`madvise` (pendência 6 de `rust-memoria-e-desempenho.md`) | I/O serializado com o resto |
| memcpy para staging write-combined | custo já medido no projeto (memória "custo de carga") | banda de WC |

Referência do llama.cpp: mmap com prefetch e, no caminho sem mmap, upload assíncrono
com 4 staging buffers pinned reusados (`llama-model-loader.cpp:1443-1650`) — a mesma
forma da tarefa 3.

## Tarefa 1 — Medir por fase antes de mexer

**Arquivos:** `crates/llama-model/src/gpu.rs`, `crates/llama-vulkan/src/resident_forward.rs`
(`new_shard`), `crates/llama-cli/src/runner.rs`.

- [ ] Cronometrar e logar (atrás de `LLAMA_RS_LOAD_PROFILE=1`): mmap+parse, dequant de
      aux/embd, e por shard: repack+staging memcpy vs espera de GPU, com o total.
- [ ] Rodar 2× (frio: `echo 3 > /proc/sys/vm/drop_caches`; quente) e registrar a
      tabela aqui. Sem essa linha de base o resto da frente não tem critério.

## Tarefa 2 — `token_embd` preguiçoso: −5,1 GB de RAM e segundos de CPU

**Arquivos:** `crates/llama-model/src/gpu.rs` (`GpuAuxWeights::from_gguf`), caminho do
`PlannedOp::Embed`.

O embedding só precisa de **uma linha por token** (5120 valores, ~3 KB quantizados).

- [ ] Guardar o `token_embd` como bytes quantizados (slice do mmap, zero cópia) e
      dequantizar a(s) linha(s) pedida(s) no momento do `Embed` — decode usa 1 linha,
      prefill usa `n_batch`. Dequant de uma linha é microssegundos; some da carga o
      passo inteiro de 5,1 GB.
- [ ] Teste: logits idênticos (±ε) antes/depois num prompt curto do modelo real.

## Tarefa 3 — Upload em pipeline: staging fixo + submits em lote

**Arquivos:** `crates/llama-vulkan/src/tensor.rs`, `resident_forward.rs` (`new_shard`),
`crates/llama-vulkan/src/alloc.rs`.

- [ ] Memória device-local dos pesos via `GpuAllocator` (chunks de 1,5 GB) em vez de
      uma alocação por tensor.
- [ ] Dois staging buffers persistentes de 256 MB por device: CPU preenche o B (memcpy
      + repack fundidos — o repack escreve direto no staging) enquanto a GPU copia o A;
      um command buffer com todas as cópias do lote e **um** fence por lote, não por
      tensor. É o desenho do loader do llama.cpp com 4 buffers pinned.
- [ ] O memcpy para WC deve ser sequencial e sem releitura (repack lê do mmap, escreve
      no staging uma vez — nunca ler de volta do staging).
- [ ] Os dois shards carregam **em paralelo** (uma thread por device; cada um já tem
      fila própria).

## Tarefa 4 — Repack paralelo e page-in antecipado

**Arquivos:** `tensor.rs`, `runner.rs` (`map_model`).

- [ ] Rayon nos laços de repack Q8_0 e pad Q6_K (por faixas de blocos; são
      transformações posicionais puras).
- [ ] `MmapOptions::populate()` + `madvise(Sequential)` no `map_model` (pendência 6 do
      doc de memória) — o page-in vira readahead do kernel em vez de falta de página no
      meio do memcpy.
- [ ] Conferir com a tarefa 1 que o tempo frio ficou ≈ tempo de ler 16,3 GB do disco.

## Critério de aceite da frente

- [ ] Carga quente ≤6 s (hoje ~35 s); fria ≤ leitura do disco +20%.
- [ ] Pico de RAM da carga cai ~5 GB (sem o f32 do `token_embd`).
- [ ] `cargo test --workspace` verde; logits inalterados no modelo real.
- [ ] Tabela de tempos por fase (antes/depois) registrada neste arquivo.
