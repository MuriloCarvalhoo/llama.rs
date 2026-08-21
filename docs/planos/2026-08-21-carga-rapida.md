# Frente 4 — Carga do modelo: ~35 s → ≤6 s (cache quente)

> **Para execução:** independente das outras frentes. Cada tarefa fecha com a medição
> do tempo de carga separado por fase e registro aqui.
>
> **Estado:** as quatro tarefas estão implementadas; o que falta é **medir**. Nenhum
> número foi escrito aqui sem medição — a tabela de tempos continua vazia de propósito.

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

- [x] Cronometrar e logar (atrás de `LLAMA_RS_LOAD_PROFILE=1`): mmap+parse, dequant de
      aux/embd, e por shard: repack+staging memcpy vs espera de GPU, com o total.
      Acumulador em `crates/llama-model/src/perfil_carga.rs`; fases de mesmo nome se
      somam, e a tabela sai no stderr quando a geração começa, com a **soma** e o tempo de
      **parede** (que passam a divergir agora que os shards carregam em paralelo).
- [ ] **Pendente de medição** (é do orquestrador, não deste agente): rodar 2× (frio:
      `echo 3 > /proc/sys/vm/drop_caches`; quente) e registrar a tabela aqui.

Fases emitidas: `mmap+parse`, `pesos crus (mmap)`, `aux f32 (dequant)`,
`GPU{n} repack+staging`, `GPU{n} espera GPU`, `aux → VRAM`.

> O `llama-server` não imprime a tabela: `imprimir()` está no início de
> `gerar_streaming_residente`, que o servidor não usa (ele fala com o backend pelo
> `Motor`). Medir pelo `llama-cli`.

## Tarefa 2 — `token_embd` preguiçoso: −5,1 GB de RAM e segundos de CPU

**Arquivos:** `crates/llama-model/src/gpu.rs` (`GpuAuxWeights::from_gguf`), caminho do
`PlannedOp::Embed`.

O embedding só precisa de **uma linha por token** (5120 valores, ~3 KB quantizados).

- [x] Guardar o `token_embd` como bytes quantizados (slice do mmap, zero cópia) e
      dequantizar a(s) linha(s) pedida(s) no momento do `Embed` — decode usa 1 linha,
      prefill usa `n_batch`. Tipo novo `llama_model::TokenEmbd`; o `ResidentState` passou a
      ter lifetime de pesos porque empresta do GGUF. Some da carga o passo de 5,1 GB **e**
      o `to_vec()` que o `new_shard` fazia por cima dele (era 2× a tabela em RAM).
      A cabeça MTP passou a ler a linha da mesma tabela — já dequantizava por linha, só
      faltava a fonte.
- [x] Teste: `TokenEmbd::linha` é **idêntica** (±0) à fatia do dequant completo num GGUF
      sintético (`gpu.rs`, sem modelo), e o gate `resident_forward_logits_iguais_a_cpu_qwen`
      (GPU × CPU, qwen2.5-0.5B real) continua verde.

## Tarefa 3 — Upload em pipeline: staging fixo + submits em lote

**Arquivos:** `crates/llama-vulkan/src/tensor.rs`, `resident_forward.rs` (`new_shard`),
`crates/llama-vulkan/src/alloc.rs`.

- [x] Memória device-local dos pesos via `GpuAllocator` (chunks de 1,5 GB) em vez de
      uma alocação por tensor. O **último** chunk é dimensionado pelo que ainda falta
      subir (`tamanho_do_chunk`): reservar 1,5 GB para guardar 300 MB tiraria VRAM da outra
      placa, e VRAM curta empurra pesos para GTT.
- [x] Dois staging buffers persistentes de 256 MB por device, mapeados uma vez; um command
      buffer com todas as cópias do lote e **um** fence por lote. Tensor maior que o
      staging (a projeção de vocabulário do 27B tem ~1 GB) é fatiado em blocos inteiros e
      atravessa vários lotes. Os auxiliares f32 entram na mesma fila — eram ~450 fences.
- [x] O repack escreve **direto** no staging, uma vez e em ordem crescente; nada é lido de
      volta. O staging deixou de pedir `HOST_CACHED` (write-combining é o tipo certo para
      memória que só se escreve).
- [x] Os dois shards carregam **em paralelo** (`std::thread::scope` em `layer_split.rs`,
      uma thread por device).

## Tarefa 4 — Repack paralelo e page-in antecipado

**Arquivos:** `tensor.rs`, `runner.rs` (`map_model`).

- [x] Rayon nos laços de repack Q8_0 e pad Q6_K (por faixas de 4096 blocos; são
      transformações posicionais puras).
- [x] `MmapOptions::populate()` + `madvise(Sequential)` no `map_model` (pendência 6 do
      doc de memória) — o page-in vira readahead do kernel em vez de falta de página no
      meio do memcpy.
- [ ] **Pendente de medição:** conferir que o tempo frio ficou ≈ tempo de ler 16,3 GB do
      disco. Dois efeitos a vigiar na medição, porque podem ir na direção contrária:
      `MAP_POPULATE` lê o arquivo inteiro **antes** de qualquer upload (tira a
      sobreposição entre I/O e repack no caso frio), e `MADV_SEQUENTIAL` autoriza o kernel
      a liberar as páginas logo após o acesso (pode esfriar o page cache entre execuções,
      justo o que o caso quente depende). Se a medição acusar, o conserto é tirar uma das
      duas — são duas linhas em `map_model`.

## Critério de aceite da frente

- [ ] **Pendente de medição:** carga quente ≤6 s (hoje ~35 s); fria ≤ leitura do disco +20%.
- [ ] **Pendente de medição:** pico de RAM da carga cai ~5 GB (sem o f32 do `token_embd`).
- [x] `cargo test --workspace` verde; `cargo clippy --workspace --all-targets` sem aviso
      novo. Com os modelos à mão, os 34 testes de `integration.rs` (qwen2.5-0.5B real) e o
      `estado_recorrente_acumula_entre_tokens` (Qwen3.8-27B Q5_K_M em layer-split nas duas
      placas) passaram — este último é o que exercita a carga paralela dos dois shards.
- [ ] **Pendente de medição:** tabela de tempos por fase (antes/depois) registrada neste
      arquivo.

## O que não foi feito e por quê

- **Não há teste automático do `new_shard` inteiro sem modelo.** O `Uploader` é testado com
  dados sintéticos na GPU (staging de 64 KiB, cinco tensores, um maior que o staging), mas
  montar um GGUF sintético que atravesse a stack de shaders do `ResidentForward` custaria
  mais do que rende — o gate real continua sendo `integration.rs` com o 0.5B.
- **`GpuTensor::upload_quant` foi removido** (só o `new_shard` usava, e ele agora usa o
  `Uploader`). `upload_q8_0` continua: `matmul.rs`, `resident.rs` e `model_gpu.rs` ainda o
  usam, com alocação própria por tensor.
