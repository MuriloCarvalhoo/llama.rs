# Fase 8.0 — Baseline multi-GPU + spike de all-reduce MI50↔MI50

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) ou superpowers:executing-plans para implementar este plano tarefa-a-tarefa. Os passos usam checkbox (`- [ ]`) para tracking.

**Goal:** Estabelecer os números-alvo do llama.cpp no 14B Q8_0 (1× MI50, 2× MI50 layer-split, 2× MI50 row-split — pegar o melhor) e **medir a latência de all-reduce MI50↔MI50** (host-bounce e, se disponível, peer-to-peer) para decidir o mecanismo da Fase 2 **antes** de escrever uma linha de decode multi-GPU.

**Architecture:** Trabalho de **medição e spike** — não toca no caminho de decode de produção. Estende `scripts/benchmark-gpu.sh` para reportar split-mode do llama.cpp; estende `scripts/get-model.sh` para o 14B; adiciona um teste-spike isolado em `crates/llama-vulkan/tests/` que cronometra transferências entre as duas `VulkanDevice` já existentes. A saída é um documento de baseline em `bench-results/` e uma decisão registrada (host-bounce vs peer-to-peer).

**Tech Stack:** Rust + `ash` (Vulkan/RADV), `llama.cpp` (`llama-bench` Vulkan), bash, 2× AMD Instinct MI50 (gfx906, wave64).

---

## Contexto herdado (leia antes de começar)

- A Fase 1 (single-GPU resident decode) ficou em **~80 tok/s** no Qwen2.5-0.5B (`bench-results/gpu-20260616-131520.md`), contra **301 tok/s** do llama.cpp 1× MI50. Por isso a **Fase 3 (kernel)** vem antes da Fase 2 (row-split). **Esta Fase 0 não desbloqueia a Fase 2 sozinha** — ela só levanta os números e de-risca o all-reduce, para que a Fase 2 (depois da Fase 3) já comece com o mecanismo decidido.
- O risco nº 1 da spec (§6) é a **latência** de all-reduce sem NVLink: 2 all-reduces × 48 camadas = **96 sincronizações/token** no 14B. Se cada round-trip custar, digamos, 50 µs, são ~4.8 ms/token só de all-reduce → teto de ~200 tok/s **antes** de qualquer compute. É isso que o spike mede.
- Já existem duas `VulkanDevice` independentes sendo criadas em `crates/llama-vulkan/src/dual_gpu.rs` (`DualGpuMatmul::new`) — o spike reusa esse padrão (`VulkanDevice::create(ctx, &phys[i])`).
- `llama-bench` aceita `-sm <none|layer|row|tensor>` e `-ts <a/b>` (confirmado via `--help`).

---

## Task 1: Baixar o modelo 14B Q8_0 (descobrindo o nome real do arquivo)

O `Qwen2.5-14B-Instruct-GGUF` distribui o Q8_0 em **arquivos split**; não inventamos o nome — consultamos a API do Hugging Face e baixamos exatamente os arquivos `*q8_0*.gguf`.

**Files:**
- Modify: `scripts/get-model.sh` (acrescentar bloco opcional do 14B, ativado por `GET_14B=1`)

- [ ] **Step 1: Acrescentar o bloco de download do 14B ao final do script (antes do `ls -lh`)**

Editar `scripts/get-model.sh`. Substituir a linha final `ls -lh models/` por:

```bash
# 14B Q8_0 (opcional; ~15.6 GB, split em vários .gguf). Ative com GET_14B=1.
# O nome exato dos arquivos é descoberto via API do HF — não hardcodado.
if [[ "${GET_14B:-0}" == "1" ]]; then
    REPO="Qwen/Qwen2.5-14B-Instruct-GGUF"
    echo "Consultando arquivos q8_0 em $REPO..." >&2
    files=$(curl -fsSL "https://huggingface.co/api/models/$REPO" \
        | python3 -c 'import json,sys; [print(s["rfilename"]) for s in json.load(sys.stdin)["siblings"] if "q8_0" in s["rfilename"].lower() and s["rfilename"].endswith(".gguf")]')
    if [[ -z "$files" ]]; then
        echo "ERRO: nenhum arquivo q8_0 .gguf encontrado em $REPO" >&2
        exit 1
    fi
    echo "Arquivos q8_0 a baixar:" >&2
    echo "$files" >&2
    while IFS= read -r f; do
        [ -f "models/$f" ] || curl -fL --retry 3 -o "models/$f" \
            "https://huggingface.co/$REPO/resolve/main/$f"
    done <<< "$files"
fi

ls -lh models/
```

- [ ] **Step 2: Rodar e confirmar que os arquivos chegaram**

Run: `GET_14B=1 ./scripts/get-model.sh`
Expected: imprime a lista de arquivos `*q8_0*.gguf` e, ao final, `ls -lh models/` mostra ~15–16 GB de `.gguf` do 14B (1 ou mais partes). Anote o **nome da primeira parte** (ex.: `qwen2.5-14b-instruct-q8_0-00001-of-0000N.gguf`) — é o caminho que o `llama-bench` recebe (ele auto-localiza as demais partes).

- [ ] **Step 3: Sanidade — o llama.cpp carrega o modelo**

Run (substitua `<PRIMEIRA_PARTE>` pelo nome anotado):
```bash
GGML_VK_VISIBLE_DEVICES=0 ./build-vulkan/bin/llama-bench \
  -m models/<PRIMEIRA_PARTE> -ngl 99 -p 0 -n 4 -r 1 2>&1 | tail -15
```
Expected: carrega sem erro e reporta um tok/s de geração (não importa o valor ainda). Se acusar VRAM insuficiente em 1 GPU, é esperado para o 14B — pule para a Task 3 (multi-GPU). Não comite o modelo (é grande); confirme que `models/*.gguf` está no `.gitignore`:
```bash
grep -q "models/" .gitignore || echo "models/" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add scripts/get-model.sh .gitignore
git commit -m "feat(bench): download opcional do Qwen2.5-14B Q8_0 (GET_14B=1)"
```

---

## Task 2: benchmark-gpu.sh reporta layer-split E row-split do llama.cpp

A spec (§8) exige reportar **layer-split e row-split e tomar o melhor**. Hoje `run_cpp` usa só o default (`-sm layer`). Adicionamos uma run com `-sm row` e uma linha de resultado para cada.

**Files:**
- Modify: `scripts/benchmark-gpu.sh`

- [ ] **Step 1: Generalizar `run_cpp` para aceitar o split-mode**

Em `scripts/benchmark-gpu.sh`, localizar a função `run_cpp()` (começa em `run_cpp() {`). Trocar a assinatura/corpo para receber o split-mode como `$3`:

```bash
# $1 = índices Vulkan ("0" ou "0,1"); $2 = log; $3 = split-mode (layer|row|none).
run_cpp() {
    local devs=$1 log=$2 sm=${3:-layer} json
    json=$(GGML_VK_VISIBLE_DEVICES="$devs" "$CPP_BENCH" \
        -m "$MODEL" -ngl 99 -p 0 -n "$N_TOKENS" -r "$REPS" -sm "$sm" -o json 2>"$log")
    assert_no_nvidia "$log" "llama.cpp (Vulkan dev=$devs sm=$sm)"
    python3 - "$json" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
for row in data:
    if int(row.get("n_gen", 0)) > 0:
        print(f'{float(row["avg_ts"]):.2f} ± {float(row["stddev_ts"]):.2f}')
        break
PY
}
```

- [ ] **Step 2: Adicionar as runs layer-split e row-split na seção de execução**

Localizar o bloco que chama `cpp2=$(run_cpp "$VK_AMD_DEVICES" /tmp/bench-cpp2.err)`. Substituir as duas linhas do `cpp2` por três runs (1×, 2× layer, 2× row):

```bash
echo "Rodando llama.cpp 1x MI50..." >&2
cpp1=$(run_cpp "$VK_AMD_FIRST"  /tmp/bench-cpp1.err layer)
echo "Rodando llama.cpp 2x MI50 (layer-split)..." >&2
cpp2_layer=$(run_cpp "$VK_AMD_DEVICES" /tmp/bench-cpp2l.err layer)
echo "Rodando llama.cpp 2x MI50 (row-split)..." >&2
cpp2_row=$(run_cpp "$VK_AMD_DEVICES" /tmp/bench-cpp2r.err row)
```

- [ ] **Step 3: Computar o melhor dos dois splits e ajustar a razão**

Localizar o bloco `ratio="-"` ... `fi`. Substituí-lo por (calcula o melhor entre layer/row e usa-o na razão):

```bash
# Melhor dos dois split-modes (maior avg_ts).
best_cpp2_avg=$(awk -v l="${cpp2_layer%% *}" -v r="${cpp2_row%% *}" \
    'BEGIN { print (r+0 > l+0) ? r : l }')
best_cpp2_label=$(awk -v l="${cpp2_layer%% *}" -v r="${cpp2_row%% *}" \
    'BEGIN { print (r+0 > l+0) ? "row" : "layer" }')

ratio="-"
if [[ -n "$rs" && "$best_cpp2_avg" != "0" ]]; then
    ratio=$(awk "BEGIN { if ($best_cpp2_avg>0) printf \"%.3fx\", $rs/$best_cpp2_avg; else print \"-\" }")
fi
```

- [ ] **Step 4: Imprimir as duas linhas (layer e row) na tabela de resultados**

Na seção `echo "## Resultados ...`, localizar a linha que imprime `llama.cpp — 2x MI50`. Substituí-la por duas linhas:

```bash
printf "| %-28s | %-16s |\n" "llama.cpp — 2x MI50 (layer)   " "${cpp2_layer:-erro}"
printf "| %-28s | %-16s |\n" "llama.cpp — 2x MI50 (row)     " "${cpp2_row:-erro}"
```

E ajustar a linha da razão para citar o melhor split:

```bash
echo "**Razão llama-rs / melhor llama.cpp 2x MI50 ($best_cpp2_label): $ratio**"
```

- [ ] **Step 5: Rodar no 0.5B (sanidade do script) e conferir as duas linhas**

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -20`
Expected: a tabela agora tem **duas** linhas `llama.cpp — 2x MI50 (layer)` e `(row)`, ambas com `avg ± stddev`, e a razão cita `(layer)` ou `(row)`. (No 0.5B o layer-split costuma ganhar — modelo pequeno; é só sanidade.)

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark-gpu.sh
git commit -m "feat(bench): reporta llama.cpp layer-split e row-split (toma o melhor)"
```

---

## Task 3: Baseline do 14B — o número a bater

**Files:**
- Create: `bench-results/` recebe um novo `gpu-<stamp>.md` (gerado pelo script; commitado como artefato)

- [ ] **Step 1: Rodar o benchmark apontando para o 14B**

Run (substitua `<PRIMEIRA_PARTE>` pelo nome da Task 1):
```bash
BENCH_MODEL=models/<PRIMEIRA_PARTE> ./scripts/benchmark-gpu.sh 2>&1 | tail -30
```
Expected: gera `bench-results/gpu-<stamp>.md`. As linhas `llama-rs` provavelmente vão falhar ou ficar muito lentas no 14B (esperado — a impl ainda não escala); o que importa são as **três linhas do llama.cpp**: `1x MI50`, `2x MI50 (layer)`, `2x MI50 (row)`. **Anote os três tok/s.** Se `1x MI50` falhar por VRAM (15.6 GB > 16 GB com overhead), registre "OOM" — é informação útil (risco nº 3).

- [ ] **Step 2: Medir a folga de VRAM por GPU (risco nº 3)**

Run:
```bash
GGML_VK_VISIBLE_DEVICES=0,1 ./build-vulkan/bin/llama-bench \
  -m models/<PRIMEIRA_PARTE> -ngl 99 -p 0 -n 8 -r 1 -sm row 2>&1 | grep -iE "buffer|VRAM|MiB|alloc" | head -20
rocm-smi --showmeminfo vram 2>/dev/null | head -20 || echo "rocm-smi indisponível"
```
Expected: confirma que o 14B Q8_0 em row-split cabe em ~7.8 GB/GPU + KV + overhead dentro dos 16 GB. Anote o pico de uso por GPU.

- [ ] **Step 3: Anotar o baseline no topo do arquivo de resultado**

Abrir o `bench-results/gpu-<stamp>.md` recém-gerado e acrescentar, logo após o cabeçalho, um bloco:

```markdown
## Baseline 14B (alvo da Fase 2 — número a bater)

- llama.cpp 1x MI50      : <X> tok/s   (ou OOM)
- llama.cpp 2x MI50 layer: <Y> tok/s
- llama.cpp 2x MI50 row  : <Z> tok/s
- **Melhor a bater (2x MI50): max(Y, Z) = <W> tok/s**
- VRAM/GPU em row-split   : <pico> de 16 GB
```

- [ ] **Step 4: Commit**

```bash
git add bench-results/
git commit -m "bench(14b): baseline llama.cpp 1x/2x MI50 (layer+row) — alvo da Fase 2"
```

---

## Task 4: Spike de all-reduce MI50↔MI50 — host-bounce

Mede a latência real de mover um vetor de ativações (`n_embd` f32 ≈ 2 KB no 14B, n_embd=5120) entre as duas GPUs **via host** (download GPU0→host, upload host→GPU1), com os fences corretos. Projeta o custo de 96 transferências/token.

**Files:**
- Create: `crates/llama-vulkan/tests/allreduce_spike.rs`

- [ ] **Step 1: Escrever o teste-spike (host-bounce) que falha por não existir ainda**

Criar `crates/llama-vulkan/tests/allreduce_spike.rs`:

```rust
//! Spike da Fase 8.0 (risco nº 1): latência de all-reduce MI50↔MI50.
//! NÃO é teste de correção — é medição. Roda só com `--ignored` em hardware 2x MI50.
//! Imprime números; o "assert" só garante que o caminho completa.

use llama_vulkan::device::{VulkanContext, VulkanDevice};
use std::time::Instant;

/// Tamanho do payload: n_embd do 14B (5120 f32 = 20 KB). Ajuste se quiser o 0.5B (896).
const N_EMBD: usize = 5120;
const ITERS: u64 = 1000;
const LAYERS: u64 = 48; // 14B
const ALLREDUCES_PER_LAYER: u64 = 2;

#[test]
#[ignore = "requer 2x MI50 — rode com: cargo test -p llama-vulkan --test allreduce_spike -- --ignored --nocapture"]
fn spike_allreduce_host_bounce() {
    let ctx = VulkanContext::new().expect("VulkanContext");
    let phys = ctx.amd_compute_devices();
    assert!(phys.len() >= 2, "spike exige 2 devices AMD");
    let dev0 = VulkanDevice::create(&ctx, &phys[0]).expect("dev0");
    let dev1 = VulkanDevice::create(&ctx, &phys[1]).expect("dev1");

    // Buffers host-visible em cada GPU + um staging no host.
    let bytes = (N_EMBD * 4) as u64;
    let src = host_visible_buf(&ctx, &phys[0], &dev0, bytes);
    let dst = host_visible_buf(&ctx, &phys[1], &dev1, bytes);
    let mut staging = vec![0u8; N_EMBD * 4];

    // Warmup.
    for _ in 0..50 {
        download(&dev0, &src, &mut staging);
        upload(&dev1, &dst, &staging);
    }

    let t0 = Instant::now();
    for _ in 0..ITERS {
        download(&dev0, &src, &mut staging); // GPU0 -> host
        upload(&dev1, &dst, &staging); // host -> GPU1
    }
    let elapsed = t0.elapsed();

    let per_transfer_us = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    let per_token_ms = per_transfer_us * (LAYERS * ALLREDUCES_PER_LAYER) as f64 / 1000.0;
    let ceiling_tok_s = 1000.0 / per_token_ms;

    println!("\n=== ALL-REDUCE SPIKE (host-bounce) ===");
    println!("payload          : {} f32 ({} bytes)", N_EMBD, bytes);
    println!("por transferência: {:.2} µs", per_transfer_us);
    println!(
        "por token (96x)  : {:.3} ms  -> teto de {:.0} tok/s só de all-reduce",
        per_token_ms, ceiling_tok_s
    );

    // Não é gate de valor — só garante que o caminho roda.
    assert!(per_transfer_us > 0.0);

    dev0.destroy();
    dev1.destroy();
}
```

> Nota de implementação: `host_visible_buf`, `download`, `upload`, `VulkanDevice::destroy` e os campos usados podem ter nomes ligeiramente diferentes no crate. Use como referência o `Buf` privado em `crates/llama-vulkan/src/resident_forward.rs` (linhas 1–60: `Buf::host`, `Buf::device`, `upload_f32`, e o padrão de `map_memory`/`copy`/`unmap`). Se esses helpers forem `pub(crate)`, exponha-os para teste com um pequeno módulo `pub mod spike_support` em `lib.rs` **ou** reimplemente as três funções localmente no arquivo de teste usando `ash` diretamente (preferível — mantém o teste auto-contido e não vaza API).

- [ ] **Step 2: Implementar os helpers locais (`host_visible_buf`, `download`, `upload`) no próprio arquivo de teste**

Acrescentar ao mesmo arquivo, abaixo do teste, usando o mesmo padrão de `Buf` de `resident_forward.rs` (mapear memória host-visible, `copy_from_slice`, submeter um `cmd_copy_buffer` quando necessário). Espelhar exatamente a criação de buffer host-visible já usada em `resident_forward.rs::Buf::host` (procure por `MEMORY_PROPERTY_HOST_VISIBLE` no arquivo). Para o host-bounce puro de medição, basta:
- `host_visible_buf`: cria um `vk::Buffer` + `vk::DeviceMemory` host-visible/coherent do tamanho dado (copie o corpo de `Buf::host`).
- `download`: `map_memory` da `src`, `copy_from_slice` para `staging`, `unmap`.
- `upload`: `map_memory` da `dst`, `copy_from_slice` de `staging`, `unmap`.

(Para esta primeira medição não há cópia device-local↔host real — é o **piso** da latência de mapeamento/coerência. A Task 5 mede a variante com `cmd_copy_buffer` device-local, que é o número honesto. Mantemos as duas para enquadrar o intervalo.)

- [ ] **Step 3: Rodar o spike e capturar o número**

Run: `cargo test -p llama-vulkan --test allreduce_spike spike_allreduce_host_bounce -- --ignored --nocapture 2>&1 | tail -15`
Expected: imprime `por transferência`, `por token (96x)` e o `teto de N tok/s`. **Anote o teto.**

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/tests/allreduce_spike.rs
git commit -m "test(spike): all-reduce MI50<->MI50 host-bounce (Fase 8.0, risco nº1)"
```

---

## Task 5: Spike de all-reduce — variante device-local + probe peer-to-peer

Mede o número **honesto**: payload em buffer **device-local** (como na prática), copiado GPU0→host→GPU1 com `cmd_copy_buffer` + fence. E **probe** da capacidade peer-to-peer (`VK_KHR_external_memory_fd`) — se o RADV expuser, mede; senão, registra que a Fase 2 usa host-staged com double-buffering (mitigação do risco nº 1 prevista na spec §6).

**Files:**
- Modify: `crates/llama-vulkan/tests/allreduce_spike.rs`

- [ ] **Step 1: Adicionar o teste device-local (cópia real com fence)**

Acrescentar ao `allreduce_spike.rs` um segundo teste `spike_allreduce_device_local` que, por iteração:
1. `cmd_copy_buffer` de um buffer **device-local** em GPU0 para um buffer **host-visible** em GPU0; submit + `queue_wait_idle(dev0)`.
2. `map`/copy host→host (staging).
3. `cmd_copy_buffer` de host-visible em GPU1 para device-local em GPU1; submit + `queue_wait_idle(dev1)`.

Cronometrar `ITERS` iterações e imprimir o mesmo bloco (`por transferência`, `por token (96x)`, `teto tok/s`). Reusar o padrão de submit+fence de `dispatch1`/`record_and_submit` em `resident_forward.rs`.

```rust
#[test]
#[ignore = "requer 2x MI50 — rode com --ignored --nocapture"]
fn spike_allreduce_device_local() {
    // ... cria dev0/dev1, src_dev (device-local GPU0), stage0 (host GPU0),
    //     stage1 (host GPU1), dst_dev (device-local GPU1) ...
    // loop ITERS: copy(src_dev->stage0)+wait; map stage0->buf; map buf->stage1;
    //             copy(stage1->dst_dev)+wait;
    // imprime per_transfer_us, per_token_ms (x96), ceiling_tok_s
}
```

- [ ] **Step 2: Adicionar o probe de peer-to-peer**

Acrescentar um teste `probe_peer_to_peer_caps` que apenas inspeciona e imprime se as extensões de memória externa estão disponíveis nos devices (não precisa transferir):

```rust
#[test]
#[ignore = "requer 2x MI50 — rode com --ignored --nocapture"]
fn probe_peer_to_peer_caps() {
    let ctx = VulkanContext::new().expect("ctx");
    let phys = ctx.amd_compute_devices();
    assert!(phys.len() >= 2);
    for (i, p) in phys.iter().take(2).enumerate() {
        let exts = unsafe {
            ctx.instance
                .enumerate_device_extension_properties(p.handle)
                .expect("exts")
        };
        let has = |name: &str| {
            exts.iter().any(|e| {
                let s = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
                s.to_string_lossy() == name
            })
        };
        println!("GPU{i}: external_memory_fd={} external_memory={} dma_buf?={}",
            has("VK_KHR_external_memory_fd"),
            has("VK_KHR_external_memory"),
            has("VK_EXT_external_memory_dma_buf"));
    }
}
```

> Ajuste `ctx.instance` / `p.handle` aos nomes reais em `device.rs` (procure por `struct VulkanContext` e `struct VulkanPhysicalDevice`). Se os campos forem privados, adicione um getter `pub(crate)` ou um helper de probe em `device.rs`.

- [ ] **Step 3: Rodar os dois e capturar os números**

Run: `cargo test -p llama-vulkan --test allreduce_spike -- --ignored --nocapture 2>&1 | tail -30`
Expected: imprime o teto device-local (o número honesto) e a tabela de extensões P2P por GPU.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/tests/allreduce_spike.rs
git commit -m "test(spike): all-reduce device-local + probe peer-to-peer (Fase 8.0)"
```

---

## Task 6: Registrar a decisão de all-reduce (entrega da Fase 0)

**Files:**
- Create: `bench-results/fase8-0-allreduce-decisao.md`

- [ ] **Step 1: Escrever o documento de decisão**

Criar `bench-results/fase8-0-allreduce-decisao.md` preenchendo com os números medidos:

```markdown
# Fase 8.0 — Decisão de all-reduce MI50↔MI50

**Data:** <stamp>  |  **Hardware:** 2× MI50 (gfx906/RADV)

## Baseline 14B a bater (Task 3)
- Melhor llama.cpp 2× MI50: **<W> tok/s** (split = <layer|row>)

## Latência de all-reduce medida (Tasks 4–5)
- Host-bounce (piso de map):   <a> µs/transfer → teto <A> tok/s (96x/token)
- Device-local (honesto):      <b> µs/transfer → teto <B> tok/s (96x/token)
- Peer-to-peer disponível?:    <sim/não> (VK_KHR_external_memory_fd: <...>)

## Decisão para a Fase 2
- [ ] Mecanismo: <host-staged double-buffered | peer-to-peer>
- [ ] O teto de all-reduce (<B> tok/s) está <acima/abaixo> do alvo (<W> tok/s)?
      - Se ABAIXO: a Fase 2 precisa sobrepor compute/transfer (spec §6 mitigação)
        e/ou reduzir o nº de all-reduces (fusão de camadas).
      - Se ACIMA: host-staged simples basta; otimizar depois se preciso.

## Pré-condição lembrada
A Fase 2 só deve começar após a **Fase 3** trazer o single-GPU para perto do
llama.cpp 1× MI50 (~301 tok/s no 0.5B). Hoje está em ~80 tok/s.
```

- [ ] **Step 2: Commit**

```bash
git add bench-results/fase8-0-allreduce-decisao.md
git commit -m "docs(fase8-0): decisão de all-reduce + baseline 14B registrados"
```

---

## Self-Review (cobertura vs spec)

- **§3 / §8 (layer-split vs row-split, tomar o melhor):** Task 2 + Task 3. ✓
- **Risco nº 1 (latência all-reduce):** Tasks 4–5 (host-bounce, device-local, probe P2P) + Task 6 (decisão). ✓
- **Risco nº 3 (VRAM 14B):** Task 3 Step 2. ✓
- **§7 Fase 0 ("baseline & spikes, sem código de produção"):** nenhuma alteração no caminho de decode; só script de bench, get-model e testes `--ignored`. ✓
- **Gate herdado (Fase 1 < 314):** registrado no contexto e na Task 6 — a Fase 0 **não** libera a Fase 2; a Fase 3 vem antes.
```
