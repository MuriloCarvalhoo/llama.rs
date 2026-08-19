# Debugging e profiling

## Perfil por operação

```bash
LLAMA_RS_PROFILE=1 ./target/release/llama-cli -m modelo.gguf -p "..." --gpu-layer-split --timings
```

Imprime ms/token por operação de GPU (matvec, attention, norm, ...) e o custo de host por fase
(gravação do command buffer, submit+fence, leitura do resultado) — ver `LayerSplitForward::print_profile`.

## Timeline cronológica CPU + GPU

```bash
cargo build --release -p llama-cli --features "gpu profiling"
LLAMA_RS_PROFILE=1 ./target/release/llama-cli -m modelo.gguf -p "..." -n 8 \
    --gpu-resident --trace /tmp/t.json
```

Abre em <https://ui.perfetto.dev>. Produz trilhas separadas e alinhadas no mesmo eixo de tempo:

- **CPU** — `carregar_modelo`, `subir_pesos`, `shard`, `gravar_cmdbuf`, `submit+fence`
- **uma trilha por GPU** — cada `matvec`, `rmsnorm`, `attention`, `add`... do token

As zonas de GPU caem **dentro** da janela de `submit+fence`, que é o ponto: um span de CPU em
volta de `vkQueueSubmit` mede o custo de *enfileirar*, não de *executar* — sozinho, mostraria só
uma barra opaca de dezenas de ms.

`LLAMA_RS_TRACE_TOKENS=N` limita quantos tokens entram no arquivo (padrão 8). Sem a feature
`profiling` nenhum código de instrumentação entra no binário — custo zero em produção.

## Captura RGP (RADV)

Para saber *por que* um dispatch é lento (ocupação de wave, instruction timing) sem escrever
código — a captura de RGP embutida no driver Mesa/RADV:

```bash
MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 MESA_VK_TRACE_TRIGGER=/tmp/trigger \
RADV_THREAD_TRACE_BUFFER_SIZE=134217728 ./target/release/llama-cli ...
```

`RADV_PROFILE_PSTATE` é `peak` por padrão nessa captura — os números do RGP **não** são
comparáveis com os de uma execução normal (clock diferente).

## Variáveis de ambiente úteis

| Variável | Efeito |
|---|---|
| `LLAMA_RS_GPU=N` | Força o índice da GPU, em vez da seleção automática por VRAM livre |
| `LLAMA_RS_SPLIT=N` | Fixa a fronteira do layer-split na camada N, em vez de derivar da VRAM livre |
| `LLAMA_RS_PROFILE=1` | Liga a coleta de timestamps de GPU |
| `LLAMA_RS_TRACE_TOKENS=N` | Quantos tokens entram no `--trace` (padrão 8) |
| `LLAMA_RS_MATVEC_GEOM=wg,linhas` | Geometria do matvec K-quant (padrão 256,2) — ver `scripts/tune-matvec.sh` |
| `LLAMA_RS_STOP_LAYER=N` | Executa só as N primeiras camadas do shard (diagnóstico) |
| `RAYON_NUM_THREADS=N` | Threads do caminho CPU |
