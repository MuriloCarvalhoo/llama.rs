# Conversa longa até perto de encher o contexto — llama.rs contra llama.cpp

**Resumo:** no **decode** o llama.rs é ~10% mais rápido que o llama.cpp HIP em todas as
profundidades (22,2 contra 20,4 tok/s com 30,6k tokens no contexto). No **prefill** ele perde por
~2,8×, e **numa conversa de vários turnos ele refaz o prefill do contexto inteiro a cada turno**:
com 30k tokens, a primeira palavra da resposta sai em 387 s, contra ~11 s do llama.cpp. Isso é o
que mais pesa no uso com agentes, e é o item 1 de [`planos/2026-09-25-proximos-ganhos.md`](planos/2026-09-25-proximos-ganhos.md).

Medido em 2026-09-25, llama.rs `8edf6da`, llama.cpp `cd83f27ef` (build HIP/ROCm para gfx906).

## Condições

| | |
|---|---|
| Hardware | 2× MI50 (gfx906), layer-split, GPU do monitor com ≥ 5 GB livres o tempo todo |
| Modelo | `Qwen3.8-27B-Q4_K_M.gguf` (qwen35: 64 camadas, 16 de atenção, 48 delta-net) |
| Contexto | 32.768 tokens nos dois servidores |
| Conversa | turnos com trechos de `docs/` (~1,2–1,8k tokens cada), 5 profundidades calibradas pelo tokenizador do llama.cpp: 1.448, 7.656, 15.745, 22.378, 30.644 tokens |
| Geração | greedy (`temperature: 0`), 256 tokens, raciocínio ligado (padrão do modelo) |
| Servidores | `llama-server --ctx 32768 --gpu-layer-split [--mtp]` e `llama-server -ngl 99 -c 32768 -fa on --jinja -np 1 [--spec-type draft-mtp --spec-draft-n-max 2]` |
| Térmica | antes de cada pedido de medida, espera a junction mais quente cair abaixo de 65 °C |

**O decode é medido com a GPU fria e o contexto já no cache, nos dois:** no llama.cpp o pedido
do marco chega depois de a conversa ter crescido turno a turno (o prefill dele é só a pergunta
final, ~70 tokens); no llama.rs o mesmo pedido é mandado duas vezes, e a segunda acerta o
snapshot inteiro (0 tokens de prefill).

## Decode (tok/s)

| contexto | llama.rs | llama.rs + MTP | llama.cpp HIP | llama.cpp HIP + MTP | llama.rs ÷ llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 1.448 | **26,0** | **33,5** | 23,4 | 21,3 | 1,11× |
| 7.656 | **25,1** | **27,8** | 22,8 | 21,5 | 1,10× |
| 15.745 | **24,0** | **25,2** | 21,9 | 19,7 | 1,10× |
| 22.378 | **23,1** | **23,4** | 21,1 | 19,4 | 1,10× |
| 30.644 | **22,2** | 20,3 | 20,4 | 19,5 | 1,09× |

- O MTP do llama.rs vale **+29% no contexto curto e vira prejuízo perto de encher**: o verify de 3
  tokens paga a atenção sobre o contexto inteiro 3 vezes (−9% em 30k).
- O MTP do llama.cpp é prejuízo em todas as profundidades neste modelo denso, apesar de 63–71% de
  aceitação — o mesmo que a medição de 2026-08 já tinha visto.

## Prefill

| contexto | llama.rs frio | llama.rs TTFT | llama.cpp frio (clock auto) | llama.cpp por turno (1,2–1,8k novos) |
|---:|---:|---:|---:|---:|
| 1.448 | 134 tok/s | 10,8 s | 184 tok/s | 171 tok/s |
| 7.656 | 115 tok/s | 66,9 s | **318 tok/s** | 157 tok/s |
| 15.745 | 100 tok/s | 157 s | — (card1 passou de 100 °C) | 176 tok/s |
| 22.378 | 90 tok/s | 249 s | — | 165 tok/s |
| 30.644 | 79 tok/s | **387 s** | — | 151 tok/s |

- **Numa conversa real o llama.rs refaz o prefill inteiro a cada turno.** O log do servidor
  mostra `0 do cache` em todo turno novo. A sessão guarda **um** snapshot do estado recorrente, no
  **fim exato** do prompt (`crates/llama-model/src/sessao.rs:201`). No turno seguinte o template
  re-renderiza o fim do turno anterior (a abertura `<|im_start|>assistant` da resposta), a
  divergência cai antes do snapshot e a sessão reinicia. O llama.cpp guarda vários checkpoints e
  processou só os tokens novos em todos os turnos (`prompt_n` de 1,2–1,8k).
- O prefill frio do llama.cpp sobe com o tamanho do bloco (318 tok/s em 7,6k); o do llama.rs cai
  com o contexto (a atenção do prefill cresce e a escada térmica atua).
- O llama.cpp **não passou de 15k de prefill frio**: no clock automático a card1 foi a 101–104 °C
  (193–202 W) e o guarda derrubou o servidor. A coluna "por turno" foi medida com o pstate
  `standard` fixo por fora só durante os turnos de prefill (`scripts/bench-conversa/segura_pstate.py`);
  o decode dele ficou no automático, que é o melhor clock dele (com `standard` fixo o decode do
  llama.cpp cai de 23,7 para 20,2 tok/s).

## Térmica

A card1 (PCI 05:00, sem monitor) é o limite dos dois motores: em repouso fica ~12 °C acima da
card2, e em prefill sustentado vai a 100 °C em segundos com o DPM automático. O llama.rs chegou a
101–102 °C em picos isolados (a escada do `pstate.rs` segura o resto); o llama.cpp, sem controle
próprio, chegou a 104 °C. Ver o item 3.1 dos próximos ganhos.

## Como repetir

```bash
O=/tmp/bench-conversa; B=scripts/bench-conversa
# 1. calibrar as conversas com o tokenizador do llama.cpp (servidor dele ouvindo em 8093)
python3 $B/conversa.py calibrar 8093
# 2. llama.cpp: conversa crescente, clock standard só nos turnos de prefill
HOLD_PREFILL=standard $B/rodar_config.sh lcpp llamacpp 8093 nenhum crescer -- \
    /home/murilo/llama.cpp/build/bin/llama-server -m models/Qwen3.8-27B-Q4_K_M.gguf \
    -ngl 99 -c 32768 -fa on --jinja -np 1 --host 127.0.0.1 --port 8093
# 3. llama.rs: frio + quente por profundidade
$B/rodar_config.sh lrs llamars 8095 nenhum frioquente -- ./target/release/llama-server \
    -m models/Qwen3.8-27B-Q4_K_M.gguf --bind 127.0.0.1:8095 --ctx 32768 --gpu-layer-split
```

Os resultados ficam em `$BENCH_SAIDA` (padrão `/tmp/bench-conversa`); o decode do llama.rs sai
das linhas `[gen]` do log do servidor.
