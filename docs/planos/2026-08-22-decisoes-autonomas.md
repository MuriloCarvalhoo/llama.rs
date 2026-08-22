# Decisões autônomas — 2026-08-22

Registro das decisões tomadas sem consulta durante a sessão autônoma (pedido do
Murilo: "se precisar tomar decisão tome e anote em um arquivo").

## 1. Escopo desta rodada

Com as frentes 1–4 medidas (ver resultado no plano geral), o que resta de maior
retorno e risco controlado, nesta ordem:

1. **n=2 encadeado no verify** (`n_tok=3`, dois pontos de snapshot) — o critério do
   plano (a₂ ≥ 40 %) foi batido com 41,7 % medido → 1,80 tokens/passo. Teto com a
   base atual: ~34 tok/s.
2. **MTP no motor do servidor** — o `--mtp` do servidor monta o backend mas o laço
   ainda decodifica 1 token/passo; o alvo real (opencode) passa pelo servidor.
3. Gates completos, docs, merge `50-toks` → `master`, push.

**Não** ataco nesta rodada: geometria do matvec do decode (as quatro tentativas
anteriores pioraram; risco alto de queimar horas sem ganho) e atenção de contexto
longo (frente de pesquisa, não de execução).

## 2. Desenho do n=2 encadeado

- `VERIFY_TOK` sobe de 2 para 3 e passa a morar no `llama-model` (o trait e o laço
  compartilham a largura). Bloco = [amostrado, proposta, proposta encadeada].
- A proposta encadeada reusa o plano da cabeça na GPU: `CopiaHidden` com o sentinel
  `HIDDEN_CABECA` lê o residual do próprio bloco (`m.b_x`, que o `eh_proj` só
  sobrescreve depois da cópia) em vez do hidden do tronco — zero shader novo.
- Dois pontos de snapshot (`snap[t-1]` depois do token `t-1`) e dois command
  buffers de rollback pré-gravados; `rollback_verify(manter)` restaura
  `snap[manter-1]` e recua o KV em `3 - manter`. Custo: +155 MB de VRAM com MTP
  (total 310 MB de snapshots) e uma cópia a mais por passo.
- **Deriva do KV da cabeça mantida como está** (o cache da cabeça não desfaz
  propostas rejeitadas — agora até 2 entradas fantasma por rejeição em vez de 1).
  Os 60,9 %/41,7 % foram medidos com esse comportamento; consertar é experimento
  próprio, não parte desta entrega. Fica como ponto aberto no plano do MTP.

**Resultado medido:** greedy real 2,20 tokens/passo (140 aceitos/117 passos, lossless
em 256 tokens); CLI 34,5 tok/s de média (31,4 36,2 36,0) contra 31,4 do n=1 e 21,8
sem MTP. Verify de 3 tokens: 56,2 ms de GPU (27,7 + 28,5) — só 1,4 ms a mais que o
verify de 2.

## 3. MTP no motor do servidor

- `Sessao` ganhou `hidden` (índice do hidden que produziu os logits guardados, em
  todos os caminhos: decode, blocos do prefill, passo MTP) e `passo_mtp(...)`, que
  delega ao `passo_mtp` do `gpu.rs` (agora público) e mantém a escrituração de
  tokens/logits — os logits do último token válido seguem guardados para o reuso
  de prefixo responder de graça.
- O laço do `motor.rs` usa uma fila de pendentes: os aceitos do passo saem um a um
  pelo mesmo caminho de emissão (detok/saida/stop); o último da fila é sempre o
  `seguinte`, e é a saída dele que dispara o próximo passo. Sem MTP o laço é o
  mesmo de antes (fila sempre vazia).
- Amostragem dentro do passo não é cronometrada por token (`ms_amostragem`
  subconta no modo MTP) — aceito como limitação de telemetria.

**Resultado medido:** `[gen] decode 80 tok (36.2 tok/s)` greedy; 32,6 tok/s com
temp 0,8. Texto coerente nos dois casos.

## 4. Rodada "caminho restante" — o que os perfis por-op decidiram

Perfis completos capturados (curto e 9,4k, `scratchpad/perfil-{curto,9k}.log`):

- **Atenção a 9,4k: 11,5 ms/token (214 GB/s efetivos), 22-25 % do decode; 24,4 ms
  por passo de verify; 25 % do bloco de prefill.** → **Frente A: KV-cache em f16
  empacotado em u32** (`packHalf2x16`/`unpackHalf2x16`, core GLSL — o device é
  criado sem features e não quero arriscar `storage_buffer_16bit`). Paga três
  vezes e devolve metade da VRAM de contexto.
- **matvec_q4k: 57 % do decode curto a 461-518 GB/s, contra 636 do q6k.** O
  cabeçalho do shader prova que 3 loads/lane é o piso e o LDS pad já falhou dos
  dois lados. Hipótese restante: paralelismo de memória — com ROWS_PER_WAVE=2 as
  3 cargas da linha 0 são consumidas antes de emitir as da linha 1. → **Frente B
  (uma tentativa única):** içar as cargas das duas linhas antes do cômputo.
  Regra: se não melhorar o `TOTAL GPU`, reverter e parar — é a 5ª tentativa numa
  frente com 4 fracassos. **Resultado: 456-478 GB/s, igual ao baseline (revertido).**
  O compilador já agrupava as cargas; os ~470 GB/s do Q4_K neste hardware são o que
  esta estrutura de kernel entrega. Frente fechada — o caminho para a base é outra
  quantização (plano B, Q3_K) ou nada.
- **Não ataco:** fusão das ops pequenas (~2,7 ms/GPU a 4-7 GB/s, mas são ~10 µs
  de latência de dispatch cada — ganho ~1 ms com risco alto de regressão);
  gravação de host (0,6 ms < 1 ms, regra da frente 1); GEMM para Q5_K/Q6_K no
  prefill (25 ms/bloco, fica anotado como próxima frente de prefill).

## 5. Implementação do KV f16 (frente A)

- Cache como pares f16 num u32 (`kv_pack.comp` empacota; a atenção desempacota
  com `unpackHalf2x16` — índices continuam em elementos, par = `idx>>1`).
- O `kv_append` por `cmd_copy_buffer` virou o dispatch `KvPack` com
  `PushSpec::KvPack{slot, n_tok}` resolvido na gravação (padrão do `Attention`);
  o cache da cabeça MTP usa o mesmo formato (a atenção é o mesmo shader).
- **`rope_kv` removido de vez** (shader, PipeId, knob): já era 0,4 ms pior e
  escreveria f32 num cache de pares f16 — a remoção é consequência da mudança,
  não limpeza gratuita. O teste dele virou o teste do `kv_pack`.
- Referências de CPU nos testes de atenção passam a arredondar K/V por f16
  antes (mesma lição da ativação int8: comparar partindo dos mesmos valores).
- Nota: o oráculo (llama.cpp) usa KV f16 por padrão — o formato novo aproxima
  os dois, não afasta.

**Resultado medido:** os 40 testes com modelo passam (logits GPU×CPU inclusive);
contexto curto inalterado; **a atenção a 9,4k NÃO mudou** (11,5 ms/token decode,
24,6 no verify) — metade dos bytes, mesmo tempo. Conclusão: ela não é limitada
por banda. A aritmética fecha com **throughput de VALU** (wave64 = 4 ciclos por
instrução, ~13 waves/SIMD residentes): os bytes caíram, as instruções não. O f16
fica pelo que entrega — metade da VRAM de contexto (32k: 4,4 → 2,2 GB) e
paridade com o formato do llama.cpp — e o próximo ganho de atenção longa seria
matemática f16 nativa (`V_DOT2_F32_F16`), que exige habilitar `shaderFloat16`
no device: fica anotado como frente futura, não desta rodada.

