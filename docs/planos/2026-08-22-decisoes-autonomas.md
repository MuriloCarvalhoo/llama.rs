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

