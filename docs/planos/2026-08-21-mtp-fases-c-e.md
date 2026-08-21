# Frente 2 — MTP fases C–E: verify em batch, rollback e ligação

> **Para execução:** depende só da frente 0 (diff pendente commitado). Fases A e B já
> estão prontas (`docs/mtp-implementacao.md`): bloco carregado (`MtpRaw`/`MtpAux`),
> cabeça rodando na CPU, **aceitação 60,9% medida** em greedy com propostas de 1 token.

**Meta:** `--mtp` ligável no CLI e no servidor, desligado por padrão. Com a base atual
(22,3) entrega ~34 tok/s; com a frente 1 completa, ≥46; a decisão n=1 vs n=2 (tarefa 6)
define se 50 fecha sem exigir base 33.

**Invariante que vale por todos os testes:** com greedy, speculative decoding é
*lossless* — a sequência gerada com MTP ligado tem de ser **idêntica** à gerada com MTP
desligado, token a token. Esse é o teste de aceite de cada fase.

## O fluxo alvo (draft n=1)

```text
passo:  h_t já existe do passo anterior
        proposta = cabeça_mtp(token t+1, h_t)              # 0,40 ms na GPU
        logits[2] = verify_batch([t+1, proposta])          # lê os pesos UMA vez
        se argmax(logits[0]) == proposta:                  # aceitou
            emite t+1, proposta; h avança; segue de argmax(logits[1])
        senão:                                             # rejeitou
            emite t+1, argmax(logits[0])                   # o token certo já veio!
            rollback do estado recorrente em 1 token; KV recua 1
```

Note que a rejeição não desperdiça forward: `argmax(logits[0])` **é** o token correto
seguinte, então todo passo emite ≥1 token novo além do t+1 — a conta
`tok/s = (1+aceitação)/t_passo` já reflete isso.

## Tarefa 1 (fase C) — Plano de verify: batch 2 com logits dos 2 tokens

**Arquivos:** `crates/llama-vulkan/src/resident_forward.rs`,
`crates/llama-model/src/gpu.rs`.

Hoje o batch tem duas travas (mapeadas): `decode_shard_batch` exige
`tokens.len() == cfg.n_batch` (`resident_forward.rs:3343`) e o plano de batch só
calcula logits do **último** token (`build_plan:3124-3147`, `b_logits` de 1×vocab).
O llama.cpp confirma o desenho: verify de até 8 tokens usa o mesmo kernel matvec com N
colunas (`mul_mat_vec_max_cols = 8`, `ggml-vulkan.cpp:389`), não GEMM.

- [ ] Construir um **terceiro plano**, `plan_verify`, com `n_tok = 2` fixo, irmão do
      `plan_batch` (mesma função `build_plan(n_tok)`, largura diferente). Buffers de
      ativação já são dimensionados por `n_batch`; o verify reusa os mesmos (2 ≤ 8).
- [ ] Generalizar o fim do `build_plan`: com `logits_todos = true`, requantizar os
      `n_tok` tokens e rodar a projeção `output` com `COLS = n_tok`;
      `b_logits` passa a `vocab × n_tok` (o custo é 1 leitura do peso Q6_K de 0,63 GB
      para as 2 colunas — ~0,3 ms a mais que hoje, não 2×).
- [ ] `decode_shard_batch` aceita a largura do plano escolhido; expor
      `decode_verify(&[t1, t2], pos) -> [logits; 2]` em `GpuResidentDecode`
      (`gpu.rs`, ao lado de `decode_batch`).
- [ ] Teste de integração: `decode_verify([a,b])` devolve logits[0] idêntico (±ε) ao
      `decode(a)` token-a-token e logits[1] idêntico ao `decode(b)` na sequência —
      rodado no modelo real, 1 camada de cada tipo já cobre (usar o harness dos testes
      existentes de batch).

## Tarefa 2 (fase C) — Snapshot do estado entre os tokens do verify

**Arquivos:** `resident_forward.rs` (`plano_delta`, `PlannedOp`).

A recorrência do delta-net no plano de batch já roda **um dispatch por token, em ordem**
(`plano_delta:1802-1824`) — isso torna o rollback trivial, sem mudar shader:

- [ ] No `plan_verify`, entre o dispatch `DeltaNet` do token 0 e o do token 1 de cada
      camada linear, inserir um `cmd_copy_buffer` do `DeltaBufs.estado` (3,1 MB) e da
      `DeltaBufs.janela` da conv (120 KB) para buffers de snapshot pré-alocados.
      Custo total: 48 × 3,2 MB ≈ 155 MB ≈ 0,45 ms. VRAM extra: 155 MB residentes.
- [ ] O KV-cache não precisa de snapshot: rejeitar é recuar o comprimento em 1
      (as posições do verify são consecutivas; o próximo append sobrescreve).
- [ ] Alternativa a considerar **só se** 0,45 ms doer na medição final: o shader
      `gated_delta_net.comp` do llama.cpp escreve snapshots de dentro do kernel
      (parâmetro K, slot `n_tokens-1-t` — `ggml.h:2564-2585`); mesma banda, um dispatch
      a menos. Não começar por aqui.

## Tarefa 3 (fase D) — Rollback

**Arquivos:** `resident_forward.rs`, `crates/llama-model/src/sessao.rs`.

- [ ] `rollback_verify()`: copia os snapshots de volta (`estado` + `janela` das 48
      camadas locais de cada shard) e recua o comprimento do KV em 1. Executado só na
      rejeição (~39% dos passos) — na média custa 0,39 × 0,45 ≈ 0,18 ms/passo.
- [ ] `Sessao`/cache de posição: o contrato de `planejar_reuso` não muda — o verify
      avança `pos` em 2 na aceitação e em 1 na rejeição; quem chama só vê a posição
      final.
- [ ] Teste: forçar rejeição (propor token errado de propósito) e verificar que o
      estado pós-rollback produz os mesmos logits que o caminho token-a-token.

## Tarefa 4 (fase C) — Cabeça MTP na GPU

**Arquivos:** `crates/llama-model/src/mtp.rs` (referência CPU já validada),
`resident_forward.rs` (plano da cabeça), `crates/llama-model/src/gpu.rs`.

Todas as ops existem (`docs/mtp-implementacao.md`, tabela da fase B). O plano da cabeça
é uma camada de atenção qwen35 normal mais o prólogo:

- [ ] `enorm(emb(t+1))` e `hnorm(h_t)` com `NormFused`+`NormP2`, escrevendo nos offsets
      0 e `n_embd` do mesmo buffer de entrada (a concatenação é só binding com offset).
- [ ] `eh_proj` (matvec Q8_0, 10240→5120), camada de atenção completa (o bloco MTP **é
      de atenção**, mesmo estando numa posição "linear" — pegadinha já documentada),
      `shared_head_norm`, projeção `output` (o GGUF do 27B não tem
      `nextn.shared_head.head` próprio — usa o `output.weight` compartilhado, como o
      llama.cpp faz com fallback em `qwen35.cpp:634-643`).
- [ ] KV-cache próprio do bloco: 8 KB/token (4 KV heads × 256 × K,V × f32) — 268 MB em
      ctx 32k. Alocar junto do cache principal; o llama.cpp confirma que a cabeça não
      tem estado SSM nenhum (contexto MTP usa KV puro filtrado — `llama-model.cpp:2362`).
- [ ] A entrada `h_t`: exportar do plano principal o hidden **pós-norma final** do
      último token (já existe como entrada da projeção de logits); no layer-split, isso
      mora no shard 2 — a cabeça roda inteira no shard que tem a norma final, zero
      tráfego extra entre GPUs.
- [ ] Validação: a proposta da cabeça GPU tem de bater com `MtpHead::propor` da CPU
      token a token (o teste `mtp_aceitacao` vira o oráculo). Cuidado redobrado com
      contagem de bindings — o modo de falha silencioso já mordeu o projeto.

## Tarefa 5 (fase E) — Ligar, medir, registrar

**Arquivos:** `crates/llama-cli/src/args.rs`, `crates/llama-server/src/motor.rs`,
`docs/decode-por-configuracao.md`.

- [ ] Flag `--mtp` no CLI e campo na config do servidor; **desligado por padrão** (o
      caminho sem MTP continua o padrão do repositório).
- [ ] Laço de geração: com `--mtp`, o passo vira propor→verify→aceitar/rollback. Com
      sampling (temp >0), aceitar a proposta só se ela == token amostrado dos
      logits[0] — mantém a distribuição exata e a aceitação cai; registrar os dois
      números (greedy e temp 0.8) na tabela.
- [ ] Teste de aceite lossless: 256 tokens greedy com e sem `--mtp`, sequências
      idênticas, na CI de GPU (`scripts/gate.sh`).
- [ ] Medir pares com/sem MTP na mesma build, `LLAMA_RS_SPLIT` fixo, e registrar em
      `docs/decode-por-configuracao.md` — as linhas "MTP ligado/desligado" estão
      reservadas lá desde a fase B.

## Tarefa 6 — Experimento barato: aceitação encadeada (n=2)

**Arquivos:** `crates/llama-vulkan/tests/mtp_aceitacao.rs` (estender).

Antes de qualquer código de produção para n=2, medir na CPU o que a fase B mediu para
n=1: realimentar a cabeça com a própria previsão (o llama.cpp realimenta o
`t_h_nextn` da cabeça — `common/speculative.cpp:1672-1675`) e contar acertos do 2º
token condicionados ao 1º ter acertado.

- [ ] Se `a₂ ≥ 40%`: `tokens/passo = 1 + 0,61 + 0,61·a₂ ≥ 1,85` — implementar
      `plan_verify` com n_tok=3 e dois pontos de snapshot vira a forma mais barata de
      chegar a 50 (orçamento de passo sobe de 32 para ~37 ms).
- [ ] Se `a₂ < 40%`: ficar em n=1 e fechar os 50 pela frente 1 (base ≥33) — registrar o
      número medido para a decisão não ser revisitada.

## Critério de aceite da frente

- [ ] Greedy lossless com `--mtp` (sequências idênticas, 256 tokens).
- [ ] tok/s com MTP ≥ 1,5× a base da mesma build em greedy.
- [ ] Overhead do passo MTP (cabeça+snapshot+logits extra) ≤ 2,5 ms medido no perfil.
- [ ] Tabela de decode atualizada com os pares com/sem MTP e a aceitação encadeada.
