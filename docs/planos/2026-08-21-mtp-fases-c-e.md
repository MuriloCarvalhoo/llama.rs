# Frente 2 — MTP fases C–E: verify em batch, rollback e ligação

> **Estado (2026-08-21):** tarefas 1–6 implementadas sobre a frente 1 mergeada. O que falta
> é **medir**: nenhum número desta frente foi levantado no modelo real ainda, e os testes que
> o exigem dão skip limpo sem o GGUF. Ver "O que ficou pendente de medição" no fim.

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

- [x] Construir um **terceiro plano**, `plan_verify`, com `n_tok = 2` fixo, irmão do
      `plan_batch` (mesma `build_plan`, agora parametrizada por um `Modo` em vez de
      `n_tok`). Buffers de ativação seguem dimensionados por `n_batch`, com o piso subindo
      para 2 quando o MTP está ligado — `LLAMA_RS_BATCH=1` não deve apertá-los.
      **Quatro pipelines de matvec novas** (`COLS = 2`): `COLS` é specialization constant,
      e escolher a pipeline por `cols` em vez de pelo modo daria a geometria errada com
      `LLAMA_RS_BATCH=2`. A geometria delas é a do **decode** (`matvec_geom`), não a do
      prefill — duas colunas ficam perto de uma em pressão de registrador. *Pendente de
      medição: o par (wg, linhas) do verify nunca foi varrido.*
- [x] Fim do `build_plan` no modo verify: **nenhuma requantização**. O `norm_p2` da norma
      final já escreve os dois tokens em `b_xq`/`b_xd` no layout `t * n_blk + b`, que é
      exatamente o que o matvec de duas colunas lê; a projeção `output` roda com `COLS = 2`
      e `b_logits` passa a `vocab × 2`. O custo é 1 leitura do peso Q6_K de 0,63 GB para as
      duas colunas. *Pendente de medição: os ~0,3 ms projetados.*
- [x] `record_token`/`record_and_submit` escolhem o plano por `Modo`, e não por
      `tokens.len()`; `verify_shard` no backend e `decode_verify(&[t1, t2], pos0)` em
      `GpuResidentDecode`, com o `LayerSplitForward` atravessando os shards como no batch.
- [x] Teste de integração: `decode_verify_bate_com_o_decode_token_a_token` em
      `crates/llama-vulkan/tests/mtp_verify.rs` — argmax igual e erro relativo < 1e-2 nas
      duas metades. **Gated no modelo real** (skip limpo sem o GGUF).

## Tarefa 2 (fase C) — Snapshot do estado entre os tokens do verify

**Arquivos:** `resident_forward.rs` (`plano_delta_verify`, `PlannedOp::Copia`).

- [x] `plano_delta_verify`: a recorrência roda `n_tok` dispatches em ordem e, entre o
      token 0 e o token 1 de cada camada linear, entra um `PlannedOp::Copia` do
      `DeltaBufs.estado` (3,1 MB) e outro da `DeltaBufs.janela` (120 KB) para buffers de
      snapshot pré-alocados. `marcar_barreiras` serializa sozinho (RAW no estado, WAR na
      cópia). Custo projetado: 48 × 3,2 MB ≈ 155 MB ≈ 0,45 ms. *Pendente de medição.*
- [x] É uma **cópia** de `plano_delta`, deliberada: o caminho de batch vai migrar para um
      kernel multi-token (frente 3) e aí o ponto de parada entre os dois tokens deixa de
      existir. O verify não pode herdar essa mudança sem perder o rollback.
- [x] Os buffers de snapshot só são alocados com MTP ligado (`new_shard_com`): 155 MB de
      VRAM que o caminho padrão não deve pagar.
- [x] O KV-cache não precisa de snapshot: rejeitar é recuar o comprimento em 1
      (as posições do verify são consecutivas; o próximo append sobrescreve).
- [ ] Alternativa a considerar **só se** 0,45 ms doer na medição final: o shader
      `gated_delta_net.comp` do llama.cpp escreve snapshots de dentro do kernel
      (parâmetro K, slot `n_tokens-1-t` — `ggml.h:2564-2585`); mesma banda, um dispatch
      a menos. Não começar por aqui.

## Tarefa 3 (fase D) — Rollback

**Arquivos:** `resident_forward.rs`.

- [x] `rollback_verify()`: um command buffer gravado **uma única vez** na construção com as
      cópias de volta (`estado` + `janela` de cada camada linear do shard), mais o
      comprimento do KV recuando 1. A rejeição custa só um submit, sem gravação. Executado
      em ~39% dos passos pela aceitação medida na fase B — na média 0,39 × 0,45 ≈ 0,18
      ms/passo. *Pendente de medição.*
- [x] `Sessao` não mudou: o verify avança `pos` em 2 na aceitação e em 1 na rejeição, e
      quem chama só vê a posição final. O laço de geração do `llama-model` é quem faz essa
      conta.
- [x] Teste `rollback_restaura_o_estado_de_um_token`: propõe um token errado de propósito e
      confere que o passo seguinte dá os mesmos logits do caminho que nunca viu a proposta.
      **Gated no modelo real.**
- [x] Teste sintético `snapshot_do_estado_desfaz_exatamente_um_token` (roda sem modelo, na
      GPU): a sequência (A, B, rollback, C) tem de dar bit a bit o mesmo que (A, C) — o que
      prende que copiar `estado`/`janela` é *suficiente* para desfazer um token.

## Tarefa 4 (fase C) — Cabeça MTP na GPU

**Arquivos:** `crates/llama-model/src/mtp.rs` (referência CPU já validada),
`resident_forward.rs` (plano da cabeça), `crates/llama-model/src/gpu.rs`.

Todas as ops existem (`docs/mtp-implementacao.md`, tabela da fase B). O plano da cabeça
é uma camada de atenção qwen35 normal mais o prólogo:

- [x] `enorm(emb(t+1))` e `hnorm(h_t)` com `NormFused`+`NormP2`, escrevendo nos offsets
      0 e `n_embd` do mesmo buffer (`b_eh`) — a concatenação é só binding com offset. Um
      `QuantizeX` sobre os `2 × n_embd` fecha, porque cada `norm_p2` quantizou só a sua
      metade.
- [x] `eh_proj` (matvec Q8_0, 10240→5120), camada de atenção completa (o bloco MTP **é
      de atenção**), `shared_head_norm`, projeção `output` — o GGUF do 27B não tem
      `nextn.shared_head.head` próprio, então é a `output.weight` compartilhada, o mesmo
      fallback do llama.cpp.
- [x] KV-cache próprio do bloco: 8 KB/token, buffers dele. O contador anda **1 por
      proposta**, como a referência de CPU. *Ponto em aberto, a medir:* em produção a
      cabeça propõe uma vez por passo mas o modelo pode avançar dois tokens, então as
      posições da cabeça ficam para trás das do modelo — algo que a medição da fase B (uma
      proposta por token) não exercitou. O llama.cpp trata o cache da cabeça do mesmo jeito.
- [x] A entrada `h_t` sai de `b_x` do último shard (o residual pós-camadas, pré-norma
      final), copiado para `b_h` por um `PlannedOp::CopiaHidden` cujo offset é escolhido na
      gravação: 0 depois de um decode ou de um verify rejeitado, 1 depois de um aceito. A
      cabeça roda inteira no shard que tem a norma final; só a linha de embedding (20 KB)
      atravessa do primeiro shard, que é quem carrega a tabela.
- [x] Validação: `cabeca_mtp_na_gpu_bate_com_a_referencia_de_cpu` compara com
      `MtpHead::propor` token a token. **Gated no modelo real.** É também a rede contra o
      modo de falha silencioso de contagem de bindings.

## Tarefa 5 (fase E) — Ligar, medir, registrar

**Arquivos:** `crates/llama-cli/src/args.rs`, `crates/llama-server/src/motor.rs`,
`docs/decode-por-configuracao.md`.

- [x] Flag `--mtp` no CLI e no servidor; **desligada por padrão**. Quando o modelo não traz
      bloco `nextn` a flag é ignorada com aviso, em vez de reservar VRAM à toa.
- [x] Laço de geração (`gerar_streaming_residente`): com MTP o passo vira
      propor→verify→aceitar/rollback. Com sampling (temp > 0) a proposta só é aceita se
      coincidir com o token que o sampler tira dos logits[0] — a distribuição continua
      exatamente a mesma do caminho sem MTP.
- [x] O prefill passou a devolver **qual token do último bloco** produziu os logits: com
      prompt múltiplo de `n_batch` o hidden de partida não é o do índice 0, e a primeira
      proposta sairia de um hidden errado.
- [ ] **Servidor:** a flag constrói o backend com MTP, mas o laço do `motor.rs` continua
      decodificando um token por passo — convertê-lo exige um `decode` de dois tokens na
      `Sessao`, que é da frente de sessão/TTFT. Até lá `--mtp` no servidor só custa VRAM.
- [x] Teste de aceite lossless: `greedy_com_mtp_gera_a_mesma_sequencia`, 256 tokens com e
      sem MTP no mesmo backend. **Gated no modelo real.** *Falta rodá-lo.*
- [ ] Medir pares com/sem MTP na mesma build, `LLAMA_RS_SPLIT` fixo, e registrar em
      `docs/decode-por-configuracao.md` — as linhas "MTP ligado/desligado" estão
      reservadas lá desde a fase B. Registrar greedy **e** temp 0.8.

## Tarefa 6 — Experimento barato: aceitação encadeada (n=2)

**Arquivos:** `crates/llama-vulkan/tests/mtp_aceitacao.rs` (estender).

Antes de qualquer código de produção para n=2, medir na CPU o que a fase B mediu para
n=1: realimentar a cabeça com a própria previsão (o llama.cpp realimenta o
`t_h_nextn` da cabeça — `common/speculative.cpp:1672-1675`) e contar acertos do 2º
token condicionados ao 1º ter acertado.

- [x] Harness pronto: `aceitacao_encadeada_do_segundo_token`. `MtpHead::propor_com_hidden`
      devolve o hidden do próprio bloco (o análogo do `b_x` do modelo), e a cabeça é
      chamada duas vezes por passo — a segunda com a própria previsão e o próprio hidden.
      O 2º token só é cobrado quando o 1º acertou, que é a condicional que interessa.
      O teste imprime `a₁`, `a₂` e os tokens/passo dos dois desenhos, e **não falha** por
      `a₂` baixo: ele existe para produzir o número.
- [x] **`a₂` medido em 2026-08-21**: 41,7 % (5/12, condicionado ao 1º) → o critério
      dos 40 % passou e o n=2 foi implementado em 2026-08-22: `VERIFY_TOK = 3`, dois
      pontos de snapshot, `rollback_verify(manter)` e a proposta encadeada reusando o
      plano da cabeça (`HIDDEN_CABECA` lê o residual do próprio bloco). Em geração
      real greedy: **2,20 tokens/passo** (140/117) e 34,5 tok/s de média (36 nas
      execuções quentes) contra 31,4 do n=1. Ver
      `docs/planos/2026-08-22-decisoes-autonomas.md`.
- [x] n=2 implementado também no motor do servidor (`Sessao::passo_mtp` + fila de
      pendentes no `motor.rs`): 36,2 tok/s greedy ponta a ponta, 32,6 com temp 0,8.

## Critério de aceite da frente

- [ ] Greedy lossless com `--mtp` (sequências idênticas, 256 tokens) — teste escrito,
      falta rodar no modelo real.
- [ ] tok/s com MTP ≥ 1,5× a base da mesma build em greedy — **não medido**.
- [ ] Overhead do passo MTP (cabeça+snapshot+logits extra) ≤ 2,5 ms no perfil — **não
      medido**. O perfil já tem uma tabela própria do verify (`LLAMA_RS_PROFILE=1` imprime
      "verify do MTP (2 tokens)" ao lado das do decode e do prefill).
- [ ] Tabela de decode atualizada com os pares com/sem MTP e a aceitação encadeada.

## O que ficou pendente de medição

Nada desta frente foi medido: a worktree não tem o `models/`. Todos os números abaixo são
projeções herdadas do plano, não observações.

| O quê | Como medir |
|---|---|
| Aceitação em geração real (greedy) | `cargo test -p llama-vulkan --test mtp_verify greedy_com_mtp -- --nocapture` — o teste imprime acertos/passos |
| Aceitação com temp 0.8 | `llama-cli --mtp --temp 0.8` contra o mesmo prompt sem a flag |
| Aceitação encadeada `a₂` | `cargo test -p llama-vulkan --test mtp_aceitacao aceitacao_encadeada -- --nocapture` |
| tok/s com/sem MTP | `llama-cli --timings` nos dois modos, `LLAMA_RS_SPLIT` fixo |
| Custo do snapshot e da cabeça | `LLAMA_RS_PROFILE=1` — linhas `copia` e a tabela do verify |
| Geometria do matvec do verify | `LLAMA_RS_MATVEC_GEOM=wg,linhas` varre decode **e** verify juntos |
