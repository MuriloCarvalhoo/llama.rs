# Frente 3 — Prefill e respostas rápidas: 73 → ≥180 tok/s

> **Para execução:** depende da frente 0 (o diff pendente é exatamente a geometria e o
> perfil do plano de batch). Compartilha infraestrutura com o MTP (frente 2): logits por
> token e delta-net multi-token servem aos dois.

**Meta:** o tempo até o primeiro token com prompt real. O opencode manda ~27k tokens no
primeiro turno; hoje isso custa **11 minutos** (73 tok/s de prefill). O llama.cpp faz
191 tok/s (HIP, p512) e 312 (com contexto fundo e `-ub 2048`) no mesmo hardware. Meta:
**≥180 tok/s** (27k em ~2,5 min), com o GEMM abrindo caminho para mais.

**Por que 73 e não 8× o decode:** três causas medidas — (1) a recorrência delta-net
roda um dispatch por token mesmo em batch; (2) a atenção em batch relê o KV; (3) o
batch está preso em 8 pelo `MAX_COLS` do shader Q6_K. Com batch 8 os matvec dominam
menos e os itens serial/quadrático aparecem.

## Tarefa 0 — Fechar o diff pendente (frente 0)

**Arquivos:** `crates/llama-vulkan/src/resident_forward.rs` (já modificado, não
commitado).

- [ ] Rodar os testes de prefill e o gate; medir um bloco com `LLAMA_RS_PROFILE=1` e
      guardar a primeira tabela "PERFIL GPU — prefill em batch" em
      `docs/decode-por-configuracao.md` (é a linha de base desta frente).
- [ ] Commitar: geometria `(64,1)` do matvec de batch (8,9 ms vs 24,1 por bloco),
      correção do `rows_q5k` no plano de batch (dispatch cobria fração das linhas) e
      perfil dos dois planos.

## Tarefa 1 — Delta-net multi-token: N dispatches → 1

**Arquivos:** `crates/llama-vulkan/shaders/delta_net.comp`, `dn_conv.comp`,
`resident_forward.rs` (`plano_delta`).

O desenho do llama.cpp resolve a serialização sem violar a recorrência: **o laço sobre
os tokens fica dentro do kernel, com o estado em registradores**
(`gated_delta_net.comp:121-182` — o estado só toca a memória global na entrada e na
saída do dispatch, não a cada token). Hoje o llama-rs relê e regrava 3,1 MB de estado
por camada **por token** do bloco.

- [ ] `delta_net.comp`: adicionar `n_tok` por push constant e o laço externo sobre
      tokens; cada lane já guarda suas colunas do estado em registrador (o layout atual
      de 1 wave por coluna se mantém — é o mesmo do kernel de referência). Entradas
      q/k/v/g/beta passam a ser indexadas por token.
- [ ] `dn_conv.comp`: mesmo movimento — a janela de conv avança token a token dentro do
      dispatch (o upstream processa 16 tokens por workgroup no `ssm_conv.comp`).
- [ ] `dn_gates` e `dn_norm` L2: batchar por `gl_WorkGroupID.y` como as ops de atenção
      já fazem (não têm recorrência; eram seriais só por simetria com o resto).
- [ ] Validar contra `crates/llama-model/src/delta_net.rs` com bloco de 8 tokens
      (extensão do teste de shader existente).
- [ ] Ganho no batch 8: ~1,3 ms/bloco da parte serial some quase todo. O ganho real é
      destravar batch grande (tarefa 2) sem explosão de dispatches: em batch 64 seriam
      48×64×4 = 12k dispatches/bloco no desenho atual.

## Tarefa 2 — Subir o teto do batch: 8 → 32

**Arquivos:** `crates/llama-vulkan/shaders/q6_k_matvec.comp` (`MAX_COLS = 8`),
`resident_forward.rs` (buffers, `batch_size`).

- [ ] Q6_K: transformar `MAX_COLS` em specialization constant como nos outros matvec
      (é a única trava de shader).
- [ ] Buffers de ativação dimensionados por `n_batch` (hoje já são; conferir os do
      caminho delta e o `b_logits` — o prefill segue calculando logits só do último
      token do bloco, isso não muda).
- [ ] Varrer `LLAMA_RS_BATCH ∈ {8, 16, 24, 32}` com o perfil de bloco e registrar
      ms/bloco e tok/s de prefill. A expectativa é o matvec por token cair até a
      pressão de registrador dos acumuladores `COLS` reverter a curva — o ponto ótimo é
      empírico e provavelmente ≤32 sem GEMM.

## Tarefa 3 — GEMM com tiling em LDS (etapa 2 do desenho antigo)

**Arquivos:** novo `crates/llama-vulkan/shaders/mul_mm.comp`, `resident_forward.rs`
(seleção matvec-COLS vs GEMM por largura do bloco).

Acima de ~32 tokens o matvec de N colunas para de pagar; o caminho é o GEMM clássico
com tile em LDS, que o prefill do llama.cpp usa (`mul_mm.comp`). Aqui a MI50 é
compute-bound e o **dot int8 empacotado passa a valer** (ao contrário do decode, onde
mediu ~0%): ~52 TOPS int8 contra 26 TFLOPS f32.

- [ ] Ponto de partida de geometria: o warptile que o upstream usa para GCN —
      workgroup 256, tile 64×64×32 (`ggml-vulkan.cpp:4287-4289`) — e ativação quantizada
      int8 (o `quantize_x` já produz), pesos K-quant desempacotados para int8 no tile.
- [ ] `dotPacked4x8AccSatEXT` no laço interno (a extensão já é usada nos matvec).
- [ ] Só para os matvec grandes (projeções e FFN); atenção e delta seguem como estão.
- [ ] Critério de adoção: GEMM vence o matvec-COLS no mesmo bloco em ≥20%; senão,
      ficar na tarefa 2 e registrar o resultado (o custo de manter dois caminhos tem de
      ser pago por medição, não por fé).

## Tarefa 4 — Servidor: reuso tolerante e TTFT medido

**Arquivos:** `crates/llama-model/src/sessao.rs`, `crates/llama-server/src/motor.rs`,
`docs/servidor-opencode.md`.

Hoje qualquer divergência no meio do prompt descarta o cache inteiro — porque o estado
recorrente não volta atrás. Um snapshot por fronteira de turno torna a divergência
barata sem tocar em shader:

- [ ] Ao fim de cada resposta, copiar `estado` + `janela` + comprimento do KV para um
      snapshot (155 MB de VRAM, 1 snapshot só — o caso real do opencode é "histórico
      igual + turno novo", e o snapshot cobre a divergência típica de re-render do
      template).
- [ ] `planejar_reuso` ganha o caso `RecuarPara{pos}`: se o prefixo diverge **depois**
      da posição do snapshot, restaurar e reprocessar só dali; se diverge antes, cai no
      `Reiniciar` de hoje.
- [ ] Registrar TTFT no log do servidor (prompt N tokens → primeiro byte de stream) e
      anotar antes/depois desta frente em `docs/servidor-opencode.md`.

## Critério de aceite da frente

- [ ] Prefill ≥180 tok/s no prompt de 9.110 tokens do teste de servidor (hoje 61,8).
- [ ] Primeiro turno do opencode (~27k tokens) ≤3 min.
- [ ] Divergência de template no turno 2 não reprocessa o histórico inteiro (teste da
      sessão com snapshot).
- [ ] Perfis de bloco antes/depois registrados.
