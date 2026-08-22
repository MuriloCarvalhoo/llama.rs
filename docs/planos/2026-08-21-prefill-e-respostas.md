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

> **Estado desta frente:** (1) e (3) resolvidos e provados por teste (tarefas 1 e 2); o
> GEMM da tarefa 3 existe atrás de knob, desligado; a tarefa 4 está fechada menos o
> registro dos números. **Nada aqui foi medido em tok/s ainda** — todas as linhas de ganho
> continuam abertas de propósito. (2), a releitura do KV pela atenção em batch, não foi
> tocada.

## Tarefa 0 — Fechar o diff pendente (frente 0)

**Arquivos:** `crates/llama-vulkan/src/resident_forward.rs` (já modificado, não
commitado).

- [ ] Rodar os testes de prefill e o gate; medir um bloco com `LLAMA_RS_PROFILE=1` e
      guardar a primeira tabela "PERFIL GPU — prefill em batch" em
      `docs/decode-por-configuracao.md` (é a linha de base desta frente).
- [x] Commitar: geometria `(64,1)` do matvec de batch (8,9 ms vs 24,1 por bloco),
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

- [x] `delta_net.comp`: adicionar `n_tok` por push constant e o laço externo sobre
      tokens; cada lane já guarda suas colunas do estado em registrador (o layout atual
      de 1 wave por coluna se mantém — é o mesmo do kernel de referência). Entradas
      q/k/v/g/beta passam a ser indexadas por token. Junto veio `v_stride`: `v` é fatia do
      buffer da convolução, com passo `conv_dim` e não `n_heads × d`. Os laços sobre
      `por_lane` viraram `[[unroll]]` com teto de compile-time e guarda de runtime — com
      limite dinâmico o array de estado ia para scratch, que é memória global.
- [x] `dn_conv.comp`: mesmo movimento — a janela de conv avança token a token dentro do
      dispatch, em registrador (`MAX_PASSOS` cobre `d_conv` até 5).
- [x] `dn_gates` e `dn_l2_qk` (a L2 fundida da frente 1): batchados por
      `gl_WorkGroupID.y` como as ops de atenção. O `dn_l2_qk` ganhou `stride` no push
      porque a entrada dele é o buffer da convolução (`q | k | v` por token).
- [x] Validar contra `crates/llama-model/src/delta_net.rs` com bloco de 8 tokens.
      Medido: 2,4e-7 no estado final e ~1e-7 nas 8 saídas do delta-net (com q/k
      normalizados em L2, como o pipeline os entrega); a janela da convolução sai bit a
      bit igual. `dn_gates` e `dn_l2_qk` rodam com `n_tok ∈ {1, 4}`.
- [ ] Ganho no batch 8: ~1,3 ms/bloco da parte serial some quase todo — **pendente de
      medição**. O ganho real é destravar batch grande (tarefa 2) sem explosão de
      dispatches: em batch 32 seriam 48×32×4 = 6k dispatches/bloco no desenho antigo.

## Tarefa 2 — Subir o teto do batch: 8 → 32

**Arquivos:** `crates/llama-vulkan/shaders/q6_k_matvec.comp` (`MAX_COLS = 8`),
`resident_forward.rs` (buffers, `batch_size`).

- [x] Q6_K: `acc`/`acc_b` dimensionados por `COLS` em vez do `MAX_COLS = 8` fixo (era a
      única trava de shader). `batch_size()` clampa em 1..=32; o padrão segue 8, que é o
      valor medido. Teste: `matvec_k_em_batch_bate_coluna_a_coluna` varre
      cols ∈ {2,4,8,16,24,32} nos três K-quants — com o `MAX_COLS` de volta, a coluna 8
      de 16 volta NaN.
- [x] Buffers de ativação dimensionados por `n_batch` — conferidos: todos passam por
      `nf`, inclusive os do caminho delta e as parciais da norma (que indexam por
      `t * n_parciais`). `b_logits` segue com um token só.
- [ ] Varrer `LLAMA_RS_BATCH ∈ {8, 16, 24, 32}` com o perfil de bloco e registrar
      ms/bloco e tok/s de prefill. A expectativa é o matvec por token cair até a
      pressão de registrador dos acumuladores `COLS` reverter a curva — o ponto ótimo é
      empírico e provavelmente ≤32 sem GEMM. **Pendente de medição.**

## Tarefa 3 — GEMM com tiling em LDS (etapa 2 do desenho antigo)

**Arquivos:** novo `crates/llama-vulkan/shaders/mul_mm.comp`, `resident_forward.rs`
(seleção matvec-COLS vs GEMM por largura do bloco).

Acima de ~32 tokens o matvec de N colunas para de pagar; o caminho é o GEMM clássico
com tile em LDS, que o prefill do llama.cpp usa (`mul_mm.comp`). Aqui a MI50 é
compute-bound e o **dot int8 empacotado passa a valer** (ao contrário do decode, onde
mediu ~0%): ~52 TOPS int8 contra 26 TFLOPS f32.

- [x] Geometria: workgroup 256, tile **128×COLS×32** em vez de 64×64×32 — com o teto de
      batch em 32, um tile de 64 colunas desperdiçaria metade do trabalho; 128 linhas por
      32 colunas mantém os 4096 resultados por workgroup e a mesma intensidade
      (`TM=4`, `TN=COLS/8`, grade 32×8). Ativação int8 do `quantize_x`, peso Q4_K
      desempacotado para int8 na LDS junto com escala e o termo afim.
- [x] `dotPacked4x8AccSatEXT` no laço interno.
- [x] Só para os matvec grandes: entra por `mv_gen`, no lugar do matvec-COLS, quando o
      peso é Q4_K e a largura cabe no tile. Atenção e delta seguem como estão.
- [ ] Critério de adoção: GEMM vence o matvec-COLS no mesmo bloco em ≥20%; senão,
      ficar na tarefa 2 e registrar o resultado (o custo de manter dois caminhos tem de
      ser pago por medição, não por fé). **Pendente de medição — o knob nasce desligado.**

**O que ficou de fora:** Q5_K e Q6_K (5,3% e 18,6% do tempo de matvec) seguem no
matvec-COLS. O Q5_K é estruturalmente igual ao Q4_K (mesma escala/mínimo de 6 bits, mais o
5º bit em `qh`) e entraria com pouca coisa; o Q6_K precisa de duas escalas por passo de K
(sub-blocos de 16, não de 32), o que muda o formato do tile. Só faz sentido pagar isso
depois de o A/B do Q4_K justificar o caminho.

## Tarefa 4 — Servidor: reuso tolerante e TTFT medido

**Arquivos:** `crates/llama-model/src/sessao.rs`, `crates/llama-server/src/motor.rs`,
`docs/servidor-opencode.md`.

Hoje qualquer divergência no meio do prompt descarta o cache inteiro — porque o estado
recorrente não volta atrás. Um snapshot por fronteira de turno torna a divergência
barata sem tocar em shader:

- [x] Copiar `estado` + `janela` + comprimento do KV para um snapshot (~155 MB de VRAM, 1
      só). **A fronteira é o fim do prefill, não o fim da resposta**: o que diverge no
      turno seguinte é o re-render da resposta que o modelo acabou de gerar, e ela vem
      toda depois do fim do prompt — guardar o fim da resposta deixaria a divergência
      antes do snapshot, ou seja, sem cobertura.
- [x] `planejar_reuso` ganha o caso `RecuarPara{pos}`: se o prefixo diverge **depois**
      da posição do snapshot, restaurar e reprocessar só dali; se diverge antes, cai no
      `Reiniciar` de hoje. `prefixo_comum` do servidor espelha a mesma decisão, senão o
      log contaria como prefill tokens que vieram do cache.
- [x] Junto: `GpuResidentDecode::reset` do backend de uma GPU só recuava o comprimento do
      KV e deixava o estado recorrente intacto — sequência nova começava sobre o estado da
      anterior. O `LayerSplitForward` já chamava `reset_len`; agora os dois coincidem.
- [x] Registrar TTFT no log do servidor (chegada do pedido → primeiro byte de stream).
- [ ] Anotar antes/depois em `docs/servidor-opencode.md`. **Pendente de medição.**

## Critério de aceite da frente

- [ ] Prefill ≥180 tok/s no prompt de 9.110 tokens do teste de servidor (hoje 61,8).
- [ ] Primeiro turno do opencode (~27k tokens) ≤3 min.
- [x] Divergência de template no turno 2 não reprocessa o histórico inteiro (teste da
      sessão com snapshot: `divergencia_depois_da_fronteira_de_turno_nao_reprocessa_o_historico`).
- [ ] Perfis de bloco antes/depois registrados.

## Varredura que falta rodar

Tudo aqui é medição no modelo real, com `LLAMA_RS_PROFILE=1`:

1. `LLAMA_RS_BATCH ∈ {8, 16, 24, 32}` — ms/bloco e tok/s de prefill. É a curva que decide
   o padrão de `batch_size()`, hoje 8 por herança.
2. Na melhor largura, `LLAMA_RS_PREFILL_GEMM ∈ {0, 1}` — o A/B do GEMM. Adotar só com
   ≥20% no bloco; senão, deixar o knob desligado e registrar o número.
3. TTFT do servidor no prompt de 9.110 tokens, antes e depois, para
   `docs/servidor-opencode.md`.
4. Turno divergente com o snapshot ligado: quantos tokens o prefill de fato reprocessa
   (a linha `[gen]` já separa "do cache" de "no prefill").
