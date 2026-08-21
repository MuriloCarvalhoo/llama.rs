# Frente 1 — Decode base: TOTAL GPU 41,35 → ≤30 ms (sem MTP)

> **Para execução:** tarefas em ordem; cada uma fecha com medição pelo protocolo de
> `docs/decode-por-configuracao.md` (3+ execuções, `LLAMA_RS_SPLIT=31`, comparar
> `TOTAL GPU` do `LLAMA_RS_PROFILE=1`, nunca tok/s isolado) e registro da linha na
> tabela daquele doc. Testes de regressão: os 28 testes de integração de
> `crates/llama-vulkan` a cada mudança de shader.

**Meta:** base sem MTP em 31–33 tok/s. É o caminho "sem MTP" do pedido e o
pré-requisito aritmético do "com MTP ≥50".

**Onde está o tempo hoje** (`LLAMA_RS_PROFILE=1`, soma das 2 GPUs, pós-`dn_gates`):

| op | ms/token | banda | teto realista |
|---|---:|---:|---|
| matvec_q4k | 22,49 | 506 GB/s | 17–19 ms a 600–680 GB/s |
| matvec_q6k | 7,91 | 573 GB/s | 6,7–7,0 ms |
| matvec_q5k | 2,26 | 460 GB/s | 1,7–1,9 ms |
| dn_gates (corrigido) | ~1,8 | — | ~1,3 ms |
| outras 13 ops | 6,91 | — | 3,5–4,5 ms com fusão |
| **TOTAL** | **41,35** | 501 agregado | **~30 ms** |

O orçamento fecha se: matvec_q4k ganhar ~20%, e metade do custo das ops pequenas sumir
por fusão. A varredura de geometria já provou que **geometria não é o gargalo**
(±1,2%); o que resta é reduzir requests/bytes por op e matar dispatches.

## Tarefa 1 — Instrumentar banda por op e por GPU

**Arquivos:** `crates/llama-vulkan/src/resident_forward.rs` (função `perfil_de`).

O perfil imprime ms/op; a banda é calculada à mão. Automatizar para não errar a conta e
para expor diferença entre as duas GPUs (a GPU do display pode estar com spill para GTT
— 95 vs 714 GB/s, `docs/performance-tuning.md:77-81`).

- [x] Anotar em cada `PipeId` os bytes lidos por dispatch (peso + ativação; o plano sabe
      as formas) e imprimir `GB/s` na tabela do perfil. Feito derivando das faixas de
      leitura de `PipeId::acessos`, sem contar duas vezes o mesmo binding; a atenção fica
      como `—` porque liga o KV inteiro e lê só `total_len` posições.
- [x] Imprimir as duas GPUs em tabelas separadas (hoje o perfil soma os shards). O
      cabeçalho da tabela passou a trazer `GPU<n> <nome> … camadas a..b`, então cada
      tabela se identifica sozinha.
- [ ] Rodar o baseline e registrar: se uma GPU sustenta banda visivelmente menor na
      mesma op, investigar GTT/clock antes de tocar em shader. **Pendente de medição.**

## Tarefa 2 — matvec_q4k: fechar 506 → ≥600 GB/s

**Arquivos:** `crates/llama-vulkan/shaders/q4_k_matvec.comp`, `scripts/tune-matvec.sh`.

O q6k sustenta 573 com a mesma estrutura; o q4k tem os mesmos bytes por superbloco lido
em `uvec4` — a diferença de 13% tem causa encontrável. Experimentos em ordem de custo,
cada um A/B pelo `TOTAL GPU` e revertido se não pagar:

- [x] **Contar requests por superbloco** no shader atual (papel e lápis, como foi feito
      no q5k): 144 B/superbloco Q4_K = 9 `uvec4` + escalas. Se as escalas (`d`/`dmin` +
      6 B de mins) saem em loads separados dos `qs`, empacotar a leitura para caber nos
      mesmos requests — foi exatamente essa conta que levou o q5k de 465 a 573.
      **Feito, e não há folga:** as escalas moram nos bytes 4..15 do mesmo `uvec4` do par
      `d|dmin`, então não custam request nenhum. São 3 loads por lane e por superbloco
      (`hdr`, `qs0`, `qs1`), e as 8 lanes cobrem os 9 `uvec4` do superbloco sem reler byte
      nenhum — é o piso da estrutura. O Q5_K faz **5** loads por lane e é mais rápido, o
      que descarta taxa de requests como causa dos 13%. Conta no topo de
      `q4_k_matvec.comp`; nenhum repack a fazer.
- [x] **Truque de ocupância do upstream**: alocar LDS morta para limitar waves por SIMD
      (o backend Vulkan do llama.cpp faz isso em GCN — `ggml-vulkan.cpp:3767-3777`,
      comentário "*too many subgroups... thrashing the cache*"). Testar 2 e 4
      subgroups/SIMD via um array `shared` não usado dimensionado por spec constant.
      Custo: ~10 linhas de shader, reversível por spec constant = 0.
      **Implementado** como `LDS_PAD_KIB` (constant_id 3) em `q4_k_matvec.comp`, ligável
      por `LLAMA_RS_MATVEC_LDS_PAD=K`, **default 0 = comportamento atual**. K=22 dá 2
      waves/SIMD e K=13 dá 4 (tabela no shader). Só a pipeline do decode; a do bloco tem
      outra geometria. **Pendente de medição** — nenhum K foi comparado ainda.
- [ ] **Distribuir a cauda**: com `(256,2)` cada dispatch cobre n_linhas/8 workgroups;
      medir se o último rank de workgroups deixa SIMDs ociosos (visível no trace
      Perfetto, `docs/debugging.md`). Se sim, testar grid 2D com menos linhas por
      workgroup **sem** mudar acumuladores (a lição do q5k: o que decide é reler a
      ativação menos, não ter mais waves).
- [ ] Registrar cada resultado (inclusive os negativos) em
      `docs/decode-por-configuracao.md` — a tabela de fracassos já evitou retrabalho
      duas vezes.

**Não fazer:** dot int8 empacotado (medido ~0%), mais linhas por wave (medido +3%),
nwarps maiores (memória do projeto: −11% a −41% no llama.cpp).

## Tarefa 3 — Fusão das ops pequenas: 6,91 → ≤4,5 ms

**Arquivos:** `crates/llama-vulkan/shaders/*.comp`, `resident_forward.rs`
(`build_plan`, `plano_delta`, tabela `PipeId::acessos`).

Cada dispatch pequeno paga tail de barreira (o motivo do +5,3% do agrupamento de
barreiras). Fusões na ordem do ganho esperado, **uma por vez**, validando cada uma
contra a referência CPU (`crates/llama-model/src/delta_net.rs` e os testes de
integração) — o risco conhecido é binding silencioso errado
(`docs/mtp-implementacao.md:82-85`):

- [x] **`dn_norm` L2 de q e k num dispatch só** (hoje são 2 por token por camada linear —
      96 dispatches/token). O shader já tem `modo` por push constant; adicionar modo que
      processa os dois tensores com offsets.
      **Feito** como shader próprio `dn_l2_qk.comp` (3 bindings: conv → qn, kn) em vez de
      um modo: q e k saem em **buffers distintos**, e o `dn_norm` só tem um binding de
      saída — acrescentar outro mudaria o número de bindings de todos os modos. 96 → 48
      dispatches/token. Teste `l2_de_q_e_k_no_mesmo_dispatch_bate_com_a_referencia`.
      **Pendente de medição.**
- [x] **`gate_mul` + `quantize_x`** (camadas de atenção): o portão sigmoide escreve e o
      quantize relê o mesmo buffer. Um shader `gate_quant` que aplica o portão e
      quantiza na mesma passada; o precedente é o `norm_p2`, que já quantiza direto.
      **Feito** — `gate_quant.comp`, uma lane por bloco de 32 como o `norm_p2`.
      `gate_mul.comp` ficou sem uso e saiu junto. Teste
      `gate_quant_bate_com_o_portao_da_cpu_e_com_o_quantize_x` (portão contra a CPU,
      `xq`/`xd` exatos contra o `quantize_x`). **Pendente de medição.**
- [ ] **`swiglu` + `quantize_x`** (todas as 65 camadas): mesmo padrão.
- [ ] **`rope` escrevendo K direto no slot do KV-cache** (camadas de atenção): hoje é
      rope in-place + `kv_append` (cópia). Rope com binding de saída no cache elimina a
      cópia de K; V continua no append. Atenção ao offset por posição — é o motivo de o
      planejador tratar o cache como buffer único em `marcar_barreiras`.
- [ ] Medir o conjunto; alvo ≤4,5 ms para "outras ops".

## Tarefa 4 — Host: só se a medição mandar

**Arquivos:** `resident_forward.rs` (`record_token`, `record_and_submit`).

O host custa ~2,4 ms/token (44,8 wall − 42,4 GPU). O perfil host já separa `gravacao` /
submit / espera.

- [ ] Medir a decomposição no baseline atual e registrar.
- [ ] Se `gravacao` ≥1 ms: pré-gravar o command buffer do decode e re-submeter,
      atualizando só o que muda por token (embed do token, push de posição do rope,
      comprimento da atenção, offset do kv_append). O plano é estático — o que varia
      cabe em push constants e num buffer de índices atualizado por `cmd_update_buffer`
      no início do CB. Se a variação de `splits_do_kv` atrapalhar, pré-gravar as 2
      variantes (curto/fatiado) e escolher no submit.
- [ ] Se sampling aparecer >1 ms com vocab 248k: já é paralelo; não tocar (medido
      0,4–1,4 ms).

## Critério de aceite da frente

- [ ] `TOTAL GPU ≤ 30 ms` no protocolo padrão (Qwen3.8-27B Q4_K_M, `LLAMA_RS_SPLIT=31`).
- [ ] tok/s ≥ 31 em 3 execuções consecutivas.
- [ ] Os 28 testes de integração passam; `cargo test --workspace` verde.
- [ ] Linhas novas na tabela de `docs/decode-por-configuracao.md` com o "como".
