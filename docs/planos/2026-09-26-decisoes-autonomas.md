# Decisões autônomas — 2026-09-26

Registro das decisões tomadas sem consulta ao executar `2026-09-25-proximos-ganhos.md` (pedido:
"se precisar tomar decisão anote que depois eu vejo"). Cada uma diz o que foi decidido, por quê
e como desfazer.

## Organização do trabalho

- **Itens que usam GPU ficam num agente só**, em série: medições concorrentes nas mesmas MI50 se
  contaminam, e duas cargas do 27B juntas quebram a margem de 2 GB da GPU do monitor.
- **Itens só de CPU vão para agentes em worktree** e entram no `master` por cherry-pick depois de
  revisados: (A) servidor — 4.2, 4.3, 4.5; (B) 4.1, 3.3, 4.4 (este último só compilado no agente;
  a verificação com a GPU é feita depois no `master`).
- **3.1 (refrigeração da card1) não é executado**: pede root (`pwm1`/`power1_cap`) e é decisão de
  hardware do usuário. Fica anotado com a recomendação no fim deste arquivo.

## Agente principal

1. **0.1 — snapshot logo depois do último token especial, e um só.** A causa do `0 do cache`
   era o BPE fundir o `\n` final do prompt de geração com o turno seguinte; a fronteira depois
   de um token especial (`Tokenizer::e_especial`) não se mexe. **Não** implementei vários
   checkpoints: cada snapshot custa ~155 MB de VRAM (estado recorrente das 48 camadas lineares)
   e o uso real (opencode) é uma conversa que só cresce. Uma conversa que **ramifica** (trocar
   uma mensagem do meio) ainda reinicia. Desfazer: `motor.rs` voltar a chamar `prefill`.
2. **A verificação do 0.1 usa turnos naturais** (`MARCO_NATURAL=1` em
   `scripts/bench-conversa/conversa.py`): o protocolo do benchmark de 2026-09-25 troca a última
   mensagem pela pergunta final nos marcos, o que é justamente uma ramificação.
3. **GEMM do Q6_K fica opcional** (`LLAMA_RS_GEMM_Q6K=1`): no bloco 32 ele foi mais lento que o
   matvec-COLS (39,5 contra 35 ms por bloco). O Q5_K vai pelo GEMM por padrão (52 → 14,7 ms).
4. **Revisei duas escolhas do agente A** (itens 1 e 2 da seção dele):
   - `model` diferente do servido **é atendido** (com uma linha `[api]` no log), como no
     llama.cpp. O 404 quebraria o opencode sempre que o servidor subisse sem `--nome`, e o
     servidor tem um modelo só. Desfazer: voltar o `return Err((404, ...))` em
     `servidor::validar_chat`.
   - `max_tokens: -1` (e `max_completion_tokens: -1`) vale como "sem limite", como no llama.cpp;
     os demais valores ≤ 0 seguem 400.
5. **"Veja como o llama.cpp faz de melhor, como o perfil, e copie em Rust"** — li como **o perfil
   do prefill**, onde o llama.cpp é 2,8× mais rápido (318 contra 115 tok/s com 7,6k): estudar o
   MMQ do HIP e portar a ideia para os shaders Vulkan, no padrão do repositório (item 1.4 + 1.2).
   Se o pedido era sobre a ferramenta de perfil em si, o `LLAMA_RS_PROFILE` já dá o tempo por op
   e o que falta é o 3.2 (memória).

## Agente A — robustez do servidor (itens 4.2, 4.3 e 4.5)

1. **`model` estrito, com o campo ausente aceito.** Vale `model` vazio/ausente ou igual ao
   `--nome` sem diferenciar maiúsculas; qualquer outro nome é 404 `model_not_found` (formato da
   OpenAI, `param: "model"`). Por quê: responder com outro modelo sem avisar esconde erro de
   configuração do cliente, e o uso documentado já casa — `scripts/run-server.sh` sobe com
   `--nome qwen3.8-27b`, o mesmo id do `opencode.json`. **Risco:** subir sem `--nome` expõe o
   nome do arquivo (`qwen3.8-27b-q4_k_m`) e o opencode passa a receber 404. Desfazer: fazer
   `api::modelo_confere` devolver `true`.
2. **`max_tokens` ≤ 0 é 400**, como na OpenAI. O llama.cpp aceita `-1` como "sem limite"; um
   cliente que dependa disso vai receber 400 aqui. `null` vale como ausente em todos os campos.
3. **`top_k` também é validado** (inteiro ≥ 0), além dos parâmetros pedidos: muda a resposta tanto
   quanto `top_p`, e antes um `-1` virava o padrão 20 em silêncio.
4. **Fila de 4 gerações** esperando o worker; cheia (ou com o worker morto), 503 com
   `Retry-After: 5`. Constante `servidor::FILA`, não flag — o pedido não pedia configurar.
5. **Prazos por leitura, não da requisição inteira.** `--timeout-leitura` (30 s) e
   `--timeout-escrita` (60 s) são `set_read_timeout`/`set_write_timeout` do socket: um cliente que
   manda um byte a cada 29 s ainda segura a thread do HTTP. Um prazo total pediria um relógio em
   `ler_requisicao`; ficou de fora por ser cliente local.
6. **A thread nova é a do HTTP; o worker é a thread principal**, a que criou o backend. Assim o
   Vulkan e a `Sessao` nunca mudam de thread e nenhum tipo precisou virar `Send`. Consequências:
   - a checagem do contexto (precisa do tokenizer) continua no worker (`Motor::preparar`), então
     um prompt longo demais só recebe o 400 quando chega a vez dele na fila;
   - o `ttft` do log conta a partir do `preparar`, sem a espera na fila;
   - cliente que desiste **enquanto está na fila** não é detectado: o worker só descobre ao
     escrever (no streaming, na primeira escrita; sem streaming, depois de gerar tudo).
7. **Limites de cabeçalho:** 8 KiB por linha e 64 KiB para linha de requisição + cabeçalhos, 431
   acima disso. `Content-Length` repetido com o mesmo valor é aceito; com valores diferentes ou
   não numérico, 400.
8. **Testes de rota passaram a mandar `"model":"modelo-teste"`** (o nome servido nos testes) em
   vez de `"m"`, que agora seria 404.

## Agente B — itens 4.1, 3.3 e 4.4

1. **4.1, laço do `llama-cli`:** a regra é a do motor do servidor, copiada, não compartilhada.
   Os dois laços têm estruturas diferentes (o servidor passa pela `Sessao` e por uma fila de
   `pendentes`; o CLI chama `passo_mtp` direto), e juntá-los seria uma refatoração maior que a
   correção. Para testar sem tokenizer, o laço foi para `gerar_ids_residente` (sobre ids).
   Prompt que enche o contexto (`len >= ctx`) virou `ModelError::ContextOverflow` antes do
   prefill — antes era um erro do backend no meio do prefill.
2. **3.3, `scripts/monitor-gpu.sh`:** lê o sysfs direto em vez de `sensors` (mais rápido, sem
   dependência) e mata só o PID. Os guardas de `scripts/bench-conversa/` ficaram como estão,
   porque o roteiro do benchmark os usa; dá para trocá-los pelo monitor depois.
3. **Monitor em repouso (achado no 3.3, corrigido também no `device.rs`):** às 2 h o
   `card2-DP-8` estava `disconnected` — o DisplayPort desconecta com a tela dormindo. A
   detecção por conector marcava então as duas GPUs como "sem monitor" (500 MiB), e o
   compositor precisa de VRAM na card2 quando a tela acorda. **Decisão:** sem nenhum conector
   ativo, 2 GiB em todas (script e `margens_vram` no Vulkan). Custo: um modelo que só coubesse
   com 500 MiB de folga na card2 não carrega com a tela dormindo — preferi o lado seguro.
4. **4.4, E06:** o mínimo nos três pontos citados (device, `Buf`, `alloc_with_flags`) e o `Drop`
   com as 31 pipelines. **Não** tratei o caso de `new_pipelines_only_on` falhar no meio da
   criação das pipelines (as já criadas sobram até o device morrer): exigiria um guarda por
   pipeline, e só acontece com o driver recusando um shader — raro e fatal de qualquer jeito.
