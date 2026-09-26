# Decisões autônomas — 2026-09-26

Registro das decisões tomadas sem consulta ao executar `2026-09-25-proximos-ganhos.md` (pedido:
"se precisar tomar decisão anote que depois eu vejo"). Cada uma diz o que foi decidido, por quê
e como desfazer.

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
