# Notas de tuning

Decisões de desempenho que exigiram medição para chegar ao desenho atual — o "porquê" por trás
de escolhas que, à primeira vista, não são a opção óbvia. Os comentários no código apontam para
cá em vez de repetir esse histórico.

## Barreiras: agrupar dispatches em vez de uma por operação

`ResidentForward::marcar_barreiras` decide, para cada operação do plano de decode, se ela precisa
de uma barreira de memória antes de rodar. Duas operações entre barreiras podem executar
concorrentes na GPU.

Antes, uma barreira era emitida depois de **todo** dispatch. Isso custa um "tail" por operação:
nenhum workgroup do próximo dispatch começa antes do último do anterior terminar, e o fim de um
matvec ocupa poucas waves dos SIMDs disponíveis — o resto da GPU fica ociosa esperando. Numa
camada densa, as projeções Q/K/V leem a mesma ativação e escrevem buffers distintos (o mesmo vale
para `ffn_gate`/`ffn_up`), então não há dependência real a respeitar entre elas.

O critério: uma barreira só é necessária quando a operação conflita com o que o grupo corrente já
fez — lê o que foi escrito (RAW), escreve o que foi lido (WAR), ou escreve o que já foi escrito
(WAW). O critério é conservador de propósito em dois pontos: compara as faixas **inteiras** dos
bindings (não o que o shader de fato toca) e trata o KV-cache como um buffer só, já que o offset
do append depende da posição do token e não é conhecido nesse ponto do planejamento.

Ganho medido: **+5.3%** no Qwen2.5-32B em layer-split. `LLAMA_RS_NO_GROUP=1` volta ao
comportamento antigo (uma barreira por operação), para comparar A/B no mesmo binário.

## Espera do fence: sondar em vez de bloquear

Um `wait_for_fences` bloqueante entrega a thread ao escalonador do SO pelos ~25 ms que o shard
leva para rodar. Nesse intervalo o governor de frequência (`schedutil`) vê baixa utilização de
CPU e reduz o clock, e a CPU entra em C-state profundo — o wakeup pela interrupção da GPU passa a
custar milissegundos em vez de microssegundos.

O sintoma era tempo por token **bimodal** (dois valores estáveis, não uma distribuição contínua),
com o tempo de GPU medido idêntico até o décimo de microssegundo nos dois casos — o que aponta
para latência de wakeup do host, não para variação real de trabalho de GPU. Qualquer busy-loop
rodando em paralelo "consertava" o número, o que fechou o diagnóstico.

A solução é sondar o fence (polling) em vez de esperar bloqueado. Dormir a maior parte do tempo
previsto e sondar só o final não resolve — a latência de wakeup já é paga ao acordar de um sono
longo, mesmo quando é por timeout, não por evento.

Medido no Qwen2.5-32B em layer-split: 17.5–18.9 tok/s oscilando com o wait bloqueante, contra
**19.4 tok/s estável** sondando. O custo é um núcleo de CPU ocupado enquanto a GPU trabalha —
o mesmo compromisso que o llama.cpp faz no laço de espera dele.

## Matvec K-quant (Q5_K/Q6_K/Q4_K): requests de memória, não bytes

O kernel de matvec Q5_K (`shaders/q5_k_matvec.comp`, com Q4_K seguindo a mesma estrutura) passou
por uma sequência de decisões que só fazem sentido olhando o que foi medido:

- **Ler em `uvec4` em vez de escalar fecha a maior parte da banda.** Com loads escalares, cada
  lane emitia 17 requests de 4 bytes por superbloco; em `uvec4` são 5 requests de 16 bytes — os
  mesmos bytes totais, com 3.4× menos requests ao cache L1. O kernel não era limitado pelos bytes
  em si (o L2 já deduplicava a redundância entre lanes vizinhas), era limitado pela **taxa de
  requests**: medimos 465 GB/s efetivos contra um teto de 717 GB/s de pico nesta arquitetura.
- **Reduzir a redundância entre lanes (em vez dos requests) foi medido 4% mais lento.** A
  alternativa óbvia — uma lane por par de sub-blocos que compartilha os mesmos bytes de `qs`, 4
  lanes por superbloco em vez de 8 — deixa 16 das 64 lanes ociosas na última rodada de um kernel
  com 20 superblocos por linha, e essa ociosidade custa mais do que a leitura duplicada que o
  cache já absorvia.
- **Duas linhas de saída por wave, não uma nem quatro.** Com 2 linhas, os loads da ativação valem
  para as duas, e os requests por linha caem de 8 para 6.5 — medido **-8%** no kernel. Com 4
  linhas, os acumuladores extras derrubam a ocupância e a curva volta a piorar (+3%).
- **Mais ocupância nem sempre ajuda aqui.** Com 1 linha por wave o kernel usa menos registradores
  (36 VGPRs) e cabe 7 waves por SIMD em vez de 6 — e mede **6% mais lento**. O que decide não é
  ter mais waves em voo, é reler a ativação menos vezes.
- **O dot empacotado em int8 (`dotPacked4x8AccSatEXT`) foi validado, mas o ganho medido é ~0%** —
  o kernel já não era limitado por ALU nesse ponto, só por requests de memória.

Ver `scripts/tune-matvec.sh` para a varredura de geometria (`LLAMA_RS_MATVEC_GEOM`), e o topo de
`shaders/q5_k_matvec.comp` para o layout de bits exato.

## GPU do display e spill para GTT

Quando um modelo quase enche a VRAM de uma GPU que também roda o display, o driver AMD pode
realocar o excedente em GTT (memória do host, acessada via PCIe) — silenciosamente, sem erro de
alocação. O sintoma é banda efetiva muito abaixo do esperado (medimos 95 GB/s contra 714 GB/s na
mesma GPU sem esse problema), sem nenhum indício direto da causa. Ver
[`hardware.md`](hardware.md) para a mitigação (seleção de GPU por VRAM livre + margem reservada).

## Delta-net: o desenho anterior do shader

O shader `delta_net.comp` (recorrência do gated delta-net, ver
[`qwen35-arquitetura.md`](qwen35-arquitetura.md)) usa uma wave por coluna do estado, com o estado
em registrador — o layout segue o kernel CUDA de referência do llama.cpp.

O desenho anterior punha uma thread por linha, cada uma percorrendo os `d` floats sozinha: sem
nenhuma redução entre lanes, o que parecia mais simples, mas com dois problemas que custam mais
do que a redução evita — o estado era lido e escrito **duas vezes** por token (uma vez para o
decaimento e o produto com a chave, outra para a atualização e o produto com a query), e as 64
lanes de uma wave liam endereços separados por `d × 4` bytes — uma linha de cache inteira por
lane, sem coalescência nenhuma.
