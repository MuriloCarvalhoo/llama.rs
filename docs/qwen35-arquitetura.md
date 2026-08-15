# Qwen3.5/3.8 (`qwen35`) — o que o llama-rs precisa implementar

Notas tiradas da implementação de referência do llama.cpp (`src/models/qwen35.cpp`,
`src/models/delta-net-base.cpp`, `ggml/src/ggml-cpu/ops.cpp`) antes de escrever código.
Modelo alvo: **Qwen3.8-27B**, 64 camadas, `n_embd = 5120`, `n_ff = 17408`, vocab 248320.

O `qwen35` **não é um transformer denso**. Três em cada quatro camadas trocam a atenção
por uma **gated delta net** — um estado recorrente por cabeça, sem KV-cache. É por isso
que o llama-rs não carrega o modelo hoje: ele só conhece RMSNorm/RoPE/GQA/SwiGLU.

## Esqueleto de uma camada

`is_recr(il) = (il + 1) % full_attention_interval != 0`, com `full_attention_interval = 4`
— então as camadas 3, 7, 11, … são de atenção completa (16 no 27B) e as outras 48 são
lineares. Fora isso as duas variantes compartilham o mesmo esqueleto, que tem **duas**
normas por camada (a segunda depois do residual da atenção, estilo Gemma2):

```
h = RMSNorm(x, attn_norm)
h = camada_linear(h)  ou  atencao_completa(h)
x = x + h
res = x
h = RMSNorm(x, attn_post_norm)
x = res + FFN_SwiGLU(h)          # ffn_gate / ffn_up / ffn_down
```

## Atenção completa (16 camadas)

Duas diferenças em relação ao Qwen2.5 que o llama-rs já roda: **QK-norm** e um **gate
sigmoide** que vem junto da projeção de Q.

```
QG   = wq @ h                    # [n_embd_head * 2 * n_head]; por cabeça, Q depois gate
Q    = RMSNorm(QG[.., 0..d], attn_q_norm)      # por cabeça
K    = RMSNorm(wk @ h, attn_k_norm)            # por cabeça
V    = wv @ h
Q, K = MRoPE(Q, K, pos)
attn = softmax(Q Kᵀ / sqrt(d)) V               # GQA
out  = wo @ (attn * sigmoid(QG[.., d..2d]))
```

**MRoPE em texto puro é RoPE 1D.** `ggml_rope_multi` divide as dimensões em 4 seções, uma
por componente de posição (t, h, w, …); sem imagem, todas as componentes valem `pos` e o
resultado é idêntico ao RoPE que já temos. O suporte multimodal fica fora de escopo.

## Camada linear — gated delta net (48 camadas)

Dimensões vêm dos metadados SSM do GGUF: `ssm_d_conv`, `ssm_d_inner`, `ssm_d_state`
(= `head_k_dim` = `head_v_dim`), `ssm_n_group` (= `num_k_heads`), `ssm_dt_rank`
(= `num_v_heads`). `head_v_dim = ssm_d_inner / num_v_heads`.

```
qkv  = wqkv @ h                  # [key_dim*2 + value_dim]
z    = wqkv_gate @ h             # [value_dim]
beta = sigmoid(ssm_beta @ h)     # [num_v_heads]
g    = ssm_a * softplus(ssm_alpha @ h + ssm_dt)   # [num_v_heads]; ssm_a = -exp(A_log)

qkv  = silu(conv1d_causal(estado_conv ++ qkv, ssm_conv1d))   # kernel ssm_d_conv
q, k, v = split(qkv)             # q,k: [head_k_dim × num_k_heads]; v: [head_v_dim × num_v_heads]
q, k = L2norm(q), L2norm(k)      # L2, não RMS
q, k = repeat_para(num_v_heads)  # quando num_k_heads < num_v_heads

# recorrência por cabeça, estado S [S_v × S_v] guardado transposto (M[j][i] = S[i][j]):
S       *= exp(g)                            # g é escalar por cabeça
delta[j] = (v[j] - dot(M[j], k)) * beta
M[j][i] += delta[j] * k[i]
out[j]   = dot(M[j], q) / sqrt(S_v)

out = RMSNorm(out, ssm_norm) * silu(z)       # norma por cabeça, ssm_norm tem head_v_dim
cur = ssm_out @ out
```

O estado recorrente substitui o KV-cache **e tem tamanho fixo**: `S_v² × num_v_heads × 4 B`
por camada linear, independente do contexto. É a razão de o modelo existir.

## O que falta no llama-rs, em ordem

1. **Parser/config**: metadados `ssm_*`, `full_attention_interval`, `rope_sections`,
   `nextn_predict_layers`; tensores novos (`ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_beta`,
   `ssm_alpha`, `ssm_norm`, `ssm_out`, `attn_gate`, `attn_q_norm`, `attn_k_norm`,
   `attn_post_norm`). Os blocos MTP/NextN no fim da pilha são **ignoráveis** no decode.
2. **Forward CPU f32** com as duas variantes de camada, validado contra o llama.cpp.
3. **Shaders**: conv1d causal com estado, delta net recorrente, L2norm, gated norm,
   QK-norm, gate sigmoide. O matvec Q5_K/Q6_K e o SwiGLU já existem.
4. **Estado residente**: o `S` de cada camada linear fica na VRAM como o KV-cache já fica.

## Por que o decode aqui é diferente do 32B

No Qwen2.5-32B, 100% do tempo é matvec e a atenção lê um KV-cache que cresce. Aqui as 48
camadas lineares fazem, por cabeça e por token, um produto matriz-vetor contra um estado
`S_v × S_v` **mais** uma atualização de posto 1 do mesmo estado — ou seja, leem e
**escrevem** `S_v² × num_v_heads` floats por camada. Isso é tráfego novo, que não existe
no transformer denso, e provavelmente decide o desempenho.
