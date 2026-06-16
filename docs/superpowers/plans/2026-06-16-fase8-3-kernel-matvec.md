# Fase 8.3 — Otimização do kernel matvec Q8_0 (single-GPU, bater o llama.cpp 1× MI50)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) ou superpowers:executing-plans para implementar este plano tarefa-a-tarefa. Os passos usam checkbox (`- [ ]`) para tracking.

**Goal:** Subir o decode single-GPU do Qwen2.5-0.5B de **~80 tok/s** para perto do llama.cpp 1× MI50 (**~301 tok/s**), otimizando **só** o kernel `q8_0_matvec.comp` e seu dispatch — sem mudar a matemática enquanto possível. Cada alavanca é uma tarefa medida; o gate de correção (igualdade de token vs CPU) tem de continuar verde.

**Architecture:** Três alavancas incrementais, da mais segura à mais arriscada: (1) **specialization constants** no `ComputePipeline`; (2) **NUM_ROWS** — várias linhas de saída por workgroup (menos workgroups, reuso de `x[]`), reorganização **bit-idêntica**; (3) **cache de `x[]` em shared memory**, ainda bit-idêntico. Depois um **spike de capacidade** (`GL_EXT_integer_dot_product`/`dotPacked4x8EXT` em RADV gfx906) que decide se vale a 4ª alavanca: **quantizar a ativação para int8 + packed dot** (muda numéricos → gate passa a ser tolerância).

**Tech Stack:** GLSL→SPIR-V (build.rs), Rust + `ash` (Vulkan/RADV), MI50 gfx906 wave64.

---

## Contexto herdado (leia antes de começar)

- **Por que Fase 3 antes da Fase 2:** a Fase 1 ficou em ~80 tok/s vs 301 do llama.cpp 1× MI50 (`bench-results/gpu-20260616-131520.md`). A spec (§6, risco nº 2) manda: *"se a Fase 1 não chegar perto dos 314 tok/s no 0.5B, o problema é kernel/arquitetura — resolver antes de seguir [para multi-GPU]"*. Esta fase é esse "resolver".
- **O que já está feito (NÃO refazer):** 1 command buffer por token, 1 submit, 1 fence (Fase 1D, `record_token`/`record_and_submit` em `resident_forward.rs`). O overhead de dispatch CPU↔GPU **já não é o gargalo**. As alavancas aqui são de **ocupação e custo aritmético do kernel**.
- **Gargalo medido (relatório de análise):** o matvec usa **1 workgroup (64 lanes) por linha de saída** e um **loop escalar de 32 multiplicações int8→float por bloco** (`q8_0_matvec.comp:49-53`), sub-ocupando a GPU e sem usar o `V_DOT4_I32_I8` nativo do gfx906. O llama.cpp processa **várias linhas/workgroup**, faz cache de ativação e usa **packed integer dot** sobre ativação quantizada (Q8_1).
- **Gates de correção (token-equality, toleram ruído de float):**
  - `resident_forward_logits_iguais_a_cpu_qwen` — argmax GPU == CPU (`integration.rs:806`).
  - `resident_forward_gera_igual_cpu_multi_token` — 8 tokens gerados GPU == CPU (`integration.rs:753`).
  - Ambos pulam sozinhos se faltar GPU/modelo; rodam com o modelo 0.5B em `models/`.
- **Estrutura atual do matvec** (`resident_forward.rs`): cada matvec é `mk(PipeId::Matvec, binds, groups, push)` com `groups = n_out` (1 workgroup/linha) e `mv_push` = `PushConstants { n_in, n_out, row_offset: 0 }`. Há 7 matvecs por camada (q, k, v, attn_output, ffn_gate, ffn_up, ffn_down) + 1 projeção final de logits.
- **Pipeline atual** (`pipeline.rs`): `ComputePipeline::with(dev, spv, n_bindings, push_size)` **não** suporta specialization constants. `ComputePipeline::new(dev)` cria o matvec.

### Protocolo de benchmark (repetido em cada tarefa)

```bash
cargo build --release -p llama-cli --features gpu -q
target/release/llama-cli -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Once upon a time" -n 64 --temp 0 --seed 42 \
  --no-display-prompt --timings --gpu-resident 2>&1 | grep -i "tok/s"
```
Anote o tok/s antes/depois de cada alavanca. Baseline atual: **~80 tok/s**.

---

## Task 1: Specialization constants no `ComputePipeline`

**Files:**
- Modify: `crates/llama-vulkan/src/pipeline.rs`
- Modify: chamadores de `ComputePipeline::with(...)` (rmsnorm/rope/attention/swiglu/add em `resident_forward.rs`)

- [ ] **Step 1: Adicionar o parâmetro `spec_consts` a `with()` e ligar `vk::SpecializationInfo`**

Em `crates/llama-vulkan/src/pipeline.rs`, alterar a assinatura de `with` e o `stage`. Trocar a assinatura:

```rust
    pub fn with(
        dev: &ash::Device,
        spv: &[u8],
        n_bindings: u32,
        push_size: u32,
        spec_consts: &[(u32, u32)], // (constant_id, valor u32)
    ) -> Result<Self, PipelineError> {
```

E, logo **antes** de `let entry_point = c"main";`, inserir:

```rust
        // Specialization constants (cada uma é um u32, layout little-endian contíguo).
        let spec_entries: Vec<vk::SpecializationMapEntry> = spec_consts
            .iter()
            .enumerate()
            .map(|(i, &(id, _))| vk::SpecializationMapEntry {
                constant_id: id,
                offset: (i * 4) as u32,
                size: 4,
            })
            .collect();
        let spec_data: Vec<u8> = spec_consts
            .iter()
            .flat_map(|&(_, v)| v.to_le_bytes())
            .collect();
        let spec_info = vk::SpecializationInfo {
            map_entry_count: spec_entries.len() as u32,
            p_map_entries: spec_entries.as_ptr(),
            data_size: spec_data.len(),
            p_data: spec_data.as_ptr().cast(),
            ..Default::default()
        };
```

E no `stage`, adicionar o ponteiro (null quando vazio, para não anexar info vazia):

```rust
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_point.as_ptr(),
            p_specialization_info: if spec_consts.is_empty() {
                std::ptr::null()
            } else {
                &spec_info
            },
            ..Default::default()
        };
```

- [ ] **Step 2: Atualizar `new()` e os demais chamadores de `with()`**

Em `pipeline.rs`, `ComputePipeline::new` passa a não ter spec consts ainda (a Task 2 adiciona):

```rust
    pub fn new(dev: &ash::Device) -> Result<Self, PipelineError> {
        Self::with(
            dev,
            crate::Q8_0_MATVEC_SPV,
            3,
            std::mem::size_of::<PushConstants>() as u32,
            &[],
        )
    }
```

Localizar todos os outros `ComputePipeline::with(` (rmsnorm, rope, attention, swiglu, add — em `resident_forward.rs`, dentro de `new`/`new_pipelines_only`) e acrescentar `&[]` como último argumento em cada um.

Run para achá-los: `grep -rn "ComputePipeline::with(" crates/llama-vulkan/src`

- [ ] **Step 3: Compilar e rodar os gates (sem regressão)**

Run:
```bash
cargo build -p llama-vulkan --features gpu -q
cargo test -p llama-vulkan --features gpu resident_forward_logits_iguais_a_cpu_qwen -- --nocapture 2>&1 | tail -8
```
Expected: compila; o teste passa (ou pula com "sem AMD/modelo"). Nenhuma mudança de comportamento — só infraestrutura.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/pipeline.rs crates/llama-vulkan/src/resident_forward.rs
git commit -m "feat(vulkan): specialization constants em ComputePipeline::with"
```

---

## Task 2: NUM_ROWS — várias linhas de saída por workgroup (bit-idêntico)

Reorganização que **não** muda a ordem de acumulação por linha (cada linha continua: soma lane-strided dos blocos + `subgroupAdd`), então é bit-idêntica → os gates de token-equality continuam verdes. Reduz o nº de workgroups e prepara o reuso de `x[]` (Task 3).

**Files:**
- Modify: `crates/llama-vulkan/shaders/q8_0_matvec.comp`
- Modify: `crates/llama-vulkan/src/pipeline.rs` (`new` passa as spec consts)
- Modify: `crates/llama-vulkan/src/resident_forward.rs` (const + `groups` do matvec)

- [ ] **Step 1: Reescrever o shader com `constant_id` e loop sobre NUM_ROWS**

Substituir **todo** o conteúdo de `crates/llama-vulkan/shaders/q8_0_matvec.comp` por:

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable

// Specialization constants (preenchidas no pipeline creation).
layout(constant_id = 0) const uint WG = 64u;       // local_size_x (wave64)
layout(constant_id = 1) const uint NUM_ROWS = 1u;  // linhas de saída por workgroup

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

// W: pesos Q8_0 row-major. Cada linha = n_blocks x 34 bytes (2 = f16 scale, 32 = i8).
layout(set = 0, binding = 0) readonly buffer WeightBuf { uint8_t w[]; } weight_buf;
// X: ativações f32 (n_in floats).
layout(set = 0, binding = 1) readonly buffer ActBuf { float x[]; } act_buf;
// Y: saída f32 (n_out floats).
layout(set = 0, binding = 2) writeonly buffer OutBuf { float y[]; } out_buf;

layout(push_constant) uniform PC {
    uint n_in;
    uint n_out;
    uint row_offset; // vestigial (saída indexada por row absoluta)
} pc;

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint base_row = gl_WorkGroupID.x * NUM_ROWS;
    uint n_blocks = pc.n_in / 32u;
    uint row_stride = n_blocks * 34u; // bytes por linha de peso

    // Cada linha mantém EXATAMENTE a mesma ordem de acumulação do kernel original
    // (lane-strided sobre blocos + subgroupAdd) → resultado bit-idêntico por linha.
    for (uint r = 0u; r < NUM_ROWS; r++) {
        uint row = base_row + r;
        if (row >= pc.n_out) { break; }
        uint roff = row * row_stride;
        float acc = 0.0;
        for (uint b = lane; b < n_blocks; b += WG) {
            uint boff = roff + b * 34u;
            uint d_lo = uint(weight_buf.w[boff + 0u]);
            uint d_hi = uint(weight_buf.w[boff + 1u]);
            float d = unpackHalf2x16((d_hi << 8u) | d_lo).x;
            float dot = 0.0;
            for (uint i = 0u; i < 32u; i++) {
                int qi = int(int8_t(weight_buf.w[boff + 2u + i]));
                float xi = act_buf.x[b * 32u + i];
                dot += float(qi) * xi;
            }
            acc += d * dot;
        }
        acc = subgroupAdd(acc);
        if (gl_SubgroupInvocationID == 0u) {
            out_buf.y[row] = acc;
        }
    }
}
```

- [ ] **Step 2: Definir a const e passar as spec consts ao criar o pipeline matvec**

Em `crates/llama-vulkan/src/resident_forward.rs`, no topo (perto dos outros itens `pub(crate)`), adicionar:

```rust
/// Linhas de saída processadas por cada workgroup do matvec Q8_0 (tunável — Task 2 Step 5).
pub(crate) const MATVEC_NUM_ROWS: u32 = 4;
/// local_size_x do matvec (wave64 no MI50).
pub(crate) const MATVEC_WG: u32 = 64;
```

Em `crates/llama-vulkan/src/pipeline.rs`, fazer `new()` passar as spec consts (id 0 = WG, id 1 = NUM_ROWS):

```rust
    pub fn new(dev: &ash::Device) -> Result<Self, PipelineError> {
        Self::with(
            dev,
            crate::Q8_0_MATVEC_SPV,
            3,
            std::mem::size_of::<PushConstants>() as u32,
            &[
                (0, crate::resident_forward::MATVEC_WG),
                (1, crate::resident_forward::MATVEC_NUM_ROWS),
            ],
        )
    }
```

> Se `MATVEC_WG`/`MATVEC_NUM_ROWS` não forem visíveis de `pipeline.rs`, declará-los `pub(crate)` (como acima) e referenciar pelo caminho do módulo. Confirme o nome do módulo em `lib.rs` (`mod resident_forward;`).

- [ ] **Step 3: Ajustar o `groups` de cada matvec para `ceil(n_out / NUM_ROWS)`**

Em `resident_forward.rs`, dentro de `build_plan` (onde está `let mv_push = |...|`), adicionar um helper de groups ao lado de `mv_push`:

```rust
        let mv_groups = |n_out: usize| -> u32 {
            (n_out as u32).div_ceil(MATVEC_NUM_ROWS)
        };
```

Depois, em **todas** as chamadas `mk(PipeId::Matvec, …, <groups>, …)`, trocar o argumento `<groups>` (que hoje é `c.n_embd as u32`, `c.kv_dim as u32`, `c.n_ff as u32`, etc.) por `mv_groups(<n_out>)`. Os call sites são (procure por `PipeId::Matvec,`):
- q proj: `mv_groups(c.n_embd)`
- k proj: `mv_groups(c.kv_dim)`
- v proj: `mv_groups(c.kv_dim)`
- attn_output: `mv_groups(c.n_embd)`
- ffn_gate: `mv_groups(c.n_ff)`
- ffn_up: `mv_groups(c.n_ff)`
- ffn_down: `mv_groups(c.n_embd)`
- projeção final de logits (após o loop de camadas): `mv_groups(<vocab>)` — use o mesmo `n_out` que já é passado lá.

Run para localizar todos: `grep -n "PipeId::Matvec," crates/llama-vulkan/src/resident_forward.rs`

> **Importante:** só mude o `groups` dos `PipeId::Matvec`. Os outros (Rmsnorm/Rope/Attention/Swiglu/Add) continuam com `Self::groups_for(...)` / `1`.

- [ ] **Step 4: Rodar os gates de correção**

Run:
```bash
cargo test -p llama-vulkan --features gpu resident_forward_logits_iguais_a_cpu_qwen resident_forward_gera_igual_cpu_multi_token -- --nocapture 2>&1 | tail -12
```
Expected: ambos passam (argmax/texto idênticos ao CPU). Como a mudança é bit-idêntica por linha, qualquer falha aqui indica bug no indexamento de `row`/`groups`, não numérico.

- [ ] **Step 5: Varredura de NUM_ROWS e benchmark**

Para cada valor em `{1, 2, 4, 8}`: editar `MATVEC_NUM_ROWS`, rodar o benchmark (protocolo acima) e anotar o tok/s.

Run (exemplo para um valor):
```bash
# editar MATVEC_NUM_ROWS, depois:
cargo build --release -p llama-cli --features gpu -q
target/release/llama-cli -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Once upon a time" -n 64 --temp 0 --seed 42 --no-display-prompt --timings --gpu-resident 2>&1 | grep -i "tok/s"
```
Expected: uma tabela mental NUM_ROWS→tok/s. **Fixar o melhor valor** em `MATVEC_NUM_ROWS`. (Se nenhum valor superar o baseline, registre isso — é resultado válido; a alavanca real pode ser a Task 5.)

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/q8_0_matvec.comp crates/llama-vulkan/src/pipeline.rs crates/llama-vulkan/src/resident_forward.rs
git commit -m "perf(vulkan): matvec NUM_ROWS linhas/workgroup via spec consts (Fase 8.3)"
```

---

## Task 3: Cache de `x[]` em shared memory (bit-idêntico)

Com NUM_ROWS linhas por workgroup, `x[]` (n_in floats) é relido do global a cada linha. Carregar `x[]` **uma vez** em LDS por workgroup e ler de lá. A ordem aritmética não muda → bit-idêntico.

**Files:**
- Modify: `crates/llama-vulkan/shaders/q8_0_matvec.comp`

- [ ] **Step 1: Adicionar o array shared e o carregamento cooperativo**

Editar `q8_0_matvec.comp`. Acrescentar, após as declarações de buffer e antes de `void main()`:

```glsl
// x[] cabe em LDS: n_in <= ~16k floats (64 KB) em 0.5B/14B. Dimensione com folga.
// MI50: 64 KB LDS/workgroup. 16384 floats = 64 KB (limite); reduza se estourar.
shared float sx[16384];
```

E no início de `main()`, **antes** do loop de linhas, carregar `x[]` cooperativamente e sincronizar:

```glsl
    uint n_in = pc.n_in;
    for (uint i = lane; i < n_in; i += WG) {
        sx[i] = act_buf.x[i];
    }
    barrier();
```

E no loop interno, trocar a leitura global por LDS:

```glsl
                float xi = sx[b * 32u + i];
```

(remover a leitura `act_buf.x[b * 32u + i]`).

> **Limite de LDS:** se `n_in` (ex.: `n_ff` do 14B) exceder 16384 floats, ou o pipeline falhar em criar por LDS insuficiente, reduza `sx` e faça o cache em **tiles** (carregue blocos de `sx` por faixa de `b`). Para o 0.5B (`n_ff=4864`) cabe folgado — valide primeiro nele.

- [ ] **Step 2: Rodar os gates**

Run:
```bash
cargo test -p llama-vulkan --features gpu resident_forward_logits_iguais_a_cpu_qwen resident_forward_gera_igual_cpu_multi_token -- --nocapture 2>&1 | tail -12
```
Expected: ambos passam (bit-idêntico).

- [ ] **Step 3: Benchmark**

Run o protocolo de benchmark. Anote o tok/s vs Task 2. Se piorar (bank conflicts / pressão de LDS reduzindo ocupação), reverta este shader e registre.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/shaders/q8_0_matvec.comp
git commit -m "perf(vulkan): cache de x[] em shared memory no matvec (Fase 8.3)"
```

---

## Task 4: Spike — `GL_EXT_integer_dot_product` / `dotPacked4x8EXT` em RADV gfx906

Decide se a Task 5 (packed int dot) é viável. Valida que o glslang compila a extensão e que o RADV/gfx906 executa o `V_DOT4_I32_I8` corretamente.

**Files:**
- Create: `crates/llama-vulkan/shaders/dot4_probe.comp`
- Create: `crates/llama-vulkan/tests/dot4_probe.rs`
- Modify: `crates/llama-vulkan/build.rs` (compilar o novo shader) e `crates/llama-vulkan/src/lib.rs` (expor o SPV)

- [ ] **Step 1: Escrever o shader de probe**

Criar `crates/llama-vulkan/shaders/dot4_probe.comp` — computa `dotPacked4x8EXT` de dois i32 (4×i8 cada) e escreve o resultado:

```glsl
#version 450
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 1) in;
layout(set = 0, binding = 0) readonly buffer A { int a; int b; } inp;
layout(set = 0, binding = 1) writeonly buffer R { int r; } outp;

void main() {
    // dotPacked4x8EXT: soma dos 4 produtos i8×i8 empacotados em a e b.
    outp.r = dotPacked4x8AccSatEXT(inp.a, inp.b, 0);
}
```

> Se `dotPacked4x8AccSatEXT` não existir no glslang instalado, use `dotPacked4x8EXT(inp.a, inp.b)` (sem acumulador). O objetivo do spike é justamente descobrir o que compila.

- [ ] **Step 2: Compilar o shader no build e expor o SPV**

Em `crates/llama-vulkan/build.rs`, localizar onde os `.comp` são compilados (procure por `q8_0_matvec` / `glslc` / `compile`) e adicionar `dot4_probe.comp` à lista, gerando `DOT4_PROBE_SPV` no mesmo padrão dos demais. Em `crates/llama-vulkan/src/lib.rs`, expor a constante `pub(crate) const DOT4_PROBE_SPV: &[u8] = ...` espelhando `Q8_0_MATVEC_SPV` (procure por `Q8_0_MATVEC_SPV` em `lib.rs`).

- [ ] **Step 3: Escrever o teste de probe (resultado conhecido)**

Criar `crates/llama-vulkan/tests/dot4_probe.rs`:

```rust
//! Spike Fase 8.3: GL_EXT_integer_dot_product roda em RADV/gfx906?
//! a = [1,2,3,4] (i8 packed), b = [5,6,7,8] → 1*5+2*6+3*7+4*8 = 70.

use llama_vulkan::{ResidentForward, VulkanContext};

#[test]
fn dot4_packed_roda_em_radv() {
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    // pack little-endian: byte0=1, byte1=2, ...
    let a: i32 = i32::from_le_bytes([1, 2, 3, 4]);
    let b: i32 = i32::from_le_bytes([5, 6, 7, 8]);
    // Usa um helper de debug que sobe a/b, despacha DOT4_PROBE_SPV e lê r.
    let r = ResidentForward::dbg_dot4_probe(&ctx, a, b)
        .expect("se a extensão não compilar/rodar, isto falha — resultado do spike");
    assert_eq!(r, 70, "dotPacked4x8 deve somar 70");
}
```

- [ ] **Step 4: Implementar o helper `dbg_dot4_probe`**

Em `resident_forward.rs`, adicionar um método de debug `pub fn dbg_dot4_probe(ctx, a: i32, b: i32) -> Result<i32, MatmulError>` espelhando os helpers `dbg_*` já existentes (`dbg_rmsnorm`, `dbg_swiglu`): cria pipeline com `DOT4_PROBE_SPV` (2 bindings, push 0), sobe `[a,b]`, despacha 1 grupo, lê o int de saída. Reuse `new_pipelines_only`/`alloc_set`/`dispatch1`/`upload_*`/readback já presentes no arquivo.

- [ ] **Step 5: Rodar o spike — DECISION GATE**

Run:
```bash
cargo test -p llama-vulkan --features gpu dot4_packed_roda_em_radv -- --nocapture 2>&1 | tail -15
```
Expected (dois desfechos, ambos válidos):
- **PASSA (r==70):** a extensão compila e roda → **prosseguir para a Task 5**.
- **FALHA (compile error no glslang ou resultado errado):** RADV/gfx906 não expõe o packed dot por este caminho → **PARAR aqui**. Registrar em `bench-results/fase8-3-dot4-spike.md` e considerar a Fase 3 encerrada com o ganho de Tasks 2–3; a paridade restante vira assunto da Fase 2/4. **Não** implementar a Task 5.

- [ ] **Step 6: Commit (e registro da decisão)**

```bash
git add crates/llama-vulkan/shaders/dot4_probe.comp crates/llama-vulkan/tests/dot4_probe.rs crates/llama-vulkan/build.rs crates/llama-vulkan/src/lib.rs crates/llama-vulkan/src/resident_forward.rs
git commit -m "test(spike): probe GL_EXT_integer_dot_product em RADV gfx906 (Fase 8.3)"
```

---

## Task 5: (Condicional — só se a Task 4 passou) Ativação int8 + packed dot

Quantiza `x[]` por bloco de 32 (escala f32 + 32 int8, estilo Q8_1) e troca o loop escalar de 32 mul por **8× `dotPacked4x8EXT`**. **Muda os numéricos** → o gate passa a ser tolerância de logits + manter a igualdade de token.

**Files:**
- Modify: `crates/llama-vulkan/shaders/q8_0_matvec.comp`
- Modify: `crates/llama-vulkan/tests/integration.rs` (novo teste de tolerância)

- [ ] **Step 1: Escrever o teste de tolerância PRIMEIRO (deve falhar por ainda não haver quantização)**

Em `crates/llama-vulkan/tests/integration.rs`, adicionar um teste que compara **logits** GPU vs CPU com erro relativo limitado (a quantização introduz ~1% de erro por bloco, que não deve mudar o argmax):

```rust
#[test]
fn resident_matvec_int8_logits_dentro_da_tolerancia() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else { eprintln!("modelo ausente — pulando"); return; };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();

    let prompt: [u32; 2] = [model.config.bos_id, 9707];
    // logits completos (precisa de um helper que retorne o vetor, não só argmax — ver Step 2)
    let cpu = model.decode_one_cpu_logits(&prompt).unwrap();
    let gpu = model.decode_one_gpu_resident_logits(&prompt, &backend).unwrap();
    assert_eq!(cpu.len(), gpu.len());
    let mut max_rel = 0.0f32;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        let denom = c.abs().max(1e-3);
        max_rel = max_rel.max((c - g).abs() / denom);
    }
    eprintln!("erro relativo máx de logits: {max_rel}");
    assert!(max_rel < 0.05, "erro relativo {max_rel} acima de 5%");
    // E o argmax tem de continuar igual:
    let amax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(amax(&cpu), amax(&gpu), "argmax deve permanecer igual após quantização");
}
```

> Se não existir `decode_one_*_logits` (que retorna o vetor completo), os gates atuais usam `decode_one_*_owned` que devolvem só o argmax. Adicione os dois helpers em `crates/llama-model/src/gpu.rs` espelhando os `_owned` (procure por `decode_one_cpu_owned`), retornando `Vec<f32>` dos logits.

- [ ] **Step 2: Rodar o teste e ver falhar pelo motivo certo**

Run: `cargo test -p llama-vulkan --features gpu resident_matvec_int8_logits_dentro_da_tolerancia -- --nocapture 2>&1 | tail -8`
Expected: compila e o teste **passa** ainda (porque o shader atual é bit-idêntico — erro ~0). Isso valida os helpers de logits. O teste vira gate de regressão para o Step 3.

- [ ] **Step 3: Quantizar x[] em LDS e usar `dotPacked4x8EXT` no shader**

Editar `q8_0_matvec.comp`. Trocar o cache `shared float sx[...]` por uma versão quantizada por bloco e o loop interno por packed dot. Esboço (ajuste tipos conforme o que a Task 4 mostrou compilar):

```glsl
#extension GL_EXT_integer_dot_product : require
// ... bindings iguais ...

shared int   sxq[16384/4]; // 32 int8 por bloco empacotados em 8 ints (4 i8 cada)
shared float sxd[512];     // escala por bloco (n_blocks <= 512 em 0.5B/14B)

void quantize_x(uint n_in, uint lane) {
    uint n_blocks = n_in / 32u;
    for (uint blk = lane; blk < n_blocks; blk += WG) {
        // amax do bloco
        float amax = 0.0;
        for (uint i = 0u; i < 32u; i++) amax = max(amax, abs(act_buf.x[blk*32u + i]));
        float d = amax / 127.0;
        float inv = (d > 0.0) ? 1.0/d : 0.0;
        sxd[blk] = d;
        for (uint g = 0u; g < 8u; g++) { // 8 grupos de 4 i8
            int packed = 0;
            for (uint j = 0u; j < 4u; j++) {
                int q = int(round(act_buf.x[blk*32u + g*4u + j] * inv));
                q = clamp(q, -127, 127);
                packed |= (q & 0xff) << (8u*j);
            }
            sxq[blk*8u + g] = packed;
        }
    }
    barrier();
}
```

E no loop de linhas, por bloco `b`:

```glsl
            float d_w = unpackHalf2x16((d_hi << 8u) | d_lo).x; // escala do peso
            int isum = 0;
            for (uint g = 0u; g < 8u; g++) {
                // peso: 4 i8 em um int (bytes boff+2 + g*4 .. +3)
                int wp = int(weight_buf.w[boff+2u+g*4u+0u])
                       | (int(weight_buf.w[boff+2u+g*4u+1u]) << 8)
                       | (int(weight_buf.w[boff+2u+g*4u+2u]) << 16)
                       | (int(weight_buf.w[boff+2u+g*4u+3u]) << 24);
                isum = dotPacked4x8AccSatEXT(wp, sxq[b*8u + g], isum);
            }
            acc += d_w * sxd[b] * float(isum);
```

Chamar `quantize_x(pc.n_in, lane);` no início de `main()` (substitui o cache f32 da Task 3).

> **Cuidado com sinal:** os bytes de peso são i8 com sinal; ao empacotar em `int`, garanta extensão de sinal correta (use `int(int8_t(...))` antes do `<<` se o packed dot exigir, ou confirme no spike da Task 4 qual empacotamento dá o resultado 70). A escala efetiva por bloco é `d_w * sxd[b]`.

- [ ] **Step 4: Rodar TODOS os gates (tolerância + token-equality)**

Run:
```bash
cargo test -p llama-vulkan --features gpu resident_matvec_int8_logits_dentro_da_tolerancia resident_forward_logits_iguais_a_cpu_qwen resident_forward_gera_igual_cpu_multi_token -- --nocapture 2>&1 | tail -16
```
Expected: erro relativo de logits < 5%, argmax igual, e a geração de 8 tokens idêntica ao CPU. Se o argmax mudar, a quantização está agressiva demais — revise o empacotamento/clamp ou reduza para Q8 simétrico por linha.

- [ ] **Step 5: Benchmark**

Run o protocolo. Anote o tok/s — esta é a alavanca com maior ganho esperado. Compare com Task 3 e com o alvo (301 tok/s).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/q8_0_matvec.comp crates/llama-vulkan/tests/integration.rs crates/llama-model/src/gpu.rs
git commit -m "perf(vulkan): ativação int8 + dotPacked4x8 no matvec (Fase 8.3)"
```

---

## Task 6: Benchmark final + decisão de gate

**Files:**
- Create: `bench-results/fase8-3-kernel-progressao.md`

- [ ] **Step 1: Rodar o benchmark comparativo completo**

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -20`
Expected: gera um `bench-results/gpu-<stamp>.md` com a linha `llama-rs — 1x MI50 (res-fwd)` atualizada.

- [ ] **Step 2: Registrar a progressão e a decisão**

Criar `bench-results/fase8-3-kernel-progressao.md`:

```markdown
# Fase 8.3 — Progressão do kernel matvec (0.5B, 1× MI50)

| Alavanca                          | tok/s | vs baseline |
| --------------------------------- | ----- | ----------- |
| Baseline (Fase 1D)                | ~80   | 1.0×        |
| Task 2 — NUM_ROWS=<best>          | <a>   | <a/80>×     |
| Task 3 — shared x[]               | <b>   | <b/80>×     |
| Task 5 — int8 + dotPacked4x8 (se) | <c>   | <c/80>×     |
| **Alvo: llama.cpp 1× MI50**       | ~301  | 3.76×       |

## Decisão
- [ ] Chegamos perto de 301 tok/s? <sim/não>
      - SIM → o gate da spec (§6 risco nº2) está satisfeito; **liberar a Fase 2** (row-split),
        usando o mecanismo de all-reduce decidido na Fase 0.
      - NÃO → identificar a próxima alavanca (ex.: dotPacked indisponível → investigar
        layout de pesos / múltiplos subgroups por workgroup) antes da Fase 2.
- Spike dot4 (Task 4): <passou/falhou>.
```

- [ ] **Step 3: Commit**

```bash
git add bench-results/
git commit -m "bench(fase8-3): progressão do kernel matvec + decisão de gate"
```

---

## Self-Review (cobertura vs spec §7 Fase 3)

- **"Matvec Q8_0 wave64 com subgroup reduction":** mantida (subgroupAdd por linha) + reorganizada (Task 2). ✓
- **"Coalescência de memória":** cache de `x[]` em LDS (Task 3) + leitura empacotada de 4 bytes de peso por vez (Task 5). ✓
- **"Fusão de ops":** **NÃO** incluída — avaliada e descartada para esta fase (a infra de 1 command buffer/token já elimina o overhead de submit; fusão de Q/K/V em 1 dispatch é ganho marginal e quebra a separação do plano de ops). Registrado aqui para não parecer omissão.
- **Re-benchmark:** Task 6. ✓
- **Aceite da spec ("≥1.3× sobre llama.cpp 2× MI50 no 14B"):** esse aceite **pressupõe a Fase 2 (multi-GPU) feita**. Como estamos fazendo a Fase 3 **antes** da Fase 2 (por causa do gate do risco nº2), o aceite **operacional desta fase** é **chegar perto do llama.cpp 1× MI50 no 0.5B (~301 tok/s)**. O aceite original de 14B/multi-GPU permanece para depois da Fase 2. Decisão registrada na Task 6.
- **Placeholders:** as Tasks 4–5 dependem de uma capacidade de hardware desconhecida a priori (`dotPacked4x8EXT` em RADV) — por isso a Task 4 é um **decision gate** explícito e a Task 5 é **condicional**, não um placeholder.
```
