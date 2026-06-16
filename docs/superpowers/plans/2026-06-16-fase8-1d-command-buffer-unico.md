# Fase 8.1D — Um command buffer por token (1 submit, 1 fence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gravar a stack inteira de um token (embedding → 48/24 camadas → norm final → logits) como **um único command buffer** com pipeline barriers entre dispatches, submetido **uma vez** com **um fence wait por token**. Elimina os ~169 `queue_submit` + `queue_wait_idle` por token que a Fase 1C ainda fazia (1 por op). É a fatia onde o tok/s salta para a vizinhança do llama.cpp single-GPU.

**Architecture:** Estende o `ResidentForward` (Fase 1C). Na 1C, `decode_step` faz, por op: `update_descriptor_sets` + `queue_submit` + `queue_wait_idle`, e reseta o descriptor pool por token. Esta fatia: (1) **pré-aloca e pré-escreve** um descriptor set por dispatch (os bindings são estáticos entre tokens — pesos por camada e buffers de ativação não mudam; o KV-cache é ligado pelo buffer inteiro, com o offset da camada/pos indo no push ou no `cmd_copy_buffer`); (2) constrói **uma vez** um "plano" de ops (lista de dados, sem closures) que descreve a sequência exata do token; (3) por token, **regrava** um command buffer persistente percorrendo o plano (push-constants e offsets de cópia dependem de `token`/`pos`, por isso a regravação), inserindo um pipeline barrier entre ops; (4) **um** `queue_submit` com fence + **um** `wait_for_fences`; readback só dos logits. A regravação por token é barata; o ganho é trocar 169 sincronizações GPU↔host por uma.

**Tech Stack:** Rust, `ash` (Vulkan), RADV/gfx906 (MI50). Reusa os 6 shaders, pipelines, `Buf`, `Cfg`, `LayerQ`/`LayerAux` e o KV-cache da Fase 1C. Nenhum shader novo.

**Barreira escolhida (correção primeiro):** um `VkMemoryBarrier` global (SHADER_WRITE|TRANSFER_WRITE → SHADER_READ|TRANSFER_READ, COMPUTE|TRANSFER → COMPUTE|TRANSFER) entre cada op. É grosso (serializa ops independentes como q/k/v), mas correto e simples. Barreiras finas por-buffer e sobreposição de dispatches independentes são otimização da Fase 3.

---

## Contexto (ler antes de começar)

Dependência dura: **a Fase 1C precisa estar concluída e o teste `resident_forward_logits_iguais_a_cpu_qwen` passando.** A 1D não muda a matemática nem os shaders — só a forma de submeter. O critério de aceite de correção é o **mesmo teste** continuar passando após a troca para command buffer único.

Pontos da 1C que a 1D substitui (em `crates/llama-vulkan/src/resident_forward.rs`):
- `decode_step` (Task 10 da 1C): 1 dispatch+wait por op, `reset_descriptor_pool` por token. → vira **1 command buffer/token**.
- `alloc_set` por op a cada token. → vira **sets pré-alocados e pré-escritos uma vez**.

Mantidos sem mudança: os métodos `dbg_*` (micro-testes por shader, Tasks 4-7 da 1C — continuam usando `dispatch1`), `dispatch1`/`readback`/`upload_f32`/`copy_region`, e toda a alocação de pesos/buffers/KV do `new` (1C Task 9).

**Por que regravar o command buffer por token (e não gravar 1 vez):** três coisas dependem de `token`/`pos` e são gravadas inline no command buffer: (a) o offset-fonte do embedding (`token·n_embd`); (b) o offset-destino do append no KV-cache (`(layer·ctx+pos)·kv_dim`); (c) os push-constants de RoPE (`pos`) e attention (`total_len`). Os **descriptor sets**, porém, têm bindings estáticos → escritos uma vez.

---

## File Structure

- **Modify:** `crates/llama-vulkan/src/resident_forward.rs` — plano de ops, sets persistentes, command buffer + fence persistentes, `record_token`, novo caminho de `decode`, `Drop` estendido.
- **Modify:** `crates/llama-vulkan/tests/integration.rs` — teste de que o decode (caminho 1D) ainda iguala a CPU + teste de que `decode` faz 1 submit (via contador opcional).
- **Modify:** `scripts/benchmark-gpu.sh` + `bench-results/` — re-rodar `--gpu-resident` (mesma flag; agora 1 cmd/token) e registrar o salto.

---

## Task 1: Plano de ops (dados) + sets persistentes pré-escritos

**Files:**
- Modify: `crates/llama-vulkan/src/resident_forward.rs`

- [ ] **Step 1: Definir os tipos do plano**

Adicionar perto do topo de `resident_forward.rs` (após `Cfg`/`LayerQ`/`LayerAux`):

```rust
/// Identifica qual pipeline um dispatch usa (resolvido em `pipe_of`).
#[derive(Clone, Copy)]
pub(crate) enum PipeId { Matvec, Rmsnorm, Rope, Attention, Swiglu, Add }

/// Como obter os bytes de push-constant de um dispatch no momento da gravação.
pub(crate) enum PushSpec {
    /// Push totalmente conhecido na construção do plano (rmsnorm, matvec, swiglu, add, bias).
    Static(Vec<u8>),
    /// RoPE: precisa de `pos` em tempo de gravação. `n_head` fixo.
    Rope { n_head: u32 },
    /// Attention: precisa de `total_len`. `kv_layer_off` fixo (offset da camada).
    Attention { kv_layer_off: u32 },
}

/// Uma op do token. `Dispatch` usa um descriptor set pré-escrito; as cópias não.
pub(crate) enum PlannedOp {
    Dispatch { pipe: PipeId, set: vk::DescriptorSet, groups: u32, push: PushSpec },
    /// Embedding lookup: copia a linha `token` de `token_embd` para `b_x`.
    Embed,
    /// Append do K e do V da camada ao KV-cache na posição `pos`.
    KvAppend { layer: usize },
}
```

- [ ] **Step 2: Helper `pipe_of` e `groups_for` (este já existe da 1C)**

Adicionar ao `impl<'ctx> ResidentForward<'ctx>`:

```rust
    pub(crate) fn pipe_of(&self, id: PipeId) -> &ComputePipeline {
        match id {
            PipeId::Matvec => &self.matvec,
            PipeId::Rmsnorm => &self.rmsnorm,
            PipeId::Rope => &self.rope,
            PipeId::Attention => &self.attention,
            PipeId::Swiglu => &self.swiglu,
            PipeId::Add => &self.add,
        }
    }
```

- [ ] **Step 3: Construir o plano e pré-escrever os sets**

Adicionar ao `impl`. Este método é chamado uma vez (no fim do `new`, Task 3) e replica **exatamente** a ordem de `decode_step` (1C), mas como dados. Cada `Dispatch` aloca um set e o escreve com seus bindings estáticos.

```rust
    /// Monta a lista de ops do token (ordem idêntica ao decode) e pré-aloca/escreve
    /// um descriptor set por dispatch (bindings estáticos entre tokens).
    fn build_plan(&self) -> Result<Vec<PlannedOp>, MatmulError> {
        use crate::pipeline::PushConstants;
        let c = &self.cfg;
        let d = &self.dev.device;
        let mut plan = Vec::new();

        // helpers de push estático
        let rms_push = || -> Vec<u8> {
            #[repr(C)]
            struct P { dim: u32, eps: f32 }
            let p = P { dim: c.n_embd as u32, eps: c.rms_eps };
            unsafe { std::slice::from_raw_parts(&p as *const P as *const u8, 8) }.to_vec()
        };
        let n_push = |n: usize| -> Vec<u8> {
            #[repr(C)]
            struct P { n: u32 }
            let p = P { n: n as u32 };
            unsafe { std::slice::from_raw_parts(&p as *const P as *const u8, 4) }.to_vec()
        };
        let mv_push = |n_in: usize, n_out: usize| -> Vec<u8> {
            let p = PushConstants { n_in: n_in as u32, n_out: n_out as u32, row_offset: 0 };
            unsafe {
                std::slice::from_raw_parts(&p as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>())
            }.to_vec()
        };

        // aloca + escreve um set para um dispatch, devolvendo o PlannedOp::Dispatch.
        let mut mk = |pipe: PipeId,
                      binds: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)],
                      groups: u32,
                      push: PushSpec|
         -> Result<PlannedOp, MatmulError> {
            let set = self.alloc_set(self.pipe_of(pipe))?;
            let buf_infos: Vec<vk::DescriptorBufferInfo> = binds
                .iter()
                .map(|&(buffer, offset, range)| vk::DescriptorBufferInfo { buffer, offset, range })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buf_infos
                .iter()
                .enumerate()
                .map(|(b, bi)| vk::WriteDescriptorSet {
                    dst_set: set,
                    dst_binding: b as u32,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: bi,
                    ..Default::default()
                })
                .collect();
            // SAFETY: d válido; writes apontam para buf_infos vivos; set nunca em uso durante o build.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
            Ok(PlannedOp::Dispatch { pipe, set, groups, push: push })
        };

        let nb = |n: usize| (n * 4) as vk::DeviceSize;

        plan.push(PlannedOp::Embed);

        for l in 0..c.n_layer {
            let lq = &self.qw[l];
            let la = &self.aux[l];

            // attn norm: b_x, attn_norm -> b_normed
            plan.push(mk(PipeId::Rmsnorm,
                &[(self.b_x.buffer, 0, nb(c.n_embd)), (la.attn_norm.buffer, 0, la.attn_norm.size),
                  (self.b_normed.buffer, 0, nb(c.n_embd))], 1, PushSpec::Static(rms_push()))?);
            // q/k/v matvec
            plan.push(mk(PipeId::Matvec,
                &[(lq.attn_q.buffer, 0, lq.attn_q.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
                  (self.b_q.buffer, 0, nb(c.n_embd))], c.n_embd as u32, PushSpec::Static(mv_push(c.n_embd, c.n_embd)))?);
            plan.push(mk(PipeId::Matvec,
                &[(lq.attn_k.buffer, 0, lq.attn_k.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
                  (self.b_k.buffer, 0, nb(c.kv_dim))], c.kv_dim as u32, PushSpec::Static(mv_push(c.n_embd, c.kv_dim)))?);
            plan.push(mk(PipeId::Matvec,
                &[(lq.attn_v.buffer, 0, lq.attn_v.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
                  (self.b_v.buffer, 0, nb(c.kv_dim))], c.kv_dim as u32, PushSpec::Static(mv_push(c.n_embd, c.kv_dim)))?);
            // bias (se houver)
            if let Some(b) = &la.q_bias {
                plan.push(mk(PipeId::Add, &[(self.b_q.buffer, 0, nb(c.n_embd)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.n_embd), PushSpec::Static(n_push(c.n_embd)))?);
            }
            if let Some(b) = &la.k_bias {
                plan.push(mk(PipeId::Add, &[(self.b_k.buffer, 0, nb(c.kv_dim)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.kv_dim), PushSpec::Static(n_push(c.kv_dim)))?);
            }
            if let Some(b) = &la.v_bias {
                plan.push(mk(PipeId::Add, &[(self.b_v.buffer, 0, nb(c.kv_dim)), (b.buffer, 0, b.size)],
                    Self::groups_for(c.kv_dim), PushSpec::Static(n_push(c.kv_dim)))?);
            }
            // rope q/k (pos em tempo de gravação)
            plan.push(mk(PipeId::Rope,
                &[(self.b_q.buffer, 0, nb(c.n_embd)), (self.freq_buf.buffer, 0, self.freq_buf.size)],
                Self::groups_for(c.n_head * (c.rope_dim / 2)), PushSpec::Rope { n_head: c.n_head as u32 })?);
            plan.push(mk(PipeId::Rope,
                &[(self.b_k.buffer, 0, nb(c.kv_dim)), (self.freq_buf.buffer, 0, self.freq_buf.size)],
                Self::groups_for(c.n_head_kv * (c.rope_dim / 2)), PushSpec::Rope { n_head: c.n_head_kv as u32 })?);
            // append KV (offset depende de pos -> gravado inline)
            plan.push(PlannedOp::KvAppend { layer: l });
            // attention (total_len em tempo de gravação)
            let layer_off = (l * c.ctx * c.kv_dim) as u32;
            plan.push(mk(PipeId::Attention,
                &[(self.b_q.buffer, 0, nb(c.n_embd)), (self.kcache.buffer, 0, self.kcache.size),
                  (self.vcache.buffer, 0, self.vcache.size), (self.b_attn.buffer, 0, nb(c.n_embd))],
                c.n_head as u32, PushSpec::Attention { kv_layer_off: layer_off })?);
            // attn output proj + residual
            plan.push(mk(PipeId::Matvec,
                &[(lq.attn_output.buffer, 0, lq.attn_output.size_bytes), (self.b_attn.buffer, 0, nb(c.n_embd)),
                  (self.b_proj.buffer, 0, nb(c.n_embd))], c.n_embd as u32, PushSpec::Static(mv_push(c.n_embd, c.n_embd)))?);
            plan.push(mk(PipeId::Add, &[(self.b_x.buffer, 0, nb(c.n_embd)), (self.b_proj.buffer, 0, nb(c.n_embd))],
                Self::groups_for(c.n_embd), PushSpec::Static(n_push(c.n_embd)))?);
            // ffn norm + gate/up + swiglu + down + residual
            plan.push(mk(PipeId::Rmsnorm,
                &[(self.b_x.buffer, 0, nb(c.n_embd)), (la.ffn_norm.buffer, 0, la.ffn_norm.size),
                  (self.b_normed.buffer, 0, nb(c.n_embd))], 1, PushSpec::Static(rms_push()))?);
            plan.push(mk(PipeId::Matvec,
                &[(lq.ffn_gate.buffer, 0, lq.ffn_gate.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
                  (self.b_gate.buffer, 0, nb(c.n_ff))], c.n_ff as u32, PushSpec::Static(mv_push(c.n_embd, c.n_ff)))?);
            plan.push(mk(PipeId::Matvec,
                &[(lq.ffn_up.buffer, 0, lq.ffn_up.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
                  (self.b_up.buffer, 0, nb(c.n_ff))], c.n_ff as u32, PushSpec::Static(mv_push(c.n_embd, c.n_ff)))?);
            plan.push(mk(PipeId::Swiglu,
                &[(self.b_gate.buffer, 0, nb(c.n_ff)), (self.b_up.buffer, 0, nb(c.n_ff)),
                  (self.b_act.buffer, 0, nb(c.n_ff))], Self::groups_for(c.n_ff), PushSpec::Static(n_push(c.n_ff)))?);
            plan.push(mk(PipeId::Matvec,
                &[(lq.ffn_down.buffer, 0, lq.ffn_down.size_bytes), (self.b_act.buffer, 0, nb(c.n_ff)),
                  (self.b_proj.buffer, 0, nb(c.n_embd))], c.n_embd as u32, PushSpec::Static(mv_push(c.n_ff, c.n_embd)))?);
            plan.push(mk(PipeId::Add, &[(self.b_x.buffer, 0, nb(c.n_embd)), (self.b_proj.buffer, 0, nb(c.n_embd))],
                Self::groups_for(c.n_embd), PushSpec::Static(n_push(c.n_embd)))?);
        }

        // norm final + logits
        plan.push(mk(PipeId::Rmsnorm,
            &[(self.b_x.buffer, 0, nb(c.n_embd)), (self.output_norm_buf.buffer, 0, self.output_norm_buf.size),
              (self.b_normed.buffer, 0, nb(c.n_embd))], 1, PushSpec::Static(rms_push()))?);
        let out_w = self.output_w.as_ref().unwrap();
        plan.push(mk(PipeId::Matvec,
            &[(out_w.buffer, 0, out_w.size_bytes), (self.b_normed.buffer, 0, nb(c.n_embd)),
              (self.b_logits.buffer, 0, nb(c.vocab))], c.vocab as u32, PushSpec::Static(mv_push(c.n_embd, c.vocab)))?);

        Ok(plan)
    }
```

> O `mk` é um closure `FnMut` porque chama `self.alloc_set` (que aloca do pool). O pool precisa de `max_sets` suficiente: ~`n_layer·14 + 2` sets (qwen 0.5b: 24·14+2 ≈ 338). O `max_sets=1024` definido em `new_pipelines_only` (1C) cobre. **Confirmar e, se preciso, elevar `max_sets`/`descriptor_count` proporcional a `n_layer`.**

- [ ] **Step 2 (build): compilar isolado**

Run: `cargo build -p llama-vulkan 2>&1 | tail -30`
Expected: compila (com warnings de `build_plan`/tipos do plano não usados até a Task 2). Sem erros de borrow no `mk`.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs
git commit -m "feat(vulkan): plano de ops + descriptor sets persistentes (Fase 1D)"
```

---

## Task 2: Command buffer + fence persistentes; `record_token`; barrier

**Files:**
- Modify: `crates/llama-vulkan/src/resident_forward.rs`

- [ ] **Step 1: Adicionar campos persistentes à struct**

Adicionar a `pub struct ResidentForward<'ctx>` (campos novos da 1D):

```rust
    pub(crate) plan: Vec<PlannedOp>,
    pub(crate) token_cmd: vk::CommandBuffer,
    pub(crate) token_fence: vk::Fence,
```

- [ ] **Step 2: Barrier global e `record_token`**

Adicionar ao `impl<'ctx> ResidentForward<'ctx>`:

```rust
    /// Barreira de memória global entre dispatches/cópias do mesmo command buffer.
    fn full_barrier(&self, cmd: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            ..Default::default()
        };
        let stage = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        // SAFETY: cmd em gravação; barreira global sem buffer/image barriers.
        unsafe {
            self.dev.device.cmd_pipeline_barrier(
                cmd, stage, stage, vk::DependencyFlags::empty(), &[mb], &[], &[],
            );
        }
    }

    /// Grava a stack inteira do token em `cmd` (já em `begin`). Push/offsets dependem de token/pos.
    fn record_token(&self, cmd: vk::CommandBuffer, token: u32, pos: usize) {
        let d = &self.dev.device;
        let c = &self.cfg;
        let total_len = (pos + 1) as u32;

        for op in &self.plan {
            match op {
                PlannedOp::Embed => {
                    let region = vk::BufferCopy {
                        src_offset: (token as usize * c.n_embd * 4) as vk::DeviceSize,
                        dst_offset: 0,
                        size: (c.n_embd * 4) as vk::DeviceSize,
                    };
                    // SAFETY: cmd em gravação; buffers vivos; offsets dentro do tamanho.
                    unsafe { d.cmd_copy_buffer(cmd, self.token_embd_buf.buffer, self.b_x.buffer, &[region]); }
                    self.full_barrier(cmd);
                }
                PlannedOp::KvAppend { layer } => {
                    let off = ((layer * c.ctx + pos) * c.kv_dim * 4) as vk::DeviceSize;
                    let sz = (c.kv_dim * 4) as vk::DeviceSize;
                    let rk = vk::BufferCopy { src_offset: 0, dst_offset: off, size: sz };
                    // SAFETY: idem.
                    unsafe {
                        d.cmd_copy_buffer(cmd, self.b_k.buffer, self.kcache.buffer, &[rk]);
                        d.cmd_copy_buffer(cmd, self.b_v.buffer, self.vcache.buffer, &[rk]);
                    }
                    self.full_barrier(cmd);
                }
                PlannedOp::Dispatch { pipe, set, groups, push } => {
                    let p = self.pipe_of(*pipe);
                    // bytes de push conforme o tipo
                    let bytes: Vec<u8> = match push {
                        PushSpec::Static(b) => b.clone(),
                        PushSpec::Rope { n_head } => {
                            #[repr(C)]
                            struct P { n_head: u32, head_dim: u32, rope_dim: u32, pos: f32 }
                            let pp = P { n_head: *n_head, head_dim: c.head_dim as u32,
                                rope_dim: c.rope_dim as u32, pos: pos as f32 };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 16) }.to_vec()
                        }
                        PushSpec::Attention { kv_layer_off } => {
                            #[repr(C)]
                            struct P { n_head: u32, n_head_kv: u32, head_dim: u32, total_len: u32, kv_dim: u32, kv_layer_off: u32 }
                            let pp = P { n_head: c.n_head as u32, n_head_kv: c.n_head_kv as u32,
                                head_dim: c.head_dim as u32, total_len, kv_dim: c.kv_dim as u32,
                                kv_layer_off: *kv_layer_off };
                            unsafe { std::slice::from_raw_parts(&pp as *const P as *const u8, 24) }.to_vec()
                        }
                    };
                    // SAFETY: cmd em gravação; pipeline/set válidos; bytes do tamanho do range.
                    unsafe {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[*set], &[]);
                        d.cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                        d.cmd_dispatch(cmd, *groups, 1, 1);
                    }
                    self.full_barrier(cmd);
                }
            }
        }
    }
```

- [ ] **Step 3: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila (warnings de `token_cmd`/`token_fence`/`record_token` não usados até a Task 3).

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs
git commit -m "feat(vulkan): record_token (1 command buffer) + barrier global (Fase 1D)"
```

---

## Task 3: Construir plano+cmd+fence no `new`; novo `decode` (1 submit/token)

**Files:**
- Modify: `crates/llama-vulkan/src/resident_forward.rs`

- [ ] **Step 1: Alocar command buffer + fence e o plano no fim do `new`**

No `ResidentForward::new` (1C Task 9), **antes** do `Ok(Self { ... })`, alocar o command buffer persistente e o fence; e **depois** de construir `Self`, montar o plano (precisa de `&self` já com buffers). Como `build_plan` usa `&self`, a ordem é: construir um `Self` parcial com `plan: Vec::new()`, `token_cmd`, `token_fence`, depois preencher o plano.

Substituir o fim do `new` (o `Ok(Self { ... })`) por:

```rust
        // Command buffer + fence persistentes (reusados a cada token).
        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: base.dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: device/pool válidos; pool tem RESET_COMMAND_BUFFER (device.rs).
        let token_cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let token_fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? };

        let mut me = Self {
            cfg,
            qw,
            output_w: Some(output_w),
            aux: aux_buf,
            output_norm_buf,
            freq_buf,
            token_embd_buf,
            kcache,
            vcache,
            b_x: nf(config.n_embd)?,
            b_normed: nf(config.n_embd)?,
            b_q: nf(config.n_embd)?,
            b_k: nf(kv_dim)?,
            b_v: nf(kv_dim)?,
            b_attn: nf(config.n_embd)?,
            b_proj: nf(config.n_embd)?,
            b_gate: nf(config.n_ff)?,
            b_up: nf(config.n_ff)?,
            b_act: nf(config.n_ff)?,
            b_logits: nf(config.vocab)?,
            len: RefCell::new(0),
            plan: Vec::new(),
            token_cmd,
            token_fence,
            ..base
        };
        me.plan = me.build_plan()?;
        Ok(me)
```

> Ajustar `output_w` para `Option<GpuTensor>` conforme a nota da 1C Task 9 Step 3 (já recomendado lá). `build_plan` toma `&self` — por isso é chamado após `me` existir, atribuindo a `me.plan`.

- [ ] **Step 2: Substituir o caminho de `decode` por 1 command buffer/token**

Substituir o método `decode_step` (1C Task 10) por um `record_and_submit` e fazer a impl da trait chamá-lo. Remover o `decode_step` antigo (e os `op_*` da 1C, que ficam sem uso — confirmar e remover para evitar dead code). Adicionar:

```rust
    /// Regrava o command buffer do token, submete uma vez, espera o fence e lê os logits.
    fn record_and_submit(&self, token: u32, pos: usize) -> Result<Vec<f32>, MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;
        let cmd = self.token_cmd;

        // SAFETY: pool RESET_COMMAND_BUFFER; cmd não está em uso (fence do token anterior já aguardado).
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            d.begin_command_buffer(cmd, &begin)?;
        }
        self.record_token(cmd, token, pos);
        // SAFETY: cmd em gravação.
        unsafe { d.end_command_buffer(cmd)?; }

        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        // SAFETY: fence resetado antes do submit; cmd válido.
        unsafe {
            d.reset_fences(&[self.token_fence])?;
            d.queue_submit(dev.queue, &[submit], self.token_fence)?;
            d.wait_for_fences(&[self.token_fence], true, u64::MAX)?;
        }
        self.readback(&self.b_logits, self.cfg.vocab)
    }
```

E a impl da trait (substituir a da 1C Task 10):

```rust
impl llama_model::GpuResidentDecode for ResidentForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let logits = self
            .record_and_submit(token, pos)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        *self.len.borrow_mut() = pos + 1;
        Ok(logits)
    }
    fn reset(&self) {
        *self.len.borrow_mut() = 0;
    }
}
```

- [ ] **Step 3: Estender o `Drop` para liberar cmd buffer + fence**

No `impl Drop` (1C Task 9 Step 3), após o `device_wait_idle`, adicionar:

```rust
        // SAFETY: GPU ociosa; cmd/fence criados por nós.
        unsafe {
            d.free_command_buffers(self.dev.cmd_pool, &[self.token_cmd]);
            d.destroy_fence(self.token_fence, None);
        }
```

(O command buffer também seria liberado ao destruir o pool, mas liberar explicitamente é correto e claro.)

- [ ] **Step 4: Compilar e limpar dead code**

Run: `cargo build -p llama-vulkan 2>&1 | tail -30`
Expected: compila. Remover `op_rmsnorm/op_matvec/op_add/op_rope/op_swiglu/op_attention` e `decode_step` da 1C se o compilador apontar dead code (o caminho 1D não os usa). Os `dbg_*` permanecem.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs
git commit -m "perf(vulkan): decode em 1 command buffer/token, 1 submit, 1 fence (Fase 1D)"
```

---

## Task 4: Correção mantida (teste 1C ainda passa) + prova de 1 submit/token

**Files:**
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Reconfirmar o teste de paridade da 1C (agora pelo caminho 1D)**

O teste `resident_forward_logits_iguais_a_cpu_qwen` (criado na 1C Task 11) exercita `decode` — que agora é o caminho de command buffer único. Rodá-lo é o gate de correção da 1D.

Run: `cargo test -p llama-vulkan --test integration resident_forward_logits_iguais_a_cpu_qwen -- --nocapture 2>&1 | tail -30`
Expected: PASS (ou "pulando"). Se falhar **agora** (passava na 1C), o bug está na 1D: ordem do plano ≠ ordem da 1C, barreira faltando entre ops dependentes, push de rope/attention errado, ou offset de KV/embedding. Comparar a ordem de `build_plan` com a de `decode_step` (1C) op a op.

- [ ] **Step 2: Teste de geração multi-token vs CPU (KV-cache ao longo de vários tokens)**

Adicionar a `integration.rs`:

```rust
#[test]
fn resident_forward_gera_igual_cpu_multi_token() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    use llama_tokenizer::Tokenizer;
    use rand::{SeedableRng, rngs::SmallRng};

    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else { eprintln!("modelo ausente — pulando"); return; };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::load(&f, &bytes).unwrap();
    let tok = Tokenizer::from_gguf(&f).unwrap();
    let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let aux = model.gpu_aux_weights().unwrap();
    let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux).unwrap();
    let sampler = llama_sampling::Sampler::Greedy;

    let mut cpu_out = String::new();
    let mut r1 = SmallRng::seed_from_u64(0);
    model.generate_streaming(&tok, "Hello", 8, &sampler, &mut r1, &mut |p| cpu_out.push_str(p)).unwrap();

    let mut gpu_out = String::new();
    let mut r2 = SmallRng::seed_from_u64(0);
    model.generate_streaming_gpu_resident(&tok, "Hello", 8, &sampler, &mut r2, &backend, &mut |p| gpu_out.push_str(p)).unwrap();

    assert_eq!(cpu_out, gpu_out, "geração GPU-resident (1D) deve igualar CPU em 8 tokens");
}
```

> Greedy (argmax) torna a comparação determinística. Se divergir só após N tokens, o bug é no append/leitura do KV-cache ao crescer `pos` (offset `(layer·ctx+pos)·kv_dim` ou `total_len`).

- [ ] **Step 2 (run): rodar**

Run: `cargo test -p llama-vulkan --test integration resident_forward_gera_igual_cpu_multi_token -- --nocapture 2>&1 | tail -30`
Expected: PASS (ou "pulando").

- [ ] **Step 3: Commit**

```bash
git add crates/llama-vulkan/tests/integration.rs
git commit -m "test(vulkan): decode 1D == CPU (1 token e geração de 8 tokens)"
```

---

## Task 5: Benchmark — o salto de tok/s

**Files:**
- Modify: `bench-results/` (script já tem a linha `--gpu-resident` da Fase 1C)

- [ ] **Step 1: Rodar o benchmark no hardware**

A flag `--gpu-resident` agora usa o caminho de command buffer único. Não há mudança no script (a linha "1x MI50 (resident-fwd)" foi adicionada na Fase 1C).

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -30`
Expected: "1x MI50 (resident-fwd)" reporta tok/s **muito maior** que na 1C (que ainda fazia ~169 `wait_idle`/token). **Meta da Fase 1** (spec §7): chegar à vizinhança dos **314 tok/s** do llama.cpp single-GPU no 0.5B. O número exato decide se a tese (row-split nas 2 GPUs no 14B vence) está madura para a Fase 2, ou se ainda falta otimização de kernel (Fase 3).

- [ ] **Step 2: Comparar a progressão 1A→1B→1C→1D**

Run: `ls -t bench-results/ | head -5`
Abrir os arquivos e montar a progressão dos números "1x MI50 (resident*)". Anotar no corpo do commit (ex.: `2.16 → … → … → N tok/s`).

- [ ] **Step 3: Commit**

```bash
git add bench-results/
git commit -m "bench(gpu): 1x MI50 resident-fwd em 1 cmd/token (Fase 1D) — progressão da Fase 1"
```

---

## Self-Review

**1. Cobertura do spec (§4.4 — 1 command buffer/token, sem sync por-op):** Task 1 pré-aloca/escreve os descriptor sets (bindings estáticos); Task 2 grava a stack inteira em `token_cmd` com `full_barrier` entre ops; Task 3 faz **1** `queue_submit` + **1** `wait_for_fences` por token e lê só os logits (`readback(b_logits)`). Os ~169 submits+waits da 1C somem. §4.5 (kernel wave64) e barreiras finas/overlap ficam para a Fase 3 (documentado na arquitetura: barreira global é grossa mas correta).

**2. Placeholders:** Sem TBD/TODO. `build_plan`, `record_token`, `record_and_submit`, `full_barrier`, o novo fim do `new` e as extensões de `Drop` têm o código real. Os pontos "confirmar `max_sets`" e "remover dead code dos `op_*`/`decode_step` da 1C" são checagens concretas contra o estado pós-1C, não lacunas de design.

**3. Consistência de tipos:** os push-constants gravados em `record_token` casam exatamente com os shaders e com os tamanhos da 1C (rope 16B, attention 24B, rmsnorm 8B via `Static`, matvec 12B via `Static`, add/swiglu 4B via `Static`). `PlannedOp`/`PushSpec`/`PipeId` definidos na Task 1 são consumidos identicamente em `build_plan` (Task 1) e `record_token` (Task 2). `token_cmd`/`token_fence`/`plan` adicionados à struct (Task 2) são inicializados no `new` (Task 3) e liberados no `Drop` (Task 3). A trait `GpuResidentDecode::{decode,reset}` permanece a mesma da 1C — só o corpo de `decode` muda (de N submits para 1).

**4. Dependência declarada:** a 1D **exige** a 1C concluída e correta. O gate de correção é o teste da 1C continuar passando pelo novo caminho (Task 4 Step 1), reforçado pelo teste de geração multi-token (Task 4 Step 2) que exercita o KV-cache residente ao longo de `pos` crescente.

---

## Conclusão da Fase 1

Com 1A→1B→1C→1D, o decode single-GPU passa de **2.16 tok/s** (protótipo da Fase 7) para o regime residente + 1 command buffer/token. Conforme o número final do benchmark (Task 5):

- **Se ≈/≥ 314 tok/s (llama.cpp 1× MI50):** a tese está validada no campo de prova; seguir para a **Fase 2** (tensor-parallel row-split nas 2 MI50 no 14B) — depende do mecanismo de all-reduce medido na **Fase 0**.
- **Se ainda abaixo:** o gargalo é kernel/arquitetura (não comunicação) → **Fase 3** (matvec wave64 com subgroup reduction otimizada, coalescência, barreiras finas, overlap) antes de escalar para 2 GPUs.

Ambas as próximas fases têm planos próprios (a serem escritos quando seus pré-requisitos — número da Fase 1, spike de banda P2P da Fase 0 — existirem; ver spec §7).
