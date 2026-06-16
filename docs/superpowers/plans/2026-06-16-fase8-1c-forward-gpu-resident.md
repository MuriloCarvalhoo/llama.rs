# Fase 8.1C — Forward 100% na GPU (fim do ping-pong CPU↔GPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rodar o decode inteiro (RMSNorm, q/k/v, bias, RoPE, attention-GQA com KV-cache, attn-output, residual, SwiGLU, FFN, norm final, logits) na GPU, mantendo todas as ativações e o KV-cache **residentes em VRAM**. Só os logits finais voltam ao host. Acaba o ping-pong CPU↔GPU que hoje força readback após cada um dos ~169 matvecs/token.

**Architecture:** Novo backend `ResidentForward` (llama-vulkan) que implementa uma nova trait `GpuResidentDecode` (llama-model). Diferente do `ResidentGpu` (1A/1B, que só faz matvec e devolve `Vec<f32>` ao host por chamada), o `ResidentForward` carrega **todos** os pesos (Q8_0 + auxiliares f32) e mantém um conjunto fixo de buffers de ativação e o KV-cache em VRAM. O método `decode(token, pos)` grava a stack de ops como uma sequência de dispatches que leem/escrevem esses buffers residentes; nada de intermediário cruza o PCIe. Cinco shaders novos (`rmsnorm`, `rope`, `attention`, `swiglu`, `add`) acompanham o `q8_0_matvec` já existente. **Nesta fatia cada op ainda é 1 submit + 1 `queue_wait_idle`** (a fusão em 1 command buffer/token é a Fase 1D); o ganho de 1C é eliminar o readback de intermediários e validar a correção de todos os shaders bit-a-bit (tolerância) vs CPU no `qwen2.5-0.5b`.

**Tech Stack:** Rust, `ash` (Vulkan), shaderc, GLSL compute (wave64/subgroup), RADV/gfx906 (MI50). Reusa `GpuTensor`, `create_buf`/`alloc_and_bind`/`one_shot_copy`. Generaliza `ComputePipeline` e `build.rs` para múltiplos shaders.

**Limitação assumida (campo de prova 0.5B):** o shader de attention usa 1 lane por dimensão de head e exige `head_dim ≤ 64` (qwen2.5-0.5b: `head_dim=64`). `head_dim>64` (ex.: 14B=128) é rejeitado em runtime com erro claro — generalização fica para a Fase 2. O offset do KV-cache (`l*ctx*kv_dim`) é passado como `u32`; cabe no 0.5B. Ambos documentados no código.

---

## Contexto (ler antes de começar)

O caminho de referência na CPU está em `crates/llama-model/src/gpu.rs::forward_gpu` (linhas 151-235) e usa os kernels de `ops.rs`/`attention.rs`. Os shaders desta fatia **replicam exatamente** essa matemática:

- `rmsnorm_and_scale` (ops.rs:56-66): `ss=Σx²; scale=1/√(ss/dim+eps); out[i]=x[i]·scale·w[i]`.
- `rope_norm` (ops.rs:934-957): por head, rotaciona pares `(2i,2i+1)` por `θ=pos·freq[i]`.
- `attention` GQA causal (attention.rs:76-166): `kv_h=h/n_rep`; `score[j]=⟨q_h,k_j⟩·1/√head_dim`; softmax; `out=Σ score[j]·v_j`.
- `swiglu` (ops.rs:960-969): `silu(g)·u`, `silu(g)=g/(1+e^{-g})`.
- residual/bias: soma elementwise.
- KV-cache (attention.rs:9-67): layout `[n_layer·ctx·kv_dim]`, token-major por camada.

A comparação de igualdade usa **tolerância ~1e-3** (não igualdade exata) — mesma razão do teste `forward_gpu_mock_identico_a_forward_cpu` (gpu.rs:287): `sin/cos`/`exp` da GPU e a ordem de soma diferem levemente da CPU.

**Não tocar:** `dual_gpu.rs`, `backend.rs`, `matmul.rs`, o `ResidentGpu` (1A/1B) e a flag `--gpu-single` (continuam válidos para comparação). `forward_gpu` permanece como caminho de referência.

---

## File Structure

- **Modify:** `crates/llama-vulkan/build.rs` — compilar os 6 shaders, emitindo um env var por shader.
- **Create:** `crates/llama-vulkan/shaders/rmsnorm.comp`, `rope.comp`, `attention.comp`, `swiglu.comp`, `add.comp`.
- **Modify:** `crates/llama-vulkan/src/lib.rs` — consts SPV dos novos shaders; `mod resident_forward;` + `pub use`.
- **Modify:** `crates/llama-vulkan/src/pipeline.rs` — `ComputePipeline::with(dev, spv, n_bindings, push_size)` genérico; `new` delega para matvec.
- **Create:** `crates/llama-vulkan/src/resident_forward.rs` — `ResidentForward`: pipelines + pesos + buffers residentes + KV-cache + `decode`.
- **Modify:** `crates/llama-model/src/gpu.rs` — trait `GpuResidentDecode`, struct `GpuAuxWeights`, `Model::gpu_aux_weights`, `Model::generate_streaming_gpu_resident`, `Model::decode_one_gpu_resident_owned`.
- **Modify:** `crates/llama-cli/src/args.rs` + `runner.rs` — flag `--gpu-resident`.
- **Modify:** `crates/llama-vulkan/tests/integration.rs` — micro-testes por shader + teste bit-exact do decode resident vs CPU.
- **Modify:** `scripts/benchmark-gpu.sh` — linha "llama-rs — 1x MI50 (resident-fwd)".

---

## Task 1: Generalizar `build.rs` e `lib.rs` para múltiplos shaders

**Files:**
- Modify: `crates/llama-vulkan/build.rs`
- Modify: `crates/llama-vulkan/src/lib.rs`

- [ ] **Step 1: Criar os 5 arquivos de shader vazios (placeholders para o build compilar)**

Para o `build.rs` generalizado não falhar antes das próximas tasks, crie os arquivos com um `main` mínimo válido. Eles serão substituídos pelo conteúdo real nas Tasks 3-7.

Criar `crates/llama-vulkan/shaders/rmsnorm.comp`, `rope.comp`, `attention.comp`, `swiglu.comp`, `add.comp`, cada um com:

```glsl
#version 450
layout(local_size_x = 64) in;
void main() {}
```

- [ ] **Step 2: Reescrever `build.rs` para compilar todos os shaders**

Substituir o conteúdo inteiro de `crates/llama-vulkan/build.rs` por:

```rust
use std::path::PathBuf;

fn main() {
    // (arquivo, env var). O matvec já existia; os demais entram na Fase 1C.
    let shaders = [
        ("q8_0_matvec.comp", "Q8_0_MATVEC_SPV"),
        ("rmsnorm.comp", "RMSNORM_SPV"),
        ("rope.comp", "ROPE_SPV"),
        ("attention.comp", "ATTENTION_SPV"),
        ("swiglu.comp", "SWIGLU_SPV"),
        ("add.comp", "ADD_SPV"),
    ];

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let compiler = shaderc::Compiler::new().expect("shaderc init falhou");
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);

    for (file, env) in shaders {
        let src_path = PathBuf::from("shaders").join(file);
        println!("cargo:rerun-if-changed=shaders/{file}");
        let src = std::fs::read_to_string(&src_path)
            .unwrap_or_else(|_| panic!("nao encontrou {}", src_path.display()));
        let artifact = compiler
            .compile_into_spirv(&src, shaderc::ShaderKind::Compute, file, "main", Some(&opts))
            .unwrap_or_else(|e| panic!("Falha ao compilar {file}: {e}"));
        let spv_path = out_dir.join(format!("{file}.spv"));
        std::fs::write(&spv_path, artifact.as_binary_u8())
            .unwrap_or_else(|e| panic!("falha ao escrever {}: {e}", spv_path.display()));
        println!("cargo:rustc-env={env}={}", spv_path.display());
    }
}
```

- [ ] **Step 3: Declarar os consts SPV em `lib.rs`**

Substituir a linha `pub(crate) const Q8_0_MATVEC_SPV: ...` (lib.rs:18) por:

```rust
pub(crate) const Q8_0_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("Q8_0_MATVEC_SPV")));
pub(crate) const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("RMSNORM_SPV")));
pub(crate) const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("ROPE_SPV")));
pub(crate) const ATTENTION_SPV: &[u8] = include_bytes!(concat!(env!("ATTENTION_SPV")));
pub(crate) const SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("SWIGLU_SPV")));
pub(crate) const ADD_SPV: &[u8] = include_bytes!(concat!(env!("ADD_SPV")));
```

Remover o `#[allow(dead_code)]` da linha anterior se passar a gerar warning (os consts serão usados na Task 9).

- [ ] **Step 4: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila; os 6 SPVs são gerados. Warnings de consts não usados são aceitáveis até a Task 9.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-vulkan/build.rs crates/llama-vulkan/src/lib.rs crates/llama-vulkan/shaders/
git commit -m "build(vulkan): compila múltiplos shaders (rmsnorm/rope/attention/swiglu/add)"
```

---

## Task 2: Generalizar `ComputePipeline`

**Files:**
- Modify: `crates/llama-vulkan/src/pipeline.rs`

- [ ] **Step 1: Adicionar o construtor genérico `with` e fazer `new` delegar**

Em `pipeline.rs`, substituir a função `pub fn new(...)` inteira (linhas 31-126) por:

```rust
    /// Pipeline do matvec Q8_0 (3 bindings STORAGE_BUFFER + push de `PushConstants`).
    pub fn new(dev: &ash::Device) -> Result<Self, PipelineError> {
        Self::with(
            dev,
            crate::Q8_0_MATVEC_SPV,
            3,
            std::mem::size_of::<PushConstants>() as u32,
        )
    }

    /// Pipeline de compute genérico: `n_bindings` STORAGE_BUFFER (bindings 0..n) +
    /// um push-constant range de `push_size` bytes (COMPUTE). `spv` é o SPIR-V já compilado.
    pub fn with(
        dev: &ash::Device,
        spv: &[u8],
        n_bindings: u32,
        push_size: u32,
    ) -> Result<Self, PipelineError> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
            .map(|b| vk::DescriptorSetLayoutBinding {
                binding: b,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        // SAFETY: dev válido; dsl_info aponta para `bindings` vivo na stack.
        let desc_set_layout = unsafe { dev.create_descriptor_set_layout(&dsl_info, None)? };

        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_size,
        };
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &desc_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        // SAFETY: dev válido; layout_info aponta para dados vivos na stack.
        let layout = unsafe { dev.create_pipeline_layout(&layout_info, None)? };

        assert_eq!(spv.len() % 4, 0, "SPIR-V size deve ser multiplo de 4 bytes");
        let spv_u32: Vec<u32> = spv
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv_u32.as_ptr(),
            ..Default::default()
        };
        // SAFETY: dev válido; shader_info aponta para `spv_u32` vivo na stack.
        let shader_module = unsafe { dev.create_shader_module(&shader_info, None)? };

        let entry_point = c"main";
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage,
            layout,
            ..Default::default()
        };
        // SAFETY: dev válido; pipeline_info aponta para dados vivos na stack.
        let pipelines = unsafe {
            dev.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines[0];
        // SAFETY: shader_module foi criado por nós; a pipeline já o consumiu.
        unsafe { dev.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            layout,
            desc_set_layout,
        })
    }
```

- [ ] **Step 2: Compilar (matmul.rs e resident.rs continuam usando `new()` sem mudança)**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila. `ResidentGpu`/`dispatch_inner` seguem chamando `ComputePipeline::new(&dev)` (matvec), agora implementado via `with`.

- [ ] **Step 3: Regressão — testes da 1A/1B continuam PASS**

Run: `cargo test -p llama-vulkan --test integration resident_gpu -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando").

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/pipeline.rs
git commit -m "refactor(vulkan): ComputePipeline::with genérico (n_bindings, push_size)"
```

---

## Task 3: Esqueleto do `ResidentForward` + helper de buffer + dispatch genérico

**Files:**
- Create: `crates/llama-vulkan/src/resident_forward.rs`
- Modify: `crates/llama-vulkan/src/lib.rs`

Esta task cria a infra (device, pipelines vazios pendentes, helper `Buf`, dispatch genérico, readback) que as Tasks 4-7 usam para testar cada shader isoladamente.

- [ ] **Step 1: Criar `resident_forward.rs` com `Buf`, struct base, `dispatch1`, `readback`, `copy_region`**

```rust
//! Backend de decode 100% na GPU: todas as ativações e o KV-cache residentes em VRAM.
//! Só os logits finais voltam ao host. Cada op é 1 dispatch + 1 wait nesta fatia (1C);
//! a fusão em 1 command buffer/token é a Fase 1D.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::matmul::MatmulError;
use crate::pipeline::ComputePipeline;
use ash::vk;

/// Buffer Vulkan simples (device-local ou host-visible) com tamanho conhecido.
pub(crate) struct Buf {
    pub buffer: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Buf {
    /// Buffer device-local STORAGE | TRANSFER_SRC | TRANSFER_DST de `bytes`.
    fn device(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, false)?;
        Ok(Self { buffer, mem, size: bytes })
    }

    /// Buffer host-visible TRANSFER_SRC | TRANSFER_DST de `bytes` (upload/readback).
    fn host(
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        bytes: vk::DeviceSize,
    ) -> Result<Self, MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = create_buf(d, bytes, usage)?;
        let mem = alloc_and_bind(ctx, phys, d, buffer, true)?;
        Ok(Self { buffer, mem, size: bytes })
    }

    fn destroy(&self, d: &ash::Device) {
        // SAFETY: handles criados por nós; chamado no Drop, sem uso concorrente.
        unsafe {
            d.destroy_buffer(self.buffer, None);
            d.free_memory(self.mem, None);
        }
    }
}

/// Backend de decode GPU-resident (1 GPU). Construído via `ResidentForward::new`.
pub struct ResidentForward<'ctx> {
    pub(crate) ctx: &'ctx VulkanContext,
    pub(crate) phys_idx: usize,
    pub(crate) dev: VulkanDevice,
    // pipelines (preenchidos na Task 9; campos públicos ao crate para as tasks de teste)
    pub(crate) matvec: ComputePipeline,
    pub(crate) rmsnorm: ComputePipeline,
    pub(crate) rope: ComputePipeline,
    pub(crate) attention: ComputePipeline,
    pub(crate) swiglu: ComputePipeline,
    pub(crate) add: ComputePipeline,
    pub(crate) desc_pool: vk::DescriptorPool,
}

impl<'ctx> ResidentForward<'ctx> {
    pub(crate) fn phys(&self) -> &VulkanPhysicalDevice {
        &self.ctx.amd_compute_devices()[self.phys_idx]
    }

    /// Aloca um descriptor set do pool com o layout da pipeline dada.
    pub(crate) fn alloc_set(&self, pipe: &ComputePipeline) -> Result<vk::DescriptorSet, MatmulError> {
        let d = &self.dev.device;
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &pipe.desc_set_layout,
            ..Default::default()
        };
        // SAFETY: d e pool válidos; layout vem da pipeline.
        Ok(unsafe { d.allocate_descriptor_sets(&info)? }[0])
    }

    /// Escreve `bindings` no `set` (na ordem 0..n) e despacha `pipe` com `push` bytes e
    /// `groups` workgroups em x. Faz 1 submit + 1 wait. Não lê de volta (dados ficam residentes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch1(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        bindings: &[(vk::Buffer, vk::DeviceSize, vk::DeviceSize)], // (buffer, offset, range)
        push: &[u8],
        groups: u32,
    ) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;

        let buf_infos: Vec<vk::DescriptorBufferInfo> = bindings
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
        // SAFETY: d válido; writes apontam para buf_infos vivos; GPU ociosa (wait por op).
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: d e cmd_pool válidos.
        let cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            // SAFETY: cmd recém-alocado; pipe/set válidos; push tem o tamanho do range do layout.
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipe.layout,
                0,
                &[set],
                &[],
            );
            if !push.is_empty() {
                d.cmd_push_constants(
                    cmd,
                    pipe.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            d.cmd_dispatch(cmd, groups, 1, 1);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            // SAFETY: queue/submit/cmd válidos.
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }
        Ok(())
    }

    /// Sobe `data` (f32) para o `dst` device-local via um staging host-visible próprio.
    pub(crate) fn upload_f32(&self, dst: &Buf, data: &[f32]) -> Result<(), MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = std::mem::size_of_val(data) as vk::DeviceSize;
        let staging = Buf::host(self.ctx, self.phys(), d, bytes)?;
        unsafe {
            // SAFETY: staging host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(staging.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
            d.unmap_memory(staging.mem);
        }
        one_shot_copy(d, self.dev.queue, self.dev.cmd_pool, staging.buffer, dst.buffer, bytes)?;
        staging.destroy(d);
        Ok(())
    }

    /// Copia `len` bytes de `src[+src_off]` para `dst[+dst_off]` (regiões dentro de buffers residentes).
    pub(crate) fn copy_region(
        &self,
        src: vk::Buffer,
        src_off: vk::DeviceSize,
        dst: vk::Buffer,
        dst_off: vk::DeviceSize,
        len: vk::DeviceSize,
    ) -> Result<(), MatmulError> {
        let d = &self.dev.device;
        let dev = &self.dev;
        let cb_info = vk::CommandBufferAllocateInfo {
            command_pool: dev.cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        // SAFETY: d/pool válidos.
        let cmd = unsafe { d.allocate_command_buffers(&cb_info)? }[0];
        let begin = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        let region = vk::BufferCopy { src_offset: src_off, dst_offset: dst_off, size: len };
        unsafe {
            // SAFETY: cmd válido; src/dst buffers vivos; offsets/len dentro dos tamanhos (garantido pelo caller).
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_copy_buffer(cmd, src, dst, &[region]);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }
        Ok(())
    }

    /// Lê `len` f32 do `src` device-local de volta ao host.
    pub(crate) fn readback(&self, src: &Buf, len: usize) -> Result<Vec<f32>, MatmulError> {
        use crate::tensor::one_shot_copy;
        let d = &self.dev.device;
        let bytes = (len * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let host = Buf::host(self.ctx, self.phys(), d, bytes)?;
        one_shot_copy(d, self.dev.queue, self.dev.cmd_pool, src.buffer, host.buffer, bytes)?;
        let out = unsafe {
            // SAFETY: host host-visible com `bytes`; ptr válido até unmap.
            let ptr = d.map_memory(host.mem, 0, bytes, vk::MemoryMapFlags::empty())?;
            let mut v = vec![0f32; len];
            std::ptr::copy_nonoverlapping(ptr as *const f32, v.as_mut_ptr(), len);
            d.unmap_memory(host.mem);
            v
        };
        host.destroy(d);
        Ok(out)
    }
}
```

> Os campos de pipeline (`matvec`..`add`) e `desc_pool` ficam `pub(crate)` para que as tasks de teste (Tasks 4-7) construam o backend parcialmente. A struct ainda não tem `new()` completo — isso vem na Task 9. Para destravar os testes por-shader, a Task 4 adiciona um `new_pipelines_only()` mínimo.

- [ ] **Step 2: Registrar o módulo em `lib.rs`**

Após `mod resident;` adicionar `mod resident_forward;` e, nos re-exports (após `pub use resident::ResidentGpu;`):

```rust
pub use resident_forward::ResidentForward;
```

- [ ] **Step 3: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: erro de "missing field" / "no function `new`" **não** ocorre porque ninguém constrói `ResidentForward` ainda. Deve compilar com warnings de campos/métodos não usados. Se reclamar de `ComputePipeline` sem `Default`, ok — a struct só é instanciada na Task 4+.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs crates/llama-vulkan/src/lib.rs
git commit -m "feat(vulkan): esqueleto ResidentForward (Buf, dispatch1, upload, readback, copy_region)"
```

---

## Task 4: Shader `rmsnorm` + construtor de pipelines + micro-teste

**Files:**
- Modify: `crates/llama-vulkan/shaders/rmsnorm.comp`
- Modify: `crates/llama-vulkan/src/resident_forward.rs`
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Escrever o teste que falha (micro-teste do rmsnorm vs CPU)**

Adicionar a `crates/llama-vulkan/tests/integration.rs`:

```rust
#[test]
fn resident_fwd_rmsnorm_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let dim = 896usize;
    let x: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
    let w: Vec<f32> = (0..dim).map(|i| 1.0 + ((i % 7) as f32) * 0.01).collect();
    let eps = 1e-6f32;

    // CPU de referência: ss=Σx²; scale=1/√(ss/dim+eps); out=x·scale·w.
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
    let cpu: Vec<f32> = x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * scale * wi).collect();

    let gpu = fwd.dbg_rmsnorm(&x, &w, eps).unwrap();
    assert_eq!(gpu.len(), dim);
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "rmsnorm[{i}]: cpu={a} gpu={b}");
    }
}
```

- [ ] **Step 2: Rodar — deve falhar (sem `new_pipelines_only`/`dbg_rmsnorm`)**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_rmsnorm_igual_cpu 2>&1 | tail -20`
Expected: erro de compilação do teste (métodos inexistentes). É o "teste que falha".

- [ ] **Step 3: Escrever o shader `rmsnorm.comp`**

Substituir o conteúdo de `crates/llama-vulkan/shaders/rmsnorm.comp` por:

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
// 1 workgroup = 1 subgroup (wave64) processa a linha inteira.
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer XBuf { float x[]; } xb;
layout(set = 0, binding = 1) readonly buffer WBuf { float w[]; } wb;
layout(set = 0, binding = 2) writeonly buffer OBuf { float o[]; } ob;

layout(push_constant) uniform PC { uint dim; float eps; } pc;

void main() {
    uint lane = gl_LocalInvocationID.x;
    float ss = 0.0;
    for (uint i = lane; i < pc.dim; i += 64u) {
        float v = xb.x[i];
        ss += v * v;
    }
    ss = subgroupAdd(ss);                       // soma total em TODAS as lanes
    float scale = inversesqrt(ss / float(pc.dim) + pc.eps);
    for (uint i = lane; i < pc.dim; i += 64u) {
        ob.o[i] = xb.x[i] * scale * wb.w[i];
    }
}
```

- [ ] **Step 4: Adicionar `new_pipelines_only`, push struct e `dbg_rmsnorm`**

Em `resident_forward.rs`, adicionar dentro de `impl<'ctx> ResidentForward<'ctx>`:

```rust
    /// Constrói só device + pipelines + descriptor pool (sem pesos/buffers). Para micro-testes.
    pub fn new_pipelines_only(ctx: &'ctx VulkanContext) -> Result<Self, MatmulError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_INITIALIZATION_FAILED));
        }
        let dev = VulkanDevice::create(ctx, &phys[0])?;
        let d = &dev.device;
        let matvec = ComputePipeline::new(d)?;
        let rmsnorm = ComputePipeline::with(d, crate::RMSNORM_SPV, 3, 8)?; // dim:u32 + eps:f32
        let rope = ComputePipeline::with(d, crate::ROPE_SPV, 2, 16)?; // n_head,head_dim,rope_dim:u32 + pos:f32
        let attention = ComputePipeline::with(d, crate::ATTENTION_SPV, 4, 24)?; // 6×u32
        let swiglu = ComputePipeline::with(d, crate::SWIGLU_SPV, 3, 4)?; // n:u32
        let add = ComputePipeline::with(d, crate::ADD_SPV, 2, 4)?; // n:u32

        // Pool generoso: muitos sets (1 por op por camada serão alocados na Task 9/1D).
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4096,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1024,
            pool_size_count: 1,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: d válido; pool_info aponta para dados vivos.
        let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };

        Ok(Self {
            ctx,
            phys_idx: 0,
            dev,
            matvec,
            rmsnorm,
            rope,
            attention,
            swiglu,
            add,
            desc_pool,
        })
    }

    /// Diagnóstico: roda só o shader rmsnorm sobre `x`,`w` e devolve a saída ao host.
    pub fn dbg_rmsnorm(&self, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P { dim: u32, eps: f32 }
        let d = &self.dev.device;
        let dim = x.len();
        let xb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(x) as vk::DeviceSize)?;
        let wb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(w) as vk::DeviceSize)?;
        let ob = Buf::device(self.ctx, self.phys(), d, (dim * 4) as vk::DeviceSize)?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&wb, w)?;
        let set = self.alloc_set(&self.rmsnorm)?;
        let push = P { dim: dim as u32, eps };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(&push as *const P as *const u8, std::mem::size_of::<P>())
        };
        self.dispatch1(
            &self.rmsnorm,
            set,
            &[
                (xb.buffer, 0, xb.size),
                (wb.buffer, 0, wb.size),
                (ob.buffer, 0, ob.size),
            ],
            push_bytes,
            1, // 1 workgroup processa a linha
        )?;
        let out = self.readback(&ob, dim)?;
        xb.destroy(d);
        wb.destroy(d);
        ob.destroy(d);
        Ok(out)
    }
```

Adicionar também um `Drop` mínimo que destrói pipelines + pool (será estendido na Task 9):

```rust
impl Drop for ResidentForward<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: wait_idle garante GPU ociosa antes de liberar.
        unsafe {
            let _ = d.device_wait_idle();
            d.destroy_descriptor_pool(self.desc_pool, None);
        }
        // ComputePipeline::destroy consome self; destruímos handles direto (campos pub(crate)).
        for p in [&self.matvec, &self.rmsnorm, &self.rope, &self.attention, &self.swiglu, &self.add] {
            // SAFETY: handles criados por nós, ordem inversa.
            unsafe {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.desc_set_layout, None);
            }
        }
    }
}
```

- [ ] **Step 5: Rodar o teste — deve passar**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_rmsnorm_igual_cpu -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando"). Se falhar por valor, checar `inversesqrt` e o push (alinhamento `dim:u32, eps:f32` = 8 bytes).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/rmsnorm.comp crates/llama-vulkan/src/resident_forward.rs crates/llama-vulkan/tests/integration.rs
git commit -m "feat(vulkan): shader rmsnorm + dbg_rmsnorm == CPU (Fase 1C)"
```

---

## Task 5: Shader `swiglu` + `add` + micro-testes

**Files:**
- Modify: `crates/llama-vulkan/shaders/swiglu.comp`, `add.comp`
- Modify: `crates/llama-vulkan/src/resident_forward.rs`
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Escrever os testes que falham**

Adicionar a `integration.rs`:

```rust
#[test]
fn resident_fwd_swiglu_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 4864usize;
    let g: Vec<f32> = (0..n).map(|i| ((i % 11) as f32) * 0.2 - 1.0).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 + 0.1).collect();
    let cpu: Vec<f32> = g.iter().zip(u.iter())
        .map(|(&gi, &ui)| (gi / (1.0 + (-gi).exp())) * ui).collect();

    let gpu = fwd.dbg_swiglu(&g, &u).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "swiglu[{i}]: cpu={a} gpu={b}");
    }
}

#[test]
fn resident_fwd_add_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n = 896usize;
    let dst: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let src: Vec<f32> = (0..n).map(|i| i as f32 * -0.25 + 1.0).collect();
    let cpu: Vec<f32> = dst.iter().zip(src.iter()).map(|(&a, &b)| a + b).collect();

    let gpu = fwd.dbg_add(&dst, &src).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "add[{i}]: cpu={a} gpu={b}");
    }
}
```

- [ ] **Step 2: Rodar — deve falhar (compilação)**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_swiglu_igual_cpu 2>&1 | tail -10`
Expected: erro de compilação (métodos `dbg_swiglu`/`dbg_add` inexistentes).

- [ ] **Step 3: Escrever os shaders**

`crates/llama-vulkan/shaders/swiglu.comp`:

```glsl
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer GBuf { float g[]; } gb;
layout(set = 0, binding = 1) readonly buffer UBuf { float u[]; } ub;
layout(set = 0, binding = 2) writeonly buffer OBuf { float o[]; } ob;
layout(push_constant) uniform PC { uint n; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    float g = gb.g[i];
    float silu = g / (1.0 + exp(-g));
    ob.o[i] = silu * ub.u[i];
}
```

`crates/llama-vulkan/shaders/add.comp`:

```glsl
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer DBuf { float d[]; } db;     // inout: dst += src
layout(set = 0, binding = 1) readonly buffer SBuf { float s[]; } sb;
layout(push_constant) uniform PC { uint n; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    db.d[i] += sb.s[i];
}
```

- [ ] **Step 4: Adicionar `dbg_swiglu` e `dbg_add`**

Em `resident_forward.rs`, dentro do `impl`:

```rust
    /// nº de workgroups para cobrir `n` elementos com local_size_x=64.
    pub(crate) fn groups_for(n: usize) -> u32 {
        ((n + 63) / 64) as u32
    }

    pub fn dbg_swiglu(&self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P { n: u32 }
        let d = &self.dev.device;
        let n = g.len();
        let gb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(g) as vk::DeviceSize)?;
        let ub = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(u) as vk::DeviceSize)?;
        let ob = Buf::device(self.ctx, self.phys(), d, (n * 4) as vk::DeviceSize)?;
        self.upload_f32(&gb, g)?;
        self.upload_f32(&ub, u)?;
        let set = self.alloc_set(&self.swiglu)?;
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        self.dispatch1(
            &self.swiglu,
            set,
            &[(gb.buffer, 0, gb.size), (ub.buffer, 0, ub.size), (ob.buffer, 0, ob.size)],
            pb,
            Self::groups_for(n),
        )?;
        let out = self.readback(&ob, n)?;
        gb.destroy(d); ub.destroy(d); ob.destroy(d);
        Ok(out)
    }

    pub fn dbg_add(&self, dst: &[f32], src: &[f32]) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P { n: u32 }
        let d = &self.dev.device;
        let n = dst.len();
        let db = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(dst) as vk::DeviceSize)?;
        let sb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(src) as vk::DeviceSize)?;
        self.upload_f32(&db, dst)?;
        self.upload_f32(&sb, src)?;
        let set = self.alloc_set(&self.add)?;
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        self.dispatch1(
            &self.add,
            set,
            &[(db.buffer, 0, db.size), (sb.buffer, 0, sb.size)],
            pb,
            Self::groups_for(n),
        )?;
        let out = self.readback(&db, n)?; // dst foi mutado in-place
        db.destroy(d); sb.destroy(d);
        Ok(out)
    }
```

- [ ] **Step 5: Rodar os testes — devem passar**

Run: `cargo test -p llama-vulkan --test integration "resident_fwd_swiglu_igual_cpu|resident_fwd_add_igual_cpu" -- --nocapture 2>&1 | tail -20`
Expected: ambos PASS (ou "pulando").

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/swiglu.comp crates/llama-vulkan/shaders/add.comp crates/llama-vulkan/src/resident_forward.rs crates/llama-vulkan/tests/integration.rs
git commit -m "feat(vulkan): shaders swiglu/add + micro-testes == CPU (Fase 1C)"
```

---

## Task 6: Shader `rope` + micro-teste

**Files:**
- Modify: `crates/llama-vulkan/shaders/rope.comp`
- Modify: `crates/llama-vulkan/src/resident_forward.rs`
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Teste que falha (rope vs CPU `rope_norm`)**

Adicionar a `integration.rs`:

```rust
#[test]
fn resident_fwd_rope_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n_head = 14usize;
    let head_dim = 64usize;
    let rope_dim = 64usize;
    let pos = 5usize;
    let freq_base = 1_000_000.0f32;
    // freq_table[i] = freq_base^(-2i/rope_dim)
    let freq: Vec<f32> = (0..rope_dim / 2)
        .map(|i| freq_base.powf(-2.0 * i as f32 / rope_dim as f32))
        .collect();
    let mut x: Vec<f32> = (0..n_head * head_dim).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();

    // CPU de referência (ops.rs::rope_norm, t=0).
    let mut cpu = x.clone();
    for h in 0..n_head {
        let base = h * head_dim;
        for i in 0..rope_dim / 2 {
            let theta = pos as f32 * freq[i];
            let (s, c) = theta.sin_cos();
            let a = cpu[base + 2 * i];
            let b = cpu[base + 2 * i + 1];
            cpu[base + 2 * i] = a * c - b * s;
            cpu[base + 2 * i + 1] = a * s + b * c;
        }
    }

    let gpu = fwd.dbg_rope(&mut x, n_head, head_dim, rope_dim, &freq, pos).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "rope[{i}]: cpu={a} gpu={b}");
    }
}
```

- [ ] **Step 2: Rodar — falha (compilação)**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_rope_igual_cpu 2>&1 | tail -10`
Expected: erro `dbg_rope` inexistente.

- [ ] **Step 3: Escrever `rope.comp`**

```glsl
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer XBuf { float x[]; } xb;        // inout
layout(set = 0, binding = 1) readonly buffer FBuf { float f[]; } fb; // freq_table[rope_dim/2]
layout(push_constant) uniform PC {
    uint n_head; uint head_dim; uint rope_dim; float pos;
} pc;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint half_rope = pc.rope_dim / 2u;
    uint total = pc.n_head * half_rope;
    if (idx >= total) return;
    uint h = idx / half_rope;
    uint i = idx % half_rope;
    uint base = h * pc.head_dim;
    float theta = pc.pos * fb.f[i];
    float s = sin(theta);
    float c = cos(theta);
    float a = xb.x[base + 2u * i];
    float b = xb.x[base + 2u * i + 1u];
    xb.x[base + 2u * i]      = a * c - b * s;
    xb.x[base + 2u * i + 1u] = a * s + b * c;
}
```

- [ ] **Step 4: Adicionar `dbg_rope`**

```rust
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_rope(
        &self,
        x: &mut [f32],
        n_head: usize,
        head_dim: usize,
        rope_dim: usize,
        freq: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        #[repr(C)]
        struct P { n_head: u32, head_dim: u32, rope_dim: u32, pos: f32 }
        let d = &self.dev.device;
        let xb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(x) as vk::DeviceSize)?;
        let fb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(freq) as vk::DeviceSize)?;
        self.upload_f32(&xb, x)?;
        self.upload_f32(&fb, freq)?;
        let set = self.alloc_set(&self.rope)?;
        let push = P {
            n_head: n_head as u32,
            head_dim: head_dim as u32,
            rope_dim: rope_dim as u32,
            pos: pos as f32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 16) };
        let pairs = n_head * (rope_dim / 2);
        self.dispatch1(
            &self.rope,
            set,
            &[(xb.buffer, 0, xb.size), (fb.buffer, 0, fb.size)],
            pb,
            Self::groups_for(pairs),
        )?;
        let out = self.readback(&xb, x.len())?;
        xb.destroy(d); fb.destroy(d);
        Ok(out)
    }
```

- [ ] **Step 5: Rodar — passa**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_rope_igual_cpu -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando").

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/rope.comp crates/llama-vulkan/src/resident_forward.rs crates/llama-vulkan/tests/integration.rs
git commit -m "feat(vulkan): shader rope + dbg_rope == CPU (Fase 1C)"
```

---

## Task 7: Shader `attention` (GQA causal, decode) + micro-teste

**Files:**
- Modify: `crates/llama-vulkan/shaders/attention.comp`
- Modify: `crates/llama-vulkan/src/resident_forward.rs`
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Teste que falha (attention vs CPU `attention`)**

Adicionar a `integration.rs`:

```rust
#[test]
fn resident_fwd_attention_igual_cpu() {
    use llama_vulkan::{ResidentForward, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let fwd = ResidentForward::new_pipelines_only(&ctx).unwrap();

    let n_head = 14usize;
    let n_head_kv = 2usize;
    let head_dim = 64usize;
    let kv_dim = n_head_kv * head_dim;
    let total_len = 7usize; // pos0=6, +1
    let n_rep = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..n_head * head_dim).map(|i| ((i % 19) as f32) * 0.05 - 0.4).collect();
    let kc: Vec<f32> = (0..total_len * kv_dim).map(|i| ((i % 23) as f32) * 0.03 - 0.3).collect();
    let vc: Vec<f32> = (0..total_len * kv_dim).map(|i| ((i % 29) as f32) * 0.02 - 0.2).collect();

    // CPU de referência (attention.rs, n_tok=1, pos0=total_len-1).
    let mut cpu = vec![0f32; n_head * head_dim];
    for h in 0..n_head {
        let kv_h = h / n_rep;
        let qoff = h * head_dim;
        let mut scores = vec![0f32; total_len];
        for j in 0..total_len {
            let koff = j * kv_dim + kv_h * head_dim;
            let dot: f32 = (0..head_dim).map(|d| q[qoff + d] * kc[koff + d]).sum();
            scores[j] = dot * scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in scores.iter_mut() { *s = (*s - m).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        for j in 0..total_len {
            let voff = j * kv_dim + kv_h * head_dim;
            for d in 0..head_dim { cpu[qoff + d] += scores[j] * vc[voff + d]; }
        }
    }

    let gpu = fwd.dbg_attention(&q, &kc, &vc, n_head, n_head_kv, head_dim, total_len).unwrap();
    for (i, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "attn[{i}]: cpu={a} gpu={b}");
    }
}
```

- [ ] **Step 2: Rodar — falha (compilação)**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_attention_igual_cpu 2>&1 | tail -10`
Expected: erro `dbg_attention` inexistente.

- [ ] **Step 3: Escrever `attention.comp` (online softmax; 1 workgroup/head, 1 lane/dim)**

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
// 1 workgroup por head; 1 lane por dimensão do head (exige head_dim <= 64).
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer QBuf { float q[]; } qb;   // [n_head*head_dim]
layout(set = 0, binding = 1) readonly buffer KBuf { float k[]; } kb;   // KV-cache K (camada via kv_layer_off)
layout(set = 0, binding = 2) readonly buffer VBuf { float v[]; } vb;   // KV-cache V
layout(set = 0, binding = 3) writeonly buffer OBuf { float o[]; } ob;  // [n_head*head_dim]

layout(push_constant) uniform PC {
    uint n_head; uint n_head_kv; uint head_dim;
    uint total_len; uint kv_dim; uint kv_layer_off; // offset em elementos f32
} pc;

void main() {
    uint h = gl_WorkGroupID.x;
    uint d = gl_LocalInvocationID.x;
    if (h >= pc.n_head) return;
    uint n_rep = pc.n_head / pc.n_head_kv;
    uint kv_h = h / n_rep;
    float scale = inversesqrt(float(pc.head_dim));
    bool active = d < pc.head_dim;
    float qd = active ? qb.q[h * pc.head_dim + d] : 0.0;

    float m = -1.0e30;   // max corrente
    float l = 0.0;       // soma corrente de exp
    float acc = 0.0;     // saída da dimensão d (online softmax)
    for (uint j = 0u; j < pc.total_len; j++) {
        uint kvbase = pc.kv_layer_off + j * pc.kv_dim + kv_h * pc.head_dim;
        float prod = active ? qd * kb.k[kvbase + d] : 0.0;
        float s = subgroupAdd(prod) * scale;   // score uniforme em todas as lanes
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * corr + p;
        float vd = active ? vb.v[kvbase + d] : 0.0;
        acc = acc * corr + p * vd;
        m = m_new;
    }
    if (active) {
        ob.o[h * pc.head_dim + d] = acc / l;
    }
}
```

- [ ] **Step 4: Adicionar `dbg_attention` (com a checagem `head_dim ≤ 64`)**

```rust
    #[allow(clippy::too_many_arguments)]
    pub fn dbg_attention(
        &self,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        total_len: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        if head_dim > 64 {
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        #[repr(C)]
        struct P {
            n_head: u32,
            n_head_kv: u32,
            head_dim: u32,
            total_len: u32,
            kv_dim: u32,
            kv_layer_off: u32,
        }
        let d = &self.dev.device;
        let kv_dim = n_head_kv * head_dim;
        let qb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(q) as vk::DeviceSize)?;
        let kb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(k_cache) as vk::DeviceSize)?;
        let vb = Buf::device(self.ctx, self.phys(), d, std::mem::size_of_val(v_cache) as vk::DeviceSize)?;
        let ob = Buf::device(self.ctx, self.phys(), d, (n_head * head_dim * 4) as vk::DeviceSize)?;
        self.upload_f32(&qb, q)?;
        self.upload_f32(&kb, k_cache)?;
        self.upload_f32(&vb, v_cache)?;
        let set = self.alloc_set(&self.attention)?;
        let push = P {
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            head_dim: head_dim as u32,
            total_len: total_len as u32,
            kv_dim: kv_dim as u32,
            kv_layer_off: 0,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 24) };
        self.dispatch1(
            &self.attention,
            set,
            &[
                (qb.buffer, 0, qb.size),
                (kb.buffer, 0, kb.size),
                (vb.buffer, 0, vb.size),
                (ob.buffer, 0, ob.size),
            ],
            pb,
            n_head as u32, // 1 workgroup por head
        )?;
        let out = self.readback(&ob, n_head * head_dim)?;
        qb.destroy(d); kb.destroy(d); vb.destroy(d); ob.destroy(d);
        Ok(out)
    }
```

- [ ] **Step 5: Rodar — passa**

Run: `cargo test -p llama-vulkan --test integration resident_fwd_attention_igual_cpu -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando"). Falha por valor → revisar o online-softmax (corr/p) ou o `subgroupAdd` (head_dim deve ser ≤ subgroup=64).

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/shaders/attention.comp crates/llama-vulkan/src/resident_forward.rs crates/llama-vulkan/tests/integration.rs
git commit -m "feat(vulkan): shader attention GQA (online softmax) + dbg == CPU (Fase 1C)"
```

---

## Task 8: `GpuAuxWeights` + trait `GpuResidentDecode` (llama-model)

**Files:**
- Modify: `crates/llama-model/src/gpu.rs`

- [ ] **Step 1: Adicionar `GpuAuxWeights`, `AuxLayer` e o builder `Model::gpu_aux_weights`**

Em `crates/llama-model/src/gpu.rs`, após a definição de `GpuRawWeights` (linha 77), adicionar:

```rust
/// Pesos auxiliares f32 que o decode GPU-resident precisa (norm/bias/freq/embd).
/// `token_embd` é emprestado do Model (grande); o resto é copiado (pequeno).
pub struct GpuAuxWeights<'a> {
    pub token_embd: &'a [f32], // [vocab * n_embd]
    pub layers: Vec<AuxLayer>,
    pub output_norm: Vec<f32>, // [n_embd]
    pub freq_table: Vec<f32>,  // [rope_dim/2]
}

/// Pesos auxiliares f32 por camada.
pub struct AuxLayer {
    pub attn_norm: Vec<f32>,      // [n_embd]
    pub ffn_norm: Vec<f32>,       // [n_embd]
    pub q_bias: Option<Vec<f32>>, // [n_embd]
    pub k_bias: Option<Vec<f32>>, // [kv_dim]
    pub v_bias: Option<Vec<f32>>, // [kv_dim]
}
```

E, dentro de `#[cfg(feature = "gpu")] impl Model` (junto aos helpers `layer_norms_f32` etc., gpu.rs:87+), adicionar:

```rust
    /// Coleta os pesos auxiliares f32 para o backend GPU-resident.
    pub fn gpu_aux_weights(&self) -> Result<GpuAuxWeights<'_>, ModelError> {
        let token_embd = self.token_embd_f32()?;
        let output_norm = self.output_norm_f32()?.to_vec();
        let freq_table = self.freq_table.clone();
        let mut layers = Vec::with_capacity(self.config.n_layer);
        for l in 0..self.config.n_layer {
            let (attn_norm, ffn_norm) = self.layer_norms_f32(l)?;
            let lw = &self.weights.layers[l];
            let bias = |b: &Option<_>| -> Result<Option<Vec<f32>>, ModelError> {
                match b {
                    Some(t) => Ok(Some(t.dequant_to_f32()?.to_vec())),
                    None => Ok(None),
                }
            };
            layers.push(AuxLayer {
                attn_norm: attn_norm.to_vec(),
                ffn_norm: ffn_norm.to_vec(),
                q_bias: bias(&lw.attn_q_bias)?,
                k_bias: bias(&lw.attn_k_bias)?,
                v_bias: bias(&lw.attn_v_bias)?,
            });
        }
        Ok(GpuAuxWeights { token_embd, layers, output_norm, freq_table })
    }
```

> Verificar os nomes exatos dos campos de bias em `weights.rs` (`attn_q_bias`/`attn_k_bias`/`attn_v_bias`) e o método `dequant_to_f32`, ambos já usados em `add_layer_biases` (model.rs:293-313). Alinhar se diferirem.

- [ ] **Step 2: Adicionar a trait `GpuResidentDecode`**

Após a trait `GpuMatmul` (gpu.rs:243-251), adicionar:

```rust
/// Decode 100% na GPU: a stack inteira do token roda na GPU, KV-cache residente.
/// Implementado por `llama_vulkan::ResidentForward`.
pub trait GpuResidentDecode {
    /// Decode de 1 token na posição absoluta `pos` (0-based). Retorna os logits ([vocab]).
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, ModelError>;
    /// Zera o comprimento do KV-cache interno (início de nova sequência).
    fn reset(&self);
}
```

- [ ] **Step 3: Adicionar o loop de geração e o helper de decode-1 GPU-resident**

Dentro de `#[cfg(feature = "gpu")] impl Model`, adicionar:

```rust
    /// Igual a `generate_streaming_gpu`, mas o token inteiro (prefill incluído) roda na
    /// GPU via `gpu` (KV-cache residente). Os logits só voltam ao host para o sampler.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_gpu_resident(
        &self,
        tokenizer: &llama_tokenizer::Tokenizer,
        prompt: &str,
        n_tokens: usize,
        sampler: &llama_sampling::Sampler,
        rng: &mut impl rand::Rng,
        gpu: &dyn GpuResidentDecode,
        on_token: &mut impl FnMut(&str),
    ) -> Result<(), ModelError> {
        let prompt_ids = tokenizer.encode(prompt, true);
        if prompt_ids.is_empty() {
            return Err(ModelError::Gpu("prompt vazio".into()));
        }
        gpu.reset();

        // Prefill na GPU: alimenta os tokens do prompt um a um (constrói o KV-cache).
        let mut logits = Vec::new();
        for (pos, &t) in prompt_ids.iter().enumerate() {
            logits = gpu.decode(t, pos)?;
        }
        let first_idx = sampler.sample(&logits, rng);
        let mut next = u32::try_from(first_idx).map_err(|_| ModelError::Overflow)?;
        let mut pos = prompt_ids.len();

        let mut count = 0usize;
        while count < n_tokens {
            if next == self.config.eos_id {
                break;
            }
            let piece = tokenizer.decode(&[next]);
            on_token(&piece);
            count += 1;
            let logits = gpu.decode(next, pos)?;
            pos += 1;
            let idx = sampler.sample(&logits, rng);
            next = u32::try_from(idx).map_err(|_| ModelError::Overflow)?;
        }
        Ok(())
    }

    /// Prefill + 1 decode 100% na GPU; argmax dos logits. Para teste de paridade vs CPU.
    pub fn decode_one_gpu_resident_owned(
        &self,
        prompt: &[u32],
        gpu: &dyn GpuResidentDecode,
    ) -> Result<u32, ModelError> {
        gpu.reset();
        let mut logits = Vec::new();
        for (pos, &t) in prompt.iter().enumerate() {
            logits = gpu.decode(t, pos)?;
        }
        u32::try_from(crate::ops::argmax(&logits)).map_err(|_| ModelError::Overflow)
    }
```

- [ ] **Step 4: Compilar o crate model com feature gpu**

Run: `cargo build -p llama-model --features gpu 2>&1 | tail -20`
Expected: compila. Erros de nome de campo de bias → alinhar com `weights.rs`.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-model/src/gpu.rs
git commit -m "feat(model): GpuAuxWeights + trait GpuResidentDecode + loop de geração resident"
```

---

## Task 9: `ResidentForward::new` — pesos + buffers + KV-cache residentes

**Files:**
- Modify: `crates/llama-vulkan/src/resident_forward.rs`

- [ ] **Step 1: Adicionar struct de config, campos de pesos/buffers e o `new` completo**

Em `resident_forward.rs`, adicionar perto do topo (após `Buf`):

```rust
use crate::tensor::GpuTensor;
use llama_model::{GpuAuxWeights, GpuRawWeights, LlamaConfig};
use std::cell::RefCell;

/// Escalares de arquitetura necessários ao decode.
pub(crate) struct Cfg {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rope_dim: usize,
    pub kv_dim: usize,
    pub vocab: usize,
    pub ctx: usize,
    pub rms_eps: f32,
}

/// Pesos Q8_0 residentes de uma camada (handles GpuTensor).
pub(crate) struct LayerQ {
    pub attn_q: GpuTensor,
    pub attn_k: GpuTensor,
    pub attn_v: GpuTensor,
    pub attn_output: GpuTensor,
    pub ffn_gate: GpuTensor,
    pub ffn_up: GpuTensor,
    pub ffn_down: GpuTensor,
}

/// Buffers f32 auxiliares residentes de uma camada.
pub(crate) struct LayerAux {
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub q_bias: Option<Buf>,
    pub k_bias: Option<Buf>,
    pub v_bias: Option<Buf>,
}
```

Estender a struct `ResidentForward` com os campos residentes (adicionar aos já existentes da Task 3):

```rust
    // --- adicionar a `pub struct ResidentForward<'ctx>` (Task 3) ---
    pub(crate) cfg: Cfg,
    pub(crate) qw: Vec<LayerQ>,
    pub(crate) output_w: GpuTensor,
    pub(crate) aux: Vec<LayerAux>,
    pub(crate) output_norm_buf: Buf,
    pub(crate) freq_buf: Buf,
    pub(crate) token_embd_buf: Buf,
    // KV-cache residente [n_layer*ctx*kv_dim] f32 para K e V.
    pub(crate) kcache: Buf,
    pub(crate) vcache: Buf,
    // Buffers de ativação (reusados por camada).
    pub(crate) b_x: Buf,
    pub(crate) b_normed: Buf,
    pub(crate) b_q: Buf,
    pub(crate) b_k: Buf,
    pub(crate) b_v: Buf,
    pub(crate) b_attn: Buf,
    pub(crate) b_proj: Buf,
    pub(crate) b_gate: Buf,
    pub(crate) b_up: Buf,
    pub(crate) b_act: Buf,
    pub(crate) b_logits: Buf,
    pub(crate) len: RefCell<usize>,
```

- [ ] **Step 2: Implementar `new(ctx, config, raw, aux)`**

Adicionar ao `impl<'ctx> ResidentForward<'ctx>`:

```rust
    /// Constrói o backend GPU-resident: sobe todos os pesos (Q8_0 + aux f32) e aloca
    /// as ativações e o KV-cache em VRAM. Após retornar, `raw`/`aux` podem ser descartados.
    pub fn new(
        ctx: &'ctx VulkanContext,
        config: &LlamaConfig,
        raw: &GpuRawWeights,
        aux: &GpuAuxWeights<'_>,
    ) -> Result<Self, MatmulError> {
        if config.head_dim > 64 {
            // Shader de attention assume head_dim <= subgroup (64). Ver doc do módulo.
            return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
        let base = Self::new_pipelines_only(ctx)?;
        let d = &base.dev.device;
        let phys = base.phys();
        let kv_dim = config.n_head_kv * config.head_dim;
        let cfg = Cfg {
            n_embd: config.n_embd,
            n_layer: config.n_layer,
            n_head: config.n_head,
            n_head_kv: config.n_head_kv,
            head_dim: config.head_dim,
            n_ff: config.n_ff,
            rope_dim: config.rope_dim,
            kv_dim,
            vocab: config.vocab,
            ctx: config.ctx,
            rms_eps: config.rms_eps,
        };

        // Pesos Q8_0 residentes (upload_q8_0 já existe; n_in/n_out p/ shape).
        let up_q = |bytes: &[u8], n_in: usize, n_out: usize| -> Result<GpuTensor, MatmulError> {
            Ok(GpuTensor::upload_q8_0(ctx, phys, &base.dev, bytes, n_in, n_out)?)
        };
        let mut qw = Vec::with_capacity(cfg.n_layer);
        for lw in &raw.layers {
            qw.push(LayerQ {
                attn_q: up_q(&lw.attn_q, cfg.n_embd, cfg.n_embd)?,
                attn_k: up_q(&lw.attn_k, cfg.n_embd, kv_dim)?,
                attn_v: up_q(&lw.attn_v, cfg.n_embd, kv_dim)?,
                attn_output: up_q(&lw.attn_output, cfg.n_embd, cfg.n_embd)?,
                ffn_gate: up_q(&lw.ffn_gate, cfg.n_embd, cfg.n_ff)?,
                ffn_up: up_q(&lw.ffn_up, cfg.n_embd, cfg.n_ff)?,
                ffn_down: up_q(&lw.ffn_down, cfg.n_ff, cfg.n_embd)?,
            });
        }
        let output_w = up_q(&raw.output, cfg.n_embd, cfg.vocab)?;

        // Aux f32 residentes.
        let mk = |data: &[f32]| -> Result<Buf, MatmulError> {
            let b = Buf::device(ctx, phys, d, std::mem::size_of_val(data) as vk::DeviceSize)?;
            base.upload_f32(&b, data)?;
            Ok(b)
        };
        let mk_opt = |o: &Option<Vec<f32>>| -> Result<Option<Buf>, MatmulError> {
            match o {
                Some(v) => Ok(Some(mk(v)?)),
                None => Ok(None),
            }
        };
        let mut aux_buf = Vec::with_capacity(cfg.n_layer);
        for al in &aux.layers {
            aux_buf.push(LayerAux {
                attn_norm: mk(&al.attn_norm)?,
                ffn_norm: mk(&al.ffn_norm)?,
                q_bias: mk_opt(&al.q_bias)?,
                k_bias: mk_opt(&al.k_bias)?,
                v_bias: mk_opt(&al.v_bias)?,
            });
        }
        let output_norm_buf = mk(&aux.output_norm)?;
        let freq_buf = mk(&aux.freq_table)?;
        let token_embd_buf = mk(aux.token_embd)?;

        // KV-cache residente.
        let kv_elems = (cfg.n_layer * cfg.ctx * kv_dim) as vk::DeviceSize;
        let kcache = Buf::device(ctx, phys, d, kv_elems * 4)?;
        let vcache = Buf::device(ctx, phys, d, kv_elems * 4)?;

        // Buffers de ativação (tamanho fixo por shape do modelo).
        let nf = |n: usize| -> Result<Buf, MatmulError> {
            Buf::device(ctx, phys, d, (n * 4) as vk::DeviceSize)
        };

        Ok(Self {
            cfg,
            qw,
            output_w,
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
            ..base
        })
    }
```

> Nota: `..base` move os campos já preenchidos por `new_pipelines_only` (ctx, phys_idx, dev, pipelines, desc_pool). `LlamaConfig` precisa ser `pub use` em `llama-model` (já é usado como `model.config` no runner). `GpuRawWeights`/`GpuAuxWeights`/`AuxLayer` precisam estar re-exportados de `llama_model` (Task 8 os definiu como `pub`; confirmar `pub use`/visibilidade no `lib.rs` do crate model — adicionar `pub use gpu::{GpuAuxWeights, AuxLayer, GpuResidentDecode};` se necessário).

- [ ] **Step 3: Estender o `Drop` para liberar pesos/buffers/KV**

Substituir o `impl Drop` (da Task 4) por uma versão que também libera os recursos residentes:

```rust
impl Drop for ResidentForward<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: GPU ociosa antes de liberar.
        unsafe { let _ = d.device_wait_idle(); }

        // Pesos Q8_0.
        for lq in self.qw.drain(..) {
            for t in [lq.attn_q, lq.attn_k, lq.attn_v, lq.attn_output, lq.ffn_gate, lq.ffn_up, lq.ffn_down] {
                t.destroy(d);
            }
        }
        // output_w: GpuTensor::destroy consome self; troca por tensor "nulo" via std::mem::take não existe,
        // então liberamos via take de um Option seria mais limpo. Aqui usamos um drop manual:
        // (GpuTensor implementa Drop que libera se handles != null — basta deixar cair.)

        // Aux + ativações + KV.
        for la in self.aux.drain(..) {
            la.attn_norm.destroy(d);
            la.ffn_norm.destroy(d);
            if let Some(b) = la.q_bias { b.destroy(d); }
            if let Some(b) = la.k_bias { b.destroy(d); }
            if let Some(b) = la.v_bias { b.destroy(d); }
        }
        for b in [
            &self.output_norm_buf, &self.freq_buf, &self.token_embd_buf,
            &self.kcache, &self.vcache,
            &self.b_x, &self.b_normed, &self.b_q, &self.b_k, &self.b_v,
            &self.b_attn, &self.b_proj, &self.b_gate, &self.b_up, &self.b_act, &self.b_logits,
        ] {
            b.destroy(d);
        }

        // Pipelines + pool.
        unsafe { d.destroy_descriptor_pool(self.desc_pool, None); }
        for p in [&self.matvec, &self.rmsnorm, &self.rope, &self.attention, &self.swiglu, &self.add] {
            // SAFETY: handles criados por nós.
            unsafe {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.desc_set_layout, None);
            }
        }
    }
}
```

> `output_w` é um `GpuTensor` campo direto; seu `Drop` (tensor.rs:99) libera os handles automaticamente quando `ResidentForward` cai, **após** o `device_wait_idle` deste `Drop` (ordem de drop dos campos é após o corpo do `Drop` manual). Confirmar que `GpuTensor::Drop` usa o mesmo `ash::Device` — ele guarda só os handles e o `VulkanDevice` (dono do `ash::Device`) é dropado por último. Se houver risco de ordem, trocar `output_w: GpuTensor` por `Option<GpuTensor>` e `take().unwrap().destroy(d)` no corpo do `Drop`. **Implementar como `Option<GpuTensor>`** para controlar a ordem explicitamente:
> - campo: `pub(crate) output_w: Option<GpuTensor>,`
> - no `new`: `output_w: Some(output_w),`
> - no `Drop`: `if let Some(t) = self.output_w.take() { t.destroy(d); }`
> - no uso (Task 10): `self.output_w.as_ref().unwrap()`.

- [ ] **Step 4: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -30`
Expected: compila. Resolver erros de `pub use`/visibilidade de `LlamaConfig`/`GpuAuxWeights` no crate model conforme Step 2.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs crates/llama-model/src/lib.rs
git commit -m "feat(vulkan): ResidentForward::new — pesos+ativações+KV residentes (Fase 1C)"
```

---

## Task 10: `decode()` — orquestração da stack na GPU + impl da trait

**Files:**
- Modify: `crates/llama-vulkan/src/resident_forward.rs`

- [ ] **Step 1: Implementar os passos elementares como métodos privados**

Adicionar ao `impl<'ctx> ResidentForward<'ctx>` helpers que despacham cada op sobre os buffers residentes. Estes encapsulam o push e o set:

```rust
    fn op_rmsnorm(&self, x: &Buf, w: &Buf, out: &Buf, dim: usize) -> Result<(), MatmulError> {
        #[repr(C)]
        struct P { dim: u32, eps: f32 }
        let push = P { dim: dim as u32, eps: self.cfg.rms_eps };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 8) };
        let set = self.alloc_set(&self.rmsnorm)?;
        self.dispatch1(&self.rmsnorm, set,
            &[(x.buffer, 0, x.size), (w.buffer, 0, w.size), (out.buffer, 0, out.size)], pb, 1)
    }

    fn op_matvec(&self, w: &GpuTensor, x: &Buf, y: &Buf, n_in: usize, n_out: usize) -> Result<(), MatmulError> {
        use crate::pipeline::PushConstants;
        let push = PushConstants { n_in: n_in as u32, n_out: n_out as u32, row_offset: 0 };
        let pb = unsafe {
            std::slice::from_raw_parts(&push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>())
        };
        let set = self.alloc_set(&self.matvec)?;
        self.dispatch1(&self.matvec, set,
            &[(w.buffer, 0, w.size_bytes), (x.buffer, 0, (n_in * 4) as vk::DeviceSize),
              (y.buffer, 0, (n_out * 4) as vk::DeviceSize)], pb, n_out as u32)
    }

    fn op_add(&self, dst: &Buf, src: &Buf, n: usize) -> Result<(), MatmulError> {
        #[repr(C)]
        struct P { n: u32 }
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        let set = self.alloc_set(&self.add)?;
        self.dispatch1(&self.add, set,
            &[(dst.buffer, 0, (n * 4) as vk::DeviceSize), (src.buffer, 0, (n * 4) as vk::DeviceSize)],
            pb, Self::groups_for(n))
    }

    fn op_rope(&self, x: &Buf, n_head: usize, pos: usize) -> Result<(), MatmulError> {
        #[repr(C)]
        struct P { n_head: u32, head_dim: u32, rope_dim: u32, pos: f32 }
        let push = P {
            n_head: n_head as u32,
            head_dim: self.cfg.head_dim as u32,
            rope_dim: self.cfg.rope_dim as u32,
            pos: pos as f32,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 16) };
        let set = self.alloc_set(&self.rope)?;
        let pairs = n_head * (self.cfg.rope_dim / 2);
        self.dispatch1(&self.rope, set,
            &[(x.buffer, 0, x.size), (self.freq_buf.buffer, 0, self.freq_buf.size)],
            pb, Self::groups_for(pairs))
    }

    fn op_swiglu(&self, g: &Buf, u: &Buf, out: &Buf, n: usize) -> Result<(), MatmulError> {
        #[repr(C)]
        struct P { n: u32 }
        let push = P { n: n as u32 };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 4) };
        let set = self.alloc_set(&self.swiglu)?;
        self.dispatch1(&self.swiglu, set,
            &[(g.buffer, 0, (n*4) as vk::DeviceSize), (u.buffer, 0, (n*4) as vk::DeviceSize),
              (out.buffer, 0, (n*4) as vk::DeviceSize)], pb, Self::groups_for(n))
    }

    fn op_attention(&self, layer: usize, total_len: usize) -> Result<(), MatmulError> {
        #[repr(C)]
        struct P { n_head: u32, n_head_kv: u32, head_dim: u32, total_len: u32, kv_dim: u32, kv_layer_off: u32 }
        let layer_off = (layer * self.cfg.ctx * self.cfg.kv_dim) as u32;
        let push = P {
            n_head: self.cfg.n_head as u32,
            n_head_kv: self.cfg.n_head_kv as u32,
            head_dim: self.cfg.head_dim as u32,
            total_len: total_len as u32,
            kv_dim: self.cfg.kv_dim as u32,
            kv_layer_off: layer_off,
        };
        let pb = unsafe { std::slice::from_raw_parts(&push as *const P as *const u8, 24) };
        let set = self.alloc_set(&self.attention)?;
        // K/V-cache ligados pelo buffer inteiro; o offset da camada vai no push.
        self.dispatch1(&self.attention, set,
            &[(self.b_q.buffer, 0, self.b_q.size),
              (self.kcache.buffer, 0, self.kcache.size),
              (self.vcache.buffer, 0, self.vcache.size),
              (self.b_attn.buffer, 0, self.b_attn.size)],
            pb, self.cfg.n_head as u32)
    }
```

- [ ] **Step 2: Implementar `decode_step` e a impl da trait**

Adicionar:

```rust
    fn decode_step(&self, token: u32, pos: usize) -> Result<Vec<f32>, MatmulError> {
        let c = &self.cfg;
        let total_len = pos + 1;

        // 1. Embedding lookup: copia a linha do token_embd residente para b_x (sem host).
        let row_bytes = (c.n_embd * 4) as vk::DeviceSize;
        let src_off = (token as usize * c.n_embd * 4) as vk::DeviceSize;
        self.copy_region(self.token_embd_buf.buffer, src_off, self.b_x.buffer, 0, row_bytes)?;

        for l in 0..c.n_layer {
            let lq = &self.qw[l];
            let la = &self.aux[l];

            // attn norm -> b_normed
            self.op_rmsnorm(&self.b_x, &la.attn_norm, &self.b_normed, c.n_embd)?;
            // q/k/v
            self.op_matvec(&lq.attn_q, &self.b_normed, &self.b_q, c.n_embd, c.n_embd)?;
            self.op_matvec(&lq.attn_k, &self.b_normed, &self.b_k, c.n_embd, c.kv_dim)?;
            self.op_matvec(&lq.attn_v, &self.b_normed, &self.b_v, c.n_embd, c.kv_dim)?;
            // bias (se houver)
            if let Some(b) = &la.q_bias { self.op_add(&self.b_q, b, c.n_embd)?; }
            if let Some(b) = &la.k_bias { self.op_add(&self.b_k, b, c.kv_dim)?; }
            if let Some(b) = &la.v_bias { self.op_add(&self.b_v, b, c.kv_dim)?; }
            // rope em q e k
            self.op_rope(&self.b_q, c.n_head, pos)?;
            self.op_rope(&self.b_k, c.n_head_kv, pos)?;
            // append k/v ao KV-cache residente (offset = (l*ctx + pos)*kv_dim)
            let kv_off = ((l * c.ctx + pos) * c.kv_dim * 4) as vk::DeviceSize;
            let kv_bytes = (c.kv_dim * 4) as vk::DeviceSize;
            self.copy_region(self.b_k.buffer, 0, self.kcache.buffer, kv_off, kv_bytes)?;
            self.copy_region(self.b_v.buffer, 0, self.vcache.buffer, kv_off, kv_bytes)?;
            // attention -> b_attn
            self.op_attention(l, total_len)?;
            // attn output proj -> b_proj; residual x += proj
            self.op_matvec(&lq.attn_output, &self.b_attn, &self.b_proj, c.n_embd, c.n_embd)?;
            self.op_add(&self.b_x, &self.b_proj, c.n_embd)?;
            // ffn norm -> b_normed; gate/up; swiglu -> b_act; down -> b_proj; residual
            self.op_rmsnorm(&self.b_x, &la.ffn_norm, &self.b_normed, c.n_embd)?;
            self.op_matvec(&lq.ffn_gate, &self.b_normed, &self.b_gate, c.n_embd, c.n_ff)?;
            self.op_matvec(&lq.ffn_up, &self.b_normed, &self.b_up, c.n_embd, c.n_ff)?;
            self.op_swiglu(&self.b_gate, &self.b_up, &self.b_act, c.n_ff)?;
            self.op_matvec(&lq.ffn_down, &self.b_act, &self.b_proj, c.n_ff, c.n_embd)?;
            self.op_add(&self.b_x, &self.b_proj, c.n_embd)?;
        }

        // norm final -> b_normed; logits = output_w · b_normed
        self.op_rmsnorm(&self.b_x, &self.output_norm_buf, &self.b_normed, c.n_embd)?;
        let out_w = self.output_w.as_ref().unwrap();
        self.op_matvec(out_w, &self.b_normed, &self.b_logits, c.n_embd, c.vocab)?;
        self.readback(&self.b_logits, c.vocab)
    }
```

E a impl da trait no fim do arquivo:

```rust
impl llama_model::GpuResidentDecode for ResidentForward<'_> {
    fn decode(&self, token: u32, pos: usize) -> Result<Vec<f32>, llama_model::ModelError> {
        let logits = self
            .decode_step(token, pos)
            .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
        *self.len.borrow_mut() = pos + 1;
        Ok(logits)
    }
    fn reset(&self) {
        *self.len.borrow_mut() = 0;
    }
}
```

> Cada `alloc_set` aloca um descriptor set novo do pool a cada op. Como `max_sets=1024` e há ~169 ops/token, o pool **esgotaria em 1 token**. Para 1C (correção, não perf) há duas opções: (a) `reset_descriptor_pool` no início de cada `decode_step`; ou (b) alocar 1 set por *tipo* de pipeline e reusá-lo (seguro porque cada op faz `wait_idle`). Usar **(a)**: adicionar no início de `decode_step`, antes do embedding:
> ```rust
> // SAFETY: GPU ociosa entre tokens; libera todos os sets do pool para reuso.
> unsafe { self.dev.device.reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())?; }
> ```
> Isso recicla os sets a cada token (169 sets/token cabem em 1024). A fusão real (1 set por dispatch pré-alocado) é tratada na Fase 1D.

- [ ] **Step 3: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -30`
Expected: compila. Atenção a tipos `vk::DeviceSize` vs `usize` nos offsets.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident_forward.rs
git commit -m "feat(vulkan): ResidentForward::decode — stack completa na GPU + KV residente (Fase 1C)"
```

---

## Task 11: Teste bit-exact (tolerância) vs CPU + flag `--gpu-resident`

**Files:**
- Modify: `crates/llama-vulkan/tests/integration.rs`
- Modify: `crates/llama-cli/src/args.rs`
- Modify: `crates/llama-cli/src/runner.rs`

- [ ] **Step 1: Teste de paridade do decode resident vs CPU no 0.5B**

Adicionar a `integration.rs`:

```rust
#[test]
fn resident_forward_logits_iguais_a_cpu_qwen() {
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
    let cpu = model.decode_one_cpu_owned(&prompt).unwrap();
    let gpu = model.decode_one_gpu_resident_owned(&prompt, &backend).unwrap();
    assert_eq!(cpu, gpu, "argmax do decode GPU-resident deve igualar CPU");
}
```

> `Model::load(&f, &bytes)` é o construtor (model.rs:36); `decode_one_cpu_owned` existe (gpu.rs:127). `GpuAuxWeights`/`AuxLayer`/`GpuResidentDecode` precisam estar em `pub use gpu::{...}` no `lib.rs` do crate model (a linha hoje exporta `GpuLayerRaw, GpuMatmul, GpuRawWeights` — adicionar os três novos).

- [ ] **Step 2: Rodar — deve passar**

Run: `cargo test -p llama-vulkan --test integration resident_forward_logits_iguais_a_cpu_qwen -- --nocapture 2>&1 | tail -30`
Expected: PASS (ou "pulando"). Se o argmax diferir, comparar os logits intermediários: rodar uma versão do teste que compara `decode(prompt_last)` logits contra `forward(prompt)` logits com tolerância 1e-2 para achar a 1ª op divergente. (Os micro-testes das Tasks 4-7 já isolam cada shader; uma divergência aqui aponta erro de orquestração: offset de KV, ordem de bias/rope, ou pos/total_len.)

- [ ] **Step 3: Adicionar a flag `--gpu-resident`**

Em `crates/llama-cli/src/args.rs`, após o campo `pub gpu_single: bool`, adicionar:

```rust
    /// Backend Vulkan com decode 100% na GPU (Fase 1C). Requer feature "gpu".
    #[arg(long = "gpu-resident", default_value_t = false)]
    pub gpu_resident: bool,
```

- [ ] **Step 4: Branch no `runner.rs`**

No bloco `#[cfg(feature = "gpu")]`, adicionar um ramo `args.gpu_resident` **antes** de `args.gpu_single` (mesma estrutura). Inserir antes de `let used_gpu = if args.gpu_single {`:

```rust
    let used_gpu = if args.gpu_resident {
        use llama_vulkan::{ResidentForward, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if !ctx.amd_compute_devices().is_empty() => {
                let dev0 = ctx.amd_compute_devices()[0].name().to_owned();
                eprintln!("[GPU] {dev0} — decode 100% na GPU (resident-fwd)");
                let raw = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let aux = model.gpu_aux_weights()?;
                let backend = ResidentForward::new(&ctx, &model.config, &raw, &aux)
                    .map_err(|e| llama_model::ModelError::Gpu(e.to_string()))?;
                model.generate_streaming_gpu_resident(
                    &tokenizer, &args.prompt, args.n, &sampler, &mut rng,
                    &backend, &mut on_token,
                )?;
                true
            }
            Ok(_) => { eprintln!("[GPU] nenhum device AMD — fallback CPU"); false }
            Err(e) => { eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU"); false }
        }
    } else if args.gpu_single {
```

> Casar os identificadores locais (`&tokenizer`, `&args.prompt`, `args.n`, `&sampler`, `&mut rng`, `&mut on_token`) com o ramo `args.gpu_single` existente (Fase 1A). `ResidentForward::new` retorna `MatmulError`; mapear para `ModelError::Gpu` como acima.

- [ ] **Step 5: Estender o fallback `#[cfg(not(feature="gpu"))]`**

Onde houver `if args.gpu || args.gpu_single` (runner.rs:262), estender para `args.gpu || args.gpu_single || args.gpu_resident`.

- [ ] **Step 6: Compilar com e sem feature gpu**

Run: `cargo build -p llama-cli --features gpu 2>&1 | tail -20 && cargo build -p llama-cli 2>&1 | tail -10`
Expected: ambos compilam.

- [ ] **Step 7: Commit**

```bash
git add crates/llama-vulkan/tests/integration.rs crates/llama-cli/src/args.rs crates/llama-cli/src/runner.rs
git commit -m "feat(cli): --gpu-resident + teste ResidentForward == CPU no 0.5B (Fase 1C)"
```

---

## Task 12: Benchmark do decode GPU-resident

**Files:**
- Modify: `scripts/benchmark-gpu.sh` + `bench-results/`

- [ ] **Step 1: Adicionar a run resident-fwd**

Em `scripts/benchmark-gpu.sh`, após `run_rs_single()` (Fase 1A), adicionar:

```bash
# ── Rust (--gpu-resident, decode 100% na GPU) — tok/s de decode ──
run_rs_resident() {
    local log=$1
    "$RS_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt --timings --gpu-resident \
        2>"$log" >/dev/null || true
    assert_no_nvidia "$log" "llama-rs (--gpu-resident)"
    grep -oE "[0-9]+\.[0-9]+ tok/s" "$log" | grep -oE "^[0-9]+\.[0-9]+" | head -1
}
```

Na seção de execução, após `rs1=$(run_rs_single ...)`:

```bash
echo "Rodando llama-rs 1x MI50 (resident-fwd)..." >&2
rsf=$(run_rs_resident /tmp/bench-rsf.err)
```

Na tabela, após a linha "1x MI50 (resident)":

```bash
printf "| %-28s | %-16s |\n" "llama-rs  — 1x MI50 (resident-fwd)" "${rsf:-erro}"
```

- [ ] **Step 2: Rodar o benchmark no hardware**

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -30`
Expected: a linha "1x MI50 (resident-fwd)" aparece. **Esperado:** correção mantida; tok/s **pode não saltar** vs 1B (ainda há ~169 `wait_idle`/token — agora também das ops de norm/rope/attn/etc.). O ganho de perf vem na 1D (1 command buffer/token). 1C entrega **correção do forward inteiro na GPU**, pré-requisito da 1D.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark-gpu.sh bench-results/
git commit -m "bench(gpu): 1x MI50 decode 100% na GPU (resident-fwd, Fase 1C)"
```

---

## Self-Review

**1. Cobertura do spec (§4.3 — forward 100% na GPU; §4.1 restante — KV-cache residente):** Tasks 4-7 portam RMSNorm/SwiGLU/add/RoPE/attention-GQA para shaders (cada um validado vs CPU); Task 9 aloca KV-cache + ativações residentes; Task 10 orquestra a stack inteira na GPU com só os logits voltando ao host (`readback(b_logits)`). §4.4 (1 command buffer/token) é explicitamente a Fase 1D — aqui cada op ainda é 1 submit + `wait_idle`, documentado na arquitetura. §4.5 (kernel wave64) reusa o `q8_0_matvec` da Fase 7.

**2. Placeholders:** Sem TBD/TODO. Todo shader tem GLSL completo; todo passo Rust tem o código real (helpers `op_*`, `decode_step`, `new`, `Drop`). Os pontos "alinhar nomes de campo/assinatura" (bias em weights.rs, `Model::load`) são checagens contra o arquivo real com instrução concreta — não lacunas de design. A nota sobre `output_w: Option<GpuTensor>` (Task 9 Step 3) resolve a ordem de `Drop` de forma determinística.

**3. Consistência de tipos:** push-constants casam shader↔Rust em tamanho e ordem em cada op: rmsnorm `{dim:u32,eps:f32}`=8B; rope `{n_head,head_dim,rope_dim:u32,pos:f32}`=16B; attention 6×u32=24B; swiglu/add `{n:u32}`=4B; matvec `PushConstants`=12B. Bindings batem com `n_bindings` passado a `ComputePipeline::with` (rmsnorm 3, rope 2, attention 4, swiglu 3, add 2, matvec 3). `Buf`/`Cfg`/`LayerQ`/`LayerAux`/`ResidentForward` usados de forma idêntica em Tasks 3-11. A trait `GpuResidentDecode::{decode,reset}` (Task 8) é implementada exatamente na Task 10. `groups_for` usado em todos os shaders elementwise.

**4. Risco coberto:** o pool de descriptors esgotaria em 1 token (169 sets > 1024 só após 6 tokens) — resolvido com `reset_descriptor_pool` por token (Task 10 Step 2). `head_dim>64` e overflow de `kv_layer_off` em u32 (modelos grandes) são rejeitados/documentados; o alvo da fatia é o 0.5B (head_dim=64).

---

## Próxima fatia

- **1D** — gravar a stack inteira do token como **1 command buffer** com pipeline barriers entre dispatches; 1 submit + 1 fence/token; descriptor sets pré-alocados (1 por dispatch, sem reset por token); readback só dos logits. É onde o tok/s salta para a vizinhança do llama.cpp. Ver `2026-06-16-fase8-1d-command-buffer-unico.md`.
