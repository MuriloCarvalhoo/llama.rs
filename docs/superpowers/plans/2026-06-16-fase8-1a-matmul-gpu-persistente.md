# Fase 8.1A — Matmul GPU persistente (single-GPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar o re-upload de pesos e a recriação de pipeline por matvec no decode GPU, criando um backend single-GPU com pesos e pipeline residentes — o primeiro salto de tok/s rumo a bater o llama.cpp.

**Architecture:** Novo `ResidentGpu` (llama-vulkan) que implementa `llama_model::GpuMatmul`. Faz upload de cada matriz de peso **uma vez** (cache por ponteiro do slice `w_bytes`) e cria a `ComputePipeline` **uma vez** em `new()`. Cada `matvec_q8_0` reusa o `GpuTensor` residente e a pipeline; só `x`/`y` e o command buffer são por-chamada. Validação: saída bit-exact vs CPU no `qwen2.5-0.5b` (campo de prova local). Norm/RoPE/attention/SwiGLU permanecem na CPU nesta fatia (vão para a GPU na Fase 1C).

**Tech Stack:** Rust, `ash` (Vulkan), shaderc, RADV/gfx906 (MI50). Reusa `GpuTensor`, `ComputePipeline`, `create_buf`/`alloc_and_bind`/`one_shot_copy`, `PushConstants` e o shader `q8_0_matvec.comp` existentes.

---

## Contexto do diagnóstico (ler antes de começar)

Caminho atual do decode GPU:

`runner.rs` (`--gpu`) → `Model::generate_streaming_gpu` → `Model::forward_gpu` (`crates/llama-model/src/gpu.rs:151`) → por camada chama 8× `gpu.matvec_q8_0(w_bytes, x, n_in, n_out)` → `DualGpuBackend` (`backend.rs`) → `DualGpuMatmul::matvec_q8_0` (`dual_gpu.rs`) → `dispatch_inner` (`matmul.rs:58`).

`dispatch_inner` faz **por chamada**: `GpuTensor::upload_q8_0` (re-upload de TODOS os bytes do peso), `ComputePipeline::new` (recria shader/pipeline), descriptor pool/set novos, command buffer novo, `queue_wait_idle`, readback, e `destroy` de tudo. São ~169 chamadas/token → 2.16 tok/s.

A trait que força isso (`crates/llama-model/src/gpu.rs:243`):

```rust
pub trait GpuMatmul {
    fn matvec_q8_0(&self, w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize)
        -> Result<Vec<f32>, ModelError>;
}
```

`forward_gpu` passa sempre os MESMOS slices (`&gw.attn_q`, etc., campos de `GpuRawWeights` carregado uma vez) → o ponteiro de `w_bytes` é estável entre tokens. Isso permite cachear o `GpuTensor` por `w_bytes.as_ptr()` sem mudar a trait nem `forward_gpu` (mudança cirúrgica).

**Não tocar nesta fatia:** `dual_gpu.rs`, `backend.rs` (caminho dual fica para a Fase 2), `forward_gpu`, a trait `GpuMatmul`, os shaders.

---

## File Structure

- **Create:** `crates/llama-vulkan/src/resident.rs` — `ResidentGpu` single-GPU com pesos+pipeline residentes. Responsabilidade única: implementar `GpuMatmul` reusando recursos persistentes.
- **Modify:** `crates/llama-vulkan/src/lib.rs` — declarar `mod resident;` e `pub use resident::ResidentGpu;`.
- **Modify:** `crates/llama-cli/src/args.rs` — adicionar flag `--gpu-single`.
- **Modify:** `crates/llama-cli/src/runner.rs:179-224` — branch que usa `ResidentGpu` quando `--gpu-single`.
- **Modify:** `crates/llama-vulkan/tests/integration.rs` — teste de hardware: `ResidentGpu` == CPU no 0.5B + prova de reuso do cache.
- **Modify:** `scripts/benchmark-gpu.sh` — registrar linha "llama-rs — 1x MI50 (resident)".

---

## Task 1: Esqueleto do `ResidentGpu` (device + pipeline residente + cache de pesos)

**Files:**
- Create: `crates/llama-vulkan/src/resident.rs`
- Modify: `crates/llama-vulkan/src/lib.rs`

- [ ] **Step 1: Criar `resident.rs` com a struct, `new()` e `Drop`**

```rust
//! Backend single-GPU com pesos Q8_0 e pipeline residentes em VRAM.
//!
//! Diferente de `dispatch_inner` (que re-faz upload + pipeline por chamada), aqui:
//! - a `ComputePipeline` é criada uma vez em `new()`;
//! - cada matriz de peso é enviada à VRAM uma vez e cacheada por `w_bytes.as_ptr()`
//!   (os slices vindos de `GpuRawWeights` têm ponteiro estável entre tokens);
//! - só `x`/`y` e o command buffer são alocados por chamada.

use crate::device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice};
use crate::matmul::MatmulError;
use crate::pipeline::ComputePipeline;
use crate::tensor::GpuTensor;
use ash::vk;
use llama_model::{GpuMatmul, ModelError};
use std::cell::RefCell;
use std::collections::HashMap;

/// Backend de matmul Q8_0 numa única GPU AMD, com pesos+pipeline residentes.
pub struct ResidentGpu<'ctx> {
    ctx: &'ctx VulkanContext,
    phys_idx: usize,
    dev: VulkanDevice,
    pipeline: ComputePipeline,
    /// key = `w_bytes.as_ptr() as usize`; value = peso já residente na VRAM.
    weights: RefCell<HashMap<usize, GpuTensor>>,
}

impl<'ctx> ResidentGpu<'ctx> {
    /// Inicializa no primeiro device AMD. A pipeline é criada uma única vez.
    pub fn new(ctx: &'ctx VulkanContext) -> Result<Self, ModelError> {
        let phys = ctx.amd_compute_devices();
        if phys.is_empty() {
            return Err(ModelError::Gpu("nenhum device AMD".into()));
        }
        let dev = VulkanDevice::create(ctx, &phys[0])
            .map_err(|e| ModelError::Gpu(format!("device create: {e}")))?;
        let pipeline = ComputePipeline::new(&dev.device)
            .map_err(|e| ModelError::Gpu(format!("pipeline: {e}")))?;
        Ok(Self {
            ctx,
            phys_idx: 0,
            dev,
            pipeline,
            weights: RefCell::new(HashMap::new()),
        })
    }

    fn phys(&self) -> &VulkanPhysicalDevice {
        &self.ctx.amd_compute_devices()[self.phys_idx]
    }
}

impl Drop for ResidentGpu<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        for (_, t) in self.weights.borrow_mut().drain() {
            t.destroy(d);
        }
        // ComputePipeline::destroy consome self; troca por um valor "vazio" não existe,
        // então destruímos os handles diretamente (mesma ordem de ComputePipeline::destroy).
        unsafe {
            d.destroy_pipeline(self.pipeline.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline.layout, None);
            d.destroy_descriptor_set_layout(self.pipeline.desc_set_layout, None);
        }
    }
}
```

> Nota: `ComputePipeline`/`GpuTensor` têm campos `pub(crate)` e `resident.rs` está no mesmo crate, então o acesso direto em `Drop` é válido. `VulkanContext::amd_compute_devices` já é `pub`.

- [ ] **Step 2: Registrar o módulo em `lib.rs`**

Em `crates/llama-vulkan/src/lib.rs`, após `pub mod matmul;` adicionar:

```rust
mod resident;
```

e na seção de re-exports (após `pub use model_gpu::GpuWeights;`):

```rust
pub use resident::ResidentGpu;
```

- [ ] **Step 3: Compilar (sem o método ainda; deve faltar a impl da trait — esperado)**

Run: `cargo build -p llama-vulkan --features gpu 2>&1 | tail -20`
Expected: compila o esqueleto; ainda **não** implementa `GpuMatmul` (Task 2). Se aparecer só warning de `unused`, ok. Erros de trait não-implementada não ocorrem aqui pois `ResidentGpu` ainda não é usado como `dyn GpuMatmul`.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs crates/llama-vulkan/src/lib.rs
git commit -m "feat(vulkan): esqueleto ResidentGpu (device + pipeline residente + cache pesos)"
```

---

## Task 2: Implementar `GpuMatmul` reusando peso residente + pipeline persistente

**Files:**
- Modify: `crates/llama-vulkan/src/resident.rs`

- [ ] **Step 1: Adicionar helper de upload-once e a impl da trait**

Adicionar dentro de `impl<'ctx> ResidentGpu<'ctx>` (antes do `impl Drop`):

```rust
    /// Garante que o peso identificado por `w_bytes.as_ptr()` está residente.
    /// Faz upload na primeira vez; chamadas seguintes são cache-hit.
    fn ensure_weight(&self, w_bytes: &[u8], n_in: usize, n_out: usize)
        -> Result<(), MatmulError>
    {
        let key = w_bytes.as_ptr() as usize;
        if self.weights.borrow().contains_key(&key) {
            return Ok(());
        }
        let t = GpuTensor::upload_q8_0(self.ctx, self.phys(), &self.dev, w_bytes, n_in, n_out)?;
        self.weights.borrow_mut().insert(key, t);
        Ok(())
    }
```

Adicionar a impl da trait no fim do arquivo:

```rust
impl GpuMatmul for ResidentGpu<'_> {
    fn matvec_q8_0(&self, w_bytes: &[u8], x: &[f32], n_in: usize, n_out: usize)
        -> Result<Vec<f32>, ModelError>
    {
        self.ensure_weight(w_bytes, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))?;
        self.dispatch(w_bytes.as_ptr() as usize, x, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))
    }
}
```

- [ ] **Step 2: Implementar `dispatch` (x/y + descriptor + cmd por chamada; peso e pipeline reusados)**

Adicionar dentro de `impl<'ctx> ResidentGpu<'ctx>`:

```rust
    fn dispatch(&self, weight_key: usize, x_f32: &[f32], n_in: usize, n_out: usize)
        -> Result<Vec<f32>, MatmulError>
    {
        use crate::pipeline::PushConstants;
        use crate::tensor::{alloc_and_bind, create_buf, one_shot_copy};

        let d = &self.dev.device;
        let dev = &self.dev;
        let weights = self.weights.borrow();
        let w_tensor = weights.get(&weight_key).expect("peso garantido por ensure_weight");

        // X: staging host-visible -> device-local STORAGE
        let x_size = std::mem::size_of_val(x_f32) as vk::DeviceSize;
        let x_staging = create_buf(d, x_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let x_staging_mem = alloc_and_bind(self.ctx, self.phys(), d, x_staging, true)?;
        unsafe {
            // SAFETY: x_staging_mem é host-visible com x_size bytes; ptr válido até unmap.
            let ptr = d.map_memory(x_staging_mem, 0, x_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(x_f32.as_ptr(), ptr as *mut f32, x_f32.len());
            d.unmap_memory(x_staging_mem);
        }
        let x_buf = create_buf(
            d, x_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let x_mem = alloc_and_bind(self.ctx, self.phys(), d, x_buf, false)?;
        one_shot_copy(d, dev.queue, dev.cmd_pool, x_staging, x_buf, x_size)?;
        unsafe {
            // SAFETY: staging já copiado; criado por nós nesta função.
            d.destroy_buffer(x_staging, None);
            d.free_memory(x_staging_mem, None);
        }

        // Y: device-local STORAGE | TRANSFER_SRC
        let y_size = (n_out * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let y_buf = create_buf(
            d, y_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let y_mem = alloc_and_bind(self.ctx, self.phys(), d, y_buf, false)?;

        // Descriptor pool/set (por chamada; barato relativo ao upload de peso)
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        // SAFETY: d válido; pool_info aponta para dados válidos na stack.
        let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.pipeline.desc_set_layout,
            ..Default::default()
        };
        // SAFETY: d e desc_pool válidos.
        let desc_set = unsafe { d.allocate_descriptor_sets(&alloc_info)? }[0];

        let buf_infos = [
            vk::DescriptorBufferInfo { buffer: w_tensor.buffer, offset: 0, range: w_tensor.size_bytes },
            vk::DescriptorBufferInfo { buffer: x_buf, offset: 0, range: x_size },
            vk::DescriptorBufferInfo { buffer: y_buf, offset: 0, range: y_size },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(binding, bi)| vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: binding as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: bi,
                ..Default::default()
            })
            .collect();
        // SAFETY: d válido; writes apontam para buf_infos vivos na stack.
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        // Command buffer
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
        let push = PushConstants { n_in: n_in as u32, n_out: n_out as u32, row_offset: 0 };
        unsafe {
            // SAFETY: cmd recém-alocado e válido.
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.layout, 0, &[desc_set], &[],
            );
            d.cmd_push_constants(
                cmd, self.pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0,
                // SAFETY: PushConstants é #[repr(C)] 3×u32; slice de bytes válido.
                std::slice::from_raw_parts(
                    &push as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );
            d.cmd_dispatch(cmd, n_out as u32, 1, 1);
            d.end_command_buffer(cmd)?;
        }
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        unsafe {
            // SAFETY: queue, submit e cmd válidos.
            d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
            d.queue_wait_idle(dev.queue)?;
            d.free_command_buffers(dev.cmd_pool, &[cmd]);
        }

        // Readback Y
        let y_read = create_buf(d, y_size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let y_read_mem = alloc_and_bind(self.ctx, self.phys(), d, y_read, true)?;
        one_shot_copy(d, dev.queue, dev.cmd_pool, y_buf, y_read, y_size)?;
        let out = unsafe {
            // SAFETY: y_read_mem host-visible com y_size; ptr válido até unmap.
            let ptr = d.map_memory(y_read_mem, 0, y_size, vk::MemoryMapFlags::empty())?;
            let mut v = vec![0f32; n_out];
            std::ptr::copy_nonoverlapping(ptr as *const f32, v.as_mut_ptr(), n_out);
            d.unmap_memory(y_read_mem);
            v
        };

        // Cleanup do que é por-chamada (NÃO destrói peso nem pipeline)
        unsafe {
            d.destroy_buffer(y_read, None);
            d.free_memory(y_read_mem, None);
            d.destroy_descriptor_pool(desc_pool, None);
            d.destroy_buffer(y_buf, None);
            d.free_memory(y_mem, None);
            d.destroy_buffer(x_buf, None);
            d.free_memory(x_mem, None);
        }
        Ok(out)
    }
```

- [ ] **Step 2b: Garantir visibilidade dos helpers de `tensor.rs`**

Run: `grep -nE "pub(\\(crate\\))? fn (create_buf|alloc_and_bind|one_shot_copy)" crates/llama-vulkan/src/tensor.rs`
Expected: as três funções existem com visibilidade ao menos `pub(crate)` (matmul.rs já as importa). Se alguma for privada de módulo, eleve para `pub(crate)`.

- [ ] **Step 3: Compilar**

Run: `cargo build -p llama-vulkan --features gpu 2>&1 | tail -20`
Expected: compila sem erros. Warnings de `phys_idx`/campos só se algo ficou sem uso — não deve.

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs
git commit -m "feat(vulkan): ResidentGpu impl GpuMatmul reusando peso+pipeline residentes"
```

---

## Task 3: Wiring CLI `--gpu-single`

**Files:**
- Modify: `crates/llama-cli/src/args.rs`
- Modify: `crates/llama-cli/src/runner.rs`

- [ ] **Step 1: Adicionar a flag em `args.rs`**

Após o campo `pub gpu: bool` (`args.rs:48`), adicionar:

```rust
    /// Backend Vulkan single-GPU com pesos residentes (Fase 1A). Requer feature "gpu".
    #[arg(long = "gpu-single", default_value_t = false)]
    pub gpu_single: bool,
```

- [ ] **Step 2: Branch no `runner.rs`**

Em `runner.rs`, dentro do bloco `#[cfg(feature = "gpu")]` que hoje começa em `let used_gpu = if args.gpu {` (linha ~180), tratar o caso single ANTES do dual. Substituir a linha:

```rust
    let used_gpu = if args.gpu {
```

por:

```rust
    let used_gpu = if args.gpu_single {
        use llama_vulkan::{ResidentGpu, VulkanContext};
        match VulkanContext::new() {
            Ok(ctx) if !ctx.amd_compute_devices().is_empty() => {
                let dev0 = ctx.amd_compute_devices()[0].name().to_owned();
                eprintln!("[GPU] {dev0} — decode na GPU (single, resident)");
                let gpu_w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config)?;
                let backend = ResidentGpu::new(&ctx)?;
                model.generate_streaming_gpu(
                    &tokenizer, &args.prompt, args.n, &sampler, &mut rng,
                    &backend, &gpu_w, &mut on_token,
                )?;
                true
            }
            Ok(_) => {
                eprintln!("[GPU] nenhum device AMD — fallback CPU");
                false
            }
            Err(e) => {
                eprintln!("[GPU] Vulkan indisponivel ({e}) — fallback CPU");
                false
            }
        }
    } else if args.gpu {
```

> **Atenção:** os nomes exatos dos argumentos de `generate_streaming_gpu` (tokenizer, prompt, n_tokens, sampler, rng, gpu, w, on_token) devem casar com o branch `args.gpu` logo abaixo (linhas ~192-200). Copie os mesmos identificadores locais já usados lá (`&tokenizer`, `&args.prompt`, `args.n`, `&sampler`, `&mut rng`, `&mut on_token`). Ajuste se os nomes diferirem no arquivo real.

- [ ] **Step 3: Compilar CLI com feature gpu**

Run: `cargo build -p llama-cli --features gpu 2>&1 | tail -20`
Expected: compila. Se houver erro de nome de variável no branch, alinhar com o branch `args.gpu` existente.

- [ ] **Step 4: Compilar SEM feature gpu (não pode quebrar)**

Run: `cargo build -p llama-cli 2>&1 | tail -10`
Expected: compila; `--gpu-single` é aceito pelo parser mas cai no aviso de build-sem-gpu (mesmo tratamento de `--gpu`). Se o `#[cfg(not(feature="gpu"))]` checar só `args.gpu`, estender para `args.gpu || args.gpu_single`.

- [ ] **Step 5: Commit**

```bash
git add crates/llama-cli/src/args.rs crates/llama-cli/src/runner.rs
git commit -m "feat(cli): flag --gpu-single (backend ResidentGpu single-MI50)"
```

---

## Task 4: Teste de hardware — `ResidentGpu` == CPU + reuso de cache

**Files:**
- Modify: `crates/llama-vulkan/tests/integration.rs`

> Estes testes exigem as MI50 presentes. Seguir o padrão dos testes existentes em `integration.rs`: pular com `eprintln!` + `return` quando o hardware/modelo não está disponível (não falhar).

- [ ] **Step 1: Escrever o teste de igualdade vs CPU**

Adicionar ao fim de `crates/llama-vulkan/tests/integration.rs`:

```rust
#[test]
fn resident_gpu_logits_iguais_a_cpu_qwen() {
    use llama_vulkan::{ResidentGpu, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("Vulkan indisponível — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("nenhum device AMD — pulando");
        return;
    }
    let path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf";
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("modelo ausente — pulando");
        return;
    };
    let f = gguf::GgufFile::parse(&bytes).unwrap();
    let model = llama_model::Model::from_gguf(&f, &bytes).unwrap();
    let gpu_w = llama_model::GpuRawWeights::from_gguf(&f, &bytes, &model.config).unwrap();
    let backend = ResidentGpu::new(&ctx).unwrap();

    // prompt curto de ids estáveis do vocab.
    let prompt: [u32; 2] = [model.config.bos_id, 9707];
    let cpu = model.decode_one_cpu_owned(&prompt).unwrap();
    let gpu = model.decode_one_gpu_owned(&prompt, &backend, &gpu_w).unwrap();
    assert_eq!(cpu, gpu, "argmax do decode GPU residente deve igualar CPU");
}
```

> Se `Model::from_gguf` tiver outra assinatura, alinhar com o uso em `crates/llama-cli/src/runner.rs`. `decode_one_cpu_owned`/`decode_one_gpu_owned` já existem em `gpu.rs:127,134`.

- [ ] **Step 2: Rodar e verificar igualdade**

Run: `cargo test -p llama-vulkan --features gpu --test integration resident_gpu_logits_iguais_a_cpu_qwen -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando" se sem hardware). Se FALHAR por valor, é bug de dispatch — depurar antes de seguir.

- [ ] **Step 3: Teste de reuso de cache (mesma chamada 2× → 1 upload)**

Para validar que o peso NÃO é re-enviado, expor a contagem do cache. Adicionar em `resident.rs` dentro de `impl ResidentGpu`:

```rust
    /// Nº de pesos residentes (uploads efetuados). Para testes/diagnóstico.
    pub fn resident_count(&self) -> usize {
        self.weights.borrow().len()
    }
```

Adicionar o teste em `integration.rs`:

```rust
#[test]
fn resident_gpu_nao_re_uploada_peso() {
    use llama_vulkan::{ResidentGpu, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else { eprintln!("sem Vulkan — pulando"); return; };
    if ctx.amd_compute_devices().is_empty() { eprintln!("sem AMD — pulando"); return; }
    let backend = ResidentGpu::new(&ctx).unwrap();

    // Peso Q8_0 sintético 1 linha × 32 col: 34 bytes (2 scale + 32 quants).
    let w = vec![0u8; 34];
    let x = vec![0f32; 32];
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.resident_count(), 1, "primeiro uso = 1 upload");
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.resident_count(), 1, "mesmo ponteiro = cache-hit, sem novo upload");
}
```

- [ ] **Step 4: Rodar o teste de cache**

Run: `cargo test -p llama-vulkan --features gpu --test integration resident_gpu_nao_re_uploada_peso -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando").

- [ ] **Step 5: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs crates/llama-vulkan/tests/integration.rs
git commit -m "test(vulkan): ResidentGpu == CPU e prova de reuso de peso (sem re-upload)"
```

---

## Task 5: Medir o ganho (benchmark single-GPU resident)

**Files:**
- Modify: `scripts/benchmark-gpu.sh`

- [ ] **Step 1: Adicionar run single-GPU resident**

Em `scripts/benchmark-gpu.sh`, após a função `run_rs()` (linha ~98), adicionar:

```bash
# ── Rust (--gpu-single, 1x MI50 resident) — retorna tok/s de decode ──
run_rs_single() {
    local log=$1
    "$RS_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" \
        --temp 0 --seed "$SEED" --no-display-prompt --timings --gpu-single \
        2>"$log" >/dev/null || true
    assert_no_nvidia "$log" "llama-rs (--gpu-single)"
    grep -oE "[0-9]+\.[0-9]+ tok/s" "$log" | grep -oE "^[0-9]+\.[0-9]+" | head -1
}
```

E na seção de execução, após `rs=$(run_rs /tmp/bench-rs.err)` (linha ~120):

```bash
echo "Rodando llama-rs 1x MI50 (resident)..." >&2
rs1=$(run_rs_single /tmp/bench-rs1.err)
```

E na tabela de resultados, após a linha do `llama-rs  — 2x MI50`:

```bash
printf "| %-28s | %-16s |\n" "llama-rs  — 1x MI50 (resident)" "${rs1:-erro}"
```

- [ ] **Step 2: Rodar o benchmark (no hardware)**

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -30`
Expected: a nova linha "llama-rs — 1x MI50 (resident)" aparece. **Meta da fatia 1A: ≫ 2.16 tok/s** (esperado salto de 1-2 ordens de grandeza só por matar o re-upload + recriação de pipeline). O número exato orienta a Fase 1B.

- [ ] **Step 3: Commit do script + resultado**

```bash
git add scripts/benchmark-gpu.sh bench-results/
git commit -m "feat(bench): mede llama-rs single-MI50 resident (Fase 1A)"
```

---

## Self-Review (preenchido)

**1. Cobertura do spec (§4.1, §4.2 — residência de pesos e pipeline):** Task 1+2 entregam pesos residentes (cache por ponteiro) e pipeline persistente — núcleo de §4.1/§4.2 para single-GPU. §4.3 (forward na GPU), §4.4 (1 command buffer/token) e residência de KV/ativações (§4.1 restante) **são explicitamente fora desta fatia** → Fases 1B/1C/1D (planos próprios). Coberto e delimitado.

**2. Placeholders:** Sem TBD/TODO. Todo passo de código tem o código real. Os dois pontos de "ajustar se os nomes diferirem" (Task 3 Step 2, Task 4 Step 1) são checagens de alinhamento com o arquivo real, com instrução concreta de como alinhar — não são lacunas de design.

**3. Consistência de tipos:** `ResidentGpu::new(&ctx) -> Result<_, ModelError>`, `matvec_q8_0` (assinatura idêntica à trait `GpuMatmul`), `ensure_weight`/`dispatch`/`resident_count`/`phys` coerentes entre tasks. `GpuTensor`/`ComputePipeline`/`PushConstants`/helpers referenciados existem em `tensor.rs`/`pipeline.rs` (verificados na leitura do código).

---

## Próximas fatias (não nesta plan; ver spec §7)

- **1B** — buffers de `x`/`y`/staging persistentes + descriptor sets pré-alocados (elimina o churn por-chamada restante).
- **1C** — shaders RMSNorm/RoPE/attention-GQA/SwiGLU na GPU + KV-cache residente (fim do ping-pong CPU↔GPU).
- **1D** — 1 command buffer por token (pipeline barriers; 1 fence/token; readback só dos logits).
