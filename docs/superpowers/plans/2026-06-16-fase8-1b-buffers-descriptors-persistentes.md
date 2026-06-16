# Fase 8.1B — Buffers x/y/staging + descriptor set persistentes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar o churn por-chamada que sobrou na Fase 1A — alocação/liberação dos buffers `x`/`y`/staging e a criação de descriptor pool/set a cada matvec — tornando-os residentes e reusados em todos os matvecs de todos os tokens.

**Architecture:** Estende o `ResidentGpu` (criado na 1A). Hoje `dispatch` cria e destrói 4 buffers (`x_staging`, `x_buf`, `y_buf`, `y_read`) + 1 descriptor pool + 1 descriptor set **por matvec** (~169×/token). Esta fatia introduz: (1) uma struct `Buffers` residente, com capacidades em bytes que **crescem sob demanda** e nunca encolhem — após o 1º token todas as capacidades atingem o máximo do modelo e estabilizam; (2) um descriptor pool + descriptor set persistentes, criados em `new()` e apenas **re-escritos** (`update_descriptor_sets`) a cada chamada (o binding 0 = peso muda; bindings 1/2 = x/y apontam para os buffers residentes). O `queue_wait_idle` por chamada e o readback **permanecem** (vão para a Fase 1D). Como há `wait_idle` após cada dispatch, é seguro re-escrever o set e (re)alocar buffers entre chamadas: a GPU está ociosa.

**Tech Stack:** Rust, `ash` (Vulkan), RADV/gfx906 (MI50). Reusa `GpuTensor`, `ComputePipeline`, `PushConstants`, `create_buf`/`alloc_and_bind`/`one_shot_copy` e o shader `q8_0_matvec.comp` existentes. Nenhum shader novo.

---

## Contexto (ler antes de começar)

Estado pós-1A (`crates/llama-vulkan/src/resident.rs`): pesos Q8_0 e `ComputePipeline` já são residentes. O que ainda é por-chamada em `dispatch` (resident.rs:69-252):

- `create_buf` + `alloc_and_bind` de `x_staging` (host-visible), `x_buf` (device-local), `y_buf` (device-local), `y_read` (host-visible) — **4 alocações Vulkan/matvec**;
- `create_descriptor_pool` + `allocate_descriptor_sets` — **pool+set novos/matvec**;
- ao final, `destroy_buffer`/`free_memory`/`destroy_descriptor_pool` de tudo.

A pipeline e o peso (binding 0) já vêm de `new()`/cache. Esta fatia move buffers e descriptor set para residentes.

**Por que cresce sob demanda e não pré-aloca pelo config:** `ResidentGpu::new(&ctx)` não recebe o `LlamaConfig`. Em vez de mudar a assinatura (cirurgia maior), os buffers começam vazios e crescem na 1ª vez que um tamanho maior aparece. A sequência de decode é fixa, então após o 1º token não há mais realloc — provado por teste (Task 4).

**Não tocar:** `dual_gpu.rs`, `backend.rs`, `matmul.rs`, `forward_gpu`, a trait `GpuMatmul`, os shaders, a flag `--gpu-single` (o caminho continua o mesmo; só fica mais rápido).

---

## File Structure

- **Modify:** `crates/llama-vulkan/src/resident.rs` — adicionar struct `Buffers` (residente, grow-on-demand), campos persistentes em `ResidentGpu` (`buffers`, `desc_pool`, `desc_set`, contador de grows), reescrever `dispatch` para reusar tudo, e estender `Drop`.
- **Modify:** `crates/llama-vulkan/tests/integration.rs` — teste de que buffers não realocam após warmup (mesmo tamanho 2× → 0 grows extras; tamanho maior → grow).
- **Modify:** `scripts/benchmark-gpu.sh` — nada estrutural; apenas re-rodar e registrar o novo número na mesma linha "1x MI50 (resident)".

---

## Task 1: Struct `Buffers` residente (grow-on-demand)

**Files:**
- Modify: `crates/llama-vulkan/src/resident.rs`

- [ ] **Step 1: Adicionar a struct `Buffers` e seus métodos**

No topo de `resident.rs`, após os `use` existentes, adicionar:

```rust
/// Buffers de ativação residentes, compartilhados por todos os matvecs.
/// Capacidades em bytes crescem sob demanda e nunca encolhem: após o 1º token
/// atingem o máximo do modelo e estabilizam (zero realloc nos tokens seguintes).
///
/// Lado X (entrada): `x_staging` host-visible + `x_dev` device-local STORAGE.
/// Lado Y (saída):   `y_dev` device-local STORAGE|TRANSFER_SRC + `y_read` host-visible.
struct Buffers {
    x_staging: vk::Buffer,
    x_staging_mem: vk::DeviceMemory,
    x_dev: vk::Buffer,
    x_dev_mem: vk::DeviceMemory,
    x_cap: vk::DeviceSize,
    y_dev: vk::Buffer,
    y_dev_mem: vk::DeviceMemory,
    y_read: vk::Buffer,
    y_read_mem: vk::DeviceMemory,
    y_cap: vk::DeviceSize,
    /// Nº de vezes que um lado (X ou Y) cresceu. Para teste/diagnóstico.
    grows: usize,
}

impl Buffers {
    /// Começa vazio (handles nulos). A 1ª chamada de `ensure` aloca.
    fn empty() -> Self {
        Self {
            x_staging: vk::Buffer::null(),
            x_staging_mem: vk::DeviceMemory::null(),
            x_dev: vk::Buffer::null(),
            x_dev_mem: vk::DeviceMemory::null(),
            x_cap: 0,
            y_dev: vk::Buffer::null(),
            y_dev_mem: vk::DeviceMemory::null(),
            y_read: vk::Buffer::null(),
            y_read_mem: vk::DeviceMemory::null(),
            y_cap: 0,
            grows: 0,
        }
    }

    /// Garante capacidade >= `x_size` no lado X e `y_size` no lado Y.
    /// (Re)aloca apenas o lado insuficiente. Seguro porque o caller fez
    /// `queue_wait_idle` antes (GPU ociosa, nenhum buffer em uso).
    fn ensure(
        &mut self,
        ctx: &VulkanContext,
        phys: &VulkanPhysicalDevice,
        d: &ash::Device,
        x_size: vk::DeviceSize,
        y_size: vk::DeviceSize,
    ) -> Result<(), MatmulError> {
        use crate::tensor::{alloc_and_bind, create_buf};

        if x_size > self.x_cap {
            // SAFETY: handles ou são nulos (no-op) ou foram criados por nós e não
            // estão em uso (wait_idle precedeu esta chamada).
            unsafe {
                d.destroy_buffer(self.x_staging, None);
                d.free_memory(self.x_staging_mem, None);
                d.destroy_buffer(self.x_dev, None);
                d.free_memory(self.x_dev_mem, None);
            }
            self.x_staging = create_buf(d, x_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.x_staging_mem = alloc_and_bind(ctx, phys, d, self.x_staging, true)?;
            self.x_dev = create_buf(
                d,
                x_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.x_dev_mem = alloc_and_bind(ctx, phys, d, self.x_dev, false)?;
            self.x_cap = x_size;
            self.grows += 1;
        }

        if y_size > self.y_cap {
            // SAFETY: idem lado X.
            unsafe {
                d.destroy_buffer(self.y_dev, None);
                d.free_memory(self.y_dev_mem, None);
                d.destroy_buffer(self.y_read, None);
                d.free_memory(self.y_read_mem, None);
            }
            self.y_dev = create_buf(
                d,
                y_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            self.y_dev_mem = alloc_and_bind(ctx, phys, d, self.y_dev, false)?;
            self.y_read = create_buf(d, y_size, vk::BufferUsageFlags::TRANSFER_DST)?;
            self.y_read_mem = alloc_and_bind(ctx, phys, d, self.y_read, true)?;
            self.y_cap = y_size;
            self.grows += 1;
        }
        Ok(())
    }

    /// Libera todos os handles. Chamar uma vez no Drop do `ResidentGpu`.
    fn destroy(&mut self, d: &ash::Device) {
        // SAFETY: handles criados por nós; chamado no Drop, sem uso concorrente.
        unsafe {
            d.destroy_buffer(self.x_staging, None);
            d.free_memory(self.x_staging_mem, None);
            d.destroy_buffer(self.x_dev, None);
            d.free_memory(self.x_dev_mem, None);
            d.destroy_buffer(self.y_dev, None);
            d.free_memory(self.y_dev_mem, None);
            d.destroy_buffer(self.y_read, None);
            d.free_memory(self.y_read_mem, None);
        }
    }
}
```

> Nota: `vk::Buffer::null()` / `vk::DeviceMemory::null()` são handles inválidos; `destroy_buffer(null)`/`free_memory(null)` são no-ops válidos na spec Vulkan, então o 1º `ensure` (que destrói antes de criar) é seguro mesmo com handles nulos.

- [ ] **Step 2: Compilar (struct ainda não usada — warning esperado)**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila; warning de `Buffers`/campos não usados (ainda não ligados à `ResidentGpu`). Sem erros.

- [ ] **Step 3: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs
git commit -m "feat(vulkan): struct Buffers residente grow-on-demand (Fase 1B)"
```

---

## Task 2: Campos persistentes em `ResidentGpu` (buffers + descriptor pool/set)

**Files:**
- Modify: `crates/llama-vulkan/src/resident.rs`

- [ ] **Step 1: Adicionar campos à struct `ResidentGpu`**

Localizar a definição (resident.rs:19-26) e substituí-la por:

```rust
/// Backend de matmul Q8_0 numa única GPU AMD, com pesos+pipeline+buffers+descriptor residentes.
pub struct ResidentGpu<'ctx> {
    ctx: &'ctx VulkanContext,
    phys_idx: usize,
    dev: VulkanDevice,
    pipeline: ComputePipeline,
    /// key = `w_bytes.as_ptr() as usize`; value = peso já residente na VRAM.
    weights: RefCell<HashMap<usize, GpuTensor>>,
    /// Buffers x/y/staging residentes (crescem sob demanda).
    buffers: RefCell<Buffers>,
    /// Descriptor pool/set persistentes (re-escritos a cada dispatch).
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
}
```

- [ ] **Step 2: Criar pool+set em `new()`**

Localizar `new()` (resident.rs:30-46). Substituir o bloco do `Ok(Self { ... })` por uma versão que cria o descriptor pool/set persistentes antes de construir a struct. Substituir de `let pipeline = ComputePipeline::new(...)?;` até o fim do `Ok(Self {...})` por:

```rust
        let pipeline = ComputePipeline::new(&dev.device)
            .map_err(|e| ModelError::Gpu(format!("pipeline: {e}")))?;

        // Descriptor pool/set persistentes: 1 set com 3 bindings STORAGE_BUFFER.
        let d = &dev.device;
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
        let desc_pool = unsafe {
            d.create_descriptor_pool(&pool_info, None)
                .map_err(|e| ModelError::Gpu(format!("desc pool: {e}")))?
        };
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: desc_pool,
            descriptor_set_count: 1,
            p_set_layouts: &pipeline.desc_set_layout,
            ..Default::default()
        };
        // SAFETY: d e desc_pool válidos; layout vem da pipeline recém-criada.
        let desc_set = unsafe {
            d.allocate_descriptor_sets(&alloc_info)
                .map_err(|e| ModelError::Gpu(format!("desc set: {e}")))?[0]
        };

        Ok(Self {
            ctx,
            phys_idx: 0,
            dev,
            pipeline,
            weights: RefCell::new(HashMap::new()),
            buffers: RefCell::new(Buffers::empty()),
            desc_pool,
            desc_set,
        })
```

- [ ] **Step 3: Expor contador de grows para teste**

Adicionar dentro de `impl<'ctx> ResidentGpu<'ctx>`, junto a `resident_count`:

```rust
    /// Nº de (re)alocações de buffer já feitas. Para testes/diagnóstico.
    pub fn buffer_grows(&self) -> usize {
        self.buffers.borrow().grows
    }
```

- [ ] **Step 4: Estender `Drop` para liberar buffers + pool**

Localizar `impl Drop for ResidentGpu<'_>` (resident.rs:255-269) e substituir o corpo por:

```rust
impl Drop for ResidentGpu<'_> {
    fn drop(&mut self) {
        let d = &self.dev.device;
        // SAFETY: wait_idle garante que nenhum recurso está em uso pela GPU.
        unsafe {
            let _ = d.device_wait_idle();
        }
        self.buffers.borrow_mut().destroy(d);
        for (_, t) in self.weights.borrow_mut().drain() {
            t.destroy(d);
        }
        // SAFETY: desc_pool/pipeline criados por nós; ordem inversa de criação.
        unsafe {
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_pipeline(self.pipeline.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline.layout, None);
            d.destroy_descriptor_set_layout(self.pipeline.desc_set_layout, None);
        }
    }
}
```

- [ ] **Step 5: Compilar (dispatch ainda usa o caminho antigo — ok)**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila. `dispatch` ainda cria buffers/pool locais (Task 3 reescreve). Pode haver warning de `desc_pool`/`desc_set`/`buffers` "campo não lido" — some na Task 3.

- [ ] **Step 6: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs
git commit -m "feat(vulkan): ResidentGpu com buffers+descriptor pool/set persistentes (Fase 1B)"
```

---

## Task 3: Reescrever `dispatch` para reusar buffers + descriptor set residentes

**Files:**
- Modify: `crates/llama-vulkan/src/resident.rs`

- [ ] **Step 1: Substituir o corpo inteiro de `dispatch`**

Localizar `fn dispatch(...)` (resident.rs:69-252) e substituir a função inteira (assinatura idêntica) por:

```rust
    fn dispatch(
        &self,
        weight_key: usize,
        x_f32: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, MatmulError> {
        use crate::pipeline::PushConstants;
        use crate::tensor::one_shot_copy;

        let d = &self.dev.device;
        let dev = &self.dev;

        let x_size = std::mem::size_of_val(x_f32) as vk::DeviceSize;
        let y_size = (n_out * std::mem::size_of::<f32>()) as vk::DeviceSize;

        // 1. Garantir capacidade dos buffers residentes (grow-on-demand).
        let mut buffers = self.buffers.borrow_mut();
        buffers.ensure(self.ctx, self.phys(), d, x_size, y_size)?;

        // 2. Carregar X no staging host-visible e copiar para o device-local residente.
        unsafe {
            // SAFETY: x_staging_mem host-visible com x_cap >= x_size; ptr válido até unmap.
            let ptr = d.map_memory(buffers.x_staging_mem, 0, x_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(x_f32.as_ptr(), ptr as *mut f32, x_f32.len());
            d.unmap_memory(buffers.x_staging_mem);
        }
        one_shot_copy(d, dev.queue, dev.cmd_pool, buffers.x_staging, buffers.x_dev, x_size)?;

        // 3. Re-escrever o descriptor set persistente (binding 0 = peso muda; 1/2 residentes).
        let weights = self.weights.borrow();
        let w_tensor = weights
            .get(&weight_key)
            .expect("peso garantido por ensure_weight");
        let buf_infos = [
            vk::DescriptorBufferInfo {
                buffer: w_tensor.buffer,
                offset: 0,
                range: w_tensor.size_bytes,
            },
            vk::DescriptorBufferInfo {
                buffer: buffers.x_dev,
                offset: 0,
                range: x_size,
            },
            vk::DescriptorBufferInfo {
                buffer: buffers.y_dev,
                offset: 0,
                range: y_size,
            },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(binding, bi)| vk::WriteDescriptorSet {
                dst_set: self.desc_set,
                dst_binding: binding as u32,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: bi,
                ..Default::default()
            })
            .collect();
        // SAFETY: d válido; writes apontam para buf_infos vivos na stack; GPU ociosa (wait_idle anterior).
        unsafe { d.update_descriptor_sets(&writes, &[]) };

        // 4. Command buffer (ainda por-chamada nesta fatia; fusão vira 1 cmd/token na Fase 1D).
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
        let push = PushConstants {
            n_in: n_in as u32,
            n_out: n_out as u32,
            row_offset: 0,
        };
        unsafe {
            // SAFETY: cmd recém-alocado; desc_set/pipeline válidos.
            d.begin_command_buffer(cmd, &begin)?;
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.desc_set],
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
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

        // 5. Readback de Y do device-local residente para o host-visible residente.
        one_shot_copy(d, dev.queue, dev.cmd_pool, buffers.y_dev, buffers.y_read, y_size)?;
        let out = unsafe {
            // SAFETY: y_read_mem host-visible com y_cap >= y_size; ptr válido até unmap.
            let ptr = d.map_memory(buffers.y_read_mem, 0, y_size, vk::MemoryMapFlags::empty())?;
            let mut v = vec![0f32; n_out];
            std::ptr::copy_nonoverlapping(ptr as *const f32, v.as_mut_ptr(), n_out);
            d.unmap_memory(buffers.y_read_mem);
            v
        };
        Ok(out)
    }
```

> Diferença-chave vs 1A: **nenhum** `create_buf`/`alloc_and_bind`/`create_descriptor_pool`/`destroy_*` de buffers ou pool dentro de `dispatch`. Só `map`/`copy`/`update_descriptor_sets`/cmd/submit/readback. Os buffers e o set são residentes.

- [ ] **Step 2: Compilar**

Run: `cargo build -p llama-vulkan 2>&1 | tail -20`
Expected: compila sem erros e **sem** warnings de campo não usado (`buffers`/`desc_pool`/`desc_set` agora são lidos).

- [ ] **Step 3: Regressão numérica — o teste da 1A deve continuar PASS**

Run: `cargo test -p llama-vulkan --test integration resident_gpu_logits_iguais_a_cpu_qwen -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando" sem hardware). Se FALHAR por valor, há bug na re-escrita do descriptor/range — depurar antes de seguir (provável causa: `range` errado no `DescriptorBufferInfo` ou capacidade não garantida).

- [ ] **Step 4: Commit**

```bash
git add crates/llama-vulkan/src/resident.rs
git commit -m "perf(vulkan): dispatch reusa buffers+descriptor set residentes (Fase 1B)"
```

---

## Task 4: Teste — buffers não realocam após warmup

**Files:**
- Modify: `crates/llama-vulkan/tests/integration.rs`

- [ ] **Step 1: Escrever o teste de estabilidade do cache de buffers**

Adicionar ao fim de `crates/llama-vulkan/tests/integration.rs`:

```rust
#[test]
fn resident_gpu_buffers_estabilizam_apos_warmup() {
    use llama_vulkan::{ResidentGpu, VulkanContext};
    let Ok(ctx) = VulkanContext::new() else {
        eprintln!("sem Vulkan — pulando");
        return;
    };
    if ctx.amd_compute_devices().is_empty() {
        eprintln!("sem AMD — pulando");
        return;
    }
    let backend = ResidentGpu::new(&ctx).unwrap();

    // Peso Q8_0 sintético 1 linha × 32 col: 34 bytes (2 scale + 32 quants).
    let w = vec![0u8; 34];
    let x = vec![0f32; 32];

    // 1ª chamada: cresce lado X (1) e lado Y (1) = 2 grows.
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.buffer_grows(), 2, "1ª chamada aloca X e Y");

    // 2ª chamada idêntica: cache-hit total, nenhum grow novo.
    let _ = backend.matvec_q8_0(&w, &x, 32, 1).unwrap();
    assert_eq!(backend.buffer_grows(), 2, "mesmo tamanho => sem realloc");

    // Saída maior (n_out=2): só o lado Y cresce (+1).
    let w2 = vec![0u8; 2 * 34];
    let _ = backend.matvec_q8_0(&w2, &x, 32, 2).unwrap();
    assert_eq!(backend.buffer_grows(), 3, "n_out maior => 1 grow só no lado Y");
}
```

- [ ] **Step 2: Rodar o teste**

Run: `cargo test -p llama-vulkan --test integration resident_gpu_buffers_estabilizam_apos_warmup -- --nocapture 2>&1 | tail -20`
Expected: PASS (ou "pulando"). Falha em "mesmo tamanho => sem realloc" indica que `ensure` está crescendo sem necessidade (checar comparações `>` vs `>=`).

- [ ] **Step 3: Commit**

```bash
git add crates/llama-vulkan/tests/integration.rs
git commit -m "test(vulkan): buffers residentes estabilizam após warmup (Fase 1B)"
```

---

## Task 5: Medir o ganho (benchmark single-GPU resident, pós-1B)

**Files:**
- Modify: `scripts/benchmark-gpu.sh` (somente se necessário) + `bench-results/`

- [ ] **Step 1: Rodar o benchmark no hardware**

O caminho `--gpu-single` já existe (Fase 1A) e agora usa o `dispatch` otimizado. Não há mudança estrutural no script.

Run: `./scripts/benchmark-gpu.sh 2>&1 | tail -30`
Expected: a linha "llama-rs — 1x MI50 (resident)" reporta tok/s **maior** que o número da 1A (registrado no último arquivo de `bench-results/`). A 1A matou re-upload+pipeline; a 1B mata o churn de buffers/descriptors. Ganho esperado: incremental sobre 1A (a maior parte do tempo agora é `queue_wait_idle`/readback por matvec — alvo da Fase 1D).

- [ ] **Step 2: Comparar com o número da 1A**

Run: `ls -t bench-results/ | head -3`
Abrir o arquivo mais recente e o anterior; confirmar que o número da 1B ≥ 1A. Anotar ambos no corpo do commit.

- [ ] **Step 3: Commit do resultado**

```bash
git add bench-results/
git commit -m "bench(gpu): 1x MI50 resident pós-1B (buffers+descriptors persistentes)"
```

---

## Self-Review

**1. Cobertura do spec (§4.2 — pipelines/descriptors persistentes):** Task 2 cria descriptor pool/set persistentes; Task 3 os reusa via `update_descriptor_sets` (fim do churn de pool por dispatch). §4.1 (residência de buffers de ativação) é coberto pela struct `Buffers` (Tasks 1-3). KV-cache residente e forward-na-GPU (§4.1 restante, §4.3) são explicitamente da Fase 1C; 1 command buffer/token (§4.4) é da 1D — fora desta fatia, delimitado nos comentários do código (`dispatch` ainda faz 1 submit + wait_idle por matvec).

**2. Placeholders:** Sem TBD/TODO. Todo passo de código tem o código real, incluindo o corpo completo reescrito de `dispatch` e a struct `Buffers`.

**3. Consistência de tipos:** `Buffers::{empty, ensure, destroy}` e os campos (`x_staging`, `x_dev`, `y_dev`, `y_read`, `x_cap`, `y_cap`, `grows`) são usados de forma idêntica em Tasks 1-4. `ResidentGpu` ganha `buffers: RefCell<Buffers>`, `desc_pool`, `desc_set` e o método `buffer_grows()`, todos referenciados de forma coerente. `create_buf`/`alloc_and_bind`/`one_shot_copy` mantêm as assinaturas verificadas em `tensor.rs`. A trait `GpuMatmul` e `matvec_q8_0` não mudam — `--gpu-single` continua válido.

---

## Próxima fatia

- **1C** — shaders RMSNorm/RoPE/attention-GQA/SwiGLU/residual na GPU + KV-cache residente: fim do ping-pong CPU↔GPU (só os logits voltam ao host). Ver `2026-06-16-fase8-1c-forward-gpu-resident.md`.
