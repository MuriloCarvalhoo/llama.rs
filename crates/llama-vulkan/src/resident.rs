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
            // Zera os handles antes de qualquer alocação falível: se um `?` abaixo
            // disparar, um `ensure`/`destroy` posterior vê nulos (no-op), não double-free.
            self.x_staging = vk::Buffer::null();
            self.x_staging_mem = vk::DeviceMemory::null();
            self.x_dev = vk::Buffer::null();
            self.x_dev_mem = vk::DeviceMemory::null();
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
            // Zera os handles antes da alocação falível (ver lado X).
            self.y_dev = vk::Buffer::null();
            self.y_dev_mem = vk::DeviceMemory::null();
            self.y_read = vk::Buffer::null();
            self.y_read_mem = vk::DeviceMemory::null();
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
    }

    fn phys(&self) -> &VulkanPhysicalDevice {
        &self.ctx.amd_compute_devices()[self.phys_idx]
    }

    /// Nº de pesos residentes (uploads efetuados). Para testes/diagnóstico.
    pub fn resident_count(&self) -> usize {
        self.weights.borrow().len()
    }

    /// Nº de (re)alocações de buffer já feitas. Para testes/diagnóstico.
    pub fn buffer_grows(&self) -> usize {
        self.buffers.borrow().grows
    }

    /// Garante que o peso identificado por `w_bytes.as_ptr()` está residente.
    /// Faz upload na primeira vez; chamadas seguintes são cache-hit.
    fn ensure_weight(&self, w_bytes: &[u8], n_in: usize, n_out: usize) -> Result<(), MatmulError> {
        let key = w_bytes.as_ptr() as usize;
        if self.weights.borrow().contains_key(&key) {
            return Ok(());
        }
        let t = GpuTensor::upload_q8_0(self.ctx, self.phys(), &self.dev, w_bytes, n_in, n_out)?;
        self.weights.borrow_mut().insert(key, t);
        Ok(())
    }

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
            let ptr = d.map_memory(
                buffers.x_staging_mem,
                0,
                x_size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(x_f32.as_ptr(), ptr as *mut f32, x_f32.len());
            d.unmap_memory(buffers.x_staging_mem);
        }
        one_shot_copy(
            d,
            dev.queue,
            dev.cmd_pool,
            buffers.x_staging,
            buffers.x_dev,
            x_size,
        )?;

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
        // SAFETY: d válido; writes apontam para buf_infos vivos na stack. O desc_set não
        // está em uso: na 1ª chamada nenhum cmd o referenciou; nas demais o queue_wait_idle
        // do dispatch anterior garantiu GPU ociosa.
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
        one_shot_copy(
            d,
            dev.queue,
            dev.cmd_pool,
            buffers.y_dev,
            buffers.y_read,
            y_size,
        )?;
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
}

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

impl GpuMatmul for ResidentGpu<'_> {
    fn matvec_q8_0(
        &self,
        w_bytes: &[u8],
        x: &[f32],
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, ModelError> {
        self.ensure_weight(w_bytes, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))?;
        self.dispatch(w_bytes.as_ptr() as usize, x, n_in, n_out)
            .map_err(|e| ModelError::Gpu(e.to_string()))
    }
}
