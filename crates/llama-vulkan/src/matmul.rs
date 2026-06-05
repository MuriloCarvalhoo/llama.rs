use crate::{
    device::{VulkanContext, VulkanDevice, VulkanPhysicalDevice},
    pipeline::{ComputePipeline, PipelineError, PushConstants},
    tensor::{GpuTensor, TensorError, alloc_and_bind, create_buf, one_shot_copy},
};
use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MatmulError {
    #[error("Tensor: {0}")]
    Tensor(#[from] TensorError),
    #[error("Pipeline: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("Vulkan: {0}")]
    Vulkan(#[from] vk::Result),
}

/// Executa Q8_0 matvec em GPU single-device.
///
/// `w_bytes`: pesos Q8_0 row-major, n_out × (n_in/32 × 34) bytes.
/// `x_f32`: ativações, n_in floats.
/// Retorna Vec<f32> de tamanho n_out.
pub fn dispatch_q8_0_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
) -> Result<Vec<f32>, MatmulError> {
    dispatch_inner(ctx, phys, dev, w_bytes, x_f32, n_in, n_out, 0, n_out)
}

/// Versão interna com row_offset para suporte a row-split (multi-GPU).
pub(crate) fn dispatch_inner(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    _n_out_total: usize,
    row_offset: usize,
    n_out_local: usize,
) -> Result<Vec<f32>, MatmulError> {
    let d = &dev.device;

    // 1. Upload W (n_out_local linhas de pesos) via GpuTensor::upload_q8_0
    let w_tensor = GpuTensor::upload_q8_0(ctx, phys, dev, w_bytes, n_in, n_out_local)?;

    // 2. Upload X (activations f32) via staging → STORAGE_BUFFER
    let x_size = (x_f32.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;

    let x_staging = create_buf(d, x_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let x_staging_mem = alloc_and_bind(ctx, phys, d, x_staging, true)?;
    unsafe {
        // SAFETY: x_staging_mem é host-visible com tamanho x_size; ptr é válido até unmap.
        let ptr = d.map_memory(x_staging_mem, 0, x_size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(x_f32.as_ptr(), ptr as *mut f32, x_f32.len());
        d.unmap_memory(x_staging_mem);
    }

    let x_buf = create_buf(
        d,
        x_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let x_mem = alloc_and_bind(ctx, phys, d, x_buf, false)?;
    one_shot_copy(d, dev.queue, dev.cmd_pool, x_staging, x_buf, x_size)?;
    unsafe {
        // SAFETY: staging já foi copiado; staging_buf e staging_mem foram criados por nós.
        d.destroy_buffer(x_staging, None);
        d.free_memory(x_staging_mem, None);
    }

    // 3. Criar buffer Y (output) — TRANSFER_SRC para readback posterior
    let y_size = (n_out_local * std::mem::size_of::<f32>()) as vk::DeviceSize;
    let y_buf = create_buf(
        d,
        y_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let y_mem = alloc_and_bind(ctx, phys, d, y_buf, false)?;

    // 4. Criar ComputePipeline, descriptor pool, descriptor set, escrever descriptors
    let pipe = ComputePipeline::new(d)?;

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
    // SAFETY: d é válido; pool_info aponta para dados válidos na stack frame.
    let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };

    let alloc_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool: desc_pool,
        descriptor_set_count: 1,
        p_set_layouts: &pipe.desc_set_layout,
        ..Default::default()
    };
    // SAFETY: d e desc_pool são válidos; alloc_info aponta para dados válidos.
    let desc_sets = unsafe { d.allocate_descriptor_sets(&alloc_info)? };
    let desc_set = desc_sets[0];

    // Escrever descriptors: binding 0=weights, 1=activations, 2=output
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: w_tensor.buffer,
            offset: 0,
            range: w_tensor.size_bytes,
        },
        vk::DescriptorBufferInfo {
            buffer: x_buf,
            offset: 0,
            range: x_size,
        },
        vk::DescriptorBufferInfo {
            buffer: y_buf,
            offset: 0,
            range: y_size,
        },
    ];

    let writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(binding, buf_info)| vk::WriteDescriptorSet {
            dst_set: desc_set,
            dst_binding: binding as u32,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: buf_info,
            ..Default::default()
        })
        .collect();

    // SAFETY: d é válido; writes aponta para dados válidos na stack frame.
    unsafe { d.update_descriptor_sets(&writes, &[]) };

    // 5. Command buffer: bind pipeline → bind descriptor set → push constants → cmd_dispatch
    let cb_alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: dev.cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    // SAFETY: d e cmd_pool são válidos.
    let cmd_bufs = unsafe { d.allocate_command_buffers(&cb_alloc_info)? };
    let cmd = cmd_bufs[0];

    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    let push = PushConstants {
        n_in: n_in as u32,
        n_out: n_out_local as u32,
        row_offset: row_offset as u32,
    };

    unsafe {
        // SAFETY: cmd é válido e recém alocado.
        d.begin_command_buffer(cmd, &begin_info)?;

        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        d.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipe.layout,
            0,
            &[desc_set],
            &[],
        );
        d.cmd_push_constants(
            cmd,
            pipe.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            // SAFETY: PushConstants é #[repr(C)] com 3×u32; o slice de bytes é válido.
            std::slice::from_raw_parts(
                &push as *const PushConstants as *const u8,
                std::mem::size_of::<PushConstants>(),
            ),
        );
        // Cada workgroup computa 1 linha de output
        d.cmd_dispatch(cmd, n_out_local as u32, 1, 1);

        d.end_command_buffer(cmd)?;
    }

    // 6. Submit → wait_idle
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        ..Default::default()
    };
    unsafe {
        // SAFETY: queue, submit_info e cmd são válidos.
        d.queue_submit(dev.queue, &[submit_info], vk::Fence::null())?;
        d.queue_wait_idle(dev.queue)?;
        d.free_command_buffers(dev.cmd_pool, &[cmd]);
    }

    // Readback Y via buffer staging TRANSFER_DST | HOST_VISIBLE | HOST_COHERENT
    let y_read_buf = create_buf(d, y_size, vk::BufferUsageFlags::TRANSFER_DST)?;
    let y_read_mem = alloc_and_bind(ctx, phys, d, y_read_buf, true)?;
    one_shot_copy(d, dev.queue, dev.cmd_pool, y_buf, y_read_buf, y_size)?;

    let result = unsafe {
        // SAFETY: y_read_mem é host-visible com tamanho y_size; ptr é válido até unmap.
        let ptr = d.map_memory(y_read_mem, 0, y_size, vk::MemoryMapFlags::empty())?;
        let mut out = vec![0f32; n_out_local];
        std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), n_out_local);
        d.unmap_memory(y_read_mem);
        out
    };

    // 7. Cleanup de todos os recursos Vulkan
    unsafe {
        d.destroy_buffer(y_read_buf, None);
        d.free_memory(y_read_mem, None);
    }
    pipe.destroy(d);
    unsafe {
        // SAFETY: desc_pool foi criado por nós; os descriptor sets são destruídos com o pool.
        d.destroy_descriptor_pool(desc_pool, None);
        d.destroy_buffer(y_buf, None);
        d.free_memory(y_mem, None);
        d.destroy_buffer(x_buf, None);
        d.free_memory(x_mem, None);
    }
    w_tensor.destroy(d);

    Ok(result)
}
