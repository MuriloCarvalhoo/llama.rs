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

/// Argumentos para `dispatch_inner`, agrupados para evitar `too_many_arguments`.
pub(crate) struct DispatchArgs<'a> {
    pub ctx: &'a VulkanContext,
    pub phys: &'a VulkanPhysicalDevice,
    pub dev: &'a VulkanDevice,
    pub w_bytes: &'a [u8],
    pub x_f32: &'a [f32],
    pub n_in: usize,
    pub row_offset: usize,
    pub n_out_local: usize,
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
    dispatch_inner(DispatchArgs {
        ctx,
        phys,
        dev,
        w_bytes,
        x_f32,
        n_in,
        row_offset: 0,
        n_out_local: n_out,
    })
}

/// Versão interna com row_offset para suporte a row-split (multi-GPU).
pub(crate) fn dispatch_inner(args: DispatchArgs<'_>) -> Result<Vec<f32>, MatmulError> {
    let DispatchArgs {
        ctx,
        phys,
        dev,
        w_bytes,
        x_f32,
        n_in,
        row_offset,
        n_out_local,
    } = args;
    let d = &dev.device;

    // 1. Upload W (n_out_local linhas de pesos) via GpuTensor::upload_q8_0
    let w_tensor = GpuTensor::upload_q8_0(ctx, phys, dev, w_bytes, n_in, n_out_local)?;

    // 2. Quantizar X no host e subir (xq, xd) — o shader consome a ativacao ja em int8.
    let (xq, xd) = crate::tensor::quantize_x_host(x_f32);

    // Sobe um Vec<T> para um STORAGE_BUFFER device-local via staging descartavel.
    let upload = |bytes_of: &[u8]| -> Result<(vk::Buffer, vk::DeviceMemory), MatmulError> {
        let size = bytes_of.len() as vk::DeviceSize;
        let staging = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_and_bind(ctx, phys, d, staging, true)?;
        unsafe {
            // SAFETY: staging_mem é host-visible com `size`; ptr válido até unmap.
            let ptr = d.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes_of.as_ptr(), ptr.cast::<u8>(), bytes_of.len());
            d.unmap_memory(staging_mem);
        }
        let buf = create_buf(
            d,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let mem = alloc_and_bind(ctx, phys, d, buf, false)?;
        one_shot_copy(d, dev.queue, dev.cmd_pool, staging, buf, size)?;
        unsafe {
            // SAFETY: staging já copiado; ambos criados por nós.
            d.destroy_buffer(staging, None);
            d.free_memory(staging_mem, None);
        }
        Ok((buf, mem))
    };

    // SAFETY: Vec<u32>/Vec<f32> são POD contíguos; reinterpretar como bytes é válido.
    let xq_bytes = unsafe {
        std::slice::from_raw_parts(xq.as_ptr().cast::<u8>(), std::mem::size_of_val(&xq[..]))
    };
    let xd_bytes = unsafe {
        std::slice::from_raw_parts(xd.as_ptr().cast::<u8>(), std::mem::size_of_val(&xd[..]))
    };
    let (xq_buf, xq_mem) = upload(xq_bytes)?;
    let (xd_buf, xd_mem) = upload(xd_bytes)?;
    let xq_size = xq_bytes.len() as vk::DeviceSize;
    let xd_size = xd_bytes.len() as vk::DeviceSize;

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
        descriptor_count: 5,
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

    // Escrever descriptors: 0=weights, 1=x quantizado, 2=escalas de x, 3=output, 4=bias.
    // Este caminho não soma bias: liga `xd_buf` no 4 só para o layout ficar completo, e
    // `tem_bias: 0` faz o shader ignorá-lo.
    let buf_infos = [
        vk::DescriptorBufferInfo {
            buffer: w_tensor.buffer,
            offset: 0,
            range: w_tensor.size_bytes,
        },
        vk::DescriptorBufferInfo {
            buffer: xq_buf,
            offset: 0,
            range: xq_size,
        },
        vk::DescriptorBufferInfo {
            buffer: xd_buf,
            offset: 0,
            range: xd_size,
        },
        vk::DescriptorBufferInfo {
            buffer: y_buf,
            offset: 0,
            range: y_size,
        },
        vk::DescriptorBufferInfo {
            buffer: xd_buf,
            offset: 0,
            range: xd_size,
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
        tem_bias: 0,
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
        d.destroy_buffer(xq_buf, None);
        d.free_memory(xq_mem, None);
        d.destroy_buffer(xd_buf, None);
        d.free_memory(xd_mem, None);
    }
    w_tensor.destroy(d);

    Ok(result)
}

/// Matvec Q5_K numa GPU: `y = W * x`, com W em superblocos Q5_K crus (176 B / 256 elementos).
///
/// Diferente do caminho Q8_0, os pesos sobem **sem repack**: o superbloco Q5_K já tem 176
/// bytes = 44 uints, então as leituras de 32 bits ficam alinhadas naturalmente. E a ativação
/// vai em f32 — as escalas por sub-bloco do K-quant tornariam o dot empacotado em int8 bem
/// mais complicado, e o ganho dele foi medido em ~0% neste kernel.
///
/// `cols`: ver `dispatch_q4_k_matvec`.
// Contexto Vulkan (3) + pesos/ativacoes (2) + dimensoes (3): agrupa-los numa struct
// so daria um wrapper de uso unico. Mesmo criterio do `plano_delta`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q5_k_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> Result<Vec<f32>, MatmulError> {
    // Q5_K sobe cru: 176 bytes por superbloco já são 44 uints alinhados.
    let (wg, rows) = crate::resident_forward::matvec_geom();
    let cols_u32 =
        u32::try_from(cols).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))?;
    dispatch_k_matvec(
        ctx,
        phys,
        dev,
        crate::Q5_K_MATVEC_SPV,
        w_bytes,
        x_f32,
        n_in,
        n_out,
        cols,
        wg / 64 * rows,
        &[(0, wg), (1, rows), (2, cols_u32)],
    )
}

/// Matvec Q4_K numa GPU: mesma estrutura do Q5_K (`dispatch_q5_k_matvec`), sem o 5º bit —
/// superbloco de 144 B = 36 uints, já alinhado, sobe cru.
///
/// `cols` é quantas ativações de `n_in` estão concatenadas em `x_f32`: 1 reproduz o matvec
/// do decode, N processa N tokens contra uma única leitura de cada peso (`docs/prefill-em-batch.md`).
/// A saída sai coluna a coluna: `y[c * n_out + linha]`.
// Contexto Vulkan (3) + pesos/ativacoes (2) + dimensoes (3): agrupa-los numa struct
// so daria um wrapper de uso unico. Mesmo criterio do `plano_delta`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q4_k_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> Result<Vec<f32>, MatmulError> {
    let (wg, rows) = crate::resident_forward::matvec_geom();
    let cols_u32 =
        u32::try_from(cols).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))?;
    dispatch_k_matvec(
        ctx,
        phys,
        dev,
        crate::Q4_K_MATVEC_SPV,
        w_bytes,
        x_f32,
        n_in,
        n_out,
        cols,
        wg / 64 * rows,
        &[(0, wg), (1, rows), (2, cols_u32)],
    )
}

/// Matvec Q6_K. O superbloco tem 210 bytes, que não é múltiplo de 4 — cada um é alinhado
/// em 212 bytes no upload para as leituras de 32 bits do shader ficarem alinhadas, o mesmo
/// motivo do repack 34 → 36 do Q8_0.
///
/// `cols`: ver `dispatch_q4_k_matvec`. Aqui a geometria é fixa no shader, então `COLS` é o
/// `constant_id` 0 — e não o 2, como nos outros dois K-quant.
// Contexto Vulkan (3) + pesos/ativacoes (2) + dimensoes (3): agrupa-los numa struct
// so daria um wrapper de uso unico. Mesmo criterio do `plano_delta`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_q6_k_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
    cols: usize,
) -> Result<Vec<f32>, MatmulError> {
    let n_sb = w_bytes.len() / 210;
    let mut padded = vec![0u8; n_sb * 212];
    for i in 0..n_sb {
        padded[i * 212..i * 212 + 210].copy_from_slice(&w_bytes[i * 210..(i + 1) * 210]);
    }
    let cols_u32 =
        u32::try_from(cols).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))?;
    dispatch_k_matvec(
        ctx,
        phys,
        dev,
        crate::Q6_K_MATVEC_SPV,
        &padded,
        x_f32,
        n_in,
        n_out,
        cols,
        8, // 4 waves x 2 linhas por wave, fixas no shader
        &[(0, cols_u32)],
    )
}

/// Caminho comum dos matvecs K-quant: sobe pesos e ativação, despacha
/// `n_out / rows_por_wg` workgroups e lê o resultado. `w_bytes` já vem no layout que o
/// shader espera.
#[allow(clippy::too_many_arguments)]
fn dispatch_k_matvec(
    ctx: &VulkanContext,
    phys: &VulkanPhysicalDevice,
    dev: &VulkanDevice,
    spv: &[u8],
    w_bytes: &[u8],
    x_f32: &[f32],
    n_in: usize,
    n_out: usize,
    // `cols`: quantas ativações de `n_in` vêm concatenadas em `x_f32`. 1 no decode; N no
    // prefill em batch, onde a saída sai como `cols` blocos de `n_out` (coluna a coluna).
    cols: usize,
    rows_por_wg: u32,
    // `spec`: specialization constants da geometria. Têm de ser as mesmas com que a pipeline
    // residente é criada — o shader não tem default utilizável para `local_size_x_id`.
    spec: &[(u32, u32)],
) -> Result<Vec<f32>, MatmulError> {
    use crate::pipeline::{ComputePipeline, PushConstants};
    use crate::tensor::{alloc_and_bind, create_buf, one_shot_copy};

    if !n_in.is_multiple_of(256) {
        return Err(MatmulError::Vulkan(vk::Result::ERROR_FEATURE_NOT_PRESENT));
    }
    let d = &dev.device;

    // Sobe um slice de bytes para um STORAGE_BUFFER device-local via staging descartável.
    let upload = |bytes: &[u8]| -> Result<(vk::Buffer, vk::DeviceMemory), MatmulError> {
        let size = bytes.len() as vk::DeviceSize;
        let staging = create_buf(d, size, vk::BufferUsageFlags::TRANSFER_SRC)?;
        let staging_mem = alloc_and_bind(ctx, phys, d, staging, true)?;
        unsafe {
            // SAFETY: staging_mem é host-visible com `size`; ptr válido até unmap.
            let ptr = d.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            d.unmap_memory(staging_mem);
        }
        let buf = create_buf(
            d,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let mem = alloc_and_bind(ctx, phys, d, buf, false)?;
        one_shot_copy(d, dev.queue, dev.cmd_pool, staging, buf, size)?;
        unsafe {
            // SAFETY: staging já copiado; ambos criados por nós.
            d.destroy_buffer(staging, None);
            d.free_memory(staging_mem, None);
        }
        Ok((buf, mem))
    };

    let (w_buf, w_mem) = upload(w_bytes)?;
    // Os shaders K-quant consomem a ativação em int8, como o caminho Q8_0: o sub-bloco de
    // 32 elementos do K-quant coincide com o bloco de quantização.
    let (xq, xd) = crate::tensor::quantize_x_host(x_f32);
    // SAFETY: Vec<u32>/Vec<f32> são POD contíguos; reinterpretar como bytes é válido.
    let xq_bytes = unsafe { std::slice::from_raw_parts(xq.as_ptr().cast::<u8>(), xq.len() * 4) };
    let xd_bytes = unsafe { std::slice::from_raw_parts(xd.as_ptr().cast::<u8>(), xd.len() * 4) };
    let (xq_buf, xq_mem) = upload(xq_bytes)?;
    let (xd_buf, xd_mem) = upload(xd_bytes)?;

    let y_size = (cols * n_out * size_of::<f32>()) as vk::DeviceSize;
    let y_buf = create_buf(
        d,
        y_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let y_mem = alloc_and_bind(ctx, phys, d, y_buf, false)?;

    let pipe = ComputePipeline::with(d, spv, 5, size_of::<PushConstants>() as u32, spec)?;
    let push = PushConstants {
        n_in: u32::try_from(n_in).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))?,
        n_out: u32::try_from(n_out).map_err(|_| MatmulError::Vulkan(vk::Result::ERROR_UNKNOWN))?,
        row_offset: 0,
        tem_bias: 0,
    };
    // Descriptor pool + set com os 5 bindings (pesos, xq, xd, saída, bias). Sem bias
    // aqui: o binding 4 recebe `xd` só para completar o layout, e `tem_bias: 0` o ignora.
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 5,
    }];
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets: 1,
        pool_size_count: 1,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };
    // SAFETY: d válido; pool_info aponta para dados vivos nesta frame.
    let desc_pool = unsafe { d.create_descriptor_pool(&pool_info, None)? };
    let set_alloc = vk::DescriptorSetAllocateInfo {
        descriptor_pool: desc_pool,
        descriptor_set_count: 1,
        p_set_layouts: &pipe.desc_set_layout,
        ..Default::default()
    };
    // SAFETY: pool e layout válidos.
    let desc_set = unsafe { d.allocate_descriptor_sets(&set_alloc)? }[0];

    let infos = [
        vk::DescriptorBufferInfo {
            buffer: w_buf,
            offset: 0,
            range: w_bytes.len() as vk::DeviceSize,
        },
        vk::DescriptorBufferInfo {
            buffer: xq_buf,
            offset: 0,
            range: xq_bytes.len() as vk::DeviceSize,
        },
        vk::DescriptorBufferInfo {
            buffer: xd_buf,
            offset: 0,
            range: xd_bytes.len() as vk::DeviceSize,
        },
        vk::DescriptorBufferInfo {
            buffer: y_buf,
            offset: 0,
            range: y_size,
        },
        vk::DescriptorBufferInfo {
            buffer: xd_buf,
            offset: 0,
            range: xd_bytes.len() as vk::DeviceSize,
        },
    ];
    let writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .enumerate()
        .map(|(b, i)| vk::WriteDescriptorSet {
            dst_set: desc_set,
            dst_binding: b as u32,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: i,
            ..Default::default()
        })
        .collect();
    // SAFETY: writes aponta para `infos`, vivo nesta frame.
    unsafe { d.update_descriptor_sets(&writes, &[]) };

    let cb_alloc = vk::CommandBufferAllocateInfo {
        command_pool: dev.cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    // SAFETY: device e pool válidos; cmd gravado e liberado aqui.
    let out = unsafe {
        let cmd = d.allocate_command_buffers(&cb_alloc)?[0];
        d.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            },
        )?;
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
            std::slice::from_raw_parts(
                std::ptr::from_ref(&push).cast::<u8>(),
                size_of::<PushConstants>(),
            ),
        );
        d.cmd_dispatch(cmd, push.n_out.div_ceil(rows_por_wg), 1, 1);
        d.end_command_buffer(cmd)?;
        let submit = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            ..Default::default()
        };
        d.queue_submit(dev.queue, &[submit], vk::Fence::null())?;
        d.queue_wait_idle(dev.queue)?;
        d.free_command_buffers(dev.cmd_pool, &[cmd]);

        // Readback.
        let read = create_buf(d, y_size, vk::BufferUsageFlags::TRANSFER_DST)?;
        let read_mem = alloc_and_bind(ctx, phys, d, read, true)?;
        one_shot_copy(d, dev.queue, dev.cmd_pool, y_buf, read, y_size)?;
        let ptr = d.map_memory(read_mem, 0, y_size, vk::MemoryMapFlags::empty())?;
        let mut v = vec![0f32; cols * n_out];
        std::ptr::copy_nonoverlapping(ptr.cast::<f32>(), v.as_mut_ptr(), cols * n_out);
        d.unmap_memory(read_mem);
        d.destroy_buffer(read, None);
        d.free_memory(read_mem, None);
        d.destroy_descriptor_pool(desc_pool, None);
        Ok(v)
    };

    unsafe {
        // SAFETY: GPU ociosa após o readback; handles criados por nós.
        pipe.destroy(d);
        d.destroy_buffer(w_buf, None);
        d.free_memory(w_mem, None);
        d.destroy_buffer(xq_buf, None);
        d.free_memory(xq_mem, None);
        d.destroy_buffer(xd_buf, None);
        d.free_memory(xd_mem, None);
        d.destroy_buffer(y_buf, None);
        d.free_memory(y_mem, None);
    }
    out
}
