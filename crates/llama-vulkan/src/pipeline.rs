use ash::vk;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("Vulkan API error: {0}")]
    Vulkan(#[from] vk::Result),
}

/// Push constants para o shader Q8_0 matvec.
#[repr(C)]
pub(crate) struct PushConstants {
    pub n_in: u32,
    pub n_out: u32,
    pub row_offset: u32,
}

/// Pipeline Vulkan para o shader Q8_0 matmul-vector.
///
/// O `ash::Device` não é armazenado aqui pois não implementa `Clone`.
/// O caller deve chamar `destroy(dev)` antes de dropar.
pub struct ComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) desc_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    /// Cria a pipeline Vulkan para o shader Q8_0 matmul-vector.
    /// `dev` deve ser mantido vivo enquanto esta struct existir.
    pub fn new(dev: &ash::Device) -> Result<Self, PipelineError> {
        // 1. DescriptorSetLayout com 3 bindings STORAGE_BUFFER
        //    binding 0 = weights, 1 = activations, 2 = output
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        // SAFETY: dev é válido; dsl_info aponta para dados válidos na stack frame.
        let desc_set_layout = unsafe { dev.create_descriptor_set_layout(&dsl_info, None)? };

        // 2. PipelineLayout com o descriptor set layout + push constants (3× u32 = 12 bytes)
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };
        let layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &desc_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        // SAFETY: dev é válido; layout_info aponta para dados válidos na stack frame.
        let layout = unsafe { dev.create_pipeline_layout(&layout_info, None)? };

        // 3. ShaderModule carregado do SPIR-V
        // Copia para Vec<u32> para garantir alinhamento a 4 bytes exigido por Vulkan.
        let spv = crate::Q8_0_MATVEC_SPV;
        let spv_u32: Vec<u32> = spv
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv.len(),
            p_code: spv_u32.as_ptr(),
            ..Default::default()
        };
        // SAFETY: dev é válido; shader_info aponta para SPIR-V válido na memória estática.
        let shader_module = unsafe { dev.create_shader_module(&shader_info, None)? };

        // 4. ComputePipelineCreateInfo → create_compute_pipelines
        let entry_name = std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage,
            layout,
            ..Default::default()
        };
        // SAFETY: dev é válido; pipeline_info aponta para dados válidos na stack frame.
        let pipelines = unsafe {
            dev.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e)?
        };
        let pipeline = pipelines[0];

        // 5. Destruir o ShaderModule após criar a pipeline
        // SAFETY: shader_module foi criado por nós; a pipeline já foi criada acima.
        unsafe { dev.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            layout,
            desc_set_layout,
        })
    }

    /// Libera os recursos Vulkan. Deve ser chamado antes de dropar.
    pub fn destroy(self, dev: &ash::Device) {
        unsafe {
            // SAFETY: pipeline, layout e desc_set_layout foram criados por nós nesta ordem inversa.
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.layout, None);
            dev.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
