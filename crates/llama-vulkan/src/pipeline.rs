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
