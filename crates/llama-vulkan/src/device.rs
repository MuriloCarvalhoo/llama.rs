use ash::{Entry, Instance, vk};
use std::ffi::CStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error("Falha ao carregar biblioteca Vulkan")]
    LibraryLoad,
    #[error("Falha ao criar instancia Vulkan: {0}")]
    InstanceCreate(vk::Result),
    #[error("Nenhum physical device encontrado")]
    NoDevices,
    #[error("Vulkan API error: {0}")]
    Api(#[from] vk::Result),
}

const AMD_VENDOR_ID: u32 = 0x1002;

pub struct VulkanContext {
    #[allow(dead_code)] // mantido para garantir que Entry não seja dropada antes de Instance
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    physical_devices: Vec<VulkanPhysicalDevice>,
}

pub struct VulkanPhysicalDevice {
    pub(crate) handle: vk::PhysicalDevice,
    name: String,
    subgroup_size: u32,
    pub(crate) queue_family: u32,
}

impl VulkanPhysicalDevice {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }
}

impl VulkanContext {
    pub fn new() -> Result<Self, VulkanError> {
        // SAFETY: carrega biblioteca Vulkan dinamicamente via ash; nenhum invariante de
        // memória é violado — a função apenas dlopen/LoadLibrary a lib do sistema.
        let entry = unsafe { Entry::load().map_err(|_| VulkanError::LibraryLoad)? };

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 1, 0), // Vulkan 1.1 para subgroup ops
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        // SAFETY: `app_info` e `create_info` são referências válidas na mesma stack frame;
        // ambas vivem até `create_instance` retornar, satisfazendo os requisitos de lifetime do FFI.
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(VulkanError::InstanceCreate)?
        };

        let physical_devices = Self::enumerate_amd_devices(&instance)?;
        Ok(Self {
            entry,
            instance,
            physical_devices,
        })
    }

    pub fn amd_compute_devices(&self) -> &[VulkanPhysicalDevice] {
        &self.physical_devices
    }

    fn enumerate_amd_devices(
        instance: &Instance,
    ) -> Result<Vec<VulkanPhysicalDevice>, VulkanError> {
        // SAFETY: `instance` é válida — foi criada com sucesso pela função chamadora.
        let phys_devs = unsafe { instance.enumerate_physical_devices()? };
        let mut result = Vec::new();

        for pd in phys_devs {
            // SAFETY: `pd` é um handle válido retornado por `enumerate_physical_devices`.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.vendor_id != AMD_VENDOR_ID {
                continue;
            }

            // SAFETY: `pd` é um handle válido retornado por `enumerate_physical_devices`.
            let qfams = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qfam_idx) = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };

            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2 {
                p_next: &mut subgroup_props as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            // SAFETY: `pd` é válido; `p_next` aponta para `subgroup_props` que vive na mesma
            // stack frame durante toda a chamada, satisfazendo o requisito de validade do ponteiro.
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

            // SAFETY: `device_name` é garantido nul-terminado pela spec Vulkan
            // (VkPhysicalDeviceProperties.deviceName tem VK_MAX_PHYSICAL_DEVICE_NAME_SIZE bytes
            // com nul terminator obrigatório).
            let name = unsafe {
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            result.push(VulkanPhysicalDevice {
                handle: pd,
                name,
                subgroup_size: subgroup_props.subgroup_size,
                queue_family: qfam_idx as u32,
            });
        }
        Ok(result)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        // SAFETY: instance foi criada por nós e não foi destruída antes.
        unsafe { self.instance.destroy_instance(None) };
    }
}

/// Device lógico Vulkan + fila de compute + command pool.
pub struct VulkanDevice {
    pub(crate) device: ash::Device,
    #[allow(dead_code)] // usado em tasks futuras (dispatch de compute)
    pub(crate) queue: vk::Queue,
    pub(crate) cmd_pool: vk::CommandPool,
    #[allow(dead_code)] // usado em tasks futuras (pipeline creation)
    pub(crate) queue_family: u32,
}

impl VulkanDevice {
    /// Retorna referencia ao `ash::Device` logico.
    pub fn as_device(&self) -> &ash::Device {
        &self.device
    }

    pub fn create(ctx: &VulkanContext, phys: &VulkanPhysicalDevice) -> Result<Self, vk::Result> {
        let queue_priority = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: phys.queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priority.as_ptr(),
            ..Default::default()
        };
        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            ..Default::default()
        };
        // SAFETY: `phys.handle` é um handle válido retornado por `enumerate_physical_devices`;
        // `create_info` e `queue_info` vivem na mesma stack frame até a chamada retornar.
        let device = unsafe {
            ctx.instance
                .create_device(phys.handle, &create_info, None)?
        };
        // SAFETY: `device` foi criado com sucesso acima; queue_family e index 0 são válidos
        // pois foram verificados durante a enumeração dos physical devices.
        let queue = unsafe { device.get_device_queue(phys.queue_family, 0) };
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: phys.queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        // SAFETY: `device` é válido e `pool_info` aponta para dados válidos na stack frame atual.
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        Ok(Self {
            device,
            queue,
            cmd_pool,
            queue_family: phys.queue_family,
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: cmd_pool e device foram criados por nós nesta ordem.
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
        }
    }
}
