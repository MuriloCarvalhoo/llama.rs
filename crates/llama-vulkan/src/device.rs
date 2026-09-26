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

/// VRAM que o llama-rs deixa livre num device: **2 GiB** na GPU que dirige um monitor e
/// 500 MiB nas outras. Sem VRAM livre a GPU do display trava a sessão gráfica (o compositor
/// não consegue alocar, ou o driver reseta a GPU). Sem como saber se há monitor, vale a maior.
const MARGEM_COM_MONITOR: u64 = 2 << 30;
const MARGEM_SEM_MONITOR: u64 = 500 << 20;

fn margem_vram_de(tem_monitor: Option<bool>) -> u64 {
    match tem_monitor {
        Some(false) => MARGEM_SEM_MONITOR,
        _ => MARGEM_COM_MONITOR,
    }
}

/// A margem de cada device a partir do que o sysfs diz de todos. Com o monitor em repouso o
/// DisplayPort passa a `disconnected` e **nenhuma** GPU parece dirigir tela — mas uma delas
/// volta a dirigir quando ele acordar, e o compositor precisa de VRAM nela. Sem saber qual,
/// vale a margem maior em todas.
fn margens_vram(monitores: &[Option<bool>]) -> Vec<u64> {
    let algum = monitores.contains(&Some(true));
    monitores
        .iter()
        .map(|&m| {
            if algum {
                margem_vram_de(m)
            } else {
                MARGEM_COM_MONITOR
            }
        })
        .collect()
}

/// Algum conector do card DRM no endereço PCI `pci` (`dddd:bb:dd.f`) está `connected`?
/// `None` quando o sysfs não diz (sem DRM, outro SO). `raiz` é `/sys/bus/pci/devices` fora
/// dos testes.
///
/// Não há como perguntar isso ao Vulkan: o RADV não diz qual physical device é qual card do
/// DRM, só o endereço PCI (`VK_EXT_pci_bus_info`) — e é por ele que se chega aos conectores.
fn tem_monitor(raiz: &std::path::Path, pci: &str) -> Option<bool> {
    let mut algum_card = false;
    for card in std::fs::read_dir(raiz.join(pci).join("drm"))
        .ok()?
        .flatten()
    {
        let card_nome = card.file_name().to_string_lossy().into_owned();
        if !card_nome.starts_with("card") {
            continue;
        }
        algum_card = true;
        // Conectores são `cardN-<tipo>-<n>/status`; o resto do diretório (`device`,
        // `power`, ...) não tem `status` de conector.
        let prefixo = format!("{card_nome}-");
        for con in std::fs::read_dir(card.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            if con.file_name().to_string_lossy().starts_with(&prefixo)
                && std::fs::read_to_string(con.path().join("status"))
                    .is_ok_and(|s| s.trim() == "connected")
            {
                return Some(true);
            }
        }
    }
    algum_card.then_some(false)
}

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
    margem_vram: u64,
    /// `dddd:bb:dd.f`, de `VK_EXT_pci_bus_info` — liga o device ao sysfs.
    pub(crate) pci: Option<String>,
}

impl VulkanPhysicalDevice {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Bytes de VRAM que têm de sobrar livres neste device depois da carga: 2 GiB se ele
    /// dirige um monitor, 500 MiB se não (ver `MARGEM_COM_MONITOR`).
    pub fn margem_vram(&self) -> u64 {
        self.margem_vram
    }

    /// Bytes livres nos heaps DEVICE_LOCAL, via `VK_EXT_memory_budget`.
    /// Retorna `None` se a extensão não estiver disponível.
    ///
    /// Importa porque a GPU que roda o display já tem ~1.6 GB ocupados: num modelo que
    /// quase enche a VRAM, o driver realoca o excedente em GTT (memória do host, via
    /// PCIe) e a banda efetiva do matvec despenca — medimos 95 GB/s contra 714 GB/s
    /// para o mesmo modelo na GPU sem display.
    pub fn free_device_memory(&self, ctx: &VulkanContext) -> Option<u64> {
        self.orcamento_vram(ctx).map(|(_, livre)| livre)
    }

    /// VRAM que **este processo** ocupa nos heaps DEVICE_LOCAL, pelo mesmo
    /// `VK_EXT_memory_budget` (é o `heapUsage`, que conta só as alocações do processo).
    pub fn vram_do_processo(&self, ctx: &VulkanContext) -> Option<u64> {
        self.orcamento_vram(ctx).map(|(usada, _)| usada)
    }

    /// `(usada pelo processo, livre para ele)` nos heaps DEVICE_LOCAL.
    fn orcamento_vram(&self, ctx: &VulkanContext) -> Option<(u64, u64)> {
        let exts = unsafe {
            ctx.instance
                .enumerate_device_extension_properties(self.handle)
        }
        .ok()?;
        let has_budget = exts.iter().any(|e| {
            // SAFETY: extension_name é nul-terminado pela spec Vulkan.
            let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            n.to_bytes() == b"VK_EXT_memory_budget"
        });
        if !has_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2 {
            p_next: std::ptr::from_mut(&mut budget).cast(),
            ..Default::default()
        };
        // SAFETY: handle válido; p_next aponta para `budget`, vivo durante a chamada.
        unsafe {
            ctx.instance
                .get_physical_device_memory_properties2(self.handle, &mut props2)
        };
        let mem = props2.memory_properties;
        let mut free = 0u64;
        let mut usada = 0u64;
        for i in 0..mem.memory_heap_count as usize {
            if mem.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                free += budget.heap_budget[i].saturating_sub(budget.heap_usage[i]);
                usada += budget.heap_usage[i];
            }
        }
        Some((usada, free))
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
        let mut monitores = Vec::new();

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

            // A struct de PCI só pode entrar na cadeia se o device anunciar a extensão.
            // SAFETY: `pd` é um handle válido retornado por `enumerate_physical_devices`.
            let tem_pci_info = unsafe { instance.enumerate_device_extension_properties(pd) }
                .is_ok_and(|exts| {
                    exts.iter().any(|e| {
                        // SAFETY: extension_name é nul-terminado pela spec Vulkan.
                        let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                        n.to_bytes() == b"VK_EXT_pci_bus_info"
                    })
                });
            let mut pci_props = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
            let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
            if tem_pci_info {
                subgroup_props.p_next = std::ptr::from_mut(&mut pci_props).cast();
            }
            let mut props2 = vk::PhysicalDeviceProperties2 {
                p_next: &mut subgroup_props as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            // SAFETY: `pd` é válido; `p_next` aponta para `subgroup_props` (e este, com a
            // extensão, para `pci_props`), que vivem na mesma stack frame durante toda a
            // chamada, satisfazendo o requisito de validade do ponteiro.
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
            let pci = tem_pci_info.then(|| {
                format!(
                    "{:04x}:{:02x}:{:02x}.{:x}",
                    pci_props.pci_domain,
                    pci_props.pci_bus,
                    pci_props.pci_device,
                    pci_props.pci_function
                )
            });
            let monitor = pci
                .as_deref()
                .and_then(|pci| tem_monitor(std::path::Path::new("/sys/bus/pci/devices"), pci));
            monitores.push(monitor);

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
                // Preenchida depois do laço: depende do que os outros devices disserem.
                margem_vram: 0,
                pci,
            });
        }
        for (p, m) in result.iter_mut().zip(margens_vram(&monitores)) {
            p.margem_vram = m;
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
        let cmd_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                // O `Drop` que destruiria o device ainda não existe: destruí-lo aqui, senão
                // ele sobra até o fim do processo.
                // SAFETY: device criado acima, sem nenhum objeto filho vivo.
                unsafe { device.destroy_device(None) };
                return Err(e);
            }
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `raiz/<pci>/drm/card<N>/card<N>-<con>/status` num diretório temporário próprio.
    fn sysfs(nome: &str, conectores: &[(&str, &str)]) -> std::path::PathBuf {
        let raiz =
            std::env::temp_dir().join(format!("llama-rs-sysfs-{nome}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&raiz);
        let card = raiz.join("0000:85:00.0/drm/card2");
        std::fs::create_dir_all(card.join("power")).unwrap();
        for (con, status) in conectores {
            let d = card.join(format!("card2-{con}"));
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("status"), format!("{status}\n")).unwrap();
        }
        raiz
    }

    #[test]
    fn conector_ligado_e_monitor() {
        let raiz = sysfs("ligado", &[("DP-7", "disconnected"), ("DP-8", "connected")]);
        assert_eq!(tem_monitor(&raiz, "0000:85:00.0"), Some(true));
        std::fs::remove_dir_all(raiz).unwrap();
    }

    #[test]
    fn card_sem_conector_ligado_nao_e_monitor() {
        let raiz = sysfs("desligado", &[("DP-7", "disconnected")]);
        assert_eq!(tem_monitor(&raiz, "0000:85:00.0"), Some(false));
        std::fs::remove_dir_all(raiz).unwrap();
    }

    #[test]
    fn pci_desconhecido_nao_decide() {
        let raiz = sysfs("ausente", &[]);
        assert_eq!(tem_monitor(&raiz, "0000:05:00.0"), None);
        std::fs::remove_dir_all(raiz).unwrap();
    }

    #[test]
    fn monitor_em_repouso_vale_a_margem_maior_em_todas() {
        let g = 2 << 30;
        let p = 500 << 20;
        assert_eq!(margens_vram(&[Some(true), Some(false)]), vec![g, p]);
        assert_eq!(margens_vram(&[Some(false), Some(false)]), vec![g, g]);
        assert_eq!(margens_vram(&[None, Some(false)]), vec![g, g]);
    }

    #[test]
    fn sem_saber_do_monitor_vale_a_margem_maior() {
        assert_eq!(margem_vram_de(Some(true)), 2 << 30);
        assert_eq!(margem_vram_de(None), 2 << 30);
        assert_eq!(margem_vram_de(Some(false)), 500 << 20);
    }
}
