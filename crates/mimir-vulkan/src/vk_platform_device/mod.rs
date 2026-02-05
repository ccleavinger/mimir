use std::sync::{Arc, RwLock};

use ash::vk;
use mimir_runtime::{
    generic::device::{DeviceID, DeviceNameID, MimirDevice, MimirPlatformDevices, PCIBusID},
    platforms::MimirPlatform,
};

use crate::runtime::{MIMIR_VK_RUNTIME, error::VulkanError};

#[derive(Clone)]
pub struct MimirVkPlatformDevices {
    pub(crate) devices: Vec<MimirVkDevice>,
}

impl MimirPlatformDevices for MimirVkPlatformDevices {
    fn get_devices(&self) -> Vec<Box<dyn MimirDevice + Send + Sync>> {
        self.devices
            .iter()
            .map(|device| {
                let boxed: Box<dyn MimirDevice + Send + Sync> = Box::new(device.clone());
                boxed
            })
            .collect()
    }

    fn get_platform(&self) -> MimirPlatform {
        MimirPlatform::Vulkan
    }
}

#[derive(Clone)]
pub struct MimirVkDevice {
    pub log_device: Arc<ash::Device>,
    pub phys_device: Arc<vk::PhysicalDevice>,
    pub allocator: Arc<vk_mem::Allocator>,
    pub phys_dev_props: Arc<RwLock<vk::PhysicalDeviceProperties>>,
    pub phys_dev_limits: Arc<vk::PhysicalDeviceLimits>,
    pub pci_bus_info: Option<PCIBusID>,
}

impl MimirVkDevice {
    pub fn new(
        log_device: Arc<ash::Device>,
        phys_device: Arc<vk::PhysicalDevice>,
        instance: &ash::Instance,
    ) -> Result<Self, VulkanError> {
        let (phys_dev_props, phys_dev_limits) = {
            let props = unsafe { instance.get_physical_device_properties(*phys_device) };

            let limits = Arc::new(props.limits);
            let arc_props = Arc::new(RwLock::new(props));

            (arc_props, limits)
        };

        let pci_bus_info = {
            let extensions_opt =
                unsafe { instance.enumerate_device_extension_properties(*phys_device) }.ok();

            if let Some(extensions) = extensions_opt {
                let has_pci_info = extensions.iter().any(|ext| {
                    let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                    name == ash::ext::pci_bus_info::NAME
                });

                if has_pci_info {
                    let mut pci_bus_info = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();

                    let mut properties2 =
                        vk::PhysicalDeviceProperties2::default().push_next(&mut pci_bus_info);

                    unsafe {
                        instance.get_physical_device_properties2(*phys_device, &mut properties2);
                    }

                    let pci = pci_bus_info;

                    Some(PCIBusID {
                        pci_domain: pci.pci_domain,
                        pci_bus: pci.pci_bus,
                        pci_device: pci.pci_device,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        };

        let allocator = Arc::new(
            match unsafe {
                vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
                    instance,
                    log_device.as_ref(),
                    *phys_device,
                ))
            } {
                Ok(alloc) => alloc,
                Err(err) => {
                    return Err(VulkanError::Generic(format!(
                        "Failed to create an allocator for a Vulkan device: {err:?}"
                    )));
                }
            },
        );

        Ok(Self {
            log_device,
            phys_device,
            allocator,
            phys_dev_props,
            phys_dev_limits,
            pci_bus_info,
        })
    }
}

impl MimirDevice for MimirVkDevice {
    fn device_name(&self) -> String {
        let props = self.phys_dev_props.read().unwrap(); // Get read lock
        match props.device_name_as_c_str() {
            Ok(c_str) => match c_str.to_str() {
                Ok(str) => String::from(str),
                Err(_) => "UNKNOWN_DEVICE".to_owned(),
            },
            Err(_) => "UNKNOWN_DEVICE".to_owned(),
        }
    }

    fn vram_size(&self) -> usize {
        let mem_props = unsafe {
            let runtime = match MIMIR_VK_RUNTIME.as_ref() {
                Some(rtim) => rtim,
                None => return 0, // if the runtime is unsupported and we somehow have a device (very very unlikely) just say the vram/mem is 0
            };

            runtime
                .instance
                .get_physical_device_memory_properties(*self.phys_device)
        };

        // Sum all device-local heaps
        let mut total_vram = 0u64;

        for i in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[i];

            // Check if this heap is device-local (VRAM)
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total_vram += heap.size;
            }
        }

        total_vram as usize
    }

    fn as_platform_device(&self) -> Box<dyn std::any::Any> {
        Box::new(self.clone())
    }

    fn platform(&self) -> mimir_runtime::platforms::MimirPlatform {
        MimirPlatform::Vulkan
    }

    fn get_id(&self) -> mimir_runtime::generic::device::DeviceID {
        if let Some(pci) = self.pci_bus_info {
            DeviceID::PCI(PCIBusID {
                pci_domain: pci.pci_domain,
                pci_bus: pci.pci_bus,
                pci_device: pci.pci_device,
            })
        } else {
            DeviceID::NAME(DeviceNameID::new(self.device_name()))
        }
    }
}
