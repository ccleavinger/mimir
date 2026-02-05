pub mod error;
pub mod jit;
pub mod program;

use ash::{Entry, vk};
use mimir_ir::ir::MimirIRKind;
use parking_lot::Mutex;
use std::{
    cell::Cell,
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, LazyLock},
};
use vk_mem::AllocatorCreateInfo;

use crate::{
    runtime::{error::VulkanError, program::Program},
    vk_platform_device::{MimirVkDevice, MimirVkPlatformDevices},
};

pub static MIMIR_VK_RUNTIME: LazyLock<Option<MimirVkRuntime>> = LazyLock::new(MimirVkRuntime::init);

#[derive(PartialEq, Eq, Hash)]
pub struct SpirvCacheKey {
    name: String,
    const_generic_hash: u64,
}

impl SpirvCacheKey {
    pub fn new(name: &str, const_generics: &[u32]) -> Self {
        let mut s = DefaultHasher::new();
        const_generics.hash(&mut s);
        let hash = s.finish();

        Self {
            name: name.to_string(),
            const_generic_hash: hash,
        }
    }
}

#[allow(dead_code)]
pub struct MimirVkRuntime {
    // instance members
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,

    // device members
    pub(crate) phys_dev: vk::PhysicalDevice,
    pub(crate) device: Mutex<ash::Device>,
    pub(crate) queue_family_index: u32, // might not need this
    pub(crate) queue: Mutex<vk::Queue>,
    pub(crate) descriptor_pool: Mutex<vk::DescriptorPool>,

    // allocator
    pub(crate) allocator: Mutex<vk_mem::Allocator>,

    // Cache
    pub(crate) ir_cache: Mutex<Vec<Arc<MimirIRKind>>>,
    pub(crate) spirv: Mutex<Cell<HashMap<SpirvCacheKey, Vec<u32>>>>,
    pub(crate) programs: Mutex<HashMap<String, Arc<Program>>>,
    platform_devices: MimirVkPlatformDevices,
}

impl MimirVkRuntime {
    pub fn init() -> Option<Self> {
        match Self::init_err() {
            Ok(runtime) => Some(runtime),
            Err(err) => {
                log::debug!(
                    "Failed to establish Vulkan Backend.\nERR: {:?}",
                    err.to_string()
                );

                None
            }
        }
    }

    pub fn init_err() -> Result<Self, VulkanError> {
        let entry = {
            let file = if cfg!(target_os = "windows") {
                "vulkan-1.dll"
            } else {
                "libvulkan.so (many variations on Unix systems)"
            };

            unsafe { Entry::load() }
                .map_err(|_| VulkanError::BackendInitialization(
                    format!("Failed to load Vulkan. Device(s) is/are either not supported or {:?} couldn't be found.", file)
                ))?
        };

        let instance = {
            let app_info = vk::ApplicationInfo {
                p_application_name: c"Mimir".as_ptr(),
                application_version: 0,
                p_engine_name: c"Mimir".as_ptr(),
                engine_version: 0,
                api_version: vk::API_VERSION_1_3,
                ..Default::default()
            };

            let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

            let create_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: layer_names.as_ptr(),
                ..Default::default()
            };

            unsafe { entry.create_instance(&create_info, None)? }
        };

        let phys_devs = unsafe { instance.enumerate_physical_devices()? };

        let (phys_dev, queue_fam_idx) = phys_devs
            .iter()
            .filter_map(|&physical_dev| {
                let queue_families =
                    unsafe { instance.get_physical_device_queue_family_properties(physical_dev) };
                let compute_queue_family_idx = queue_families
                    .iter()
                    .enumerate()
                    .find(|(_idx, queue_family)| {
                        queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .map(|(index, _queue_family)| index as u32);

                compute_queue_family_idx.map(|index| (physical_dev, index))
            })
            .next()
            .expect("No compute capable device found");

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index: queue_fam_idx,
            p_queue_priorities: priorities.as_ptr(),
            queue_count: priorities.len() as u32,
            ..Default::default()
        };

        let device_extension_names_raw = [ash::vk::KHR_MAINTENANCE4_NAME.as_ptr()];
        let mut maintenance4_features =
            vk::PhysicalDeviceMaintenance4FeaturesKHR::default().maintenance4(true);
        let features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features)
            .push_next(&mut maintenance4_features);

        let device = unsafe { instance.create_device(phys_dev, &device_create_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_fam_idx, 0) };

        let descriptor_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16)];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(8)
            .pool_sizes(&descriptor_sizes);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;

        let allocator = unsafe {
            vk_mem::Allocator::new(AllocatorCreateInfo::new(&instance, &device, phys_dev))
        }?;

        let mimir_vk_device =
            MimirVkDevice::new(Arc::new(device.clone()), Arc::new(phys_dev), &instance)?;

        let platform_devices = MimirVkPlatformDevices {
            devices: vec![mimir_vk_device],
        };

        Ok(Self {
            entry,
            instance,
            phys_dev,
            device: Mutex::new(device),
            queue_family_index: queue_fam_idx,
            queue: Mutex::new(queue),
            descriptor_pool: Mutex::new(descriptor_pool),
            allocator: Mutex::new(allocator),
            ir_cache: Mutex::new(Vec::new()),
            spirv: Mutex::new(Cell::new(HashMap::new())),
            programs: Mutex::new(HashMap::new()),
            platform_devices,
        })
    }
}
