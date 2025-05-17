use std::{collections::HashMap, sync::{Arc, LazyLock, Mutex}};

use ash::{vk, Entry};
use mimir_ast::MimirGlobalAST;
use vk_mem::AllocatorCreateInfo;

use super::{error::VulkanError, program::Program};

pub static BACKEND: LazyLock<MimirVkBackend> = LazyLock::new(MimirVkBackend::init);

#[allow(dead_code)]
pub struct MimirVkBackend {
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
    pub(crate) asts: Mutex<Vec<Arc<MimirGlobalAST>>>,
    pub(crate) spirv: Mutex<HashMap<String, Vec<u32>>>,
    pub(crate) programs: Mutex<HashMap<String, Arc<Program>>>
}

impl MimirVkBackend {
    fn init_err() -> Result<MimirVkBackend, VulkanError> {

        let entry = {
            let file = if cfg!(target_os = "windows") {
                "vulkan-1.dll"
            } else {
                "libvulkan.so (many variations on Unix systems)"
            };

            unsafe { Entry::load() }
                .map_err(|_| VulkanError::BackendInitialization(
                    format!("Failed to load Vulkan. Device is either not supported or {:?} couldn't be found.", file)
                ))?
        };

        let instance= {
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

            unsafe {
                entry.create_instance(&create_info, None)?
            }
        };

        let phys_devs = unsafe {
            instance.enumerate_physical_devices()?
        };

        let (phys_dev, queue_fam_idx) = phys_devs
            .iter()
            .filter_map(|&physical_dev| {
                let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_dev) };
                let compute_queue_family_idx = queue_families
                    .iter()
                    .enumerate()
                    .find(|(_idx, queue_family)| queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(index, _queue_family)| index as u32 );

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

        let device_extension_names_raw = [
            ash::vk::KHR_MAINTENANCE4_NAME.as_ptr(),
        ];
        let mut maintenance4_features = vk::PhysicalDeviceMaintenance4FeaturesKHR::default()
            .maintenance4(true);
        let features = vk::PhysicalDeviceFeatures::default();
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features)
            .push_next(&mut maintenance4_features);

        let device = unsafe { instance.create_device(phys_dev, &device_create_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_fam_idx, 0) };

        let descriptor_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(16),
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(8)
            .pool_sizes(&descriptor_sizes);
            
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;


        let allocator = unsafe { vk_mem::Allocator::new(
            AllocatorCreateInfo::new(&instance, &device, phys_dev)
        ) }?;

        Ok(Self {
            entry,
            instance,
            phys_dev,
            device: Mutex::new(device),
            queue_family_index: queue_fam_idx,
            queue: Mutex::new(queue),
            descriptor_pool: Mutex::new(descriptor_pool),
            allocator: Mutex::new(allocator),
            asts: Mutex::new(vec![]),
            spirv: Mutex::new(HashMap::new()),
            programs: Mutex::new(HashMap::new())
        })
    }
    pub fn init() -> MimirVkBackend {
        match MimirVkBackend::init_err() {
            Ok(backend) => backend,
            Err(err) => panic!("Failed to establish Vulkan Backend.\nERR: {:?}", err.to_string()),
        }
    }
}

