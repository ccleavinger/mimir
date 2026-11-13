use ash::{Entry, vk};
use mimir_ir::flat::*;
use mimir_runtime_core::{
    Argument, BufferBackend, DeviceBackend, DeviceId, DeviceLuid, ExecutionError, ExecutionResult,
    HardwareInfo, Launch, PciAddress, validate_buffer_owner, validate_buffer_values,
};
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    sync::{Arc, Mutex},
};

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

struct Instance {
    _entry: Entry,
    raw: ash::Instance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.raw.destroy_instance(None) }
    }
}

struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor: vk::DescriptorSetLayout,
}

struct State {
    pipelines: HashMap<VariantKey, Pipeline>,
}

pub struct VulkanDevice {
    id: DeviceId,
    hardware: HardwareInfo,
    instance: Arc<Instance>,
    physical: vk::PhysicalDevice,
    raw: ash::Device,
    queue: vk::Queue,
    family: u32,
    pub name: String,
    properties: vk::PhysicalDeviceProperties,
    state: Mutex<State>,
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            for (_, p) in self.state.get_mut().unwrap().pipelines.drain() {
                self.raw.destroy_pipeline(p.pipeline, None);
                self.raw.destroy_pipeline_layout(p.layout, None);
                self.raw.destroy_descriptor_set_layout(p.descriptor, None);
            }
            self.raw.destroy_device(None);
        }
    }
}
pub struct VulkanBuffer {
    element: ScalarType,
    length: usize,
    device: Arc<VulkanDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn discover() -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
    unsafe {
        let entry = Entry::load().map_err(err)?;
        let name = CString::new("Mimir").unwrap();
        let app = vk::ApplicationInfo::default()
            .application_name(&name)
            .api_version(vk::API_VERSION_1_1);
        let raw = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .map_err(err)?;
        let instance = Arc::new(Instance { _entry: entry, raw });
        let mut devices: Vec<Arc<dyn DeviceBackend>> = vec![];
        for physical in instance.raw.enumerate_physical_devices().map_err(err)? {
            let properties = instance.raw.get_physical_device_properties(physical);
            if properties.api_version < vk::API_VERSION_1_1 {
                continue;
            }
            let hardware = hardware_info(&instance.raw, physical, &properties);
            let families = instance
                .raw
                .get_physical_device_queue_family_properties(physical);
            let Some(family) = families
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family as u32)
                .queue_priorities(&priorities)];
            let raw = instance
                .raw
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                    None,
                )
                .map_err(err)?;
            let queue = raw.get_device_queue(family as u32, 0);
            let name = CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();
            devices.push(Arc::new(VulkanDevice {
                id: DeviceId::new(),
                hardware,
                instance: instance.clone(),
                physical,
                raw,
                queue,
                family: family as u32,
                name,
                properties,
                state: Mutex::new(State {
                    pipelines: HashMap::new(),
                }),
            }));
        }
        Ok(devices)
    }
}

// Physical-device identity is distinct from pipelineCacheUUID (which identifies
// compatible pipeline binaries and cannot correlate different driver APIs).
unsafe fn hardware_info(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
) -> HardwareInfo {
    let pci_supported = unsafe { instance.enumerate_device_extension_properties(physical) }
        .unwrap_or_default().iter().any(|extension|
            unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) } == vk::EXT_PCI_BUS_INFO_NAME
        );
    let mut ids = vk::PhysicalDeviceIDProperties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    {
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
        if pci_supported {
            properties = properties.push_next(&mut pci);
        }
        unsafe {
            instance.get_physical_device_properties2(physical, &mut properties);
        }
    }
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let total_memory_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .try_fold(0u64, |sum, heap| sum.checked_add(heap.size))
        .filter(|bytes| *bytes > 0);
    HardwareInfo {
        uuid: (ids.device_uuid != [0; 16]).then_some(ids.device_uuid),
        luid: (ids.device_luid_valid == vk::TRUE
            && ids.device_luid != [0; 8]
            && ids.device_node_mask.count_ones() == 1)
            .then_some(DeviceLuid {
                bytes: ids.device_luid,
                node_mask: ids.device_node_mask,
            }),
        pci: pci_supported
            .then_some(PciAddress {
                domain: pci.pci_domain,
                bus: pci.pci_bus,
                device: pci.pci_device,
                function: pci.pci_function,
            })
            .filter(|address| address.is_valid()),
        vendor_id: (properties.vendor_id != 0).then_some(properties.vendor_id),
        total_memory_bytes,
    }
}

impl VulkanDevice {
    fn allocate(
        self: Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        validate_buffer_values(element, values)?;
        unsafe {
            let bytes = (values.len() as u64)
                .checked_mul(4)
                .ok_or_else(|| err("buffer size overflow"))?;
            if bytes > self.properties.limits.max_storage_buffer_range as u64 {
                return Err(err("buffer exceeds device storage range"));
            }
            let info = vk::BufferCreateInfo::default()
                .size(bytes)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self.raw.create_buffer(&info, None).map_err(err)?;
            let requirements = self.raw.get_buffer_memory_requirements(buffer);
            let properties = self
                .instance
                .raw
                .get_physical_device_memory_properties(self.physical);
            let memory_type = (0..properties.memory_type_count).find(|i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[*i as usize]
                        .property_flags
                        .contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
            });

            let Some(memory_type) = memory_type else {
                self.raw.destroy_buffer(buffer, None);
                return Err(err("device lacks host coherent buffer memory"));
            };

            let memory = match self.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            ) {
                Ok(m) => m,
                Err(e) => {
                    self.raw.destroy_buffer(buffer, None);
                    return Err(err(e));
                }
            };

            let owned = Arc::new(VulkanBuffer {
                element,
                length: values.len(),
                device: self.clone(),
                buffer,
                memory,
                bytes,
            });

            self.raw
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(err)?;

            let mapped = self
                .raw
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
                .map_err(err)?;

            std::ptr::copy_nonoverlapping(values.as_ptr(), mapped.cast(), values.len());
            self.raw.unmap_memory(memory);

            Ok(owned)
        }
    }

    fn read_buffer(&self, buffer: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        let buffer = self.native_buffer(buffer)?;
        let _guard = self.state.lock().map_err(err)?;
        unsafe {
            let mapped = self
                .raw
                .map_memory(buffer.memory, 0, buffer.bytes, vk::MemoryMapFlags::empty())
                .map_err(err)?;
            let values = std::slice::from_raw_parts(mapped.cast::<u32>(), buffer.length).to_vec();
            self.raw.unmap_memory(buffer.memory);
            Ok(values)
        }
    }

    fn submit(&self, launch: &Launch<'_>) -> ExecutionResult<()> {
        launch.validate_device(self)?;
        let handle = launch.kernel();
        let grid = launch.grid();
        let workgroup = launch.workgroup();
        let args = launch.arguments();
        let limits = &self.properties.limits;
        if workgroup.contains(&0) || grid.contains(&0) {
            return Err(err("launch dimensions must be positive"));
        }
        let invocations = workgroup
            .iter()
            .try_fold(1u64, |a, &b| a.checked_mul(b as u64))
            .ok_or_else(|| err("workgroup size overflow"))?;
        if invocations > limits.max_compute_work_group_invocations as u64
            || (0..3).any(|i| {
                workgroup[i] > limits.max_compute_work_group_size[i]
                    || grid[i] > limits.max_compute_work_group_count[i]
            })
        {
            return Err(err("launch exceeds device dimensions"));
        }
        let module = handle.ir();
        let mut push = vec![];
        let mut buffers = vec![];
        for argument in args {
            match argument {
                Argument::Scalar(value) => push.push(match value {
                    Literal::Bool(v) => u32::from(*v),
                    Literal::I32(v) => *v as u32,
                    Literal::U32(v) | Literal::F32(v) => *v,
                }),
                Argument::Buffer(buffer) => buffers.push(self.native_buffer(*buffer)?),
            }
        }
        if push.len() * 4 > limits.max_push_constants_size as usize {
            return Err(err("scalar arguments exceed push constant limit"));
        }
        if buffers.len() > limits.max_per_stage_descriptor_storage_buffers as usize {
            return Err(err("buffer arguments exceed descriptor limit"));
        }
        let specialized = launch.specialized();
        specialized
            .kernel
            .check_allocation_limits(u64::MAX, limits.max_compute_shared_memory_size as u64)
            .map_err(err)?;
        let target = format!(
            "vulkan:{}:{}:{}:{}",
            self.properties.vendor_id,
            self.properties.device_id,
            self.properties.driver_version,
            self.properties.api_version
        );
        let mut options = concat!("mimir.spirv.v1:", env!("CARGO_PKG_VERSION"))
            .as_bytes()
            .to_vec();
        for v in workgroup {
            options.extend(v.to_le_bytes())
        }
        let key = specialized.variant_key(target.as_bytes(), &options);
        // This lock serializes queue access and compilation of the same variant.
        let mut state = self.state.lock().map_err(err)?;
        if let std::collections::hash_map::Entry::Vacant(entry) = state.pipelines.entry(key) {
            let words =
                mimir_codegen_spirv::compile(module, specialized, workgroup).map_err(err)?;
            let pipeline = self.pipeline(&words, buffers.len(), push.len() * 4)?;
            entry.insert(pipeline);
        }
        let pipeline = &state.pipelines[&key];
        unsafe {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: buffers.len().max(1) as u32,
            }];
            let descriptor_pool = self
                .raw
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(err)?;
            let command_pool = match self.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                None,
            ) {
                Ok(p) => p,
                Err(e) => {
                    self.raw.destroy_descriptor_pool(descriptor_pool, None);
                    return Err(err(e));
                }
            };
            let result = (|| -> ExecutionResult<()> {
                let layouts = [pipeline.descriptor];
                let set = self
                    .raw
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&layouts),
                    )
                    .map_err(err)?[0];
                let infos = buffers
                    .iter()
                    .map(|b| vk::DescriptorBufferInfo {
                        buffer: b.buffer,
                        offset: 0,
                        range: b.bytes,
                    })
                    .collect::<Vec<_>>();
                let writes = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(std::slice::from_ref(info))
                    })
                    .collect::<Vec<_>>();
                self.raw.update_descriptor_sets(&writes, &[]);
                let command = self
                    .raw
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(err)?[0];
                self.raw
                    .begin_command_buffer(
                        command,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(err)?;
                self.raw.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline,
                );
                self.raw.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[set],
                    &[],
                );
                if !push.is_empty() {
                    let bytes =
                        std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), push.len() * 4);
                    self.raw.cmd_push_constants(
                        command,
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytes,
                    );
                }
                self.raw.cmd_dispatch(command, grid[0], grid[1], grid[2]);
                let barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::HOST_READ
                            | vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE,
                    )];
                self.raw.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
                self.raw.end_command_buffer(command).map_err(err)?;
                let fence = self
                    .raw
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(err)?;
                let commands = [command];
                let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
                let submitted = self
                    .raw
                    .queue_submit(self.queue, &submits, fence)
                    .map_err(err);
                let result = match submitted {
                    Ok(()) => self
                        .raw
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(err),
                    Err(e) => Err(e),
                };
                if result.is_err() {
                    let _ = self.raw.device_wait_idle();
                }
                self.raw.destroy_fence(fence, None);
                result
            })();
            self.raw.destroy_command_pool(command_pool, None);
            self.raw.destroy_descriptor_pool(descriptor_pool, None);
            result
        }
    }

    fn pipeline(&self, words: &[u32], buffers: usize, push: usize) -> ExecutionResult<Pipeline> {
        unsafe {
            let bindings = (0..buffers)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let descriptor = self
                .raw
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(err)?;
            let layouts = [descriptor];
            let ranges = if push == 0 {
                vec![]
            } else {
                vec![vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: push as u32,
                }]
            };
            let layout = match self.raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.raw.destroy_descriptor_set_layout(descriptor, None);
                    return Err(err(e));
                }
            };
            let shader = match self
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
            {
                Ok(v) => v,
                Err(e) => {
                    self.raw.destroy_pipeline_layout(layout, None);
                    self.raw.destroy_descriptor_set_layout(descriptor, None);
                    return Err(err(e));
                }
            };
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(c"main"),
                )
                .layout(layout)];
            let result = self
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None);
            self.raw.destroy_shader_module(shader, None);
            match result {
                Ok(p) => Ok(Pipeline {
                    pipeline: p[0],
                    layout,
                    descriptor,
                }),
                Err((partial, e)) => {
                    for p in partial {
                        self.raw.destroy_pipeline(p, None)
                    }
                    self.raw.destroy_pipeline_layout(layout, None);
                    self.raw.destroy_descriptor_set_layout(descriptor, None);
                    Err(err(e))
                }
            }
        }
    }
}

impl VulkanDevice {
    fn native_buffer<'a>(
        &self,
        buffer: &'a dyn BufferBackend,
    ) -> ExecutionResult<&'a VulkanBuffer> {
        validate_buffer_owner(self, buffer)?;
        let native = buffer
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| err("buffer is not a Vulkan allocation"))?;
        // Check native ownership too: a third-party trait adapter can expose
        // a different object's Any reference or report misleading metadata.
        if !std::ptr::eq(self, native.device.as_ref()) {
            return Err(err("buffer belongs to another device"));
        }
        if buffer.element_type() != native.element || buffer.len() != native.length {
            return Err(err("buffer metadata does not match Vulkan allocation"));
        }
        Ok(native)
    }
}
impl DeviceBackend for VulkanDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn id(&self) -> DeviceId {
        self.id.clone()
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn backend_name(&self) -> &'static str {
        "vulkan"
    }
    fn priority(&self) -> usize {
        100
    }
    fn hardware_info(&self) -> HardwareInfo {
        self.hardware.clone()
    }
    fn buffer(
        self: Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        self.allocate(element, values)
    }
    fn read(&self, buffer: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        self.read_buffer(buffer)
    }
    unsafe fn launch(&self, launch: &Launch<'_>) -> ExecutionResult<()> {
        self.submit(launch)
    }
}
impl BufferBackend for VulkanBuffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn device(&self) -> Arc<dyn DeviceBackend> {
        self.device.clone()
    }
    fn element_type(&self) -> ScalarType {
        self.element
    }
    fn len(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_runtime_core::KernelHandle;
    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn concurrent_requests_compile_one_pipeline() {
        let devices = discover().unwrap();
        let device = devices.first().unwrap();
        let module = Module {
            kernels: vec![Kernel {
                routine: Routine {
                    header: Header::default(),
                    body: Body {
                        entry: BlockId(0),
                        blocks: vec![Block {
                            id: BlockId(0),
                            arguments: vec![],
                            instructions: vec![],
                            merge: None,
                            terminator: Terminator::Return(None),
                        }],
                    },
                },
                debug: DebugInfo::default(),
            }],
            functions: vec![],
        }
        .validate()
        .unwrap();
        let kernel = KernelHandle::new(Arc::new(module), 0).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let kernel = &kernel;
                scope.spawn(move || {
                    let launch =
                        Launch::new(device.as_ref(), kernel, &[], [1, 1, 1], [1, 1, 1], &[])
                            .unwrap();
                    unsafe {
                        device.launch(&launch).unwrap();
                    }
                });
            }
        });
        assert_eq!(
            device
                .as_any()
                .downcast_ref::<VulkanDevice>()
                .unwrap()
                .state
                .lock()
                .unwrap()
                .pipelines
                .len(),
            1
        );
    }
}
