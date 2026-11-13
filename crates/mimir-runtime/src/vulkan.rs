use crate::execution::{BufferBackend, DeviceBackend};
use crate::{Argument, Buffer, Device, ExecutionError, ExecutionResult, KernelHandle};
use ash::{Entry, vk};
use mimir_ir::flat::*;
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
pub(crate) struct VulkanDevice {
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
pub(crate) struct VulkanBuffer {
    device: Arc<VulkanDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
}
impl VulkanBuffer {
    pub(crate) fn owner(&self) -> Arc<VulkanDevice> {
        self.device.clone()
    }
}
impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}
pub(crate) fn discover() -> ExecutionResult<Vec<Device>> {
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
        let mut devices = vec![];
        for physical in instance.raw.enumerate_physical_devices().map_err(err)? {
            let properties = instance.raw.get_physical_device_properties(physical);
            if properties.api_version < vk::API_VERSION_1_1 {
                continue;
            }
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
            devices.push(Device {
                inner: DeviceBackend::Vulkan(Arc::new(VulkanDevice {
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
                })),
            });
        }
        Ok(devices)
    }
}

impl VulkanDevice {
    pub fn buffer(
        self: &Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Buffer> {
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

            Ok(Buffer {
                element,
                length: values.len(),
                inner: BufferBackend::Vulkan(owned),
            })
        }
    }

    pub fn read(self: &Arc<Self>, buffer: &Buffer) -> ExecutionResult<Vec<u32>> {
        if !Arc::ptr_eq(self, &buffer.vulkan().device) {
            return Err(err("buffer belongs to another device"));
        }
        let _guard = self.state.lock().map_err(err)?;
        unsafe {
            let mapped = self
                .raw
                .map_memory(
                    buffer.vulkan().memory,
                    0,
                    buffer.vulkan().bytes,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(err)?;
            let values = std::slice::from_raw_parts(mapped.cast::<u32>(), buffer.length).to_vec();
            self.raw.unmap_memory(buffer.vulkan().memory);
            Ok(values)
        }
    }

    pub fn launch(
        self: &Arc<Self>,
        handle: &KernelHandle,
        tuning: &[u32],
        grid: [u32; 3],
        workgroup: [u32; 3],
        args: &[Argument<'_>],
    ) -> ExecutionResult<()> {
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
        let kernel = &module.module().kernels[handle.index()];
        if args.len() != kernel.routine.header.parameters.len() {
            return Err(err("kernel argument count mismatch"));
        }
        let mut push = vec![];
        let mut buffers = vec![];
        for (p, a) in kernel.routine.header.parameters.iter().zip(args) {
            match (p.ty, a) {
                (Type::Scalar(t), Argument::Scalar(v)) if t == v.ty() => push.push(match v {
                    Literal::Bool(v) => u32::from(*v),
                    Literal::I32(v) => *v as u32,
                    Literal::U32(v) | Literal::F32(v) => *v,
                }),
                (Type::Buffer { element, .. }, Argument::Buffer(b)) if element == b.element => {
                    if !Arc::ptr_eq(self, &b.vulkan().device) {
                        return Err(err("buffer belongs to another device"));
                    }
                    buffers.push(*b)
                }
                _ => return Err(err("kernel argument type mismatch")),
            }
        }
        if push.len() * 4 > limits.max_push_constants_size as usize {
            return Err(err("scalar arguments exceed push constant limit"));
        }
        if buffers.len() > limits.max_per_stage_descriptor_storage_buffers as usize {
            return Err(err("buffer arguments exceed descriptor limit"));
        }
        let specialized = module
            .specialize_kernel(handle.index(), tuning)
            .map_err(err)?;
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
                mimir_codegen_spirv::compile(module, &specialized, workgroup).map_err(err)?;
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
                        buffer: b.vulkan().buffer,
                        offset: 0,
                        range: b.vulkan().bytes,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn concurrent_requests_compile_one_pipeline() {
        let devices = discover().unwrap();
        let device = devices.first().unwrap();
        let runtime = crate::Runtime::default();
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
        };
        let hash = module.clone().validate().unwrap().kernel_hash(0).unwrap();
        let kernel = runtime.register(hash, module).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let kernel = &kernel;
                scope.spawn(move || unsafe {
                    device
                        .launch(kernel, &[], [1, 1, 1], [1, 1, 1], &[])
                        .unwrap();
                });
            }
        });
        assert_eq!(
            match &device.inner {
                DeviceBackend::Vulkan(device) => device.state.lock().unwrap().pipelines.len(),
            },
            1
        );
    }
}
