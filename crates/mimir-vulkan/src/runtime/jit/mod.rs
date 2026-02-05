use std::sync::Arc;

use ash::vk::{self, PhysicalDeviceLimits};
use mimir_runtime::{
    generic::{
        compiler::{MimirJITCompiler, MimirPlatformBytecode},
        device::{MimirDevice, MimirPlatformDevices},
        launch::{MimirBuffer, MimirVar, Param},
        runtime::{MimirJITExecutionError, MimirJITRuntime},
    },
    jit_exec_gen_err,
    platforms::MimirPlatform,
};
use mimir_vulkan_spirv_compiler::{SpirvBytecode, VulkanSpirVCompiler};

use crate::{
    runtime::{MIMIR_VK_RUNTIME, MimirVkRuntime, SpirvCacheKey},
    vk_platform_device::MimirVkDevice,
};

// TODO: ensure the launch configuration matches with all the physical device limits
fn validate_launch_config(
    _grid_size: &[u32; 3],
    block_size: &[u32; 3],
    limits: Arc<PhysicalDeviceLimits>,
) -> Result<(), MimirJITExecutionError> {
    let total_threads = block_size[0] * block_size[1] * block_size[2];
    if total_threads > limits.max_compute_work_group_invocations {
        return Err(MimirJITExecutionError::TooManyThreadsPerBlock(
            total_threads,
            limits.max_compute_work_group_invocations,
            *block_size,
        ));
    }

    Ok(())
}

impl MimirJITRuntime for MimirVkRuntime {
    fn platform(&self) -> mimir_runtime::platforms::MimirPlatform {
        MimirPlatform::Vulkan
    }

    fn execute_bytecode(
        &self,
        device: Box<&(dyn MimirDevice + Send + Sync)>,
        bytecode: Box<dyn MimirPlatformBytecode>,
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), mimir_runtime::generic::runtime::MimirJITExecutionError> {
        let (log_device, limits) = if MimirPlatform::Vulkan == device.platform() {
            let binding = device.as_platform_device();
            let vk_device = binding.downcast_ref::<MimirVkDevice>().unwrap();
            (
                vk_device.log_device.clone(),
                vk_device.phys_dev_limits.clone(),
            )
        } else {
            return Err(jit_exec_gen_err!(
                "`{:?}` is an invalid platform device for the Mimir Vulkan backend",
                device.platform()
            ));
        };

        validate_launch_config(grid_dim, block_dim, limits)?;

        let (push_consts, buffers) =
            params
                .iter()
                .fold((Vec::new(), Vec::new()), |(mut pcs, mut bufs), param| {
                    match param {
                        Param::Var(var) => pcs.push(var.clone()),
                        Param::Buffer(buffer) => bufs.push(buffer),
                    }
                    (pcs, bufs)
                });

        let binding = bytecode.bytecode();

        let spirv = if MimirPlatform::Vulkan == bytecode.platform() {
            // should be safe, if platform is reported to be a MimirPlatform::Vulkan
            &binding.downcast_ref::<SpirvBytecode>().unwrap().bytes
        } else {
            return Err(jit_exec_gen_err!(
                "`{:?}` is an invalid platform bytecode for the Mimir Vulkan backend",
                bytecode.platform()
            ));
        };

        let shader_module = {
            let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(spirv);

            unsafe { log_device.create_shader_module(&shader_module_create_info, None) }
                .map_err(|x| MimirJITExecutionError::Generic(x.to_string()))?
        };

        let descriptor_set_layout = {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = buffers
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();

            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe {
                self.device
                    .lock()
                    .create_descriptor_set_layout(&create_info, None)
            }
            .map_err(|err| MimirJITExecutionError::Generic(err.to_string()))
        }?;

        let push_constant_size = calculate_push_constant_size(&push_consts);
        let push_constant_ranges = if push_constant_size > 0 {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_size),
            ]
        } else {
            Vec::new()
        };

        let pipeline_layout = {
            let layouts = [descriptor_set_layout];
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push_constant_ranges);
            unsafe {
                self.device
                    .lock()
                    .create_pipeline_layout(&create_info, None)
            }
            .map_err(|err| MimirJITExecutionError::Generic(err.to_string()))
        }?;

        let pipeline = {
            let mut specialization = SpecializationInfoBuilder::new();
            specialization.add(0, block_dim[0]);
            specialization.add(1, block_dim[1]);
            specialization.add(2, block_dim[2]);

            let map_entries = specialization.get_map_entries();
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&map_entries)
                .data(&specialization.data);

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main") // Entry point
                .specialization_info(&spec_info);

            let create_info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);

            unsafe {
                log_device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            }
            .map_err(|(_, e)| MimirJITExecutionError::Generic(e.to_string()))?[0] // We create one pipeline
        };

        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffers.len() as u32)]; // One descriptor for each buffer
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1); // We need one descriptor set
            unsafe {
                self.device
                    .lock()
                    .create_descriptor_pool(&create_info, None)
            }
            .map_err(|err| MimirJITExecutionError::Generic(err.to_string()))
        }?;

        let descriptor_set = {
            let set_layouts_array = [descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts_array);
            unsafe { self.device.lock().allocate_descriptor_sets(&alloc_info) }
                .map_err(|err| MimirJITExecutionError::Generic(err.to_string()))?[0]
        };

        // Update descriptor sets
        let descriptor_buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|buffer| {
                let vk_buffer = match buffer {
                    MimirBuffer::Int32(buf) => {
                        (**buf).get_inner_as_any().downcast_ref::<vk::Buffer>()
                    }
                    MimirBuffer::Int64(buf) => {
                        (**buf).get_inner_as_any().downcast_ref::<vk::Buffer>()
                    }
                    MimirBuffer::Float32(buf) => {
                        (**buf).get_inner_as_any().downcast_ref::<vk::Buffer>()
                    }
                    MimirBuffer::Bool(buf) => {
                        (**buf).get_inner_as_any().downcast_ref::<vk::Buffer>()
                    }
                }
                .ok_or_else(|| {
                    jit_exec_gen_err!("Failed to downcast a buffer to become a vulkan buffer!")
                })?;
                Ok(vk::DescriptorBufferInfo::default()
                    .buffer(*vk_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE))
            })
            .collect::<Result<Vec<_>, MimirJITExecutionError>>()?;

        let write_descriptor_sets = descriptor_buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();

        unsafe {
            self.device
                .lock()
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // Allows command buffers to be reset
                .queue_family_index(self.queue_family_index);
            unsafe { self.device.lock().create_command_pool(&create_info, None) }
                .map_err(|err| MimirJITExecutionError::Generic(err.to_string()))
        }?;

        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.device.lock().allocate_command_buffers(&alloc_info) }
                .map_err(|e| MimirJITExecutionError::Generic(e.to_string()))?[0]
        };

        // Record command buffer
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .lock()
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| jit_exec_gen_err!("{}", e))?;

            log_device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            log_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            if push_constant_size > 0 {
                let pc_data = create_push_constant_buffer(&push_consts);
                log_device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &pc_data,
                );
            }

            log_device.cmd_dispatch(command_buffer, grid_dim[0], grid_dim[1], grid_dim[2]);

            // Memory barrier to ensure shader writes are visible to the host
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Writes from compute shader
                .dst_access_mask(vk::AccessFlags::HOST_READ); // Make visible to host for reading

            log_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER, // Stage where writes occurred
                vk::PipelineStageFlags::HOST, // Stage where reads will occur (or just ensure visibility)
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[], // No buffer memory barriers needed if using a global memory barrier
                &[], // No image memory barriers needed
            );

            self.device
                .lock()
                .end_command_buffer(command_buffer)
                .map_err(|e| jit_exec_gen_err!("{}", e))?;
        }

        // Create a fence
        let fence = unsafe {
            self.device
                .lock()
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| jit_exec_gen_err!("{}", e))?;

        // Submit to queue
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            log_device
                .queue_submit(
                    *self.queue.lock(),
                    std::slice::from_ref(&submit_info),
                    fence, // Use the created fence
                )
                .map_err(|e| jit_exec_gen_err!("{}", e))?;

            // Wait for the fence
            log_device
                .wait_for_fences(
                    &[fence],
                    true,     // Wait for all fences
                    u64::MAX, // Timeout (effectively infinite)
                )
                .map_err(|e| jit_exec_gen_err!("{}", e))?;

            // Reset the fence
            log_device
                .reset_fences(&[fence])
                .map_err(|e| jit_exec_gen_err!("{}", e))?;
        }

        // Cleanup
        unsafe {
            log_device.destroy_fence(fence, None); // Destroy the fence
            self.device.lock().destroy_command_pool(command_pool, None);
            // Descriptor sets are freed when the pool is destroyed
            self.device
                .lock()
                .destroy_descriptor_pool(descriptor_pool, None);
            self.device.lock().destroy_pipeline(pipeline, None);
            self.device
                .lock()
                .destroy_pipeline_layout(pipeline_layout, None);
            self.device
                .lock()
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            self.device
                .lock()
                .destroy_shader_module(shader_module, None);
        }

        Ok(())
    }

    fn get_platform_devices(&self) -> Box<dyn MimirPlatformDevices> {
        Box::new(self.platform_devices.clone())
    }

    fn new_jit_compiler(&self) -> Box<dyn MimirJITCompiler> {
        Box::new(VulkanSpirVCompiler::new())
    }

    fn get_bytecode(
        &self,
        name: &str,
        const_generics: &[u32],
    ) -> Result<Option<Box<dyn MimirPlatformBytecode>>, MimirJITExecutionError> {
        let runtime = match MIMIR_VK_RUNTIME.as_ref() {
            Some(runtime) => runtime,
            None => {
                return Err(MimirJITExecutionError::RuntimeUnsupported(
                    MimirPlatform::Vulkan,
                ));
            }
        };

        let mut spirv_bind = runtime.spirv.lock();
        let spirv = spirv_bind.get_mut();
        let bytes_opt = spirv.get(&SpirvCacheKey::new(name, const_generics));
        Ok(match bytes_opt {
            Some(bytes) => Some(Box::new(mimir_vulkan_spirv_compiler::SpirvBytecode {
                bytes: bytes.clone(),
            })),
            None => None,
        })
    }

    fn insert_bytecode(
        &self,
        name: &str,
        const_generics: &[u32],
        bytecode: Box<&dyn MimirPlatformBytecode>,
    ) -> Result<(), MimirJITExecutionError> {
        if bytecode.platform() == MimirPlatform::Vulkan {
            let binding = bytecode.bytecode();
            let bytes = match binding.downcast_ref::<SpirvBytecode>() {
                Some(spv) => spv,
                None => return Err(MimirJITExecutionError::InnapropriateBytecode),
            }
            .bytes
            .clone();

            let runtime = match MIMIR_VK_RUNTIME.as_ref() {
                Some(runtime) => runtime,
                None => {
                    return Err(MimirJITExecutionError::RuntimeUnsupported(
                        MimirPlatform::Vulkan,
                    ));
                }
            };
            runtime
                .spirv
                .lock()
                .get_mut()
                .insert(SpirvCacheKey::new(name, const_generics), bytes);
            Ok(())
        } else {
            Err(jit_exec_gen_err!(
                "Invalid bytecode platform of `{:?}` was submitted to the Mimir vulkan backend.",
                bytecode.platform()
            ))
        }
    }
}

fn calculate_push_constant_size(pc_vec: &[MimirVar]) -> u32 {
    let mut size = 0;
    for pc in pc_vec {
        size += match pc {
            MimirVar::Int32(_) => std::mem::size_of::<i32>() as u32,
            MimirVar::Int64(_) => std::mem::size_of::<i64>() as u32,
            MimirVar::Float32(_) => std::mem::size_of::<f32>() as u32,
            MimirVar::Bool(_) => 4, // bool aligned to 4 bytes
        }
    }
    size
}

fn create_push_constant_buffer(pc_vec: &[MimirVar]) -> Vec<u8> {
    let total_size = pc_vec.iter().fold(0, |acc, pc| {
        acc + match pc {
            MimirVar::Int32(_) => std::mem::size_of::<i32>(),
            MimirVar::Int64(_) => std::mem::size_of::<i64>(),
            MimirVar::Float32(_) => std::mem::size_of::<f32>(),
            MimirVar::Bool(_) => 4, // bool normalized to 4 bytes
        }
    });

    let mut buffer = vec![0u8; total_size];
    let mut offset = 0;

    for pc in pc_vec {
        match pc {
            MimirVar::Int32(val) => {
                buffer[offset..(offset + 4)].copy_from_slice(&val.to_ne_bytes());
                offset += 4;
            }
            MimirVar::Int64(val) => {
                buffer[offset..(offset + 8)].copy_from_slice(&val.to_ne_bytes());
                offset += 8;
            }
            MimirVar::Float32(val) => {
                buffer[offset..(offset + 4)].copy_from_slice(&val.to_ne_bytes());
                offset += 4;
            }
            MimirVar::Bool(val) => {
                // store as u32
                let v = if *val { 1u32 } else { 0u32 };
                buffer[offset..offset + 4].copy_from_slice(&v.to_ne_bytes());
                offset += 4;
            }
        }
    }

    buffer
}

struct SpecializationInfoBuilder {
    pub constant_ids: Vec<u32>,
    pub data: Vec<u8>,
    pub sizes: Vec<usize>,
}

impl SpecializationInfoBuilder {
    pub fn new() -> Self {
        SpecializationInfoBuilder {
            constant_ids: Vec::new(),
            data: Vec::new(),
            sizes: Vec::new(),
        }
    }

    pub fn add<T: Copy>(&mut self, constant_id: u32, value: T) {
        let size = std::mem::size_of::<T>();
        let mut bytes = vec![0u8; size];

        unsafe {
            let value_ptr = &value as *const T as *const u8;
            std::ptr::copy_nonoverlapping(value_ptr, bytes.as_mut_ptr(), size);
        }

        self.constant_ids.push(constant_id);
        self.sizes.push(size);
        self.data.extend_from_slice(&bytes);
    }

    pub fn get_map_entries(&self) -> Vec<vk::SpecializationMapEntry> {
        let mut entries = Vec::with_capacity(self.constant_ids.len());
        let mut offset = 0;

        for i in 0..self.constant_ids.len() {
            entries.push(vk::SpecializationMapEntry {
                constant_id: self.constant_ids[i],
                offset,
                size: self.sizes[i],
            });

            offset += self.sizes[i] as u32;
        }

        entries
    }
}
