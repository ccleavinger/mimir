use crate::vulkan::backend::BACKEND;

use super::error::VulkanError;
use ash::vk::{self};
use mimir_ast::MimirGlobalAST;
use mimir_runtime::generic::launch::{MimirBuffer, MimirLaunch, MimirPushConst, Param};
use log;
use spirv_compiler::compile_to_spirv; // Import the log crate

pub struct VulkanKernel;

impl VulkanKernel {
    fn calculate_push_constant_size(pc_vec: &[MimirPushConst]) -> u32 {
        let mut size = 0;
        for pc in pc_vec {
            size += match pc {
                MimirPushConst::Int32(_) => std::mem::size_of::<i32>() as u32,
                MimirPushConst::Int64(_) => std::mem::size_of::<i64>() as u32,
                MimirPushConst::Float32(_) => std::mem::size_of::<f32>() as u32,
                MimirPushConst::Bool(_) => 4, // bool aligned to 4 bytes
            }
        }
        size
    }

    fn create_push_constant_buffer(pc_vec: &[MimirPushConst]) -> Vec<u8> {
        let total_size = pc_vec.iter().fold(0, |acc, pc| {
            acc + match pc {
                MimirPushConst::Int32(_) => std::mem::size_of::<i32>(),
                MimirPushConst::Int64(_) => std::mem::size_of::<i64>(),
                MimirPushConst::Float32(_) => std::mem::size_of::<f32>(),
                MimirPushConst::Bool(_) => 4, // bool normalized to 4 bytes
            }
        });

        let mut buffer = vec![0u8; total_size];
        let mut offset = 0;

        for pc in pc_vec {
            match pc {
                MimirPushConst::Int32(val) => {
                    buffer[offset..(offset+4)].copy_from_slice(&val.to_ne_bytes());
                    offset += 4;
                },
                MimirPushConst::Int64(val) => {
                    buffer[offset..(offset+8)].copy_from_slice(&val.to_ne_bytes());
                    offset += 8;
                },
                MimirPushConst::Float32(val) => {
                    buffer[offset..(offset+4)].copy_from_slice(&val.to_ne_bytes());
                    offset += 4;
                },
                MimirPushConst::Bool(val) => {
                    // store as u32
                    let v = if *val {1u32} else {0u32};
                    buffer[offset..offset+4].copy_from_slice(&v.to_ne_bytes());
                    offset += 4;
                },
            }
        }

        buffer
    }
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
            sizes: Vec::new()
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


impl MimirLaunch<VulkanError> for VulkanKernel {
    fn launch_ast(
        ast: &MimirGlobalAST,
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), VulkanError> {
        
        let (push_consts, buffers) = {
            let mut push_consts = Vec::new();
            let mut buffers = Vec::new();

            for param in params {
                match param {
                    Param::PushConst(pc) => push_consts.push(pc.clone()),
                    Param::Buffer(buffer) => buffers.push(buffer),
                }
            }

            (push_consts, buffers)
        };

        // Revised SPIR-V caching logic:
        // First, try to get the SPIR-V code from the cache.
        // We clone it here if found, so the lock is released quickly.
        let cached_spirv = {
            let cache_guard = BACKEND.spirv.lock().unwrap();
            cache_guard.get(&ast.name).cloned()
        };

        let spirv = match cached_spirv {
            Some(code) => code, // Use the already cloned code.
            None => {
                // Not in cache. Compile outside the lock.
                let compiled_spirv_code = compile_to_spirv(ast)
                    .map_err(|e| VulkanError::SpirvCompilation(e.to_string()))?;

                // Lock again to insert.
                let mut cache_guard = BACKEND.spirv.lock().unwrap();
                // Use entry API: inserts compiled_spirv_code if ast.name is not present,
                // otherwise compiled_spirv_code is dropped and it returns a mutable reference
                // to the existing value.
                // Then, clone the value from the cache (either newly inserted or existing).
                cache_guard.entry(ast.name.clone())
                           .or_insert(compiled_spirv_code)
                           .clone()
            }
        };

        let shader_module = {
            let shader_module_create_info = vk::ShaderModuleCreateInfo::default()
                .code(&spirv);

            unsafe {
                BACKEND.device.lock().unwrap().create_shader_module(
                    &shader_module_create_info,
                    None
                )
            }?
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
            unsafe { BACKEND.device.lock().unwrap().create_descriptor_set_layout(&create_info, None) }?
        };

        let push_constant_size = Self::calculate_push_constant_size(&push_consts);
        let push_constant_ranges = if push_constant_size > 0 {
            vec![vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size)]
        } else {
            Vec::new()
        };

        let pipeline_layout = {
            let layouts = [descriptor_set_layout];
            let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);
            unsafe { BACKEND.device.lock().unwrap().create_pipeline_layout(&create_info, None) }?
        };

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
                BACKEND.device.lock().unwrap().create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[create_info],
                    None,
                )
            }
            .map_err(|(_, e)| VulkanError::VkResult(e))?[0] // We create one pipeline
        };

        let descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffers.len() as u32)]; // One descriptor for each buffer
            let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1); // We need one descriptor set
            unsafe { BACKEND.device.lock().unwrap().create_descriptor_pool(&create_info, None) }?
        };

        let descriptor_set = {
            let set_layouts_array = [descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts_array);
            unsafe { BACKEND.device.lock().unwrap().allocate_descriptor_sets(&alloc_info) }?[0]
        };

        // Update descriptor sets
        let descriptor_buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|buffer| {
                let vk_buffer = match buffer {
                    MimirBuffer::Int32(buf) => (**buf).get_inner_as_any()
                        .downcast_ref::<vk::Buffer>(),
                    MimirBuffer::Int64(buf) => (**buf).get_inner_as_any()
                        .downcast_ref::<vk::Buffer>(),
                    MimirBuffer::Float32(buf) => (**buf).get_inner_as_any()
                        .downcast_ref::<vk::Buffer>(),
                    MimirBuffer::Bool(buf) => (**buf).get_inner_as_any()
                        .downcast_ref::<vk::Buffer>(),
                }.ok_or(VulkanError::BufferDowncast)?;
                Ok(vk::DescriptorBufferInfo::default()
                    .buffer(*vk_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE))
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;

        let write_descriptor_sets: Vec<vk::WriteDescriptorSet> = descriptor_buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe {
            BACKEND
            .device
            .lock().unwrap()
            .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // Allows command buffers to be reset
            .queue_family_index(BACKEND.queue_family_index);
            unsafe { BACKEND.device.lock().unwrap().create_command_pool(&create_info, None) }?
        };

        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
            unsafe { BACKEND.device.lock().unwrap().allocate_command_buffers(&alloc_info) }?[0]
        };

        // Record command buffer
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            BACKEND
            .device
            .lock().unwrap()
            .begin_command_buffer(command_buffer, &begin_info)?;

            BACKEND.device
                .lock().unwrap()
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);

            BACKEND.device.lock().unwrap().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            if push_constant_size > 0 {
                let pc_data = Self::create_push_constant_buffer(&push_consts);
                BACKEND.device.lock().unwrap().cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &pc_data,
                );
            }

            BACKEND.device
                .lock().unwrap()
                .cmd_dispatch(command_buffer, grid_dim[0], grid_dim[1], grid_dim[2]);

            // Memory barrier to ensure shader writes are visible to the host
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE) // Writes from compute shader
                .dst_access_mask(vk::AccessFlags::HOST_READ);   // Make visible to host for reading

            BACKEND.device.lock().unwrap().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER, // Stage where writes occurred
                vk::PipelineStageFlags::HOST,          // Stage where reads will occur (or just ensure visibility)
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[], // No buffer memory barriers needed if using a global memory barrier
                &[], // No image memory barriers needed
            );

            BACKEND.device.lock().unwrap().end_command_buffer(command_buffer)?;
        }

        // Create a fence
        let fence = unsafe {
            BACKEND.device.lock().unwrap().create_fence(&vk::FenceCreateInfo::default(), None)
        }?;

        // Submit to queue
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            BACKEND.device.lock().unwrap().queue_submit(
                *BACKEND.queue.lock().unwrap(),
                std::slice::from_ref(&submit_info),
                fence, // Use the created fence
            )?;

            // Wait for the fence
            BACKEND.device.lock().unwrap().wait_for_fences(
                &[fence],
                true, // Wait for all fences
                u64::MAX, // Timeout (effectively infinite)
            )?;

            // Reset the fence
            BACKEND.device.lock().unwrap().reset_fences(&[fence])?;
        }

        // Cleanup
        unsafe {
            BACKEND.device.lock().unwrap().destroy_fence(fence, None); // Destroy the fence
            BACKEND.device.lock().unwrap().destroy_command_pool(command_pool, None);
            // Descriptor sets are freed when the pool is destroyed
            BACKEND.device.lock().unwrap().destroy_descriptor_pool(descriptor_pool, None);
            BACKEND.device.lock().unwrap().destroy_pipeline(pipeline, None);
            BACKEND.device
                .lock().unwrap()
                .destroy_pipeline_layout(pipeline_layout, None);
            BACKEND.device
                .lock().unwrap()
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
            BACKEND.device
                .lock().unwrap()
                .destroy_shader_module(shader_module, None);
        }



        Ok(())
        
    }

    fn launch_kernel_name(
        kernel_name: &str,
        grid_dim: &[u32; 3],
        block_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), VulkanError> {
        // TODO: add cfg_if!() for bin and json modes, rn just bin
        const FILEPATH: &str = "mimir.bin";
        log::debug!("Launching kernel by name: {}", kernel_name);

        let backend = &BACKEND;

        let asts = {
            let asts_guard = backend.asts.lock().unwrap().to_owned();

            match asts_guard.is_empty() {
                true => {
                    log::debug!("AST cache empty, loading from file: {}", FILEPATH);
                    MimirGlobalAST::load_from_file(FILEPATH)?
                },
                false => {
                    log::trace!("Using cached ASTs");
                    asts_guard.iter().map(|arc| (**arc).clone()).collect()
                }
            }
        };

        let ast = asts.iter().find(|ast| ast.name == kernel_name).ok_or_else(|| {
            log::error!("Kernel '{}' not found in loaded ASTs", kernel_name);
            VulkanError::Generic(format!("Kernel {} not found in ASTs", kernel_name))
        })?;

        VulkanKernel::launch_ast(ast, block_dim, grid_dim, params)
    }
}