use ash::vk::{self, BufferCreateInfo};
use mimir_runtime::generic::array::MimirArray;
use vk_mem::{Alloc, AllocationCreateFlags};
use std::marker::PhantomData;

use super::backend::BACKEND;

pub struct VulkanArray<T: Copy + Sized + 'static> {
    //device: ash::Device,
    size: usize,
    //properties: vk::MemoryPropertyFlags,
    //usage: vk::BufferUsageFlags,
    buffer: vk::Buffer,
    allocation: Option<vk_mem::Allocation>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Sized + 'static> VulkanArray<T> {
    fn from_host<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let data_vec: Vec<T> = iter.into_iter().collect();
        let size = data_vec.len();
        let buffer_size_bytes = (size * std::mem::size_of::<T>()) as vk::DeviceSize;

        if buffer_size_bytes == 0 {
            return VulkanArray {
                size,
                buffer: vk::Buffer::null(),
                allocation: None,
                _marker: PhantomData,
            };
        }

        let properties = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let allocator = BACKEND.allocator.lock().unwrap();        let buffer_info = BufferCreateInfo::default()
            .size(buffer_size_bytes)
            .usage(usage);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            required_flags: properties,
            usage: vk_mem::MemoryUsage::Auto,
            ..Default::default()
        };
        
        let (buffer, allocation) = match unsafe { allocator.create_buffer(&buffer_info, &allocation_create_info) } {
            Ok((buffer, allocation)) => (buffer, allocation),
            Err(e) => panic!("Failed to create buffer: {:?}", e),
        };

        let allocation_details = allocator.get_allocation_info(&allocation);
        if !allocation_details.mapped_data.is_null() {
            unsafe {
                // Copy data from the collected Vec into the mapped buffer memory
                std::ptr::copy_nonoverlapping(data_vec.as_ptr(), allocation_details.mapped_data as *mut T, size);
            }

            // Ensure data is flushed if memory is not HOST_COHERENT
            // Check actual memory properties if possible, or rely on requested 'properties'
            // For HOST_VISIBLE memory, vk-mem might provide HOST_COHERENT if available.
            // If not HOST_COHERENT, a flush is needed for device to see host writes.
            let memory_properties = unsafe { allocator.get_memory_properties() };
            let memory_type = &memory_properties.memory_types[allocation_details.memory_type as usize];
            if !memory_type.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
                let flush_info = vk::MappedMemoryRange::default()
                    .memory(allocation_details.device_memory)
                    .offset(allocation_details.offset)
                    .size(vk::WHOLE_SIZE); // Or buffer_size_bytes

                let device = BACKEND.device.lock().unwrap();
                unsafe {
                    device
                        .flush_mapped_memory_ranges(&[flush_info])
                        .expect("Failed to flush memory");
                }
            }
        } else if size > 0 {
             panic!("Buffer mapped_data is null despite MAPPED flag and non-zero size.");
        }


        VulkanArray {
            size,
            buffer,
            allocation: Some(allocation),
            _marker: PhantomData,
        }
    }
}

impl<T: Copy + Sized + 'static> Drop for VulkanArray<T> {
    fn drop(&mut self) {
        // Skip cleanup if buffer is null or allocation is None
        if self.buffer == vk::Buffer::null() || self.allocation.is_none() {
            return;
        }
        
        // Clean up the buffer and its associated memory
        let allocator = BACKEND.allocator.lock().unwrap();
        if let Some(mut allocation) = self.allocation.take() {
            // Free buffer and allocation in one call
            unsafe {
                allocator.destroy_buffer(self.buffer, &mut allocation);
            }
            // Make sure the buffer is marked as invalid after destruction
            self.buffer = vk::Buffer::null();
        }
        
        // No need to manually drop self.allocation as we've taken ownership with .take()
    }
}

impl<T: Copy + Sized + 'static> MimirArray<T> for VulkanArray<T> {
    fn len(&self) -> usize {
        self.size
    }

    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        Self: Sized,
    {
        VulkanArray::from_host(iter)
    }

    fn get_inner_as_any(&self) -> &dyn std::any::Any {
        &self.buffer
    }

    fn to_iter(&self) -> Box<dyn Iterator<Item = T>> {
        let allocator = BACKEND.allocator.lock().unwrap();
        let allocation_info = allocator.get_allocation_info(self.allocation.as_ref().unwrap());
        let mapped_ptr = allocation_info.mapped_data;

        assert!(!mapped_ptr.is_null(), "Failed to get mapped data pointer. Ensure buffer was allocated with MAPPED flag and HOST_VISIBLE property.");

        let data = unsafe {
            std::slice::from_raw_parts(
                mapped_ptr as *const T,
                self.size,
            )
        };
        Box::new(data.iter().copied())
    }
}