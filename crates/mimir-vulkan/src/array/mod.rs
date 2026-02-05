use std::{
    marker::PhantomData,
    sync::{Arc, LazyLock},
};

use ash::vk;
use mimir_runtime::{
    generic::{
        array::{MimirArray, MimirArrayError, MimirArrayFactory},
        device::MimirDevice,
    },
    platforms::MimirPlatform,
};
use vk_mem::Alloc;

use crate::vk_platform_device::MimirVkDevice;

pub struct MimirVkArray<T: Copy + Sized + 'static> {
    size: usize,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    device: Arc<MimirVkDevice>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Sized + 'static> MimirVkArray<T> {
    fn from_host<I>(
        device: &(dyn MimirDevice + Send + Sync),
        iter: I,
    ) -> Result<Self, MimirArrayError>
    where
        I: IntoIterator<Item = T>,
    {
        let data_vec = iter.into_iter().collect::<Vec<_>>();
        let size = data_vec.len();
        let buffer_size_bytes = (size * std::mem::size_of::<T>()) as vk::DeviceSize;

        let vk_device = if MimirPlatform::Vulkan == device.platform() {
            let binding = device.as_platform_device();
            Arc::new(binding.downcast_ref::<MimirVkDevice>().unwrap().clone())
        } else {
            return Err(MimirArrayError::InvalidPlatformError(device.platform()));
        };

        if buffer_size_bytes == 0 {
            return Err(MimirArrayError::UnsizedError);
        }

        let properties = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let allocator = vk_device.allocator.clone();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size_bytes)
            .usage(usage);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            required_flags: properties,
            usage: vk_mem::MemoryUsage::Auto,
            ..Default::default()
        };

        let (buffer, allocation) =
            match unsafe { allocator.create_buffer(&buffer_info, &allocation_create_info) } {
                Ok((buffer, allocation)) => (buffer, allocation),
                Err(e) => {
                    return Err(MimirArrayError::Generic(format!(
                        "Failed to create buffer: {:?}",
                        e
                    )));
                }
            };

        let allocation_details = allocator.get_allocation_info(&allocation);
        if !allocation_details.mapped_data.is_null() {
            unsafe {
                // Copy data from the collected Vec into the mapped buffer memory
                std::ptr::copy_nonoverlapping(
                    data_vec.as_ptr(),
                    allocation_details.mapped_data as *mut T,
                    size,
                );
            }

            // Ensure data is flushed if memory is not HOST_COHERENT
            // Check actual memory properties if possible, or rely on requested 'properties'
            // For HOST_VISIBLE memory, vk-mem might provide HOST_COHERENT if available.
            // If not HOST_COHERENT, a flush is needed for device to see host writes.
            let memory_properties = unsafe { allocator.get_memory_properties() };
            let memory_type =
                &memory_properties.memory_types[allocation_details.memory_type as usize];
            if !memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                let flush_info = vk::MappedMemoryRange::default()
                    .memory(allocation_details.device_memory)
                    .offset(allocation_details.offset)
                    .size(vk::WHOLE_SIZE); // Or buffer_size_bytes

                let device = vk_device.log_device.clone();
                unsafe {
                    device
                        .flush_mapped_memory_ranges(&[flush_info])
                        .expect("Failed to flush memory");
                }
            }
        } else if size > 0 {
            return Err(MimirArrayError::Generic(
                "Buffer mapped_data is null despite MAPPED flag and non-zero size.".to_owned(),
            ));
        }

        Ok(Self {
            size,
            buffer,
            allocation,
            device: vk_device,
            _marker: PhantomData,
        })
    }
}

impl<T: Copy + Sized + 'static> Drop for MimirVkArray<T> {
    fn drop(&mut self) {
        // Skip cleanup if buffer is null or allocation is None
        // Should be impossible but better safe than sorry
        if self.buffer == vk::Buffer::null() {
            return;
        }

        // Clean up the buffer and its associated memory
        let allocator = self.device.allocator.clone();
        {
            // Free buffer and allocation in one call
            unsafe {
                allocator.destroy_buffer(self.buffer, &mut self.allocation);
            }
            // Make sure the buffer is marked as invalid after destruction
            self.buffer = vk::Buffer::null();
        }

        // No need to manually drop self.allocation as we've taken ownership with .take()
    }
}

impl<T: Copy + Sized + 'static> MimirArray<T> for MimirVkArray<T> {
    fn len(&self) -> usize {
        self.size
    }

    fn get_inner_as_any(&self) -> &dyn std::any::Any {
        &self.buffer
    }

    fn to_iter(&self) -> Box<dyn Iterator<Item = T>> {
        let allocator = self.device.allocator.clone();
        let allocation_info = allocator.get_allocation_info(&self.allocation);
        let mapped_ptr = allocation_info.mapped_data;

        assert!(
            !mapped_ptr.is_null(),
            "Failed to get mapped data pointer. Ensure buffer was allocated with MAPPED flag and HOST_VISIBLE property."
        );

        let data = unsafe { std::slice::from_raw_parts(mapped_ptr as *const T, self.size) };
        Box::new(data.iter().copied())
    }

    fn get_device(&self) -> Arc<dyn MimirDevice> {
        self.device.clone()
    }
}

pub static MIMIR_VK_ARR_FACTORY: LazyLock<MimirVkArrFactory> =
    LazyLock::new(MimirVkArrFactory::default);

#[derive(Default)]
pub struct MimirVkArrFactory;

impl MimirArrayFactory for MimirVkArrFactory {
    fn create_from_iter_device<T: Copy + Send + Sync + Sized + 'static, I>(
        &self,
        mimir_device: Arc<&(dyn MimirDevice + Send + Sync)>,
        iter: I,
    ) -> Result<Arc<dyn MimirArray<T> + Send + Sync>, MimirArrayError>
    where
        I: IntoIterator<Item = T>,
    {
        Ok(Arc::new(MimirVkArray::from_host(*mimir_device, iter)?))
    }

    fn platform(&self) -> MimirPlatform {
        MimirPlatform::Vulkan
    }
}
