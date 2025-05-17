use vulkan::{
    vulkan_kernel::VulkanKernel,
    vulkan_array::VulkanArray
};

pub mod vulkan;

//    bring your own backend generic types
/* ------------------------------------------ */
pub type GPUArray<T> = VulkanArray<T>;

pub type MimirKernel = VulkanKernel;

// pub type GPUDevice = VulkanDevice;
/* ------------------------------------------ */

pub use mimir_runtime::generic::array::MimirArray;
pub use mimir_runtime::generic::launch::MimirLaunch;
pub use mimir_runtime::generic::device::MimirDevice;
pub use mimir_runtime::generic::device::MimirDeviceManager;
pub use mimir_runtime::generic::launch::Param;
pub use mimir_runtime::generic::launch::MimirPushConst;
pub use mimir_runtime::generic::launch::MimirBuffer;