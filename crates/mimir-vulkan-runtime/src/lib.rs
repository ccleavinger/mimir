//! Standalone Vulkan backend. Depends on core contracts, never the unified runtime.
//!
//! Device discovery, allocation and readback can be used without `mimir-runtime`:
//! ```no_run
//! use mimir_runtime_core::{Backend, GpuScalar};
//! use mimir_vulkan_runtime::VulkanBackend;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let devices = VulkanBackend.discover()?;
//! if let Some(device) = devices.first() {
//!     let buffer = device.clone().buffer(<u32 as GpuScalar>::TYPE, &[1, 2, 3])?;
//!     assert_eq!(device.read(buffer.as_ref())?, vec![1, 2, 3]);
//! }
//! # Ok(())
//! # }
//! ```
//! Launches use core's checked `Launch` request and retain its unsafe execution
//! obligations. Compilation caches and queue synchronization belong to each
//! Vulkan device context. Linux, Android and Windows support Vulkan 1.1 compute;
//! discovery returns no devices on other targets.
//! Vulkan reports priority 100 and best-effort physical UUID, Windows LUID/node,
//! PCI address (when VK_EXT_pci_bus_info is supported), vendor and device-local
//! memory totals. Unified discovery uses these to resolve duplicate hardware;
//! standalone discovery returns every Vulkan context. Pipeline-cache UUIDs are
//! deliberately excluded from physical-device identity.
use mimir_runtime_core::{Backend, DeviceBackend, ExecutionResult};
use std::sync::Arc;

#[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
mod vulkan;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
pub use vulkan::{VulkanBuffer, VulkanDevice};

pub struct VulkanBackend;
impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn discover(&self) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
        discover()
    }
}
/// Discover Vulkan 1.1 compute contexts. Unsupported platforms return no devices.
pub fn discover() -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
    {
        vulkan::discover()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
    {
        Ok(Vec::new())
    }
}
