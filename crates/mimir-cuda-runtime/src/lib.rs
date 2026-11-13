#![doc = include_str!("../README.md")]
use mimir_runtime_core::{Backend, DeviceBackend, ExecutionResult};
use std::sync::Arc;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_pointer_width = "64"
))]
mod cuda;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_pointer_width = "64"
))]
mod driver;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_pointer_width = "64"
))]
pub use cuda::{CudaBuffer, CudaDevice};

pub struct CudaBackend;
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn discover(&self) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
        discover()
    }
}

/// Discover CUDA contexts for compute capability 5.0 or later. Missing drivers,
/// no NVIDIA devices, and unsupported platforms return an empty list. Driver
/// initialization failures other than CUDA_ERROR_NO_DEVICE are reported.
pub fn discover() -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
    #[cfg(all(
        any(target_os = "linux", target_os = "windows"),
        target_pointer_width = "64"
    ))]
    {
        cuda::discover()
    }
    #[cfg(not(all(
        any(target_os = "linux", target_os = "windows"),
        target_pointer_width = "64"
    )))]
    {
        Ok(Vec::new())
    }
}
