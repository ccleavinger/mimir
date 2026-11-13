mod registry;
#[doc(hidden)]
pub use linkme;
pub use registry::*;

mod execution;
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
mod vulkan;
pub use execution::*;

mod array;
pub use array::GpuArray;
