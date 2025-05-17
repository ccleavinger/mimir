use thiserror::Error;
use ash::vk;
use std::sync::PoisonError;

#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan backend initialization failed: {0}")]
    BackendInitialization(String),

    #[error("Failed to lock backend mutex")]
    BackendMutex,

    #[error("Failed to lock allocator mutex")]
    AllocatorMutex,

    #[error("Failed to lock SPIR-V map mutex")]
    SpirvMapMutex,

    #[error("Failed to lock AST map mutex")]
    AstMapMutex,

    #[error("Vulkan API call failed: {0}")]
    VkResult(#[from] vk::Result),

    // #[error("Vulkan memory allocation failed: {0}")]
    // VkMem(#[from] vk::Result),

    #[error("SPIR-V compilation failed: {0}")]
    SpirvCompilation(String),

    #[error("No suitable Vulkan device found")]
    NoDevice,

    #[error("Failed to downcast device type")]
    DeviceDowncast,

    #[error("Failed to downcast buffer type")]
    BufferDowncast,

    #[error("Push constant size {0} exceeds limit {1}")]
    PushConstantTooLarge(u32, u32),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(vk::Result),

    #[error("Kernel '{0}' not found in loaded ASTs")]
    KernelNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Generic error: {0}")]
    Generic(String),

    #[error("Device lost error")]
    DeviceLost,
}

// Helper to convert PoisonError into our specific mutex errors
impl<T> From<PoisonError<T>> for VulkanError {
    fn from(_: PoisonError<T>) -> Self {
        // We can't easily know which mutex was poisoned without more context,
        // so we might need specific handling at the call site or a more generic error.
        // For now, let's default to BackendMutex, but this might need refinement.
        VulkanError::BackendMutex // Or a new generic MutexPoison variant
    }
}

// Allow converting String errors easily, though specific variants are preferred.
impl From<String> for VulkanError {
    fn from(s: String) -> Self {
        VulkanError::Generic(s)
    }
}

// Allow converting &str errors easily.
impl From<&str> for VulkanError {
    fn from(s: &str) -> Self {
        VulkanError::Generic(s.to_string())
    }
}
