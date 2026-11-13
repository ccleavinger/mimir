//! Unified entry point for Mimir.
pub use mimir_ir::flat as ir;
#[doc(hidden)]
pub use mimir_runtime::linkme;
pub use mimir_runtime::{
    KernelHandle, KernelHash, LINKED_MODULES, LinkedModule, RegistrationError, Runtime,
};

/// Embed a serialized IR module without runtime file access.
///
/// The byte expression must be static (for example, a byte literal or
/// `include_bytes!`). Registration validates its contents when the runtime starts.
#[macro_export]
macro_rules! embed_ir {
    ($hashes:expr, $bytes:expr) => {
        const _: () = {
            #[$crate::linkme::distributed_slice($crate::LINKED_MODULES)]
            #[linkme(crate = $crate::linkme)]
            static MODULE: $crate::LinkedModule = $crate::LinkedModule {
                hashes: $hashes,
                bytes: $bytes,
            };
        };
    };
}

pub use mimir_macros::mimir_global;

pub use mimir_runtime::{
    Argument, Buffer, Device, DeviceLuid, ExecutionError, ExecutionResult, HardwareInfo, PciAddress,
};

pub use mimir_macros::launch;
pub use mimir_macros::mimir_module;
pub use mimir_runtime::GpuArray;
pub use mimir_runtime::GpuScalar;
pub use mimir_runtime::{AsArgument, LaunchError};
pub use mimir_runtime::{
    MIMIR_RUNTIME, default_device, default_runtime, launch_default, launch_on_default,
};

pub use mimir_runtime::GpuBuffer;
