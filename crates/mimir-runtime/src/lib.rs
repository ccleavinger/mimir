mod registry;
#[doc(hidden)]
pub use linkme;
pub use registry::*;

mod discovery;
mod execution;
pub use execution::*;

mod array;
pub use array::GpuArray;

/// Contracts for implementing or supplying independent backends.
pub use mimir_runtime_core as backend;
