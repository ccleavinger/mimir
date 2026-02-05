use mimir_ir::ir::MimirIRData;
use std::sync::Arc;
use thiserror::Error;

use crate::{generic::runtime::MimirJITExecutionError, platforms::MimirPlatform};

use super::array::MimirArray;

pub trait MimirLaunch {
    fn launch_kernel(
        &self,
        ir: &MimirIRData,
        name: String, // name of the kernel
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
        cgs: &[u32], // currently const generics are only unsized 32 bit integers
    ) -> Result<(), MimirLaunchError>;

    fn set_device(&self, idx: usize) -> Result<(), LaunchDeviceError>;
}

#[derive(Error, Debug)]
pub enum MimirLaunchError {
    #[error("Launch device error: {0}")]
    LaunchDevError(#[from] LaunchDeviceError),

    #[error("Mimir JIT execution error: {0}")]
    JitExecutionError(#[from] MimirJITExecutionError),

    #[error("`{0:?}` is an unsupported platform and runtime")]
    UnsupportedPlatform(MimirPlatform),
}

pub enum Param<'a> {
    Var(MimirVar),
    Buffer(MimirBuffer<'a>),
}

pub enum MimirVar {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Bool(bool),
}

impl Clone for MimirVar {
    fn clone(&self) -> Self {
        match self {
            Self::Int32(val) => Self::Int32(*val),
            Self::Int64(val) => Self::Int64(*val),
            Self::Float32(val) => Self::Float32(*val),
            Self::Bool(val) => Self::Bool(*val),
        }
    }
}

pub type BufferType<'a, T> = &'a Arc<dyn MimirArray<T> + Send + Sync>;

pub enum MimirBuffer<'a> {
    Int32(BufferType<'a, i32>),
    Int64(BufferType<'a, i64>),
    Float32(BufferType<'a, f32>),
    Bool(BufferType<'a, bool>),
}

#[derive(Error, Debug)]
pub enum LaunchDeviceError {
    #[error(
        "Device at index `{0}` is out of range, only `{1}` Mimir supported devices are availible"
    )]
    OutOfRange(usize, usize),
}
