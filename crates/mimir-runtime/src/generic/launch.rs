use mimir_ast::MimirGlobalAST;
use std::error::Error; // Import the Error trait

use super::array::MimirArray;

pub trait MimirLaunch<E: Error> { // Add generic error type E constrained by std::error::Error
    fn launch_ast(
        ast: &MimirGlobalAST,
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), E>; // Use E as the error type

    // may not be implemented in all backends, i.e different language is used for kernel
    fn launch_kernel_name(
        kernel_name: &str,
        grid_dim: &[u32; 3],
        block_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), E>; // Use E as the error type
}

pub enum Param<'a> {
    PushConst(MimirPushConst),
    Buffer(MimirBuffer<'a>),
}

pub enum MimirPushConst {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Bool(bool),
}

impl Clone for MimirPushConst {
    fn clone(&self) -> Self {
        match self {
            MimirPushConst::Int32(val) => MimirPushConst::Int32(*val),
            MimirPushConst::Int64(val) => MimirPushConst::Int64(*val),
            MimirPushConst::Float32(val) => MimirPushConst::Float32(*val),
            MimirPushConst::Bool(val) => MimirPushConst::Bool(*val),
        }
    }
}

pub type BufferType<'a, T> = Box<&'a (dyn MimirArray<T> + Send + Sync)>;

pub enum MimirBuffer<'a> {
    Int32(BufferType<'a, i32>),
    Int64(BufferType<'a, i64>),
    Float32(BufferType<'a, f32>),
    Bool(BufferType<'a, bool>),
}