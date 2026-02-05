use std::{any::Any, sync::Arc};

use mimir_ir::{
    ir::{MimirIRData, MimirLit, MimirTy},
    util::error::{MimirInlinePassError, MultiPassError},
};
use thiserror::Error;

use crate::platforms::MimirPlatform;

pub trait MimirJITCompiler {
    // kernel_name is the selected kernel accessed via the hashmap in MimirIRData
    fn compile_kernel(
        &mut self,
        kernel_name: &str,
        mimir_ir: &MimirIRData,
        const_generics: Vec<u32>,
    ) -> Result<Box<dyn MimirPlatformBytecode>, MimirJITCompilationError>;

    // fn execute_bytecode(
    //     &mut self,
    //     device: Box<dyn MimirDevice>,
    //     bytecode: Box<dyn PlatformBytecode>,
    // ) -> Result<(), MimirJITExecutionError>;

    // fn new() -> Self;
}

#[derive(Error, Debug, Clone)]
pub enum MimirJITCompilationError {
    #[error("JIT compilation error: {0}")]
    Generic(String),

    #[error("Failed to find a kernel named `{0}` during JIT compilation.")]
    KernelNotFound(String),

    #[error("`{0:?}` type could not be found during JIT compilation.")]
    TypeNotFound(MimirTy),

    #[error(
        "Failed to find a variable with the uuid of `{0}` and potential variable name of `{1}` during JIT compilation"
    )]
    VarNotFound(u64, String),

    #[error("Failed to find a literal of `{0:?}`")]
    LitNotFound(MimirLit),

    #[error("Expected {0:?} const generic(s), instead received {1:?}")]
    ConstGenericSizeMismatch(usize, usize),

    #[error("Inlining error: {0}")]
    InlinePass(#[from] MimirInlinePassError),

    #[error("Error during multiple JIT IR passes: {0}")]
    MultiPass(#[from] MultiPassError),
}

pub trait MimirPlatformBytecode {
    fn bytecode(&self) -> Arc<dyn Any>; // like basically whatever the underlying bytecode is

    fn platform(&self) -> MimirPlatform;

    // for tracking versions of the bytecode
    // helpful for which features are required and optimizations that are enabled
    fn version(&self) -> [u16; 2]; // [major, minor]
}
