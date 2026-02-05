use mimir_ir::ir::{MimirIRData, MimirIRKind, MimirTy};
use thiserror::Error;

use crate::{
    generic::{
        compiler::{MimirJITCompilationError, MimirJITCompiler, MimirPlatformBytecode},
        device::{MimirDevice, MimirPlatformDevices},
        launch::Param,
    },
    platforms::MimirPlatform,
};

pub trait MimirJITRuntime {
    fn platform(&self) -> MimirPlatform;

    fn new_jit_compiler(&self) -> Box<dyn MimirJITCompiler>;

    fn execute_bytecode(
        &self,
        device: Box<&(dyn MimirDevice + Send + Sync)>,
        bytecode: Box<dyn MimirPlatformBytecode>,
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
    ) -> Result<(), MimirJITExecutionError>;

    #[allow(clippy::too_many_arguments)]
    fn execute_kernel(
        &self,
        device: Box<&(dyn MimirDevice + Send + Sync)>,
        ir: &MimirIRData,
        name: &str,
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
        const_generics: &[u32],
    ) -> Result<(), MimirJITExecutionError> {
        let bytecode = match self.get_bytecode(name, const_generics)? {
            Some(bytecode) => bytecode,
            None => {
                let mut jit_compiler = self.new_jit_compiler();

                // Safety/validity check
                {
                    match ir.irs.get(name) {
                        Some(ir_kind) => match ir_kind {
                            MimirIRKind::Kernel(mimir_kernel_ir) => {
                                if !mimir_kernel_ir.is_verified() {
                                    return Err(MimirJITExecutionError::UnverifiedIR);
                                }
                            }
                        },
                        None => {
                            return Err(MimirJITExecutionError::KernelNotFound(name.to_string()));
                        }
                    }
                }

                let bytecode = jit_compiler
                    .compile_kernel(name, ir, const_generics.to_vec())
                    .map_err(MimirJITExecutionError::JITCompilationError)?;
                self.insert_bytecode(name, const_generics, Box::new(bytecode.as_ref()))?;
                bytecode
            }
        };

        self.execute_bytecode(device, bytecode, block_dim, grid_dim, params)
    }

    fn get_platform_devices(&self) -> Box<dyn MimirPlatformDevices>;

    fn get_bytecode(
        &self,
        name: &str,
        const_generics: &[u32],
    ) -> Result<Option<Box<dyn MimirPlatformBytecode>>, MimirJITExecutionError>;

    fn insert_bytecode(
        &self,
        name: &str,
        const_generics: &[u32],
        bytecode: Box<&dyn MimirPlatformBytecode>,
    ) -> Result<(), MimirJITExecutionError>;
}

#[derive(Error, Debug)]
pub enum MimirJITExecutionError {
    #[error("JIT Execution Error: {0}")]
    Generic(String),

    #[error(
        "Expected `{0}` numer of parameters, instead got {1} number of parameters during JIT execution."
    )]
    ParamaterMismatch(usize, usize),

    #[error("Expected a type of `{0:?}` instead got a `{1:?}` during JIT execution.")]
    ParamaterTypeMismatch(MimirTy, MimirTy),

    #[error("JIT compilation error during execution:\n{0:?}")]
    JITCompilationError(MimirJITCompilationError),

    #[error(
        "Thread config issue: requested {0} threads, the max number of group/block invocations/ is {1}, the block size is {2:?}"
    )]
    TooManyThreadsPerBlock(u32, u32, [u32; 3]),

    #[error("The `{0:?}` platform is unsupported for Mimir JIT execution")]
    RuntimeUnsupported(MimirPlatform),

    #[error("The unofficial `{0}` platform is unsupported for Mimir JIT execution")]
    UnofficialRuntimeUnsupported(&'static str),

    #[error(
        "Expected valid bytecode that casted properly, instead an innapropriate bytecode was submitted"
    )]
    InnapropriateBytecode, // #[error("JIT Execution Error:\n\t{0}")]
    // GenericError(#[from])
    #[error("Expected a kernel of the name `{0}`, instead found no match")]
    KernelNotFound(String),

    #[error("Inputted IR is unverified and not to be trusted")]
    UnverifiedIR,
}

#[macro_export]
macro_rules! jit_exec_gen_err {
    ($($arg:tt)*) => {
        MimirJITExecutionError::Generic(format!($($arg)*))
    };
}
