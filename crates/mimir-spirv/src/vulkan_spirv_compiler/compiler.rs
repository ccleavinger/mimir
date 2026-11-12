use mimir_ir::ir::{MimirBuiltIn, MimirIRData, MimirIRKind, MimirLit, MimirTy};
use mimir_ir::passes::{kernel_ir_pass, KernelIRPassSettings};
use mimir_runtime::generic::compiler::{
    MimirJITCompilationError, MimirJITCompiler, MimirPlatformBytecode,
};
use mimir_runtime::platforms::MimirPlatform;
use rspirv::binary::Assemble;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use rspirv::{dr::Builder, spirv::Word};

use super::ir::{MimirPtrType, MimirVariable};

pub struct VulkanSpirVCompiler {
    pub(crate) spirv_builder: Builder,
    pub(crate) vars: HashMap<u64, MimirVariable>,
    pub(crate) name_map: HashMap<u64, String>,
    //pub shared_vars: HashMap<String, SharedVariable>, // variables mapped by name to objects for variables shared across threads
    pub(crate) ptr_types: HashMap<MimirPtrType, Word>,
    pub(crate) types: HashMap<MimirTy, Word>,
    // builtin -> type
    pub(crate) builtin_tys: HashMap<MimirBuiltIn, Word>,
    // builtin -> var Word
    pub(crate) builtins: HashMap<MimirBuiltIn, Word>,
    pub(crate) literals: HashMap<MimirLit, Word>,
    pub(crate) const_gens: Vec<u32>,
    pub(crate) pc_order: Vec<u64>,
    pub(crate) buffer_order: Vec<u64>, // Used to keep track of the order of buffers for spirv generation, if needed
    pub(crate) vec3_ptr_word: Option<Word>,
    pub(crate) ext_inst: Word,
    pub(crate) pc_var_word: Word,
}

#[derive(Clone, Debug)]
pub struct SpirvBytecode {
    pub bytes: Vec<u32>,
}

impl MimirPlatformBytecode for SpirvBytecode {
    fn bytecode(&self) -> Arc<dyn std::any::Any> {
        Arc::new(self.clone())
    }

    fn platform(&self) -> mimir_runtime::platforms::MimirPlatform {
        MimirPlatform::Vulkan
    }

    fn version(&self) -> [u16; 2] {
        [0, 0]
    }
}

impl VulkanSpirVCompiler {
    pub fn new() -> Self {
        VulkanSpirVCompiler {
            spirv_builder: Builder::new(),
            vars: HashMap::new(),
            name_map: HashMap::new(),
            //shared_vars: HashMap::new(),
            ptr_types: HashMap::new(),
            types: HashMap::new(),
            builtins: HashMap::new(),
            builtin_tys: HashMap::new(),
            literals: HashMap::new(),
            const_gens: Vec::new(),
            pc_order: Vec::new(),
            buffer_order: Vec::new(),
            ext_inst: 0,
            vec3_ptr_word: None,
            pc_var_word: 0,
        }
    }
}

// the MimirJITCompiler will be defined in mimir-vulkan
impl MimirJITCompiler for VulkanSpirVCompiler {
    fn compile_kernel(
        &mut self,
        kernel_name: &str,
        mimir_ir: &MimirIRData,
        const_generics: Vec<u32>,
    ) -> Result<Box<dyn MimirPlatformBytecode>, MimirJITCompilationError> {
        let unoptimized_kernel = match mimir_ir.irs.get(kernel_name) {
            Some(ir) => match ir {
                MimirIRKind::Kernel(mimir_kernel_ir) => mimir_kernel_ir,
            },
            None => {
                return Err(MimirJITCompilationError::KernelNotFound(
                    kernel_name.to_string(),
                ))
            }
        };

        let kernel = &kernel_ir_pass(
            unoptimized_kernel,
            KernelIRPassSettings {
                const_generics: const_generics.clone(),
                inline_pass: true,
            },
        )?;

        self.const_gens = const_generics;

        self.name_map =
            HashMap::from_iter(kernel.name_map.iter().clone().map(|(u, s)| (*u, s.clone())));

        // init builder values (params, push constants, specialization values, & types)
        self.init_builder(kernel)?;

        self.setup_vars(kernel)?;

        // println!("Compiling kernel: {}", kernel_name);
        // println!("IR: {:#?}", kernel);

        for ir in &kernel.body {
            match self.ir_to_spirv(ir) {
                Ok(_) => {},
                Err(e) => {
                    return Err(e);
                }
            }
        }

        self.spirv_builder
            .ret()
            .map_err(|e| MimirJITCompilationError::Generic(e.to_string()))?;
        self.spirv_builder
            .end_function()
            .map_err(|e| MimirJITCompilationError::Generic(e.to_string()))?;

        let shader = std::mem::take(&mut self.spirv_builder).module().assemble();

        let bytecode = Box::new(SpirvBytecode {
            bytes: shader.clone(),
        });

        Ok(bytecode)
    }
}

// fn save_spv(path: &str, buffer: &[u32]) -> std::io::Result<()> {
//     let bytes: Vec<u8> = buffer
//         .iter()
//         .flat_map(|word| word.to_le_bytes())
//         .collect();

//     let mut file = File::create(path)?;
//     file.write_all(&bytes)
// }

impl Default for VulkanSpirVCompiler {
    fn default() -> Self {
        Self::new()
    }
}
