use anyhow::Result;
use std::collections::{HashMap, HashSet};

use mimir_ast::Parameter;
use rspirv::dr::Module;
use rspirv::{dr::Builder, spirv::Word};
use syn::Block;

use super::ir::{MimirBuiltIn, MimirExprIR, MimirLit, MimirPtrType, MimirType, MimirVariable};

pub struct SpirVCompiler {
    pub spirv_builder: Builder,
    pub vars: HashMap<String, MimirVariable>,
    //pub shared_vars: HashMap<String, SharedVariable>, // variables mapped by name to objects for variables shared across threads
    pub ptr_types: HashMap<MimirPtrType, Word>,
    pub types: HashMap<MimirType, Word>,
    pub builtins: HashSet<MimirBuiltIn>,
    pub literals: HashMap<MimirLit, Word>,
    pub block_vars: HashSet<String>, // Used to track block-scoped variables (for loops, if statements, etc.)
    pub param_order: Vec<String>,
    pub buffer_order: Vec<String>, // Used to keep track of the order of buffers for spirv generation, if needed
    pub ext_inst: Word,
    pub ir: Vec<MimirExprIR>,
}

impl SpirVCompiler {
    pub fn new() -> Self {
        SpirVCompiler {
            spirv_builder: Builder::new(),
            vars: HashMap::new(),
            //shared_vars: HashMap::new(),
            ptr_types: HashMap::new(),
            types: HashMap::new(),
            builtins: HashSet::new(),
            literals: HashMap::new(),
            block_vars: HashSet::new(),
            param_order: Vec::new(),
            buffer_order: Vec::new(),
            ext_inst: 0,
            ir: Vec::new(),
        }
    }

    pub fn compile_kernel(
        &mut self,
        body: &Block,
        parameters: &[Parameter],
    ) -> Result<Module> {
        self.params_to_ir(parameters)?;
        self.body_to_ir(body)?;

        self.ir_to_spirv()?;

        // Replace the builder with a new one and get the module from the old one
        // idk what to do if I need to debug
        let builder = std::mem::replace(&mut self.spirv_builder, Builder::new());
        let module = builder.module();

        Ok(module)
    }
}

impl Default for SpirVCompiler {
    fn default() -> Self {
        Self::new()
    }
}
