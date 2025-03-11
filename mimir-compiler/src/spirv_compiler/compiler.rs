use std::collections::{HashMap, HashSet};

use mimir_ast::Parameter;
use rspirv::{binary::Assemble, dr::Builder, spirv::Word};
use syn::Block;

use super::ir::{MimirBuiltIn, MimirExprIR, MimirLit, MimirPtrType, MimirType, MimirVariable};



pub struct SpirVCompiler {
    pub spirv_builder: Builder,
    pub vars: HashMap<String, MimirVariable>,
    pub ptr_types: HashMap<MimirPtrType, Word>,
    pub types: HashMap<MimirType, Word>,
    pub builtins: HashSet<MimirBuiltIn>,
    pub literals: HashMap<MimirLit, Word>,
    pub ir: Vec<MimirExprIR>,
}

impl SpirVCompiler {

    pub fn new() -> Self {
        SpirVCompiler {
            spirv_builder: Builder::new(),
            vars: HashMap::new(),
            ptr_types: HashMap::new(),
            types: HashMap::new(),
            builtins: HashSet::new(),
            literals: HashMap::new(),
            ir: Vec::new(),
        }
    }

    pub fn compile_kernel(&mut self, body: &Block, parameters: &[Parameter]) -> Result<Vec<u32>, String> {

        self.params_to_ir(parameters);
        self.body_to_ir(body)?;
        
        

        let module = std::mem::replace(&mut self.spirv_builder, Builder::new()).module();

        Ok(module.assemble())
    }
}

impl Default for SpirVCompiler {
    fn default() -> Self {
        Self::new()
    }
}

