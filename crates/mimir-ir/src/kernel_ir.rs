use std::{collections::BTreeMap, hash::Hash};

use serde::{Deserialize, Serialize};

use crate::ir::{IRFunctionality, MimirBlock, MimirStmt, MimirTyVar};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirKernelIR {
    pub name: String,
    // String -> (hash it) -> u64 -> MimirTyVar
    pub var_map: BTreeMap<u64, MimirTyVar>,

    // order of the paramaters as uuids
    pub param_order: Vec<u64>,

    // we still need var names for debugging & whatnot.
    // multiple different variables can have the same name (i.e. scoping)
    pub name_map: BTreeMap<u64, String>,
    pub body: MimirBlock,        // body of the kernel
    pub const_generics: usize,   // number of const generics
    pub(crate) safety_sig: bool, // proof that the kernel is spec compliant
}

impl MimirKernelIR {
    pub fn new(
        name: String,
        var_map: BTreeMap<u64, MimirTyVar>,
        param_order: Vec<u64>,
        name_map: BTreeMap<u64, String>,
        body: MimirBlock,
        const_generics: usize,
    ) -> Self {
        Self {
            name,
            var_map,
            param_order,
            name_map,
            body,
            const_generics,
            safety_sig: false,
        }
    }

    pub fn is_verified(&self) -> bool {
        self.safety_sig
    }
}

impl IRFunctionality for MimirKernelIR {
    fn var_map(&self) -> BTreeMap<u64, MimirTyVar> {
        self.var_map.clone()
    }

    fn get_body(&self) -> Vec<MimirStmt> {
        self.body.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn var_name_map(&self) -> BTreeMap<u64, String> {
        self.name_map.clone()
    }
}
