use mimir_ir::ir::{MimirBuiltIn, MimirBuiltInField, MimirLit, MimirPrimitiveTy, MimirTy};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::{StorageClass, Word};

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir::MimirPtrType, util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub(crate) fn builtin_field_access_to_word(
        &mut self,
        builtin: &MimirBuiltIn,
        field: &MimirBuiltInField,
    ) -> Result<(Word, MimirTy), MimirJITCompilationError> {
        // println!("Builtin is {:?}", builtin);
        // println!("Builtins are {:?}", self.builtins);

        let ptr_ty = self.get_ptr_ty(&MimirPtrType {
            base: MimirTy::Primitive(MimirPrimitiveTy::Uint32),
            storage_class: StorageClass::Input,
        })?;

        let var = if &MimirBuiltIn::BlockDim == builtin {
            // lowk don't understand why this is the case but I'm so sick of SPIRV compiler stuff rn
            // TODO: fix this atrocious code later
            self.builtin_tys.get(builtin)
        } else {
            self.builtins.get(builtin)
        }
        .unwrap();

        let ty = self.get_mimir_ty(&MimirTy::Primitive(MimirPrimitiveTy::Uint32))?;

        let idx = field.to_num() as i32;

        let idx_lit = *self
            .literals
            .get(&MimirLit::Int32(idx))
            .ok_or(MimirJITCompilationError::LitNotFound(MimirLit::Int32(idx)))?;

        let access_word = if &MimirBuiltIn::BlockDim == builtin {
            self.spirv_builder
                .composite_extract(ty, None, *var, vec![idx as u32])
        } else {
            self.spirv_builder
                .access_chain(ptr_ty, None, *var, vec![idx_lit])
        }
        .map_err(map_err_closure)?;

        let load_word = if &MimirBuiltIn::BlockDim == builtin {
            access_word
        } else {
            self.spirv_builder
                .load(ty, None, access_word, None, vec![])
                .map_err(map_err_closure)?
        };

        Ok((load_word, MimirTy::Primitive(MimirPrimitiveTy::Uint32)))
    }
}
