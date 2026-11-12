use mimir_ir::ir::{MimirExpr, MimirLit, MimirPrimitiveTy, MimirTy};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::Word;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir::MimirPtrType, util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub(crate) fn index_to_word(
        &mut self,
        var: u64,
        index: &MimirExpr,
    ) -> Result<(Word, MimirTy), MimirJITCompilationError> {
        let (var_word, inny_ty, ptr_storage_class, is_param_buffer) = {
            let array_var = self
                .vars
                .get(&var)
                .ok_or(MimirJITCompilationError::VarNotFound(
                    var,
                    match self.name_map.get(&var) {
                        Some(name) => name.clone(),
                        None => "".to_string(),
                    },
                ))?;

            if let MimirTy::GlobalArray { element_type } = &array_var.ty.base {
                let var_word = array_var.word;
                (
                    var_word,
                    &element_type.clone(),
                    array_var.ty.storage_class,
                    true,
                )
            } else if let MimirTy::SharedMemArray { .. } = &array_var.ty.base {
                let element_type = MimirPrimitiveTy::Float32;
                let var_word = array_var.word;
                (
                    var_word,
                    &element_type.clone(),
                    array_var.ty.storage_class,
                    false,
                )
            } else {
                return Err(MimirJITCompilationError::Generic(format!(
                    "Indexing not allowed for non runtime array type `{:?}`",
                    array_var.ty.base
                )));
            }
        };

        let (index_word, _) = self.expr_to_word(index)?;
        let lit_0 = self.get_lit(&MimirLit::Int32(0))?;

        let element_ptr_ty = self.get_ptr_ty(&MimirPtrType {
            base: inny_ty.clone().to_mimir_ty(),
            storage_class: ptr_storage_class,
        })?;

        let ty = self.get_mimir_ty(&inny_ty.clone().to_mimir_ty())?;

        let ptr = self
            .spirv_builder
            .access_chain(
                element_ptr_ty,
                None,
                var_word,
                if is_param_buffer {
                    vec![lit_0, index_word]
                } else {
                    vec![index_word]
                },
            )
            .map_err(map_err_closure)?;

        let res = self
            .spirv_builder
            .load(ty, None, ptr, None, vec![])
            .map_err(map_err_closure)?;

        Ok((res, MimirTy::Primitive(inny_ty.clone())))
    }

    #[inline]
    fn get_lit(&self, lit: &MimirLit) -> Result<Word, MimirJITCompilationError> {
        self.literals
            .get(lit)
            .ok_or(MimirJITCompilationError::LitNotFound(lit.clone()))
            .copied()
    }
}
