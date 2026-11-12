use mimir_ir::ir::{
    MimirExpr, MimirIndexExpr, MimirLit, MimirPrimitiveTy, MimirStmtAssignLeft, MimirTy,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::StorageClass;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir::MimirPtrType, util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub fn assign_to_spirv(
        &mut self,
        lhs: &MimirStmtAssignLeft,
        rhs: &MimirExpr,
    ) -> Result<(), MimirJITCompilationError> {
        let (lhs_ptr_word, lhs_base_ty) = match lhs {
            MimirStmtAssignLeft::Index(MimirIndexExpr { var, index }) => {
                let mimir_var = self.get_mimir_var(var)?;

                let (idx_word, _) = self.expr_to_word(index)?;

                if let MimirTy::GlobalArray { element_type } = &mimir_var.ty.base {
                    let var_word = mimir_var.word;

                    let lit_0 = *self
                        .get_literal(MimirLit::Int32(0))
                        .map_err(|e| MimirJITCompilationError::Generic(e.to_string()))?;

                    let ptr_storage_class = mimir_var.ty.storage_class;
                    let element_ptr_type = self.get_ptr_ty(&MimirPtrType {
                        base: element_type.clone().to_mimir_ty(),
                        storage_class: ptr_storage_class,
                    })?;

                    let ptr_word = self
                        .spirv_builder
                        .access_chain(element_ptr_type, None, var_word, vec![lit_0, idx_word])
                        .map_err(map_err_closure)?;
                    (ptr_word, element_type.clone().to_mimir_ty())
                } else if let MimirTy::SharedMemArray { .. } = &mimir_var.ty.base {
                    let var_word = mimir_var.word;

                    let elem_ty = MimirPrimitiveTy::Float32;
                    let storage_class = mimir_var.ty.storage_class;

                    let (index_word, _) = self.expr_to_word(index)?;

                    let elem_ptr_ty = self.get_ptr_ty(&MimirPtrType {
                        base: elem_ty.clone().to_mimir_ty(),
                        storage_class,
                    })?;

                    let ptr_word = self
                        .spirv_builder
                        .access_chain(elem_ptr_ty, None, var_word, vec![index_word])
                        .map_err(map_err_closure)?;

                    (ptr_word, elem_ty.to_mimir_ty())
                } else {
                    return Err(MimirJITCompilationError::Generic(format!(
                        "Indexing not allowed for non global array type `{:?}`",
                        mimir_var.ty.base
                    )));
                }
            }
            MimirStmtAssignLeft::Var(uuid) => {
                let var = self.vars.get(uuid).ok_or_else(|| {
                    MimirJITCompilationError::VarNotFound(*uuid, self.get_var_name(uuid))
                })?;

                if !matches!(var.ty.storage_class, StorageClass::Function) {
                    return Err(MimirJITCompilationError::Generic(format!(
                        "(internal) invalid storage class `{:?}` was found in assignment var",
                        var.ty.storage_class
                    )));
                }

                let ptr = var.word;

                (ptr, var.ty.base.clone())
            }
        };

        let (rhs_word, rhs_ty) = self.expr_to_word(rhs)?;

        if rhs_ty != lhs_base_ty {
            return Err(MimirJITCompilationError::Generic(
                format!(
                    "left hand side type and right hand side type of assignment must be the same.\nLHS type `{:?}` != RHS type `{:?}`",
                    lhs_base_ty,
                    rhs_ty
                )
            ));
        }

        self.spirv_builder
            .store(lhs_ptr_word, rhs_word, None, vec![])
            .map_err(map_err_closure)?;

        Ok(())
    }
}
