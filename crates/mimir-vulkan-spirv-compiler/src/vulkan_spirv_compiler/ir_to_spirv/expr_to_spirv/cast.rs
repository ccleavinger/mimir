use mimir_ir::ir::{MimirExpr, MimirPrimitiveTy, MimirTy};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::Word;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir_to_spirv::expr_to_spirv::expr::ExprToWordResult,
    util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub fn cast_to_word(&mut self, from: &MimirExpr, to: &MimirTy) -> ExprToWordResult {
        let (word, ty) = self.expr_to_word(from)?;
        self.cast_to_word_impl(word, &ty, to)
    }

    fn cast_to_word_impl(
        &mut self,
        from: Word,
        orig_ty: &MimirTy,
        end_ty: &MimirTy,
    ) -> ExprToWordResult {
        if orig_ty == end_ty {
            return Ok((from, orig_ty.clone()));
        }

        Ok((
            match (orig_ty, end_ty) {
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Int32),
                    MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                ) => self
                    .spirv_builder
                    .bitcast(self.get_mimir_ty(end_ty)?, None, from),
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Int32),
                    MimirTy::Primitive(MimirPrimitiveTy::Float32),
                ) => self
                    .spirv_builder
                    .convert_s_to_f(self.get_mimir_ty(end_ty)?, None, from),
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    MimirTy::Primitive(MimirPrimitiveTy::Int32),
                ) => self
                    .spirv_builder
                    .bitcast(self.get_mimir_ty(end_ty)?, None, from),
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    MimirTy::Primitive(MimirPrimitiveTy::Float32),
                ) => self
                    .spirv_builder
                    .convert_u_to_f(self.get_mimir_ty(end_ty)?, None, from),
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Float32),
                    MimirTy::Primitive(MimirPrimitiveTy::Int32),
                ) => self
                    .spirv_builder
                    .convert_f_to_s(self.get_mimir_ty(end_ty)?, None, from),
                (
                    MimirTy::Primitive(MimirPrimitiveTy::Float32),
                    MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                ) => self
                    .spirv_builder
                    .convert_f_to_u(self.get_mimir_ty(end_ty)?, None, from),
                _ => {
                    return Err(MimirJITCompilationError::Generic(format!(
                        "Unsupported type conversion from type `{:?}` to type `{:?}`",
                        orig_ty, end_ty
                    )))
                }
            }
            .map_err(map_err_closure)?,
            end_ty.clone(),
        ))
    }
}
