use mimir_ir::ir::{MimirExpr, MimirPrimitiveTy, MimirTy, MimirUnOp};
use mimir_runtime::generic::compiler::MimirJITCompilationError;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir_to_spirv::expr_to_spirv::expr::ExprToWordResult,
    util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub fn unary_to_word(&mut self, un_op: &MimirUnOp, expr: &MimirExpr) -> ExprToWordResult {
        let (word, ty) = self.expr_to_word(expr)?;
        let ty_word = self.get_mimir_ty(&ty)?;

        let ret_word = match un_op {
            MimirUnOp::Not => {
                if !ty.is_bool() {
                    return Err(MimirJITCompilationError::Generic(format!(
                        "Logical not requires bool type, got `{:?}` instead",
                        ty
                    )));
                }

                self.spirv_builder.logical_not(ty_word, None, word)
            }
            MimirUnOp::Neg => {
                if let MimirTy::Primitive(prim) = &ty {
                    match prim {
                        MimirPrimitiveTy::Float32 => {
                            self.spirv_builder.f_negate(ty_word, None, word)
                        }
                        MimirPrimitiveTy::Int32 | MimirPrimitiveTy::Uint32 => {
                            self.spirv_builder.s_negate(ty_word, None, word)
                        }
                        _ => {
                            return Err(MimirJITCompilationError::Generic(format!(
                                "Unsupported primitive type `{:?}` for negation",
                                prim
                            )))
                        }
                    }
                } else {
                    return Err(MimirJITCompilationError::Generic(format!(
                        "Unsupported type `{:?}` for negation",
                        ty
                    )));
                }
            }
        }
        .map_err(map_err_closure)?;

        Ok((ret_word, ty))
    }
}
