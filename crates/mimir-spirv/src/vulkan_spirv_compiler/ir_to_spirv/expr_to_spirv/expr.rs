use mimir_ir::ir::{
    MathIntrinsicExpr, MimirBinOpExpr, MimirCastExpr, MimirConstExpr, MimirExpr, MimirIndexExpr,
    MimirLit, MimirPrimitiveTy, MimirTy, MimirUnOpExpr,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::Word;

use crate::vulkan_spirv_compiler::compiler::VulkanSpirVCompiler;

pub(crate) type ExprToWordResult = Result<(Word, MimirTy), MimirJITCompilationError>;

impl VulkanSpirVCompiler {
    pub(crate) fn expr_to_word(&mut self, expr: &MimirExpr) -> ExprToWordResult {
        match expr {
            MimirExpr::BinOp(MimirBinOpExpr { lhs, op, rhs, .. }) => {
                self.binop_to_word(lhs.as_ref(), op, rhs.as_ref())
            }
            MimirExpr::Index(MimirIndexExpr { var, index }) => self.index_to_word(*var, index),
            MimirExpr::BuiltinFieldAccess { built_in, field } => {
                self.builtin_field_access_to_word(built_in, field)
            }
            MimirExpr::Unary(MimirUnOpExpr { un_op, expr }) => {
                self.unary_to_word(un_op, expr.as_ref())
            }
            MimirExpr::Literal(mimir_lit) => {
                if !self.literals.contains_key(mimir_lit) {
                    let mimir_ty = self.get_mimir_ty(&mimir_lit.to_ty().to_mimir_ty())?;
                    let word = match mimir_lit {
                        MimirLit::Bool(val) => {
                            if *val {
                                self.spirv_builder.constant_true(mimir_ty)
                            } else {
                                self.spirv_builder.constant_false(mimir_ty)
                            }
                        }
                        _ => self
                            .spirv_builder
                            .constant_bit32(mimir_ty, mimir_lit.to_u32().unwrap()),
                    };
                    self.literals.insert(mimir_lit.clone(), word);

                    Ok((word, mimir_lit.to_ty().to_mimir_ty()))
                } else {
                    Ok((
                        *self.get_literal(mimir_lit.clone())?,
                        mimir_lit.to_ty().to_mimir_ty(),
                    ))
                }
            }
            MimirExpr::Var(uuid) => self.var_to_word(uuid),
            MimirExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
                self.math_intrinsic_to_word(func, args)
            }
            MimirExpr::Cast(MimirCastExpr { from, to }) => {
                self.cast_to_word(from, &to.to_mimir_ty())
            }
            MimirExpr::ConstExpr(mimir_const_expr) => match mimir_const_expr {
                MimirConstExpr::Literal(mimir_lit) => Ok((
                    *self.get_literal(mimir_lit.clone())?,
                    mimir_lit.to_ty().to_mimir_ty(),
                )),
                _ => Err(MimirJITCompilationError::Generic(
                    "All const expressions should be evaluated to literals for speed ups."
                        .to_owned(),
                )),
            },
        }
    }

    pub fn cast_to_ty(
        &mut self,
        orig_ty: &MimirTy,
        word: Word,
        to_ty: &MimirTy,
    ) -> Result<Word, MimirJITCompilationError> {
        if orig_ty == to_ty {
            return Ok(word);
        } else if orig_ty == &MimirTy::Primitive(MimirPrimitiveTy::Float32)
            && to_ty == &MimirTy::Primitive(MimirPrimitiveTy::Int32)
        {
            self.spirv_builder.convert_f_to_s(
                self.get_mimir_ty(&MimirPrimitiveTy::Int32.to_mimir_ty())?,
                None,
                word,
            )
        } else if orig_ty == &MimirTy::Primitive(MimirPrimitiveTy::Int32)
            && to_ty == &MimirTy::Primitive(MimirPrimitiveTy::Float32)
        {
            self.spirv_builder.convert_s_to_f(
                self.get_mimir_ty(&MimirPrimitiveTy::Float32.to_mimir_ty())?,
                None,
                word,
            )
        } else if orig_ty == &MimirTy::Primitive(MimirPrimitiveTy::Float32)
            && to_ty == &MimirTy::Primitive(MimirPrimitiveTy::Uint32)
        {
            self.spirv_builder.convert_f_to_u(
                self.get_mimir_ty(&MimirPrimitiveTy::Uint32.to_mimir_ty())?,
                None,
                word,
            )
        } else if orig_ty == &MimirPrimitiveTy::Uint32.to_mimir_ty()
            && to_ty == &MimirPrimitiveTy::Float32.to_mimir_ty()
        {
            self.spirv_builder.convert_u_to_f(
                self.get_mimir_ty(&MimirPrimitiveTy::Float32.to_mimir_ty())?,
                None,
                word,
            )
        } else if orig_ty == &MimirPrimitiveTy::Int32.to_mimir_ty()
            && to_ty == &MimirPrimitiveTy::Uint32.to_mimir_ty()
        {
            self.spirv_builder.u_convert(
                self.get_mimir_ty(&MimirPrimitiveTy::Uint32.to_mimir_ty())?,
                None,
                word,
            )
        } else if orig_ty == &MimirPrimitiveTy::Uint32.to_mimir_ty()
            && to_ty == &MimirPrimitiveTy::Int32.to_mimir_ty()
        {
            self.spirv_builder.bitcast(
                self.get_mimir_ty(&MimirPrimitiveTy::Int32.to_mimir_ty())?,
                None,
                word,
            )
        } else {
            return Err(MimirJITCompilationError::Generic(format!(
                "Unsupported cast from {:?} to {:?}",
                orig_ty, to_ty
            )));
        }
        .map_err(|e| MimirJITCompilationError::Generic(e.to_string()))
    }
}
