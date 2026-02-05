use mimir_ir::{
    ir::{MimirBinOp, MimirExpr, MimirPrimitiveTy, MimirTy},
    util::ty,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::Word;

use crate::vulkan_spirv_compiler::compiler::VulkanSpirVCompiler;

impl VulkanSpirVCompiler {
    pub(crate) fn binop_to_word(
        &mut self,
        lhs: &MimirExpr,
        op: &MimirBinOp,
        rhs: &MimirExpr,
    ) -> Result<(Word, MimirTy), MimirJITCompilationError> {
        // if lhs.is_const() && rhs.is_const() {
        //     let lit = self.binop_to_literal(lhs, op, rhs)?;

        //     return self.expr_to_word(&MimirExpr::Literal(lit));
        // }

        let lhs_precedence = lhs.precedence();
        let rhs_precedence = rhs.precedence();

        // sorting via precedence
        let (lhs_word, lhs_ty, rhs_word, rhs_ty) = {
            if rhs_precedence > lhs_precedence {
                let (r_w, r_ty) = self.expr_to_word(rhs)?;
                let (l_w, l_ty) = self.expr_to_word(lhs)?;

                (l_w, l_ty, r_w, r_ty)
            } else {
                let (l_w, l_ty) = self.expr_to_word(lhs)?;
                let (r_w, r_ty) = self.expr_to_word(rhs)?;

                (l_w, l_ty, r_w, r_ty)
            }
        };
        let result_ty = self.binop_ty(&lhs_ty, op, &rhs_ty);
        let op_ty = ty::type_importance(&[lhs_ty.clone(), rhs_ty.clone()]);

        // Cast operands except for boolean operations
        let (lhs_cast_word, rhs_cast_word) =
            if !matches!(op, MimirBinOp::And | MimirBinOp::Or) && !op.is_comparison() {
                (
                    self.cast_to_ty(&lhs_ty, lhs_word, &result_ty)?,
                    self.cast_to_ty(&rhs_ty, rhs_word, &result_ty)?,
                )
            } else {
                (lhs_word, rhs_word)
            };

        self.binop_word_logic(
            lhs_cast_word,
            rhs_cast_word,
            &lhs_ty,
            &rhs_ty,
            op,
            &op_ty,
            &result_ty,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn binop_word_logic(
        &mut self,
        lhs_cast_word: Word,
        rhs_cast_word: Word,
        lhs_ty: &MimirTy,
        rhs_ty: &MimirTy,
        op: &MimirBinOp,
        op_ty: &MimirTy,
        result_ty: &MimirTy,
    ) -> Result<(Word, MimirTy), MimirJITCompilationError> {
        let result_word = match op {
            // Arithmetic operations
            MimirBinOp::Add
            | MimirBinOp::Sub
            | MimirBinOp::Mul
            | MimirBinOp::Div
            | MimirBinOp::Mod => {
                let ty_word = self.mimir_ty_to_word(result_ty)?;
                match (op, result_ty) {
                    (
                        MimirBinOp::Add,
                        MimirTy::Primitive(MimirPrimitiveTy::Int32)
                        | MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    ) => self
                        .spirv_builder
                        .i_add(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Add, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_add(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (
                        MimirBinOp::Sub,
                        MimirTy::Primitive(MimirPrimitiveTy::Int32)
                        | MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    ) => self
                        .spirv_builder
                        .i_sub(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Sub, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_sub(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (
                        MimirBinOp::Mul,
                        MimirTy::Primitive(MimirPrimitiveTy::Int32)
                        | MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    ) => self
                        .spirv_builder
                        .i_mul(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Mul, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_mul(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Div, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_div(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Div, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_div(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Div, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_div(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Mod, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_mod(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Mod, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_mod(ty_word, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Mod, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_mod(ty_word, None, lhs_cast_word, rhs_cast_word),
                    _ => return Err(unsupported_type_error(result_ty, "binary operation")),
                }
            }
            // Boolean operations
            MimirBinOp::And | MimirBinOp::Or => {
                if !(lhs_ty.is_bool() && rhs_ty.is_bool()) {
                    return Err(MimirJITCompilationError::Generic(
                        "Tried to perform a boolean operation with non-boolean types".to_string(),
                    ));
                }
                let bool_ty_w =
                    self.mimir_ty_to_word(&MimirTy::Primitive(MimirPrimitiveTy::Bool))?;
                match op {
                    MimirBinOp::And => self.spirv_builder.logical_and(
                        bool_ty_w,
                        None,
                        lhs_cast_word,
                        rhs_cast_word,
                    ),
                    MimirBinOp::Or => {
                        self.spirv_builder
                            .logical_or(bool_ty_w, None, lhs_cast_word, rhs_cast_word)
                    }
                    _ => unreachable!(),
                }
            }
            // Comparison operations
            MimirBinOp::Lt
            | MimirBinOp::Lte
            | MimirBinOp::Gt
            | MimirBinOp::Gte
            | MimirBinOp::Eq
            | MimirBinOp::Ne => {
                let bool_ty_w =
                    self.mimir_ty_to_word(&MimirTy::Primitive(MimirPrimitiveTy::Bool))?;
                match (op, &op_ty) {
                    (MimirBinOp::Lt, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_less_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Lt, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_less_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Lt, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_less_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Lte, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_less_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Lte, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_less_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Lte, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_less_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gt, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_greater_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gt, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_greater_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gt, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_greater_than(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gte, MimirTy::Primitive(MimirPrimitiveTy::Int32)) => self
                        .spirv_builder
                        .s_greater_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gte, MimirTy::Primitive(MimirPrimitiveTy::Uint32)) => self
                        .spirv_builder
                        .u_greater_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Gte, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_greater_than_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (
                        MimirBinOp::Eq,
                        MimirTy::Primitive(MimirPrimitiveTy::Int32)
                        | MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    ) => self
                        .spirv_builder
                        .i_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (MimirBinOp::Eq, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    (
                        MimirBinOp::Ne,
                        MimirTy::Primitive(MimirPrimitiveTy::Int32)
                        | MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                    ) => self.spirv_builder.i_not_equal(
                        bool_ty_w,
                        None,
                        lhs_cast_word,
                        rhs_cast_word,
                    ),
                    (MimirBinOp::Ne, MimirTy::Primitive(MimirPrimitiveTy::Float32)) => self
                        .spirv_builder
                        .f_unord_not_equal(bool_ty_w, None, lhs_cast_word, rhs_cast_word),
                    _ => return Err(unsupported_type_error(op_ty, "comparison operation")),
                }
            }
        }
        .map_err(|e| MimirJITCompilationError::Generic(e.to_string()))?;

        Ok((result_word, result_ty.clone()))
    }

    pub(crate) fn binop_ty(&self, lhs: &MimirTy, op: &MimirBinOp, rhs: &MimirTy) -> MimirTy {
        match op {
            MimirBinOp::Add
            | MimirBinOp::Sub
            | MimirBinOp::Mul
            | MimirBinOp::Div
            | MimirBinOp::Mod => {
                if lhs == rhs {
                    lhs.clone()
                } else {
                    ty::type_importance(&[lhs.clone(), rhs.clone()])
                }
            }
            MimirBinOp::And
            | MimirBinOp::Or
            | MimirBinOp::Lt
            | MimirBinOp::Lte
            | MimirBinOp::Gt
            | MimirBinOp::Gte
            | MimirBinOp::Eq
            | MimirBinOp::Ne => MimirTy::Primitive(MimirPrimitiveTy::Bool),
        }
    }
}

// Helper function for error messages
#[inline]
fn unsupported_type_error(ty: &MimirTy, operation: &str) -> MimirJITCompilationError {
    MimirJITCompilationError::Generic(format!("Unsupported type `{:?}` for {}", ty, operation))
}

// impl VulkanSpirVCompiler {
//     fn binop_to_literal(
//         &self,
//         lhs: &MimirExpr,
//         op: &MimirBinOp,
//         rhs: &MimirExpr,
//     ) -> Result<MimirLit, MimirJITCompilationError> {
//         expr_to_lit_recur(
//             &MimirConstExpr::BinOp(MimirBinOpExpr {
//                 lhs: Box::new(lhs.clone()),
//                 op: op.clone(),
//                 rhs: Box::new(rhs.clone()),
//                 is_parenthesized: false,
//             }),
//             &self.const_gens,
//         )
//         .map_err(MimirJITCompilationError::InlinePass)
//     }
// }
