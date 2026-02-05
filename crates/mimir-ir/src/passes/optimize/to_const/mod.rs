use crate::{
    ir::{
        MathIntrinsicExpr, MimirBinOpExpr, MimirCastExpr, MimirConstExpr, MimirExpr, MimirUnOpExpr,
    },
    util::error::MimirInlinePassError,
};

pub(crate) mod pass;

impl MimirExpr {
    pub fn to_const_expr(&self) -> Result<MimirConstExpr, MimirInlinePassError> {
        match self {
            MimirExpr::BinOp(mimir_bin_op_expr) => {
                let lhs = mimir_bin_op_expr.lhs.to_const_expr()?;
                let rhs = mimir_bin_op_expr.rhs.to_const_expr()?;

                Ok(MimirConstExpr::BinOp(MimirBinOpExpr {
                    lhs: Box::new(lhs),
                    op: mimir_bin_op_expr.op.clone(),
                    rhs: Box::new(rhs),
                    is_parenthesized: mimir_bin_op_expr.is_parenthesized,
                }))
            }
            MimirExpr::Index(_) => Err(MimirInlinePassError::AttemptedIndex),
            MimirExpr::BuiltinFieldAccess { .. } => Err(MimirInlinePassError::AttemptedVarAccess),
            MimirExpr::Unary(mimir_un_op_expr) => {
                let expr = mimir_un_op_expr.expr.to_const_expr()?;

                Ok(MimirConstExpr::Unary(MimirUnOpExpr {
                    un_op: mimir_un_op_expr.un_op.clone(),
                    expr: Box::new(expr),
                }))
            }
            MimirExpr::Literal(mimir_lit) => Ok(MimirConstExpr::Literal(mimir_lit.clone())),
            MimirExpr::Var(_) => Err(MimirInlinePassError::AttemptedVarAccess),
            MimirExpr::MathIntrinsic(math_intrinsic_expr) => {
                let const_params = math_intrinsic_expr
                    .args
                    .iter()
                    .map(|p| p.to_const_expr())
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(MimirConstExpr::MathIntrinsic(MathIntrinsicExpr {
                    func: math_intrinsic_expr.func.clone(),
                    args: const_params,
                }))
            }
            MimirExpr::Cast(mimir_cast_expr) => {
                let const_expr = mimir_cast_expr.from.to_const_expr()?;

                Ok(MimirConstExpr::Cast(MimirCastExpr {
                    from: Box::new(const_expr),
                    to: mimir_cast_expr.to.clone(),
                }))
            }
            MimirExpr::ConstExpr(mimir_const_expr) => Ok(mimir_const_expr.clone()),
        }
    }
}
