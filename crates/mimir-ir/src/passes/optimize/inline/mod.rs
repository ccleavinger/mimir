use std::iter::zip;

use crate::{
    ir::{
        MathIntrinsicExpr, MimirBinOpExpr, MimirCastExpr, MimirConstExpr, MimirExpr,
        MimirIndexExpr, MimirStmt, MimirStmtAssignLeft, MimirTy, MimirUnOpExpr,
    },
    passes::{optimize::inline::inline_expr::expr_to_lit_recur, KernelIRJitPass},
    util::error::{MimirInlinePassError, MultiPassError},
};

pub(crate) mod inline_expr;

pub(crate) struct InlineExprPass;

impl InlineExprPass {
    fn stmt_pass_priv(
        stmt: &MimirStmt,
        const_generics: &[u32],
    ) -> Result<MimirStmt, MimirInlinePassError> {
        match stmt {
            MimirStmt::Assign { lhs, rhs } => {
                let ret_lhs = match &lhs {
                    MimirStmtAssignLeft::Index(mimir_index_expr) => {
                        let new_idx = Self::expr_inline(
                            &MimirExpr::Index(mimir_index_expr.clone()),
                            const_generics,
                        )?;
                        MimirStmtAssignLeft::Index(match new_idx {
                            MimirExpr::Index(idx) => idx,
                            _ => unreachable!(),
                        })
                    }
                    MimirStmtAssignLeft::Var(_) => lhs.clone(),
                };

                let ret_rhs = Self::expr_inline(rhs, const_generics)?;

                Ok(MimirStmt::Assign {
                    lhs: ret_lhs,
                    rhs: ret_rhs,
                })
            }
            MimirStmt::RangeFor {
                var,
                start,
                end,
                step,
                body,
            } => {
                let ret_start = Self::expr_inline(start, const_generics)?;
                let ret_end = Self::expr_inline(end, const_generics)?;

                let ret_step = match step {
                    Some(st) => Some(Self::expr_inline(st, const_generics)?),
                    None => None,
                };

                let ret_body = body
                    .iter()
                    .map(|body_stmt| Self::stmt_pass_priv(body_stmt, const_generics))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(MimirStmt::RangeFor {
                    var: *var,
                    start: ret_start,
                    end: ret_end,
                    step: ret_step,
                    body: ret_body,
                })
            }
            MimirStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let ret_cond = Self::expr_inline(condition, const_generics)?;

                let ret_then = then_branch
                    .iter()
                    .map(|then_stmt| Self::stmt_pass_priv(then_stmt, const_generics))
                    .collect::<Result<Vec<_>, _>>()?;

                let ret_else = match else_branch {
                    Some(some_else) => Some(
                        some_else
                            .iter()
                            .map(|else_stmt| Self::stmt_pass_priv(else_stmt, const_generics))
                            .collect::<Result<Vec<_>, _>>()?,
                    ),
                    None => None,
                };

                Ok(MimirStmt::If {
                    condition: ret_cond,
                    then_branch: ret_then,
                    else_branch: ret_else,
                })
            }
            MimirStmt::Return(_) => Ok(stmt.clone()),
            MimirStmt::Syncthreads => Ok(stmt.clone()),
        }
    }

    fn expr_inline(
        expr: &MimirExpr,
        const_generics: &[u32],
    ) -> Result<MimirExpr, MimirInlinePassError> {
        match expr {
            MimirExpr::BinOp(MimirBinOpExpr {
                lhs,
                op,
                rhs,
                is_parenthesized,
            }) => {
                if expr.is_const() {
                    return Ok(MimirExpr::Literal(expr_to_lit_recur(
                        &expr.to_const_expr()?,
                        const_generics,
                    )?));
                }

                let is_lhs_const = lhs.is_const();
                let is_rhs_const = rhs.is_const();

                if is_lhs_const && is_rhs_const {
                    return Ok(MimirExpr::Literal(expr_to_lit_recur(
                        &expr.to_const_expr()?,
                        const_generics,
                    )?));
                }

                let ret_lhs = if is_lhs_const {
                    &(expr_to_lit_recur(&lhs.to_const_expr()?, const_generics)?).to_expr()
                } else {
                    &Self::expr_inline(lhs.as_ref(), const_generics)?
                };
                let ret_rhs = if is_rhs_const {
                    &(expr_to_lit_recur(&rhs.to_const_expr()?, const_generics)?).to_expr()
                } else {
                    &Self::expr_inline(rhs.as_ref(), const_generics)?
                };

                Ok(MimirExpr::BinOp(MimirBinOpExpr {
                    lhs: Box::new(ret_lhs.clone()),
                    op: op.clone(),
                    rhs: Box::new(ret_rhs.clone()),
                    is_parenthesized: *is_parenthesized,
                }))
            }
            MimirExpr::Index(MimirIndexExpr { var, index }) => {
                let ret_idx = if index.is_const() {
                    (expr_to_lit_recur(&index.to_const_expr()?, const_generics)?).to_expr()
                } else {
                    Self::expr_inline(index.as_ref(), const_generics)?
                };

                Ok(MimirExpr::Index(MimirIndexExpr {
                    var: *var,
                    index: Box::new(ret_idx),
                }))
            }
            MimirExpr::BuiltinFieldAccess { .. } => Ok(expr.clone()),
            MimirExpr::Unary(mimir_un_op_expr) => {
                let ret_expr = if expr.is_const() {
                    expr_to_lit_recur(&mimir_un_op_expr.expr.to_const_expr()?, const_generics)?
                        .to_expr()
                } else {
                    Self::expr_inline(&mimir_un_op_expr.expr, const_generics)?
                };

                Ok(MimirExpr::Unary(MimirUnOpExpr {
                    un_op: mimir_un_op_expr.un_op.clone(),
                    expr: Box::new(ret_expr),
                }))
            }
            MimirExpr::Literal(_) => Ok(expr.clone()),
            MimirExpr::Var(_) => Ok(expr.clone()),
            MimirExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
                if expr.is_const() {
                    return Ok(MimirExpr::Literal(expr_to_lit_recur(
                        &expr.to_const_expr()?,
                        const_generics,
                    )?));
                }

                let is_arg_const_vec = args.iter().map(|arg| arg.is_const()).collect::<Vec<_>>();

                {
                    // lowk disgusting
                    let ret_args = zip(args, is_arg_const_vec)
                        .map(|(arg, is_const)| {
                            if is_const {
                                arg.to_const_expr().map(|x| {
                                    expr_to_lit_recur(&x, const_generics).map(MimirExpr::Literal)
                                })
                            } else {
                                Ok(Self::expr_inline(arg, const_generics))
                            }
                        })
                        .collect::<Result<Result<Vec<_>, _>, _>>()??;

                    Ok(MimirExpr::MathIntrinsic(MathIntrinsicExpr {
                        func: func.clone(),
                        args: ret_args,
                    }))
                }
            }
            MimirExpr::Cast(MimirCastExpr { from, to }) => {
                let ret_from = if from.is_const() {
                    expr_to_lit_recur(&from.to_const_expr()?, const_generics)?.to_expr()
                } else {
                    Self::expr_inline(from.as_ref(), const_generics)?
                };

                Ok(MimirExpr::Cast(MimirCastExpr {
                    from: Box::new(ret_from),
                    to: to.clone(),
                }))
            }
            MimirExpr::ConstExpr(mimir_const_expr) => {
                Ok(expr_to_lit_recur(mimir_const_expr, const_generics)?.to_expr())
            }
        }
    }
}

impl KernelIRJitPass for InlineExprPass {
    fn stmt_pass(
        stmt: &MimirStmt,
        const_generics: &[u32],
    ) -> Result<Vec<MimirStmt>, MultiPassError> {
        Ok(vec![
            Self::stmt_pass_priv(stmt, const_generics).map_err(MultiPassError::Inline)?
        ])
    }

    fn ty_pass_support() -> bool {
        true
    }

    fn ty_pass(_ty: &MimirTy, _const_generics: &[u32]) -> Result<MimirTy, MultiPassError> {
        Ok(if let MimirTy::SharedMemArray { length } = _ty {
            let lit = expr_to_lit_recur(length, _const_generics).map_err(MultiPassError::Inline)?;
            MimirTy::SharedMemArray {
                length: Box::new(MimirConstExpr::Literal(lit)),
            }
        } else {
            _ty.clone()
        })
    }
}
