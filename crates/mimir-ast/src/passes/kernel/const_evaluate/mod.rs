//! module defining const expression passes to group as many non-const expressions to const as possible
use crate::{
    ConstExpr, Expr, Intrinsic, KernelAstPass, MimirBinOpExpr, MimirExpr, MimirUnaryExpr,
    RichConstExpr, RichMimirExpr,
};

pub struct ConstEvaluator;

impl KernelAstPass for ConstEvaluator {
    type Error = ();

    type ExprOutput = Option<RichConstExpr>;

    fn expr(&mut self, expr: &mut crate::RichMimirExpr) -> Result<Self::ExprOutput, Self::Error> {
        match &mut expr.expr {
            crate::MimirExpr::BinOp(mimir_bin_op_expr) => {
                let lhs_opt = self.expr(&mut mimir_bin_op_expr.lhs)?;
                let rhs_opt = self.expr(&mut mimir_bin_op_expr.rhs)?;

                // If both are constant merge to one constant binop and pass it forward otherwise set each
                // side to be constant seperately
                if let Some(lhs) = &lhs_opt
                    && let Some(rhs) = &rhs_opt
                {
                    return Ok(Some(
                        RichConstExpr::new(
                            ConstExpr::BinOp(MimirBinOpExpr::new(
                                lhs.clone(),
                                mimir_bin_op_expr.op.clone(),
                                rhs.clone(),
                            )),
                            expr.span.clone(),
                        )
                        .set_p(expr.is_parenthesized),
                    ));
                } else if let Some(lhs) = lhs_opt {
                    mimir_bin_op_expr.lhs = Box::new(
                        RichMimirExpr::new(
                            MimirExpr::ConstExpr(lhs),
                            mimir_bin_op_expr.lhs.span.clone(),
                        )
                        .set_p(mimir_bin_op_expr.lhs.is_parenthesized),
                    );
                } else if let Some(rhs) = rhs_opt {
                    mimir_bin_op_expr.rhs = Box::new(
                        RichMimirExpr::new(
                            MimirExpr::ConstExpr(rhs),
                            mimir_bin_op_expr.rhs.span.clone(),
                        )
                        .set_p(mimir_bin_op_expr.rhs.is_parenthesized),
                    );
                }

                Ok(None)
            }
            crate::MimirExpr::Unary(mimir_unary_expr) => {
                match self.expr(&mut mimir_unary_expr.expr)? {
                    Some(un_expr) => {
                        let span = expr.span.clone();
                        Ok(Some(RichConstExpr::new(
                            ConstExpr::UnOp(MimirUnaryExpr {
                                op: mimir_unary_expr.op.clone(),
                                expr: Box::new(un_expr),
                            }),
                            span,
                        )))
                    }
                    None => Ok(None),
                }
            }
            crate::MimirExpr::Literal(literal) => Ok(Some(
                RichConstExpr::new(ConstExpr::Literal(*literal), expr.span.clone())
                    .set_p(expr.is_parenthesized),
            )),
            crate::MimirExpr::Var(_) => Ok(None),
            crate::MimirExpr::Index { .. } => Ok(None),
            crate::MimirExpr::Builtin { .. } => Ok(None),
            crate::MimirExpr::MathIntrinsic(intrinsic) => match intrinsic {
                Intrinsic::Unary(unary_intrinsic, param) => {
                    if let Some(e) = self.expr(param)? {
                        Ok(Some(
                            RichConstExpr::new(
                                ConstExpr::MathIntrinsic(Intrinsic::Unary(
                                    *unary_intrinsic,
                                    Box::new(e),
                                )),
                                expr.span.clone(),
                            )
                            .set_p(expr.is_parenthesized),
                        ))
                    } else {
                        Ok(None)
                    }
                }
                Intrinsic::Binary(binary_intrinsic, params) => {
                    let e1_opt = self.expr(unsafe { params.get_unchecked_mut(0) })?;
                    let e2_opt = self.expr(unsafe { params.get_unchecked_mut(1) })?;

                    if let Some(e1) = &e1_opt
                        && let Some(e2) = &e2_opt
                    {
                        Ok(Some(
                            RichConstExpr::new(
                                ConstExpr::MathIntrinsic(Intrinsic::Binary(
                                    *binary_intrinsic,
                                    Box::new([e1.clone(), e2.clone()]),
                                )),
                                expr.span.clone(),
                            )
                            .set_p(expr.is_parenthesized),
                        ))
                    } else if e1_opt.is_some() || e2_opt.is_some() {
                        let opt_arr = [e1_opt, e2_opt];

                        let new_params: [_; 2] = opt_arr
                            .iter()
                            .zip(params.iter())
                            .map(|(e_opt, mim_e)| match e_opt {
                                Some(e) => RichMimirExpr::new(
                                    MimirExpr::ConstExpr(e.clone()),
                                    e.span.clone(),
                                )
                                .set_p(e.is_parenthesized),
                                None => mim_e.clone(),
                            })
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap();

                        *intrinsic = Intrinsic::Binary(*binary_intrinsic, Box::new(new_params));

                        Ok(None)
                    } else {
                        Ok(None)
                    }
                }
                Intrinsic::Ternary(ternary_intrinsic, params) => {
                    let e1_opt = self.expr(unsafe { params.get_unchecked_mut(0) })?;
                    let e2_opt = self.expr(unsafe { params.get_unchecked_mut(1) })?;
                    let e3_opt = self.expr(unsafe { params.get_unchecked_mut(2) })?;

                    if let Some(e1) = &e1_opt
                        && let Some(e2) = &e2_opt
                        && let Some(e3) = &e3_opt
                    {
                        Ok(Some(
                            RichConstExpr::new(
                                ConstExpr::MathIntrinsic(Intrinsic::Ternary(
                                    *ternary_intrinsic,
                                    Box::new([e1.clone(), e2.clone(), e3.clone()]),
                                )),
                                expr.span.clone(),
                            )
                            .set_p(expr.is_parenthesized),
                        ))
                    } else if e1_opt.is_some() || e2_opt.is_some() || e3_opt.is_some() {
                        let opt_arr = [e1_opt, e2_opt, e3_opt];

                        let new_params: [_; 3] = opt_arr
                            .iter()
                            .zip(params.iter())
                            .map(|(e_opt, mim_e)| match e_opt {
                                Some(e) => RichMimirExpr::new(
                                    MimirExpr::ConstExpr(e.clone()),
                                    e.span.clone(),
                                )
                                .set_p(e.is_parenthesized),
                                None => mim_e.clone(),
                            })
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap();

                        *intrinsic = Intrinsic::Ternary(*ternary_intrinsic, Box::new(new_params));

                        Ok(None)
                    } else {
                        Ok(None)
                    }
                }
            },
            crate::MimirExpr::Cast { value, to } => {
                if let Some(e) = self.expr(value)? {
                    Ok(Some(
                        RichConstExpr::new(
                            ConstExpr::Cast {
                                value: Box::new(e),
                                to: *to,
                            },
                            expr.span.clone(),
                        )
                        .set_p(expr.is_parenthesized),
                    ))
                } else {
                    Ok(None)
                }
            }
            crate::MimirExpr::ConstExpr(rich_expr) => self.const_expr(rich_expr),
        }
    }

    fn const_expr(
        &mut self,
        expr: &mut crate::RichConstExpr,
    ) -> Result<Self::ExprOutput, Self::Error> {
        Ok(Some(expr.clone()))
    }
}
