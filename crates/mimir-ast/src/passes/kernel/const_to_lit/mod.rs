use crate::{
    BinaryIntrinsic, ConstExpr, Expr, Intrinsic, IntrinsicReport, KernelAstPass, Literal,
    MimirBinOp, MimirExpr, MimirUnOp, PrimitiveTy, RichConstExpr, RichMimirExpr, UnaryIntrinsic,
    passes::kernel::const_to_lit::util::{bin_op_math, handle_partial_opt_intrinsic},
};

mod util;

pub struct ConstToLitPass;

impl KernelAstPass for ConstToLitPass {
    type Error = ();

    type ExprOutput = Option<Literal>;

    fn expr(&mut self, expr: &mut RichMimirExpr) -> Result<Option<Literal>, ()> {
        match &mut expr.expr {
            MimirExpr::ConstExpr(rich_expr) => {
                if let Some(lit) = self.const_expr(rich_expr)? {
                    *expr = RichMimirExpr::new(MimirExpr::Literal(lit), expr.span.clone());
                }

                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn const_expr(&mut self, expr: &mut RichConstExpr) -> Result<Self::ExprOutput, Self::Error> {
        match &mut expr.expr {
            ConstExpr::BinOp(mimir_bin_op_expr) => {
                let (lhs_lit_opt, rhs_lit_opt) =
                    if mimir_bin_op_expr.rhs.priority() > mimir_bin_op_expr.lhs.priority() {
                        let rhs = self.const_expr(&mut mimir_bin_op_expr.rhs)?;
                        let lhs = self.const_expr(&mut mimir_bin_op_expr.lhs)?;

                        (lhs, rhs)
                    } else {
                        (
                            self.const_expr(&mut mimir_bin_op_expr.lhs)?,
                            self.const_expr(&mut mimir_bin_op_expr.rhs)?,
                        )
                    };

                if let Some(lhs_lit) = lhs_lit_opt
                    && let Some(rhs_lit) = rhs_lit_opt
                {
                    debug_assert!(lhs_lit.ty() == rhs_lit.ty());

                    let res: Literal = match (lhs_lit, rhs_lit) {
                        (Literal::Bool(l_b), Literal::Bool(r_b)) => match mimir_bin_op_expr.op {
                            MimirBinOp::And => Literal::Bool(l_b && r_b),
                            MimirBinOp::Or => Literal::Bool(l_b || r_b),
                            _ => unreachable!(),
                        },
                        (Literal::Int32(l_i), Literal::Int32(r_i)) => {
                            Literal::Int32(bin_op_math(l_i, &mimir_bin_op_expr.op, r_i))
                        }
                        (Literal::Uint32(l_u), Literal::Uint32(r_u)) => {
                            Literal::Uint32(bin_op_math(l_u, &mimir_bin_op_expr.op, r_u))
                        }
                        (Literal::Float32(l_f_bits), Literal::Float32(r_f_bits)) => {
                            let l_f = f32::from_bits(l_f_bits);
                            let r_f = f32::from_bits(r_f_bits);
                            Literal::Float32(bin_op_math(l_f, &mimir_bin_op_expr.op, r_f).to_bits())
                        }
                        _ => unreachable!(),
                    };

                    return Ok(Some(res));
                } else if let Some(lhs_lit) = lhs_lit_opt {
                    *mimir_bin_op_expr.lhs = RichConstExpr::new(
                        ConstExpr::Literal(lhs_lit),
                        mimir_bin_op_expr.lhs.span.clone(),
                    );
                } else if let Some(rhs_lit) = rhs_lit_opt {
                    *mimir_bin_op_expr.rhs = RichConstExpr::new(
                        ConstExpr::Literal(rhs_lit),
                        mimir_bin_op_expr.rhs.span.clone(),
                    );
                }

                Ok(None)
            }
            ConstExpr::UnOp(mimir_unary_expr) => {
                if let Some(un_expr) = self.const_expr(&mut mimir_unary_expr.expr)? {
                    Ok(Some(match un_expr {
                        Literal::Int32(i) => match mimir_unary_expr.op {
                            MimirUnOp::Neg => Literal::Int32(-i),
                            MimirUnOp::Not => unreachable!(),
                        },
                        Literal::Uint32(_) => match mimir_unary_expr.op {
                            MimirUnOp::Neg => unreachable!(),
                            MimirUnOp::Not => unreachable!(),
                        },
                        Literal::Float32(f_bits) => match mimir_unary_expr.op {
                            MimirUnOp::Neg => Literal::Float32((-f32::from_bits(f_bits)).to_bits()),
                            MimirUnOp::Not => unreachable!(),
                        },
                        Literal::Bool(b) => match mimir_unary_expr.op {
                            MimirUnOp::Neg => unreachable!(),
                            MimirUnOp::Not => Literal::Bool(!b),
                        },
                    }))
                } else {
                    Ok(None)
                }
            }
            ConstExpr::Literal(literal) => Ok(Some(*literal)),
            ConstExpr::Generic(_) => Ok(None),
            ConstExpr::MathIntrinsic(intrinsic) => match intrinsic {
                Intrinsic::Unary(unary_intrinsic, param) => {
                    if let Some(lit) = self.const_expr(param)? {
                        debug_assert!(
                            lit.ty().is_numeric() == unary_intrinsic.is_numeric()
                                || lit.ty().is_float() == unary_intrinsic.float_only()
                        );

                        let f = match lit {
                            Literal::Float32(f) => f32::from_bits(f),
                            _ => unreachable!(),
                        };

                        Ok(Some(Literal::Float32(
                            match unary_intrinsic {
                                UnaryIntrinsic::Sin => f.sin(),
                                UnaryIntrinsic::Cos => f.cos(),
                                UnaryIntrinsic::Tan => f.tan(),
                                UnaryIntrinsic::Asin => f.asin(),
                                UnaryIntrinsic::Acos => f.acos(),
                                UnaryIntrinsic::Atan => f.atan(),
                                UnaryIntrinsic::Sinh => f.sinh(),
                                UnaryIntrinsic::Cosh => f.cosh(),
                                UnaryIntrinsic::Tanh => f.tanh(),
                                UnaryIntrinsic::Asinh => f.asinh(),
                                UnaryIntrinsic::Acosh => f.acosh(),
                                UnaryIntrinsic::Atanh => f.atanh(),
                                UnaryIntrinsic::Exp => f.exp(),
                                UnaryIntrinsic::Log => f.ln(),
                                UnaryIntrinsic::Exp2 => f.exp2(),
                                UnaryIntrinsic::Log2 => f.log2(),
                                UnaryIntrinsic::Sqrt => f.sqrt(),
                                UnaryIntrinsic::Isqrt => 1.0 / f.sqrt(),
                                UnaryIntrinsic::Floor => f.floor(),
                                UnaryIntrinsic::Ceil => f.ceil(),
                            }
                            .to_bits(),
                        )))
                    } else {
                        Ok(None)
                    }
                }
                Intrinsic::Binary(binary_intrinsic, params) => {
                    let lit_opt1 = self.const_expr(unsafe { params.get_unchecked_mut(0) })?;
                    let lit_opt2 = self.const_expr(unsafe { params.get_unchecked_mut(1) })?;

                    if let Some(lit1) = lit_opt1
                        && let Some(lit2) = lit_opt2
                    {
                        debug_assert!(
                            (lit1.ty().is_numeric() == binary_intrinsic.is_numeric()
                                && lit2.ty().is_numeric() == binary_intrinsic.is_numeric())
                                && (lit1.ty().is_float() == binary_intrinsic.float_only()
                                    && lit2.ty().is_float() == binary_intrinsic.float_only())
                                && (lit1.ty() == lit2.ty())
                        );

                        if let Literal::Float32(f1_bits) = lit1
                            && let Literal::Float32(f2_bits) = lit2
                        {
                            let f1 = f32::from_bits(f1_bits);
                            let f2 = f32::from_bits(f2_bits);
                            return Ok(Some(Literal::Float32(
                                match binary_intrinsic {
                                    BinaryIntrinsic::Atan2 => f1.atan2(f2),
                                    BinaryIntrinsic::Pow => f1.powf(f2),
                                    BinaryIntrinsic::Max => f1.max(f2),
                                    BinaryIntrinsic::Min => f1.max(f2),
                                }
                                .to_bits(),
                            )));
                        } else {
                            Ok(Some(match (lit1, lit2) {
                                (Literal::Uint32(u1), Literal::Uint32(u2)) => {
                                    Literal::Uint32(match binary_intrinsic {
                                        BinaryIntrinsic::Max => u1.max(u2),
                                        BinaryIntrinsic::Min => u1.max(u2),
                                        _ => unreachable!(),
                                    })
                                }
                                (Literal::Int32(i1), Literal::Int32(i2)) => {
                                    Literal::Int32(match binary_intrinsic {
                                        BinaryIntrinsic::Max => i1.max(i2),
                                        BinaryIntrinsic::Min => i1.max(i2),
                                        _ => unreachable!(),
                                    })
                                }
                                _ => unreachable!(),
                            }))
                        }
                    } else if lit_opt1.is_some() || lit_opt2.is_some() {
                        let opt_arr = [lit_opt1, lit_opt2];

                        let new_params = handle_partial_opt_intrinsic(opt_arr, params);

                        *params = Box::new(new_params);

                        Ok(None)
                    } else {
                        Ok(None)
                    }
                }
                Intrinsic::Ternary(ternary_intrinsic, params) => {
                    let lit_opt1 = self.const_expr(unsafe { params.get_unchecked_mut(0) })?;
                    let lit_opt2 = self.const_expr(unsafe { params.get_unchecked_mut(1) })?;
                    let lit_opt3 = self.const_expr(unsafe { params.get_unchecked_mut(2) })?;

                    if let Some(lit1) = lit_opt1
                        && let Some(lit2) = lit_opt2
                        && let Some(lit3) = lit_opt3
                    {
                        debug_assert!(
                            (lit1.ty().is_numeric() == ternary_intrinsic.is_numeric()
                                && lit2.ty().is_numeric() == ternary_intrinsic.is_numeric()
                                && lit3.ty().is_numeric() == ternary_intrinsic.is_numeric())
                                && (lit1.ty().is_float() == ternary_intrinsic.float_only()
                                    && lit2.ty().is_float() == ternary_intrinsic.float_only()
                                    && lit3.ty().is_float() == ternary_intrinsic.float_only())
                                && (lit1.ty() == lit2.ty() && lit1.ty() == lit3.ty())
                        );

                        if let Literal::Float32(f1_bits) = lit1
                            && let Literal::Float32(f2_bits) = lit2
                            && let Literal::Float32(f3_bits) = lit3
                        {
                            let f1 = f32::from_bits(f1_bits);
                            let f2 = f32::from_bits(f2_bits);
                            let f3 = f32::from_bits(f3_bits);

                            return Ok(Some(Literal::Float32(match ternary_intrinsic {
                                crate::TernaryIntrinsic::Clamp => f1.max(f2).min(f3).to_bits(),
                                crate::TernaryIntrinsic::Mix => {
                                    // Result is the linear blend of x and y, i.e., x * (1 - a) + y * a.
                                    let float = f1 * (1.0 - f3) + f2 * f3;
                                    float.to_bits()
                                }
                                crate::TernaryIntrinsic::Fma => (f1 * f2 + f3).to_bits(),
                            })));
                        } else {
                            Ok(Some(match (lit1, lit2, lit3) {
                                (Literal::Int32(i1), Literal::Int32(i2), Literal::Int32(i3)) => {
                                    Literal::Int32(match ternary_intrinsic {
                                        crate::TernaryIntrinsic::Clamp => i1.max(i2).min(i3),
                                        crate::TernaryIntrinsic::Mix => unreachable!(),
                                        crate::TernaryIntrinsic::Fma => unreachable!(),
                                    })
                                }
                                (Literal::Uint32(u1), Literal::Uint32(u2), Literal::Uint32(u3)) => {
                                    Literal::Uint32(match ternary_intrinsic {
                                        crate::TernaryIntrinsic::Clamp => u1.max(u2).min(u3),
                                        crate::TernaryIntrinsic::Mix => unreachable!(),
                                        crate::TernaryIntrinsic::Fma => unreachable!(),
                                    })
                                }
                                _ => unreachable!(),
                            }))
                        }
                    } else if lit_opt1.is_some() || lit_opt2.is_some() || lit_opt3.is_some() {
                        let opt_arr = [lit_opt1, lit_opt2, lit_opt3];

                        let new_params = handle_partial_opt_intrinsic(opt_arr, params);

                        *params = Box::new(new_params);

                        Ok(None)
                    } else {
                        Ok(None)
                    }
                }
            },
            ConstExpr::Cast { value, to } => {
                if let Some(lit) = self.const_expr(value)? {
                    Ok(Some(match lit {
                        Literal::Int32(i) => match to {
                            PrimitiveTy::Int32 => lit,
                            PrimitiveTy::Uint32 => Literal::Uint32(i as u32),
                            PrimitiveTy::Float32 => Literal::Float32((i as f32).to_bits()),
                            PrimitiveTy::Bool => Literal::Bool(i != 0),
                        },
                        Literal::Uint32(u) => match to {
                            PrimitiveTy::Int32 => Literal::Int32(u as i32),
                            PrimitiveTy::Uint32 => lit,
                            PrimitiveTy::Float32 => Literal::Float32((u as f32).to_bits()),
                            PrimitiveTy::Bool => Literal::Bool(u != 0),
                        },
                        Literal::Float32(f_bits) => {
                            let f = f32::from_bits(f_bits);
                            match to {
                                PrimitiveTy::Int32 => Literal::Int32(f as i32),
                                PrimitiveTy::Uint32 => Literal::Uint32(f as u32),
                                PrimitiveTy::Float32 => lit,
                                PrimitiveTy::Bool => Literal::Bool(f != 0.0),
                            }
                        }
                        Literal::Bool(b) => match to {
                            PrimitiveTy::Int32 => Literal::Int32(b as i32),
                            PrimitiveTy::Uint32 => Literal::Uint32(b as u32),
                            PrimitiveTy::Float32 => Literal::Float32((b as i32 as f32).to_bits()),
                            PrimitiveTy::Bool => lit,
                        },
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
