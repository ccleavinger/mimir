use std::ops::{Add, Div, Mul, Sub};

use crate::{
    ir::{
        MathIntrinsicExpr, MathIntrinsicFunc, MimirBinOp, MimirBinOpExpr, MimirCastExpr,
        MimirConstExpr, MimirLit, MimirPrimitiveTy, MimirUnOp, MimirUnOpExpr,
    },
    util::{
        error::{InvalidCastError, MimirInlinePassError, PostSafetyPassSpecFailError},
        math::intrinsic_param_count,
    },
};

// use crate::VulkanSpirVCompiler;

// Helper macros to reduce boilerplate
macro_rules! apply_binary_op {
    ($lhs:expr, $rhs:expr, $op:ident) => {
        match ($lhs, $rhs) {
            (MimirLit::Int32(lhs_i), MimirLit::Int32(rhs_i)) => {
                Ok(MimirLit::Int32(lhs_i.$op(rhs_i)))
            }
            (MimirLit::Uint32(lhs_u), MimirLit::Uint32(rhs_u)) => {
                Ok(MimirLit::Uint32(lhs_u.$op(rhs_u)))
            }
            (MimirLit::Float32(lhs_b), MimirLit::Float32(rhs_b)) => {
                let lhs_f = f32::from_bits(lhs_b);
                let rhs_f = f32::from_bits(rhs_b);
                Ok(MimirLit::Float32(lhs_f.$op(rhs_f).to_bits()))
            }
            _ => Err(MimirInlinePassError::GenericTyMismatch),
        }
    };
}

macro_rules! apply_bool_binary_op {
    ($lhs:expr, $rhs:expr, $op:ident) => {
        match (&$lhs, &$rhs) {
            (MimirLit::Bool(lhs_b), MimirLit::Bool(rhs_b)) => Ok(MimirLit::Bool(lhs_b.$op(&rhs_b))),
            _ => Err(MimirInlinePassError::TypeMismatch(
                MimirPrimitiveTy::Bool,
                Box::new([$lhs.clone(), $rhs.clone()]),
            )),
        }
    };
}

macro_rules! single_var_f_math_intrinsic {
    ($lit:expr, $func:ident) => {
        return match $lit {
            MimirLit::Float32(f_bits) => {
                let f = f32::from_bits(f_bits);
                Ok(MimirLit::Float32(f.$func().to_bits()))
            },
            _ => Err(MimirInlinePassError::Generic(
                "Cannot inline a single var floating point math intrinsic on a non-floating point literal".to_owned()
            ))
        }
    };
}

macro_rules! two_var_f_math_intrinsic {
    ($lit1:expr, $lit2:expr, $func:ident) => {
        return match (&$lit1, &$lit2) {
            (MimirLit::Float32(f1_b), MimirLit::Float32(f2_b)) => {
                let f1 = f32::from_bits(*f1_b);
                let f2 = f32::from_bits(*f2_b);
                Ok(MimirLit::Float32(f1.$func(f2).to_bits()))
            },
            _ => Err(MimirInlinePassError::Generic(
                "Cannot inline a two var floating point math intrinsic on a non-floating point literal".to_owned()
            ))
        }
    };
}

macro_rules! two_var_cmp_intrinsic {
    ($lit1:expr, $lit2:expr, $func:ident) => {
        match (&$lit1, &$lit2) {
            (MimirLit::Int32(i1), MimirLit::Int32(i2)) => return Ok(MimirLit::Int32(*i1.$func(i2))),
            (MimirLit::Uint32(u1), MimirLit::Uint32(u2)) => {
                return Ok(MimirLit::Uint32(*u1.$func(u2)))
            }
            (MimirLit::Float32(f1_b), MimirLit::Float32(f2_b)) => {
                let f1 = f32::from_bits(*f1_b);
                let f2 = f32::from_bits(*f2_b);
                return Ok(MimirLit::Float32(f1.$func(f2).to_bits()));
            }
            _ => {
                return Err(MimirInlinePassError::InvalidIntrinsicParams(
                    Box::new([$lit1.clone(), $lit2.clone()]),
                    stringify!($func).to_string(),
                ))
            }
        }
    };
}

macro_rules! three_var_cmp_intrinsic {
    ($lit1:expr, $lit2:expr, $lit3:expr, $func:ident) => {
        match (&$lit1, &$lit2, &$lit3) {
            (MimirLit::Int32(i1), MimirLit::Int32(i2), MimirLit::Int32(i3)) => {
                return Ok(MimirLit::Int32(*i1.$func(i2, i3)))
            }
            (MimirLit::Uint32(u1), MimirLit::Uint32(u2), MimirLit::Uint32(u3)) => {
                return Ok(MimirLit::Uint32(*u1.$func(u2, u3)))
            }
            (MimirLit::Float32(f1_b), MimirLit::Float32(f2_b), MimirLit::Float32(f3_b)) => {
                let f1 = f32::from_bits(*f1_b);
                let f2 = f32::from_bits(*f2_b);
                let f3 = f32::from_bits(*f3_b);
                return Ok(MimirLit::Float32(f1.$func(f2, f3).to_bits()));
            }
            _ => {
                return Err(MimirInlinePassError::InvalidIntrinsicParams(
                    Box::new([$lit1.clone(), $lit2.clone(), $lit3.clone()]),
                    stringify!($func).to_string(),
                ))
            }
        }
    };
}

pub(crate) fn expr_to_lit_recur(
    expr: &MimirConstExpr,
    const_generics: &[u32],
) -> Result<MimirLit, MimirInlinePassError> {
    match expr {
        MimirConstExpr::BinOp(MimirBinOpExpr { lhs, op, rhs, .. }) => {
            let lhs_precedence = lhs.precedence();
            let rhs_precedence = rhs.precedence();

            let (lhs_lit, rhs_lit) = if rhs_precedence > lhs_precedence {
                let rhs_lit = expr_to_lit_recur(rhs.as_ref(), const_generics)?;
                let lhs_lit = expr_to_lit_recur(lhs.as_ref(), const_generics)?;

                (lhs_lit, rhs_lit)
            } else {
                let lhs_lit = expr_to_lit_recur(lhs.as_ref(), const_generics)?;
                let rhs_lit = expr_to_lit_recur(rhs.as_ref(), const_generics)?;

                (lhs_lit, rhs_lit)
            };

            let lhs_ty = lhs_lit.to_ty();
            let rhs_ty = rhs_lit.to_ty();

            match op {
                MimirBinOp::Add
                | MimirBinOp::Sub
                | MimirBinOp::Mul
                | MimirBinOp::Div
                | MimirBinOp::Mod
                | MimirBinOp::Lt
                | MimirBinOp::Lte
                | MimirBinOp::Gt
                | MimirBinOp::Gte
                | MimirBinOp::Eq
                | MimirBinOp::Ne => {
                    if !(lhs_ty.is_numeric() && rhs_ty.is_numeric()) {
                        return Err(MimirInlinePassError::Generic(
                                format!("Both types must be numeric for mathematic operations:\n\tFailed to inline a `{op:?}` binary operation")
                            ));
                    }
                }
                _ => {
                    if !(lhs_ty.is_bool() && rhs_ty.is_bool()) {
                        return Err(MimirInlinePassError::Generic(
                                format!("Both types must be booleans for logical boolean operations:\n\tFailed to inline a `{op:?}` binary operation")
                            ));
                    }
                }
            }

            if lhs_ty != rhs_ty {
                return Err(MimirInlinePassError::Generic(
                        format!(
                            "The left hand side and right hand side of a binary operation must have matching types.\n\tInstead encountered a `{lhs_ty:?}` and `{rhs_ty:?} "
                        )
                    ));
            }

            match op {
                MimirBinOp::Add => {
                    apply_binary_op!(lhs_lit, rhs_lit, add)
                }
                MimirBinOp::Sub => {
                    apply_binary_op!(lhs_lit, rhs_lit, sub)
                }
                MimirBinOp::Mul => {
                    apply_binary_op!(lhs_lit, rhs_lit, mul)
                }
                MimirBinOp::Div => {
                    apply_binary_op!(lhs_lit, rhs_lit, div)
                }
                MimirBinOp::Mod => match (lhs_lit, rhs_lit) {
                    (MimirLit::Int32(lhs_i), MimirLit::Int32(rhs_i)) => {
                        Ok(MimirLit::Int32(lhs_i % (rhs_i)))
                    }
                    (MimirLit::Uint32(lhs_u), MimirLit::Uint32(rhs_u)) => {
                        Ok(MimirLit::Uint32(lhs_u % (rhs_u)))
                    }
                    (MimirLit::Float32(lhs_b), MimirLit::Float32(rhs_b)) => {
                        let lhs_f = f32::from_bits(lhs_b);
                        let rhs_f = f32::from_bits(rhs_b);
                        Ok(MimirLit::Float32((lhs_f % rhs_f).to_bits()))
                    }
                    _ => panic!("Type mismatch"),
                },
                MimirBinOp::And => match (lhs_lit, rhs_lit) {
                    (MimirLit::Bool(lhs_b), MimirLit::Bool(rhs_b)) => {
                        Ok(MimirLit::Bool(lhs_b && rhs_b))
                    }
                    _ => panic!(),
                },
                MimirBinOp::Or => match (lhs_lit, rhs_lit) {
                    (MimirLit::Bool(lhs_b), MimirLit::Bool(rhs_b)) => {
                        Ok(MimirLit::Bool(lhs_b || rhs_b))
                    }
                    _ => panic!("Type mismatch"),
                },
                MimirBinOp::Lt => apply_bool_binary_op!(lhs_lit, rhs_lit, lt),
                MimirBinOp::Lte => apply_bool_binary_op!(lhs_lit, rhs_lit, le),
                MimirBinOp::Gt => apply_bool_binary_op!(lhs_lit, rhs_lit, gt),
                MimirBinOp::Gte => apply_bool_binary_op!(lhs_lit, rhs_lit, ge),
                MimirBinOp::Eq => apply_bool_binary_op!(lhs_lit, rhs_lit, eq),
                MimirBinOp::Ne => apply_bool_binary_op!(lhs_lit, rhs_lit, ne),
            }
        }
        MimirConstExpr::Unary(MimirUnOpExpr { un_op, expr }) => {
            let lit = expr_to_lit_recur(expr, const_generics)?;

            match un_op {
                MimirUnOp::Not => {
                    if let MimirLit::Bool(b) = lit {
                        Ok(MimirLit::Bool(!b))
                    } else {
                        Err(MimirInlinePassError::Generic(
                            "Can't apply a `not` unary operation on a non boolean literal"
                                .to_string(),
                        ))
                    }
                }
                MimirUnOp::Neg => match lit {
                    MimirLit::Int32(i) => Ok(MimirLit::Int32(-i)),
                    MimirLit::Float32(f_bits) => {
                        let f = f32::from_bits(f_bits);

                        Ok(MimirLit::Float32((-f).to_bits()))
                    }
                    MimirLit::Bool(_) => Err(MimirInlinePassError::Generic(
                        "Can't negate a boolean literal".to_owned(),
                    )),
                    MimirLit::Uint32(_) => Err(MimirInlinePassError::Generic(
                        "Can't negate an unsigned 32 bit integer literal".to_owned(),
                    )),
                },
            }
        }
        MimirConstExpr::Literal(mimir_lit) => Ok(mimir_lit.clone()),
        MimirConstExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
            let lits = args
                .iter()
                .map(|x| expr_to_lit_recur(x, const_generics))
                .collect::<Result<Vec<_>, _>>()?;

            let n_params = intrinsic_param_count(func);

            if lits.len() != n_params {
                return Err(MimirInlinePassError::AttemptedVarAccess);
            }

            match func {
                    MathIntrinsicFunc::Sin => single_var_f_math_intrinsic!(lits[0], sin),
                    MathIntrinsicFunc::Cos => single_var_f_math_intrinsic!(lits[0], cos),
                    MathIntrinsicFunc::Tan => single_var_f_math_intrinsic!(lits[0], tan),
                    MathIntrinsicFunc::Asin => single_var_f_math_intrinsic!(lits[0], asin),
                    MathIntrinsicFunc::Acos => single_var_f_math_intrinsic!(lits[0], acos),
                    MathIntrinsicFunc::Atan => single_var_f_math_intrinsic!(lits[0], atan),
                    MathIntrinsicFunc::Sinh => single_var_f_math_intrinsic!(lits[0], sinh),
                    MathIntrinsicFunc::Cosh => single_var_f_math_intrinsic!(lits[0], cosh),
                    MathIntrinsicFunc::Tanh => single_var_f_math_intrinsic!(lits[0], tanh),
                    MathIntrinsicFunc::Asinh => single_var_f_math_intrinsic!(lits[0], asinh),
                    MathIntrinsicFunc::Acosh => single_var_f_math_intrinsic!(lits[0], acosh),
                    MathIntrinsicFunc::Atanh => single_var_f_math_intrinsic!(lits[0], atanh),
                    MathIntrinsicFunc::Atan2 => two_var_f_math_intrinsic!(lits[0], lits[1], atan2),
                    MathIntrinsicFunc::Pow => two_var_f_math_intrinsic!(lits[0], lits[1], powf),
                    MathIntrinsicFunc::Exp => single_var_f_math_intrinsic!(lits[0], exp),
                    MathIntrinsicFunc::Log => single_var_f_math_intrinsic!(lits[0], ln),
                    MathIntrinsicFunc::Exp2 => single_var_f_math_intrinsic!(lits[0], exp2),
                    MathIntrinsicFunc::Log2 => single_var_f_math_intrinsic!(lits[0], log2),
                    MathIntrinsicFunc::Sqrt => single_var_f_math_intrinsic!(lits[0], sqrt),
                    MathIntrinsicFunc::Isqrt => match lits[0] {
                        MimirLit::Float32(f_bits) => {
                            let f = f32::from_bits(f_bits);
                            Ok(MimirLit::Float32(f.sqrt().recip().to_bits()))
                        },
                        _ => Err(MimirInlinePassError::Generic(
                            "Cannot inline a single var floating point math intrinsic on a non-floating point literal".to_owned()
                        ))
                    },
                    MathIntrinsicFunc::Max => two_var_cmp_intrinsic!(lits[0], lits[1], max),
                    MathIntrinsicFunc::Min => two_var_cmp_intrinsic!(lits[0], lits[1], min),
                    MathIntrinsicFunc::Floor => single_var_f_math_intrinsic!(lits[0], floor),
                    MathIntrinsicFunc::Ceil => single_var_f_math_intrinsic!(lits[0], ceil),
                    MathIntrinsicFunc::Clamp => three_var_cmp_intrinsic!(lits[0], lits[1], lits[2], clamp),
                    MathIntrinsicFunc::Mix => match(&(lits[0]), &(lits[1]), &lits[2]) {
                        (MimirLit::Float32(f1_b),MimirLit::Float32(f2_b),MimirLit::Float32(f3_b)) => {
                            let x = f32::from_bits(*f1_b);
                            let y = f32::from_bits(*f2_b);
                            let a = f32::from_bits(*f3_b);
                            let f_mix = x * (1f32-a) + y * a;
                            Ok(MimirLit::Float32(f_mix.to_bits()))
                        },
                        _ => Err(MimirInlinePassError::Generic("Cannot inline a two var floating point math intrinsic on a non-floating point literal".to_owned()))
                    },
                    MathIntrinsicFunc::Fma => match(&(lits[0]), &(lits[1]), &lits[2]) {
                        (MimirLit::Float32(f1_b),MimirLit::Float32(f2_b), MimirLit::Float32(f3_b)) => {
                            let a = f32::from_bits(*f1_b);
                            let b = f32::from_bits(*f2_b);
                            let c = f32::from_bits(*f3_b);
                            Ok(MimirLit::Float32((a * b + c).to_bits()))
                        },
                        _ => Err(MimirInlinePassError::Generic("Cannot inline a two var floating point math intrinsic on a non-floating point literal".to_owned()))
                    },
                }
        }
        MimirConstExpr::ConstGeneric { index } => {
            Ok(MimirLit::Uint32(match const_generics.get(*index) {
                Some(cg) => *cg,
                None => {
                    return Err(MimirInlinePassError::Generic(
                        "Const generic is out of bounds during const inlining process".to_owned(),
                    ))
                }
            }))
        }
        MimirConstExpr::Cast(MimirCastExpr { from, to }) => {
            let lit = expr_to_lit_recur(from, const_generics)?;

            match lit {
                MimirLit::Int32(val) => match to {
                    MimirPrimitiveTy::Int32 => Ok(lit),
                    MimirPrimitiveTy::Uint32 => Ok(MimirLit::Uint32(val as u32)),
                    MimirPrimitiveTy::Float32 => Ok(MimirLit::Float32((val as f32).to_bits())),
                    MimirPrimitiveTy::Bool => Err(MimirInlinePassError::SpecFailErr(
                        PostSafetyPassSpecFailError::CastErr(InvalidCastError::NumericalToBool),
                    )),
                },
                MimirLit::Float32(val) => {
                    let f = f32::from_bits(val);
                    match to {
                        MimirPrimitiveTy::Int32 => Ok(MimirLit::Int32(f as i32)),
                        MimirPrimitiveTy::Uint32 => Ok(MimirLit::Uint32(f as u32)),
                        MimirPrimitiveTy::Float32 => Ok(lit),
                        MimirPrimitiveTy::Bool => Err(MimirInlinePassError::SpecFailErr(
                            PostSafetyPassSpecFailError::CastErr(InvalidCastError::NumericalToBool),
                        )),
                    }
                }
                MimirLit::Bool(_) => match to {
                    MimirPrimitiveTy::Bool => Err(MimirInlinePassError::SpecFailErr(
                        PostSafetyPassSpecFailError::CastErr(InvalidCastError::BoolToNumerical),
                    )),
                    _ => Ok(lit),
                },
                MimirLit::Uint32(val) => match to {
                    MimirPrimitiveTy::Int32 => Ok(MimirLit::Int32(val as i32)),
                    MimirPrimitiveTy::Uint32 => Ok(lit),
                    MimirPrimitiveTy::Float32 => Ok(MimirLit::Float32((val as f32).to_bits())),
                    MimirPrimitiveTy::Bool => Err(MimirInlinePassError::SpecFailErr(
                        PostSafetyPassSpecFailError::CastErr(InvalidCastError::NumericalToBool),
                    )),
                },
            }
        }
    }
}
