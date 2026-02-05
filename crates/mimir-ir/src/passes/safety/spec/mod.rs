use crate::{
    ir::{
        MathIntrinsicExpr, MimirBinOp, MimirBinOpExpr, MimirCastExpr, MimirConstExpr, MimirExpr,
        MimirIndexExpr, MimirPrimitiveTy, MimirTyVar, MimirUnOp, MimirUnOpExpr,
    },
    kernel_ir::MimirKernelIR,
    util::{
        error::SpecEnforcePassError,
        math::{intrinsic_param_count, intrinsic_param_validity},
    },
};

pub(crate) fn full_expr_spec_check(
    ir: &MimirKernelIR,
    expr: &MimirExpr,
) -> Result<(), SpecEnforcePassError> {
    match expr {
        MimirExpr::BinOp(MimirBinOpExpr { lhs, op, rhs, .. }) => {
            full_expr_spec_check(ir, lhs.as_ref())?;
            full_expr_spec_check(ir, rhs.as_ref())?;

            let lhs_ty = lhs.to_ty(&ir.var_map);
            let rhs_ty = rhs.to_ty(&ir.var_map);

            binop_spec_check(lhs_ty, rhs_ty, op)?;
        }
        MimirExpr::Index(MimirIndexExpr { var, index }) => {
            full_expr_spec_check(ir, index.as_ref())?;

            let var_ty = check_var_existence(ir, var)?.ty;

            if !(var_ty.is_global_array() || var_ty.is_shared_mem()) {
                return Err(SpecEnforcePassError::InvalidVarTypeForIndex(var_ty));
            }

            let idx_ty = index.to_ty(&ir.var_map);

            if !(matches!(idx_ty, MimirPrimitiveTy::Int32)) {
                return Err(SpecEnforcePassError::InvalidTypeForIndexExpr(idx_ty));
            }
        }
        MimirExpr::BuiltinFieldAccess { .. } => {
            // honestly can't think of any spec checks I need to perform against a BuiltinFieldAccess
        }
        MimirExpr::Unary(MimirUnOpExpr { un_op, expr }) => {
            full_expr_spec_check(ir, expr)?;

            let expr_ty = expr.to_ty(&ir.var_map);

            match un_op {
                MimirUnOp::Not => {
                    if !expr_ty.is_bool() {
                        return Err(SpecEnforcePassError::InvalidTypeForNotUnExpr(expr_ty));
                    }
                }
                MimirUnOp::Neg => {
                    if !expr_ty.is_numeric() {
                        return Err(SpecEnforcePassError::InvalidTypeForNegUnExpr(expr_ty));
                    }
                }
            }
        }
        MimirExpr::Literal(_) => {
            // I can't really do any spec checks on literals
        }
        MimirExpr::Var(id) => {
            let _ = check_var_existence(ir, id)?;
        }
        MimirExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
            for arg in args {
                full_expr_spec_check(ir, arg)?;
            }

            let arg_tys = args
                .iter()
                .map(|a| a.to_ty(&ir.var_map))
                .collect::<Vec<_>>();

            {
                let expected = intrinsic_param_count(func);
                if args.len() != expected {
                    return Err(SpecEnforcePassError::InvalidNumParamsForMathIntrinsic(
                        expected,
                        func.clone(),
                        args.len(),
                    ));
                }
            }

            if !intrinsic_param_validity(func, &arg_tys) {
                return Err(SpecEnforcePassError::InvalidParamTypesForMathIntrinsic(
                    func.clone(),
                ));
            }
        }
        MimirExpr::Cast(MimirCastExpr { from, to }) => {
            full_expr_spec_check(ir, from.as_ref())?;

            let ty = from.to_ty(&ir.var_map);

            /*
            Valid:
            i32 -> u32
            i32 -> f32
            u32 -> i32
            u32 -> f32
            f32 -> i32
            f32 -> u32
            */

            if ty.is_bool() && !to.is_bool() {
                return Err(SpecEnforcePassError::InvalidCastFromBoolToNonBool(
                    to.clone(),
                ));
            } else if !ty.is_bool() && to.is_bool() {
                return Err(SpecEnforcePassError::InvalidCastFromNonBoolToBool(
                    ty.clone(),
                ));
            }
        }
        MimirExpr::ConstExpr(mimir_const_expr) => match mimir_const_expr {
            MimirConstExpr::BinOp(mimir_bin_op_expr) => {
                let lhs = mimir_bin_op_expr.lhs.to_norm_expr();
                let rhs = mimir_bin_op_expr.rhs.to_norm_expr();

                full_expr_spec_check(ir, &lhs)?;
                full_expr_spec_check(ir, &rhs)?;

                let lhs_ty = lhs.to_ty(&ir.var_map);
                let rhs_ty = rhs.to_ty(&ir.var_map);

                binop_spec_check(lhs_ty, rhs_ty, &mimir_bin_op_expr.op)?;
            }
            MimirConstExpr::Unary(MimirUnOpExpr { un_op, expr }) => {
                full_expr_spec_check(ir, &expr.to_norm_expr())?;

                let expr_ty = expr.to_norm_expr().to_ty(&ir.var_map);

                match un_op {
                    MimirUnOp::Not => {
                        if !expr_ty.is_bool() {
                            return Err(SpecEnforcePassError::InvalidTypeForNotUnExpr(expr_ty));
                        }
                    }
                    MimirUnOp::Neg => {
                        if !expr_ty.is_numeric() {
                            return Err(SpecEnforcePassError::InvalidTypeForNegUnExpr(expr_ty));
                        }
                    }
                }
            }
            MimirConstExpr::Literal(_) => {}
            MimirConstExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
                for arg in args {
                    full_expr_spec_check(ir, &arg.to_norm_expr())?;
                }

                let arg_tys = args
                    .iter()
                    .map(|a| a.to_norm_expr().to_ty(&ir.var_map))
                    .collect::<Vec<_>>();

                {
                    let expected = intrinsic_param_count(func);
                    if args.len() != expected {
                        return Err(SpecEnforcePassError::InvalidNumParamsForMathIntrinsic(
                            expected,
                            func.clone(),
                            args.len(),
                        ));
                    }
                }

                if !intrinsic_param_validity(func, &arg_tys) {
                    return Err(SpecEnforcePassError::InvalidParamTypesForMathIntrinsic(
                        func.clone(),
                    ));
                }
            }
            MimirConstExpr::Cast(MimirCastExpr { from, to }) => {
                full_expr_spec_check(ir, &from.to_norm_expr())?;

                let ty = from.to_norm_expr().to_ty(&ir.var_map);

                if ty.is_bool() && !to.is_bool() {
                    return Err(SpecEnforcePassError::InvalidCastFromBoolToNonBool(
                        to.clone(),
                    ));
                } else if !ty.is_bool() && to.is_bool() {
                    return Err(SpecEnforcePassError::InvalidCastFromNonBoolToBool(
                        ty.clone(),
                    ));
                }
            }
            MimirConstExpr::ConstGeneric { index } => {
                if *index >= ir.const_generics {
                    return Err(SpecEnforcePassError::OutOfBoundsConstGeneric(
                        *index,
                        ir.const_generics - 1,
                    ));
                }
            }
        },
    }

    Ok(())
}

pub(crate) fn check_var_existence(
    ir: &MimirKernelIR,
    id: &u64,
) -> Result<MimirTyVar, SpecEnforcePassError> {
    match ir.var_map.get(id) {
        Some(ty_var) => Ok(ty_var.clone()),
        None => Err(SpecEnforcePassError::CouldNotFindVar(*id)),
    }
}

fn binop_spec_check(
    lhs_ty: MimirPrimitiveTy,
    rhs_ty: MimirPrimitiveTy,
    op: &MimirBinOp,
) -> Result<(), SpecEnforcePassError> {
    match op {
        MimirBinOp::Add | MimirBinOp::Sub | MimirBinOp::Mul | MimirBinOp::Div | MimirBinOp::Mod => {
            if !lhs_ty.is_numeric() || !rhs_ty.is_numeric() {
                return Err(SpecEnforcePassError::InvalidTypesForNumericBinOp(
                    op.clone(),
                    lhs_ty,
                    rhs_ty,
                ));
            } else if lhs_ty != rhs_ty {
                return Err(SpecEnforcePassError::NonMatchingTypesForNumericBinOp(
                    op.clone(),
                    lhs_ty,
                    rhs_ty,
                ));
            }
        }
        MimirBinOp::And | MimirBinOp::Or => {
            if !(lhs_ty.is_bool() && rhs_ty.is_bool()) {
                return Err(SpecEnforcePassError::InvalidTypesForBooleanBinOp(
                    op.clone(),
                    lhs_ty,
                    rhs_ty,
                ));
            }
        }
        MimirBinOp::Lt
        | MimirBinOp::Lte
        | MimirBinOp::Gt
        | MimirBinOp::Gte
        | MimirBinOp::Eq
        | MimirBinOp::Ne => {
            if !lhs_ty.is_numeric() || !rhs_ty.is_numeric() {
                return Err(SpecEnforcePassError::InvalidTypesForNumericBinOp(
                    op.clone(),
                    lhs_ty,
                    rhs_ty,
                ));
            } else if lhs_ty != rhs_ty {
                return Err(SpecEnforcePassError::NonMatchingTypesForNumericBinOp(
                    op.clone(),
                    lhs_ty,
                    rhs_ty,
                ));
            }
        }
    }

    Ok(())
}
