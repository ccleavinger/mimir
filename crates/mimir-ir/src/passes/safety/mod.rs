use std::collections::HashSet;

use crate::{
    ir::{MimirExpr, MimirPrimitiveTy, MimirStmt, MimirStmtAssignLeft, MimirTy},
    kernel_ir::MimirKernelIR,
    passes::safety::spec::{check_var_existence, full_expr_spec_check},
    util::error::SpecEnforcePassError,
};

pub(crate) mod spec;

pub(crate) fn kernel_spec_stmt_pass(
    ir: &MimirKernelIR,
    stmt: &MimirStmt,
    init_list: &mut HashSet<u64>,
) -> Result<(), SpecEnforcePassError> {
    match stmt {
        MimirStmt::Assign { lhs, rhs } => {
            let lhs_ty = match lhs {
                MimirStmtAssignLeft::Index(mimir_index_expr) => {
                    {
                        let ty_var = check_var_existence(ir, &mimir_index_expr.var)?;

                        if init_list.contains(&mimir_index_expr.var) {
                            if ty_var.is_const() {
                                return Err(SpecEnforcePassError::NonMutableLHSReassignment);
                            }
                        } else {
                            init_list.insert(mimir_index_expr.var);
                        }
                    }

                    let idx_expr = MimirExpr::Index(mimir_index_expr.clone());

                    full_expr_spec_check(ir, &idx_expr)?;

                    idx_expr.to_ty(&ir.var_map).to_mimir_ty()
                }
                MimirStmtAssignLeft::Var(id) => {
                    let ty_var = check_var_existence(ir, id)?;

                    if init_list.contains(id) {
                        if ty_var.is_const() {
                            return Err(SpecEnforcePassError::NonMutableLHSReassignment);
                        }
                    } else {
                        init_list.insert(*id);
                    }

                    ty_var.ty
                }
            };

            full_expr_spec_check(ir, rhs)?;

            let rhs_ty = rhs.to_ty(&ir.var_map);

            if let MimirTy::Primitive(lhs_prim) = lhs_ty {
                if lhs_prim != rhs_ty {
                    Err(SpecEnforcePassError::MismatchedTypesInAssignStmt(
                        lhs_prim, rhs_ty,
                    ))
                } else {
                    Ok(())
                }
            } else {
                Err(SpecEnforcePassError::NonPrimitiveAssignmentLhs(lhs_ty))
            }?;
        }
        MimirStmt::RangeFor {
            var,
            start,
            end,
            step,
            body,
        } => {
            let ty = check_var_existence(ir, var)?;

            if let MimirTy::Primitive(prim) = &ty.ty {
                if prim != &MimirPrimitiveTy::Int32 {
                    return Err(SpecEnforcePassError::Non32BitIntPrimInRangeFor(
                        prim.clone(),
                    ));
                }
            } else {
                return Err(SpecEnforcePassError::NonPrimVarInRangeFor(ty.ty));
            }

            if ty.is_mutable() {
                return Err(SpecEnforcePassError::NonConstantVarInRangeFor);
            }

            full_expr_spec_check(ir, start)?;
            full_expr_spec_check(ir, end)?;

            if let Some(st_expr) = step {
                full_expr_spec_check(ir, st_expr)?;

                let step_ty = st_expr.to_ty(&ir.var_map);

                if step_ty != MimirPrimitiveTy::Int32 {
                    return Err(SpecEnforcePassError::InvalidStepExpressionTy(step_ty));
                }
            }

            for b_stmt in body {
                kernel_spec_stmt_pass(ir, b_stmt, init_list)?;
            }
        }
        MimirStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            full_expr_spec_check(ir, condition)?;

            let cond_ty = condition.to_ty(&ir.var_map);

            if !cond_ty.is_bool() {
                return Err(SpecEnforcePassError::NonBooleanConditionalExprIf(cond_ty));
            }

            for then_stmt in then_branch {
                kernel_spec_stmt_pass(ir, then_stmt, init_list)?;
            }

            if let Some(else_b) = else_branch {
                for else_stmt in else_b {
                    kernel_spec_stmt_pass(ir, else_stmt, init_list)?;
                }
            }
        }
        MimirStmt::Return(mimir_expr) => {
            if let Some(expr) = mimir_expr {
                return Err(SpecEnforcePassError::InvalidValuedErrorFromKernel(
                    expr.clone(),
                ));
            }
        }
        MimirStmt::Syncthreads => {
            // lowk can't do error pass on Syncthreads
        }
    }

    Ok(())
}
