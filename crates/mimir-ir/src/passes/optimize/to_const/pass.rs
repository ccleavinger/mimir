use crate::{
    ir::MimirStmt,
    passes::KernelIRPrePass,
    util::error::{MimirInlinePassError, MultiPassError},
};

pub(crate) struct ConstExprPass;

impl ConstExprPass {
    fn stmt_pass_priv(stmt: &MimirStmt) -> Result<Vec<MimirStmt>, MimirInlinePassError> {
        match stmt {
            MimirStmt::Assign { lhs, rhs } => {
                if rhs.is_const() {
                    let ret = vec![MimirStmt::Assign {
                        lhs: lhs.clone(),
                        rhs: rhs.to_const_expr()?.to_norm_expr(),
                    }];

                    Ok(ret)
                } else {
                    Ok(vec![stmt.clone()])
                }
            }
            MimirStmt::RangeFor {
                var,
                start,
                end,
                step,
                body,
            } => {
                let s = if start.is_const() {
                    start.to_const_expr()?.to_norm_expr()
                } else {
                    start.clone()
                };

                let e = if end.is_const() {
                    end.to_const_expr()?.to_norm_expr()
                } else {
                    end.clone()
                };

                let st = if let Some(step_) = step {
                    if step_.is_const() {
                        Some(step_.to_const_expr()?.to_norm_expr())
                    } else {
                        Some(step_.clone())
                    }
                } else {
                    None
                };

                let mut v = vec![];
                for b_stmt in body {
                    v.append(&mut Self::stmt_pass_priv(b_stmt)?);
                }

                Ok(vec![MimirStmt::RangeFor {
                    var: *var,
                    start: s,
                    end: e,
                    step: st,
                    body: v,
                }])
            }
            MimirStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = if condition.is_const() {
                    condition.to_const_expr()?.to_norm_expr()
                } else {
                    condition.clone()
                };

                let mut then = vec![];
                for th_stmt in then_branch {
                    then.append(&mut Self::stmt_pass_priv(th_stmt)?);
                }

                let else_b = match else_branch {
                    Some(else_b_v) => {
                        let mut else_b_ret = vec![];
                        for else_stmt in else_b_v {
                            else_b_ret.append(&mut Self::stmt_pass_priv(else_stmt)?);
                        }
                        Some(else_b_ret)
                    }
                    None => None,
                };

                Ok(vec![MimirStmt::If {
                    condition: cond,
                    then_branch: then,
                    else_branch: else_b,
                }])
            }
            MimirStmt::Return(_) => Ok(vec![MimirStmt::Return(None)]),
            MimirStmt::Syncthreads => Ok(vec![MimirStmt::Syncthreads]),
        }
    }
}

impl KernelIRPrePass for ConstExprPass {
    fn stmt_pass(stmt: &MimirStmt) -> Result<Vec<MimirStmt>, MultiPassError> {
        Self::stmt_pass_priv(stmt).map_err(MultiPassError::Inline)
    }
}
