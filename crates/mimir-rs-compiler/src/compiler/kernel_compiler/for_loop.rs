use mimir_ir::{
    compiler_err,
    ir::{
        MimirExpr, MimirPrimitiveTy, MimirStmt, MimirTy, MimirTyAnnotation, MimirTyScope,
        MimirTyVar, MimirUnOp, MimirUnOpExpr,
    },
    util::error::ASTError,
};
use quote::ToTokens;
use syn::{Expr, ExprForLoop, ExprMethodCall, ExprRange};

use crate::compiler::kernel_compiler;

impl kernel_compiler::Compiler {
    pub fn compile_for_loop(&mut self, expr_for_loop: &ExprForLoop) -> Result<MimirStmt, ASTError> {
        // set current scope
        self.curr_max_scope += 1;
        self.scope_stack.push(self.curr_max_scope);

        let var_name = expr_for_loop.pat.to_token_stream().to_string();

        self.add_ty_var(
            &var_name,
            MimirTyVar {
                ty: MimirTy::Primitive(MimirPrimitiveTy::Int32),
                scope: MimirTyScope::Local(*self.scope_stack.last().unwrap()),
                annotation: MimirTyAnnotation::Constant, // mutable from loop comprehension stuff but not by other operations
            },
        )?;

        let var_num = *self.get_uuid(&var_name).unwrap();

        let (start, end, step) = self.parse_range(expr_for_loop.expr.as_ref())?;

        let mut ir = vec![];

        for (i, stmt) in expr_for_loop.body.stmts.iter().enumerate() {
            let stmt_ret_err = self.parse_stmt_ret_ir(stmt);

            let stmt_ret = match stmt_ret_err {
                Ok(x) => x,
                Err(err) => {
                    return Err(ASTError::Compiler(format!(
                        "Line {} of for loop, `{}`\n{}",
                        i + 1,
                        stmt.to_token_stream(),
                        err
                    )));
                }
            };

            if let Some(mimir_ir) = stmt_ret {
                ir.push(mimir_ir);
            }
        }

        // reset the scope stack
        self.scope_stack.pop();

        Ok(MimirStmt::RangeFor {
            var: var_num,
            start,
            end,
            step,
            body: ir,
        })
    }

    fn parse_range(
        &self,
        expr: &Expr,
    ) -> Result<(MimirExpr, MimirExpr, Option<MimirExpr>), ASTError> {
        match expr {
            Expr::Paren(expr_paren) => self.parse_range(&expr_paren.expr),
            Expr::Range(expr_range) => {
                let start = match expr_range.start.as_ref() {
                    Some(start) => self.parse_expr(start)?,
                    None => {
                        return compiler_err!(
                            "When iterating over a slice in a for loop the starting value must be present."
                        );
                    }
                };

                let end = match expr_range.end.as_ref() {
                    Some(end) => self.parse_expr(end)?,
                    None => {
                        return compiler_err!(
                            "When iterating over a slice in a for loop the ending value must be present."
                        );
                    }
                };

                Ok((start, end, None))
            }
            Expr::MethodCall(method_expr) => {
                let (range, iter_methods) = self.recur_parse_iter_methods(method_expr)?;

                let num_rev = iter_methods
                    .iter()
                    .filter(|x| matches!(**x, IterMethod::Rev))
                    .count();

                // Rust (at least in Rust playground) just uses first step_by so we'll do the same
                let step_by = iter_methods
                    .iter()
                    .filter_map(|x| {
                        if let IterMethod::StepBy(expr) = x {
                            Some(expr.clone())
                        } else {
                            None
                        }
                    })
                    .next();

                let start = match range.start {
                    Some(start) => self.parse_expr(&start)?,
                    None => {
                        return compiler_err!(
                            "When iterating over a slice in a for loop the starting value must be present."
                        );
                    }
                };

                let end = match range.end {
                    Some(end) => self.parse_expr(&end)?,
                    None => {
                        return compiler_err!(
                            "When iterating over a slice in a for loop the ending value must be present."
                        );
                    }
                };

                if num_rev % 2 == 0 {
                    // num_rev cancels out or is 0, just do normal stuff

                    Ok((start, end, step_by))
                } else {
                    let reversed_step_by = step_by.map(|step_expr| {
                        MimirExpr::Unary(MimirUnOpExpr {
                            un_op: MimirUnOp::Neg,
                            expr: Box::new(step_expr),
                        })
                    });

                    Ok((end, start, reversed_step_by))
                }
            }
            _ => compiler_err!(
                "Unkown expression `{}` found in for loop",
                expr.to_token_stream().to_string()
            ),
        }
    }

    fn recur_parse_iter_methods(
        &self,
        method_expr: &ExprMethodCall,
    ) -> Result<(ExprRange, Vec<IterMethod>), ASTError> {
        let iter_method = match method_expr.method.to_string().as_str() {
            "rev" => IterMethod::Rev,
            "step_by" => {
                if method_expr.args.len() == 1 {
                    let mimir_expr = self.parse_expr(&method_expr.args[0])?;

                    IterMethod::StepBy(mimir_expr)
                } else {
                    return compiler_err!("Invalid arguments for a `step_by` method in a for loop");
                }
            }
            _ => {
                return compiler_err!(
                    "Unknown method `{}`, only `step_by` and `rev` suported with ranges in for loops",
                    method_expr.method.to_string()
                );
            }
        };

        match Self::recur_parse_iter_reciever(&method_expr.receiver)? {
            RecieverType::MethodCall(expr) => {
                let (range, mut list) = {
                    let (rng, lst) = self.recur_parse_iter_methods(&expr)?;

                    (rng, lst.to_vec())
                };
                list.push(iter_method);

                Ok((range, list))
            }
            RecieverType::Range(range) => Ok((range, vec![iter_method])),
        }
    }

    // just recursive cause of `Expr::Paren`
    fn recur_parse_iter_reciever(reciever: &Expr) -> Result<RecieverType, ASTError> {
        // None = range, Some(expr_meth_call) = method call
        match reciever {
            Expr::MethodCall(expr_meth_call) => {
                Ok(RecieverType::MethodCall(expr_meth_call.clone()))
            }
            Expr::Paren(expr) => Self::recur_parse_iter_reciever(expr.expr.as_ref()),
            Expr::Range(range) => Ok(RecieverType::Range(range.clone())),
            _ => compiler_err!(
                "`{}` is an invalid iterator/range to call methods upon in Mimir",
                reciever.to_token_stream().to_string()
            ),
        }
    }
}

enum RecieverType {
    MethodCall(ExprMethodCall),
    Range(ExprRange),
}

#[derive(Clone, Debug)]
enum IterMethod {
    Rev, // apply MimirUnOp::Neg to StepBy
    StepBy(MimirExpr),
}
