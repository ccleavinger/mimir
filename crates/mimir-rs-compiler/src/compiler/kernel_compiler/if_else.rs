use mimir_ir::{
    compiler_err,
    ir::{MimirBlock, MimirStmt},
    util::error::ASTError,
};
use quote::ToTokens as _;
use syn::{Expr, ExprIf};

use crate::compiler::kernel_compiler;

impl kernel_compiler::Compiler {
    pub fn compile_if_else(&mut self, expr_if: &ExprIf) -> Result<MimirStmt, ASTError> {
        let expr = self.parse_expr(&expr_if.cond)?;

        let then_branch = self.block_to_irs(&expr_if.then_branch)?;

        let else_branch = {
            match &expr_if.else_branch {
                Some((_, expr)) => match expr.as_ref() {
                    Expr::Block(expr_block) => Some(self.block_to_irs(&expr_block.block)?),
                    Expr::If(expr_else_if) => Some(vec![self.compile_if_else(expr_else_if)?]),
                    _ => {
                        return compiler_err!(
                            "ERR! Only Block & If expressions are allowed!\nVERY BAD"
                        );
                    }
                },
                None => None,
            }
        };

        Ok(MimirStmt::If {
            condition: expr,
            then_branch,
            else_branch,
        })
    }

    #[inline]
    fn block_to_irs(&mut self, block: &syn::Block) -> Result<MimirBlock, ASTError> {
        self.curr_max_scope += 1;
        self.scope_stack.push(self.curr_max_scope);

        let mut ret = vec![];

        for (i, stmt) in block.stmts.iter().enumerate() {
            let err_stmt = self.parse_stmt_ret_ir(stmt);

            match err_stmt {
                Ok(ok_stmt) => {
                    if let Some(mimir_ir) = ok_stmt {
                        ret.push(mimir_ir);
                    }
                }
                Err(err) => {
                    return Err(ASTError::Compiler(format!(
                        "Line {} of if else block, `{}`\n{}",
                        i + 1,
                        stmt.to_token_stream(),
                        err
                    )));
                }
            }
        }

        self.scope_stack.pop();

        Ok(ret)
    }
}
