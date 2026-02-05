use crate::compiler::kernel_compiler::Compiler;
use mimir_ir::compiler_err;
use mimir_ir::ir::{MimirBinOp, MimirBinOpExpr, MimirExpr};
use mimir_ir::util::error::ASTError;

impl Compiler {
    pub fn compile_binop_from_expr(
        &self,
        lhs: &MimirExpr,
        op: &MimirBinOp,
        rhs: &MimirExpr,
    ) -> Result<MimirExpr, ASTError> {
        let (p_lhs, p_rhs) = {
            // let lhs_precedence = lhs.precedence();
            // let rhs_precedence = rhs.precedence();

            // if ignore_precedence {
            //     (lhs, rhs)
            // } else if rhs_precedence < lhs_precedence {
            //     (rhs, lhs)
            // } else {
            //     (lhs, rhs)
            // }

            (lhs, rhs)
        };

        // let is_const = p_lhs.is_const() && p_rhs.is_const();

        let expr = MimirExpr::BinOp(MimirBinOpExpr {
            lhs: Box::new(p_lhs.clone()),
            op: op.clone(),
            rhs: Box::new(p_rhs.clone()),
            is_parenthesized: false,
        });

        let _ = self.expr_ty(&expr)?;

        Ok(expr)
    }

    pub fn compile_binop(&self, expr_binary: &syn::ExprBinary) -> Result<MimirExpr, ASTError> {
        let lhs_mimir_expr = self.parse_expr(&expr_binary.left)?;
        let rhs_mimir_expr = self.parse_expr(&expr_binary.right)?;

        let op = match expr_binary.op {
            syn::BinOp::Add(_) => MimirBinOp::Add,
            syn::BinOp::Sub(_) => MimirBinOp::Sub,
            syn::BinOp::Mul(_) => MimirBinOp::Mul,
            syn::BinOp::Div(_) => MimirBinOp::Div,
            syn::BinOp::Rem(_) => MimirBinOp::Mod,
            syn::BinOp::And(_) => MimirBinOp::And,
            syn::BinOp::Or(_) => MimirBinOp::Or,
            syn::BinOp::BitXor(_) => {
                return Err(ASTError::Compiler("BitXor isn't implemented".to_owned()));
            }
            syn::BinOp::BitAnd(_) => {
                return Err(ASTError::Compiler("BitAnd isn't implemented".to_owned()));
            }
            syn::BinOp::BitOr(_) => {
                return Err(ASTError::Compiler("BitOr isn't implemented".to_owned()));
            }
            syn::BinOp::Shl(_) => {
                return Err(ASTError::Compiler(
                    "Bit shift left isn't implemented".to_owned(),
                ));
            }
            syn::BinOp::Shr(_) => {
                return Err(ASTError::Compiler(
                    "Bit shift right isn't implemented".to_owned(),
                ));
            }
            syn::BinOp::Eq(_) => MimirBinOp::Eq,
            syn::BinOp::Lt(_) => MimirBinOp::Lt,
            syn::BinOp::Le(_) => MimirBinOp::Lte,
            syn::BinOp::Ne(_) => MimirBinOp::Ne,
            syn::BinOp::Ge(_) => MimirBinOp::Gte,
            syn::BinOp::Gt(_) => MimirBinOp::Gt,
            syn::BinOp::AddAssign(_) => {
                return compiler_err!("Add assign isn't permitted within bin op expressions");
            }
            syn::BinOp::SubAssign(_) => {
                return compiler_err!("Minus assign isn't permitted within bin op expressions");
            }
            syn::BinOp::MulAssign(_) => {
                return compiler_err!(
                    "Multiplication assign isn't permitted within bin op expressions"
                );
            }
            syn::BinOp::DivAssign(_) => {
                return compiler_err!("Divide assign isn't permitted within bin op expressions");
            }
            syn::BinOp::RemAssign(_) => {
                return compiler_err!("Modulus assign isn't permitted within bin op expressions");
            }
            syn::BinOp::BitXorAssign(_) => {
                return compiler_err!("Bit Xor assign isn't permitted within bin op expressions");
            }
            syn::BinOp::BitAndAssign(_) => {
                return compiler_err!("Bit And assign isn't permitted within bin op expressions");
            }
            syn::BinOp::BitOrAssign(_) => {
                return compiler_err!("Bit Or assign isn't permitted within bin op expressions");
            }
            syn::BinOp::ShlAssign(_) => {
                return compiler_err!(
                    "Bit Shift Left assign isn't permitted within bin op expressions"
                );
            }
            syn::BinOp::ShrAssign(_) => {
                return compiler_err!(
                    "Bit Shift Right assign isn't permitted within bin op expressions"
                );
            }
            _ => return Err(ASTError::Compiler("Unkown bin op".to_owned())),
        };

        let expr = self.compile_binop_from_expr(&lhs_mimir_expr, &op, &rhs_mimir_expr)?;

        let _ = self.expr_ty(&expr)?;

        Ok(expr)
    }
}
