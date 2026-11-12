pub mod const_evaluate;
pub mod const_to_lit;
pub mod safety;

pub use safety::ValidityPass;

use crate::{
    AssignTarget, ConstExpr, Intrinsic, MimirExpr, MimirKernelAst, MimirKernelBlock,
    MimirKernelStmt, RichConstExpr, RichKernelStmt, RichMimirExpr,
};

/// Stateful traversal of a kernel AST. Expression results can carry inferred
/// types (validation) or be `()` (rewriting passes). Statement and block hooks
/// discard those results while propagating errors.
///
/// Override a hook to customize it, and call the matching `walk_*` function to
/// recurse through its children. Hooks receive mutable nodes to support future
/// transformations; applying one to a validated kernel requires taking the AST
/// out of its proof wrapper and validating it again.
pub trait KernelAstPass {
    type Error;
    type ExprOutput;

    fn kernel(&mut self, kernel: &mut MimirKernelAst) -> Result<(), Self::Error> {
        walk_kernel(self, kernel)
    }

    fn block(&mut self, block: &mut MimirKernelBlock) -> Result<(), Self::Error> {
        walk_block(self, block)
    }

    fn stmt(&mut self, stmt: &mut RichKernelStmt) -> Result<(), Self::Error> {
        walk_stmt(self, stmt)
    }

    fn expr(&mut self, expr: &mut RichMimirExpr) -> Result<Self::ExprOutput, Self::Error>;

    fn const_expr(&mut self, expr: &mut RichConstExpr) -> Result<Self::ExprOutput, Self::Error>;
}

pub fn walk_kernel<P: KernelAstPass + ?Sized>(
    pass: &mut P,
    kernel: &mut MimirKernelAst,
) -> Result<(), P::Error> {
    pass.block(&mut kernel.body)
}

pub fn walk_block<P: KernelAstPass + ?Sized>(
    pass: &mut P,
    block: &mut MimirKernelBlock,
) -> Result<(), P::Error> {
    for stmt in block {
        pass.stmt(stmt)?;
    }
    Ok(())
}

pub fn walk_stmt<P: KernelAstPass + ?Sized>(
    pass: &mut P,
    stmt: &mut RichKernelStmt,
) -> Result<(), P::Error> {
    match &mut stmt.stmt {
        MimirKernelStmt::Let { value, .. } => {
            pass.expr(value)?;
        }
        MimirKernelStmt::SharedMem { length, .. } => {
            pass.const_expr(length)?;
        }
        MimirKernelStmt::Assign { target, value } => {
            if let AssignTarget::Index { index, .. } = target {
                pass.expr(index)?;
            }
            pass.expr(value)?;
        }
        MimirKernelStmt::RangeFor {
            start,
            end,
            step,
            body,
            ..
        } => {
            pass.expr(start)?;
            pass.expr(end)?;
            if let Some(step) = step {
                pass.expr(step)?;
            }
            pass.block(body)?;
        }
        MimirKernelStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            pass.expr(condition)?;
            pass.block(then_branch)?;
            if let Some(branch) = else_branch {
                pass.block(branch)?;
            }
        }
        MimirKernelStmt::Block(body) => pass.block(body)?,
        MimirKernelStmt::Return | MimirKernelStmt::Syncthreads => {}
    }
    Ok(())
}

pub fn walk_expr<P: KernelAstPass + ?Sized>(
    pass: &mut P,
    expr: &mut RichMimirExpr,
) -> Result<(), P::Error> {
    match &mut expr.expr {
        MimirExpr::BinOp(binary) => {
            pass.expr(&mut binary.lhs)?;
            pass.expr(&mut binary.rhs)?;
        }
        MimirExpr::Unary(unary) => {
            pass.expr(&mut unary.expr)?;
        }
        MimirExpr::Index { index, .. } => {
            pass.expr(index)?;
        }
        MimirExpr::Cast { value, .. } => {
            pass.expr(value)?;
        }
        MimirExpr::MathIntrinsic(intrinsic) => walk_intrinsic(intrinsic, |arg| {
            pass.expr(arg)?;
            Ok(())
        })?,
        MimirExpr::ConstExpr(expr) => {
            pass.const_expr(expr)?;
        }
        MimirExpr::Literal(_) | MimirExpr::Var(_) | MimirExpr::Builtin { .. } => {}
    }
    Ok(())
}

pub fn walk_const_expr<P: KernelAstPass + ?Sized>(
    pass: &mut P,
    expr: &mut RichConstExpr,
) -> Result<(), P::Error> {
    match &mut expr.expr {
        ConstExpr::BinOp(binary) => {
            pass.const_expr(&mut binary.lhs)?;
            pass.const_expr(&mut binary.rhs)?;
        }
        ConstExpr::UnOp(unary) => {
            pass.const_expr(&mut unary.expr)?;
        }
        ConstExpr::Cast { value, .. } => {
            pass.const_expr(value)?;
        }
        ConstExpr::MathIntrinsic(intrinsic) => walk_intrinsic(intrinsic, |arg| {
            pass.const_expr(arg)?;
            Ok(())
        })?,
        ConstExpr::Literal(_) | ConstExpr::Generic(_) => {}
    }
    Ok(())
}

fn walk_intrinsic<T, E>(
    intrinsic: &mut Intrinsic<T>,
    mut visit: impl FnMut(&mut T) -> Result<(), E>,
) -> Result<(), E> {
    match intrinsic {
        Intrinsic::Unary(_, arg) => visit(arg)?,
        Intrinsic::Binary(_, args) => {
            for arg in args.iter_mut() {
                visit(arg)?;
            }
        }
        Intrinsic::Ternary(_, args) => {
            for arg in args.iter_mut() {
                visit(arg)?;
            }
        }
    }
    Ok(())
}
