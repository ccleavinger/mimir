//! Type, scope and mutability checks implemented as a kernel AST pass.
use crate::{
    AssignTarget, ConstExpr, Intrinsic, IntrinsicReport, KernelAstPass, KernelParamTy, Literal,
    MimirBinOp, MimirExpr, MimirKernelAst, MimirKernelBlock, MimirKernelStmt, MimirUnOp,
    Mutability, PrimitiveTy, RichConstExpr, RichKernelStmt, RichMimirExpr, Span, ValidationError,
    passes::kernel::walk_block, util::LocalId,
};
use std::collections::{BTreeMap, BTreeSet};

fn error(span: Span, message: &str) -> ValidationError {
    ValidationError {
        span,
        message: message.into(),
    }
}

type Check<T> = Result<T, ValidationError>;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ty {
    Scalar(PrimitiveTy),
    Array(PrimitiveTy),
}

/// Type, scope and mutability validation. A kernel visit resets all state, so
/// the same pass may be reused after either success or failure.
#[derive(Default)]
pub struct ValidityPass {
    vars: BTreeMap<LocalId, (Ty, Mutability)>,
    seen: BTreeSet<LocalId>,
    generics: usize,
}

impl ValidityPass {
    fn bind(&mut self, id: LocalId, ty: Ty, access: Mutability, span: &Span) -> Check<()> {
        if !self.seen.insert(id) {
            return Err(error(span.clone(), "duplicate binding id"));
        }
        self.vars.insert(id, (ty, access));
        Ok(())
    }

    fn lookup(&self, id: LocalId, span: &Span) -> Check<(Ty, Mutability)> {
        self.vars
            .get(&id)
            .copied()
            .ok_or_else(|| error(span.clone(), "binding is not in lexical scope"))
    }

    fn check_kernel(&mut self, kernel: &mut MimirKernelAst) -> Check<()> {
        self.generics = kernel.const_generics.len();
        let mut names = BTreeSet::new();
        for name in &kernel.const_generics {
            if !names.insert(name) {
                return Err(error(0..0, "duplicate const generic name"));
            }
        }
        names.clear();
        for p in &kernel.parameters {
            if !names.insert(&p.name) {
                return Err(error(p.span.clone(), "duplicate parameter name"));
            }
            let (ty, access) = match p.ty {
                KernelParamTy::Scalar(t) => (Ty::Scalar(t), Mutability::Constant),
                KernelParamTy::Buffer { element, access } => (Ty::Array(element), access),
            };
            self.bind(p.id, ty, access, &p.span)?;
        }
        self.block(&mut kernel.body)?;
        Ok(())
    }
}

impl KernelAstPass for ValidityPass {
    type Error = ValidationError;
    type ExprOutput = PrimitiveTy;

    fn kernel(&mut self, kernel: &mut MimirKernelAst) -> Check<()> {
        *self = Self::default();
        let result = self.check_kernel(kernel);
        *self = Self::default();
        result
    }

    fn block(&mut self, body: &mut MimirKernelBlock) -> Check<()> {
        let saved = self.vars.clone();
        let result = walk_block(self, body);
        self.vars = saved;
        result
    }

    fn stmt(&mut self, s: &mut RichKernelStmt) -> Check<()> {
        let span = &s.span;
        match &mut s.stmt {
            MimirKernelStmt::Let {
                binding,
                ty,
                mutability,
                value,
            } => {
                require(
                    self.expr(value)? == *ty,
                    &value.span,
                    "initializer type mismatch",
                )?;
                self.bind(binding.id, Ty::Scalar(*ty), *mutability, &binding.span)?;
            }
            MimirKernelStmt::SharedMem {
                binding,
                element,
                length,
            } => {
                require(
                    self.const_expr(length)? == PrimitiveTy::Uint32,
                    &length.span,
                    "shared length must be u32",
                )?;
                if matches!(length.expr, ConstExpr::Literal(Literal::Uint32(0))) {
                    return Err(error(length.span.clone(), "shared length must be positive"));
                }
                self.bind(
                    binding.id,
                    Ty::Array(*element),
                    Mutability::Mutable,
                    &binding.span,
                )?;
            }
            MimirKernelStmt::Assign { target, value } => {
                let (id, index) = match target {
                    AssignTarget::Var(id) => (*id, None),
                    AssignTarget::Index { array, index } => (*array, Some(index)),
                };
                let (ty, access) = self.lookup(id, span)?;
                require(
                    access == Mutability::Mutable,
                    span,
                    "assignment to immutable binding",
                )?;
                let target_ty = match (ty, index) {
                    (Ty::Scalar(t), None) => t,
                    (Ty::Array(t), Some(i)) => {
                        require(
                            self.expr(i)?.is_integer(),
                            &i.span,
                            "array index must be integer",
                        )?;
                        t
                    }
                    _ => return Err(error(span.clone(), "invalid assignment target")),
                };
                require(
                    self.expr(value)? == target_ty,
                    &value.span,
                    "assignment type mismatch",
                )?;
            }
            MimirKernelStmt::RangeFor {
                binding,
                start,
                end,
                step,
                body,
            } => {
                let ty = self.expr(start)?;
                require(
                    ty.is_integer() && self.expr(end)? == ty,
                    span,
                    "range bounds must have the same integer type",
                )?;
                if let Some(step) = step {
                    require(
                        self.expr(step)? == ty,
                        &step.span,
                        "range step type mismatch",
                    )?;
                    require(
                        !matches!(
                            step.expr,
                            MimirExpr::Literal(Literal::Int32(0) | Literal::Uint32(0))
                        ),
                        &step.span,
                        "range step must be nonzero",
                    )?;
                }
                let outer = self.vars.clone();
                self.bind(
                    binding.id,
                    Ty::Scalar(ty),
                    Mutability::Constant,
                    &binding.span,
                )?;
                let result = self.block(body);
                self.vars = outer;
                result?;
            }
            MimirKernelStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                require(
                    self.expr(condition)? == PrimitiveTy::Bool,
                    &condition.span,
                    "condition must be bool",
                )?;
                self.block(then_branch)?;
                if let Some(e_branch) = else_branch {
                    self.block(e_branch)?;
                }
            }
            MimirKernelStmt::Block(body) => self.block(body)?,
            MimirKernelStmt::Return | MimirKernelStmt::Syncthreads => {}
        }
        Ok(())
    }
    fn expr(&mut self, e: &mut RichMimirExpr) -> Check<PrimitiveTy> {
        let span = &e.span;
        match &mut e.expr {
            MimirExpr::Literal(l) => Ok(l.ty()),
            MimirExpr::Var(id) => match self.lookup(*id, span)?.0 {
                Ty::Scalar(t) => Ok(t),
                _ => Err(error(span.clone(), "array is not a scalar value")),
            },
            MimirExpr::Index { array, index } => {
                require(
                    self.expr(index)?.is_integer(),
                    &index.span,
                    "array index must be integer",
                )?;
                match self.lookup(*array, span)?.0 {
                    Ty::Array(t) => Ok(t),
                    _ => Err(error(span.clone(), "indexing a scalar")),
                }
            }
            MimirExpr::Builtin { .. } => Ok(PrimitiveTy::Uint32),
            MimirExpr::BinOp(b) => {
                binary(&b.op, self.expr(&mut b.lhs)?, self.expr(&mut b.rhs)?, span)
            }
            MimirExpr::Unary(u) => unary(&u.op, self.expr(&mut u.expr)?, span),
            MimirExpr::Cast { value, to } => {
                self.expr(value)?;
                Ok(*to)
            }
            MimirExpr::ConstExpr(c) => self.const_expr(c),
            MimirExpr::MathIntrinsic(i) => intrinsic(i, |e| self.expr(e), span),
        }
    }

    fn const_expr(&mut self, e: &mut RichConstExpr) -> Check<PrimitiveTy> {
        match &mut e.expr {
            ConstExpr::Literal(l) => Ok(l.ty()),
            ConstExpr::Generic(id) => {
                require(
                    (id.0 as usize) < self.generics,
                    &e.span,
                    "unknown const generic",
                )?;
                Ok(PrimitiveTy::Uint32)
            }
            ConstExpr::BinOp(b) => binary(
                &b.op,
                self.const_expr(&mut b.lhs)?,
                self.const_expr(&mut b.rhs)?,
                &e.span,
            ),
            ConstExpr::UnOp(u) => unary(&u.op, self.const_expr(&mut u.expr)?, &e.span),
            ConstExpr::Cast { value, to } => {
                self.const_expr(value)?;
                Ok(*to)
            }
            ConstExpr::MathIntrinsic(i) => intrinsic(i, |e| self.const_expr(e), &e.span),
        }
    }
}
fn require(ok: bool, span: &Span, message: &str) -> Check<()> {
    if ok {
        Ok(())
    } else {
        Err(error(span.clone(), message))
    }
}

fn binary(op: &MimirBinOp, lhs: PrimitiveTy, rhs: PrimitiveTy, span: &Span) -> Check<PrimitiveTy> {
    use MimirBinOp::*;
    require(lhs == rhs, span, "binary operand type mismatch")?;
    match op {
        And | Or => {
            require(
                lhs == PrimitiveTy::Bool,
                span,
                "logical operands must be bool",
            )?;
            Ok(PrimitiveTy::Bool)
        }
        Eq | Ne => Ok(PrimitiveTy::Bool),
        Lt | Lte | Gt | Gte => {
            require(
                lhs.is_numeric(),
                span,
                "comparison operands must be numeric",
            )?;
            Ok(PrimitiveTy::Bool)
        }
        _ => {
            require(
                lhs.is_numeric(),
                span,
                "arithmetic operands must be numeric",
            )?;
            Ok(lhs)
        }
    }
}

fn unary(op: &MimirUnOp, ty: PrimitiveTy, span: &Span) -> Check<PrimitiveTy> {
    require(
        match op {
            MimirUnOp::Not => ty == PrimitiveTy::Bool,
            MimirUnOp::Neg => matches!(ty, PrimitiveTy::Int32 | PrimitiveTy::Float32),
        },
        span,
        "invalid unary operand",
    )?;
    Ok(ty)
}

fn intrinsic<T>(
    i: &mut Intrinsic<T>,
    mut infer: impl FnMut(&mut T) -> Check<PrimitiveTy>,
    span: &Span,
) -> Check<PrimitiveTy> {
    let (args, numeric): (&mut [T], bool) = {
        let is_numeric = i.is_numeric();
        (
            match i {
                Intrinsic::Unary(_, a) => std::slice::from_mut(a.as_mut()),
                Intrinsic::Binary(_, a) => a.as_mut(),
                Intrinsic::Ternary(_, a) => a.as_mut(),
            },
            is_numeric,
        )
    };

    let first = infer(&mut args[0])?;
    require(
        if numeric {
            first.is_numeric()
        } else {
            first == PrimitiveTy::Float32
        },
        span,
        "invalid intrinsic operand type",
    )?;
    for arg in &mut args[1..] {
        require(
            infer(arg)? == first,
            span,
            "intrinsic operand type mismatch",
        )?;
    }
    Ok(first)
}
