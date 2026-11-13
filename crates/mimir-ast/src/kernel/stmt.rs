use crate::{Binding, Mutability, PrimitiveTy, RichConstExpr, RichMimirExpr, Span, util::LocalId};

/// A statement with source metadata. Statement-specific behavior belongs here;
/// expression precedence and parentheses belong to `RichExpr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichKernelStmt {
    pub stmt: MimirKernelStmt,
    pub span: Span,
}

impl RichKernelStmt {
    pub fn new(stmt: MimirKernelStmt, span: Span) -> Self {
        Self { stmt, span }
    }
}

pub type MimirKernelBlock = Vec<RichKernelStmt>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MimirKernelStmt {
    Let {
        binding: Binding,
        ty: PrimitiveTy,
        mutability: Mutability,
        value: RichMimirExpr,
    },
    SharedMem {
        binding: Binding,
        element: PrimitiveTy,
        length: RichConstExpr,
    },
    Assign {
        target: AssignTarget,
        value: RichMimirExpr,
    },
    RangeFor {
        binding: Binding,
        start: RichMimirExpr,
        end: RichMimirExpr,
        step: Option<RichMimirExpr>,
        body: MimirKernelBlock,
    },
    If {
        condition: RichMimirExpr,
        then_branch: MimirKernelBlock,
        else_branch: Option<MimirKernelBlock>,
    },
    Block(MimirKernelBlock),
    Return,
    ReturnValue(RichMimirExpr),
    Syncthreads,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignTarget {
    Var(LocalId),
    Index {
        array: LocalId,
        index: RichMimirExpr,
    },
}
