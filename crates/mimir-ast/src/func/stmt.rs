use crate::{AssignTarget, Binding, Mutability, PrimitiveTy, RichConstExpr, RichMimirExpr, Span};

pub type MimirFuncBlock = Vec<RichFuncStmt>;

pub struct RichFuncStmt {
    pub stmt: MimirFuncStmt,
    pub span: Span,
}

pub enum MimirFuncStmt {
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
        body: MimirFuncBlock,
    },
    If {
        condition: RichMimirExpr,
        then_branch: MimirFuncBlock,
        else_branch: Option<MimirFuncBlock>,
    },
    Block(MimirFuncBlock),
    Return(Option<(PrimitiveTy, RichMimirExpr)>),
    Syncthreads,
}
