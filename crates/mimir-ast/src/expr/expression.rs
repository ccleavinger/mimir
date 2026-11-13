use crate::{
    Expr, Intrinsic, Literal, MimirBinOpExpr, MimirUnaryExpr, PrimitiveTy, RichConstExpr, RichExpr,
    util::LocalId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MimirExpr {
    Call {
        function: u32,
        arguments: Vec<RichMimirExpr>,
        parameters: Vec<crate::KernelParamTy>,
        returns: PrimitiveTy,
    },
    BinOp(MimirBinOpExpr<RichMimirExpr>),
    Unary(MimirUnaryExpr<RichMimirExpr>),
    Literal(Literal),
    Var(LocalId),
    Index {
        array: LocalId,
        index: Box<RichMimirExpr>,
    },
    Builtin {
        builtin: Builtin,
        field: Axis,
    },
    MathIntrinsic(Intrinsic<RichMimirExpr>),
    Cast {
        value: Box<RichMimirExpr>,
        to: PrimitiveTy,
    },
    ConstExpr(RichConstExpr),
}

pub type RichMimirExpr = RichExpr<MimirExpr>;

impl Expr for MimirExpr {
    fn priority(&self) -> u32 {
        match self {
            Self::Call { .. } => 1 << 3,
            Self::BinOp(e) => e.op.priority(),
            Self::Unary(_) => 1 << 5,
            Self::Index { .. } | Self::MathIntrinsic(_) => 1 << 3,
            Self::Builtin { .. } => 1 << 2,
            Self::Cast { .. } => 1 << 6,
            Self::ConstExpr(e) => e.priority(),
            _ => 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    BlockIdx,
    BlockDim,
    ThreadIdx,
    GlobalInvocationId,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}
