use crate::{
    ConstGenericId, Expr, Intrinsic, Literal, MimirBinOpExpr, MimirUnaryExpr, PrimitiveTy, RichExpr,
};

/// Closed grammar: runtime variables, buffers and builtins cannot occur here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstExpr {
    BinOp(MimirBinOpExpr<RichConstExpr>),
    UnOp(MimirUnaryExpr<RichConstExpr>),
    Literal(Literal),
    Generic(ConstGenericId),
    MathIntrinsic(Intrinsic<RichConstExpr>),
    Cast {
        value: Box<RichConstExpr>,
        to: PrimitiveTy,
    },
}

pub type RichConstExpr = RichExpr<ConstExpr>;

impl Expr for ConstExpr {
    fn priority(&self) -> u32 {
        match self {
            Self::BinOp(e) => e.op.priority(),
            Self::UnOp(_) => 1 << 5,
            Self::MathIntrinsic(_) => 1 << 3,
            Self::Cast { .. } => 1 << 6,
            _ => 1,
        }
    }
}
