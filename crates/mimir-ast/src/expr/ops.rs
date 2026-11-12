use crate::{Expr, MimirBinOp, MimirUnOp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimirBinOpExpr<T: Expr> {
    pub lhs: Box<T>,
    pub op: MimirBinOp,
    pub rhs: Box<T>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimirUnaryExpr<T: Expr> {
    pub op: MimirUnOp,
    pub expr: Box<T>,
}

impl<T: Expr> MimirBinOpExpr<T> {
    pub fn new(lhs: T, op: MimirBinOp, rhs: T) -> Self {
        Self {
            lhs: Box::new(lhs),
            op,
            rhs: Box::new(rhs),
        }
    }
}

impl<T: Expr> MimirUnaryExpr<T> {
    pub fn new(op: MimirUnOp, expr: T) -> Self {
        Self {
            op,
            expr: Box::new(expr),
        }
    }
}

// report what types of parameters are needed
pub trait IntrinsicReport {
    fn is_numeric(&self) -> bool;

    fn float_only(&self) -> bool {
        !self.is_numeric()
    }
}

/// Intrinsic arity is encoded in the AST rather than checked against a Vec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intrinsic<T> {
    Unary(UnaryIntrinsic, Box<T>),
    Binary(BinaryIntrinsic, Box<[T; 2]>),
    Ternary(TernaryIntrinsic, Box<[T; 3]>),
}

impl<T> IntrinsicReport for Intrinsic<T> {
    // When true all numeric primitives are supported
    fn is_numeric(&self) -> bool {
        match self {
            Intrinsic::Unary(unary, _) => unary.is_numeric(),
            Intrinsic::Binary(binary_intrinsic, _) => binary_intrinsic.is_numeric(),
            Intrinsic::Ternary(ternary_intrinsic, _) => {
                matches!(ternary_intrinsic, TernaryIntrinsic::Clamp)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryIntrinsic {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Exp,
    Log,
    Exp2,
    Log2,
    Sqrt,
    Isqrt,
    Floor,
    Ceil,
}

impl IntrinsicReport for UnaryIntrinsic {
    fn is_numeric(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryIntrinsic {
    Atan2,
    Pow,
    Max,
    Min,
}

impl IntrinsicReport for BinaryIntrinsic {
    fn is_numeric(&self) -> bool {
        matches!(self, Self::Max | Self::Min)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryIntrinsic {
    Clamp,
    Mix,
    Fma,
}

impl IntrinsicReport for TernaryIntrinsic {
    fn is_numeric(&self) -> bool {
        matches!(self, Self::Clamp)
    }
}
