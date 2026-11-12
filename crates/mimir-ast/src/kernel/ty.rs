use crate::{Mutability, PrimitiveTy, RichConstExpr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalKernelTy {
    Prim(PrimitiveTy),
    SharedMemArr {
        element: PrimitiveTy,
        length: Box<RichConstExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelParamTy {
    Scalar(PrimitiveTy),
    Buffer {
        element: PrimitiveTy,
        access: Mutability,
    },
}
