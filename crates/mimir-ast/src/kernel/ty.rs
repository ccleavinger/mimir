use crate::{Mutability, PrimitiveTy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelParamTy {
    Scalar(PrimitiveTy),
    Buffer {
        element: PrimitiveTy,
        access: Mutability,
    },
}
