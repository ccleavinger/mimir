use crate::{Mutability, PrimitiveTy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalFuncTy {
    Prim(PrimitiveTy), // Currently only primitives can be instantiated within mimir functions
}

pub enum FuncParamTy {
    Scalar(PrimitiveTy),
    Buffer {
        element: PrimitiveTy,
        access: Mutability,
    },
    // LocalArr {
    //     element: PrimitiveTy,
    // }
}
