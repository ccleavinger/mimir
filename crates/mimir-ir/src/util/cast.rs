use crate::ir::{MimirPrimitiveTy, MimirTy};

pub fn is_legal(from: &MimirTy, to: &MimirPrimitiveTy) -> bool {
    match to {
        MimirPrimitiveTy::Int32 | MimirPrimitiveTy::Uint32 | MimirPrimitiveTy::Float32 => {
            from.is_numeric()
        }
        MimirPrimitiveTy::Bool => from.is_integer(), // int_val != 0
    }
}
