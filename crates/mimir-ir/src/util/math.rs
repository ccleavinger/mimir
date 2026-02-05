use crate::{
    ir::{MathIntrinsicFunc, MimirPrimitiveTy},
    util::error::ASTError,
};

pub fn intrinsic_param_count(func: &MathIntrinsicFunc) -> usize {
    match func {
        MathIntrinsicFunc::Sin
        | MathIntrinsicFunc::Cos
        | MathIntrinsicFunc::Tan
        | MathIntrinsicFunc::Asin
        | MathIntrinsicFunc::Acos
        | MathIntrinsicFunc::Sinh
        | MathIntrinsicFunc::Cosh
        | MathIntrinsicFunc::Tanh
        | MathIntrinsicFunc::Asinh
        | MathIntrinsicFunc::Acosh
        | MathIntrinsicFunc::Atanh
        | MathIntrinsicFunc::Atan
        | MathIntrinsicFunc::Exp
        | MathIntrinsicFunc::Log
        | MathIntrinsicFunc::Exp2
        | MathIntrinsicFunc::Log2
        | MathIntrinsicFunc::Sqrt
        | MathIntrinsicFunc::Isqrt
        | MathIntrinsicFunc::Floor
        | MathIntrinsicFunc::Ceil => 1,

        MathIntrinsicFunc::Atan2
        | MathIntrinsicFunc::Pow
        | MathIntrinsicFunc::Max
        | MathIntrinsicFunc::Min => 2,

        MathIntrinsicFunc::Clamp | MathIntrinsicFunc::Mix | MathIntrinsicFunc::Fma => 3,
    }
}

// TODO: move to custom error type not bool
pub fn intrinsic_param_validity(func: &MathIntrinsicFunc, args: &[MimirPrimitiveTy]) -> bool {
    let expected_count = intrinsic_param_count(func);
    if args.len() != expected_count {
        return false;
    }

    match func {
        MathIntrinsicFunc::Max | MathIntrinsicFunc::Min | MathIntrinsicFunc::Clamp => {
            let first = args.first().unwrap(); // safe to unwrap since we checked length
            args.iter().all(|ty| ty.is_numeric()) && args.iter().all(|ty| ty == first)
            // all arguments must be of the same numeric type
        }
        _ => args.iter().all(|ty| ty.is_floating_point()),
    }
}

pub fn get_output_ty(
    func: &MathIntrinsicFunc,
    args: &[MimirPrimitiveTy],
) -> Result<MimirPrimitiveTy, ASTError> {
    if !intrinsic_param_validity(func, args) {
        return Err(ASTError::Validation(format!(
            "Invalid argument types for intrinsic function: {:?} with args: {:?}",
            func, args
        )));
    }

    match func {
        MathIntrinsicFunc::Max | MathIntrinsicFunc::Min | MathIntrinsicFunc::Clamp => {
            let first = args.first().unwrap(); // safe to unwrap since we checked length
            Ok(first.clone())
        }
        _ => {
            // default to 32 bit float if unsure
            Ok(MimirPrimitiveTy::Float32)
        }
    }
}
