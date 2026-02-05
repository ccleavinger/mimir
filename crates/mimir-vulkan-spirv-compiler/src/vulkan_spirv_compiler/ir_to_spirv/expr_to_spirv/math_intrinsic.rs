use mimir_ir::{
    ir::{MathIntrinsicFunc, MimirExpr, MimirPrimitiveTy, MimirTy},
    util::math::intrinsic_param_validity,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::dr::Operand;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir_to_spirv::expr_to_spirv::expr::ExprToWordResult,
    util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub fn math_intrinsic_to_word(
        &mut self,
        func: &MathIntrinsicFunc,
        args: &[MimirExpr],
    ) -> ExprToWordResult {
        let (arg_words, arg_tys) = args
            .iter()
            .map(|arg| self.expr_to_word(arg))
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|(word, x)| match x {
                MimirTy::Primitive(prim) => Ok((Operand::IdRef(*word), prim.clone())),
                _ => Err(MimirJITCompilationError::Generic(format!(
                    "Arguments of a math intrinsic must be primitive types, not `{x:?}`"
                ))),
            })
            .collect::<Result<(Vec<_>, Vec<_>), _>>()?;

        if !intrinsic_param_validity(func, &arg_tys) {
            return Err(MimirJITCompilationError::Generic(format!(
                "Invalid math intrinsic `{:?}` called with the following parameters: `{:?}`",
                func, args
            )));
        }

        let ret_ty = func.ret_ty(&arg_tys);

        // All instruction codes are directly from the official SPIR-V documentation of extended instructions for GLSL
        // https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
        let inst_code = match func.clone() {
            MathIntrinsicFunc::Max | MathIntrinsicFunc::Min | MathIntrinsicFunc::Clamp => {
                match (&func, &ret_ty) {
                    (MathIntrinsicFunc::Max, MimirPrimitiveTy::Float32) => 40,
                    (MathIntrinsicFunc::Max, MimirPrimitiveTy::Uint32) => 41,
                    (MathIntrinsicFunc::Max, MimirPrimitiveTy::Int32) => 42,
                    (MathIntrinsicFunc::Min, MimirPrimitiveTy::Float32) => 37,
                    (MathIntrinsicFunc::Min, MimirPrimitiveTy::Uint32) => 38,
                    (MathIntrinsicFunc::Min, MimirPrimitiveTy::Int32) => 39,
                    (MathIntrinsicFunc::Clamp, MimirPrimitiveTy::Float32) => 43,
                    (MathIntrinsicFunc::Clamp, MimirPrimitiveTy::Uint32) => 44,
                    (MathIntrinsicFunc::Clamp, MimirPrimitiveTy::Int32) => 45,
                    _ => {
                        return Err(MimirJITCompilationError::Generic(format!(
                            "Unsupported type `{:?}` for math intrinsic function `{:?}`",
                            ret_ty, func
                        )))
                    }
                }
            }
            MathIntrinsicFunc::Sin => 13,
            MathIntrinsicFunc::Cos => 14,
            MathIntrinsicFunc::Tan => 15,
            MathIntrinsicFunc::Asin => 16,
            MathIntrinsicFunc::Acos => 17,
            MathIntrinsicFunc::Sinh => 19,
            MathIntrinsicFunc::Cosh => 20,
            MathIntrinsicFunc::Tanh => 21,
            MathIntrinsicFunc::Asinh => 22,
            MathIntrinsicFunc::Acosh => 23,
            MathIntrinsicFunc::Atanh => 24,
            MathIntrinsicFunc::Atan2 => 25,
            MathIntrinsicFunc::Pow => 26,
            MathIntrinsicFunc::Exp => 27,
            MathIntrinsicFunc::Log => 28,
            MathIntrinsicFunc::Exp2 => 29,
            MathIntrinsicFunc::Log2 => 30,
            MathIntrinsicFunc::Sqrt => 31,
            MathIntrinsicFunc::Isqrt => 32,
            MathIntrinsicFunc::Floor => 8,
            MathIntrinsicFunc::Ceil => 9,
            MathIntrinsicFunc::Mix => 46,
            MathIntrinsicFunc::Atan => 18,
            MathIntrinsicFunc::Fma => 50,
        };

        let ret_ty_word = self.get_mimir_ty(&ret_ty.to_mimir_ty())?;

        let word = self
            .spirv_builder
            .ext_inst(ret_ty_word, None, self.ext_inst, inst_code, arg_words)
            .map_err(map_err_closure)?;

        Ok((word, ret_ty.to_mimir_ty()))
    }
}
