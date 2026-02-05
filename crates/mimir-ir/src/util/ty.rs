use std::collections::BTreeMap;

use crate::{
    ir::{
        MathIntrinsicExpr, MimirBinOp, MimirBinOpExpr, MimirCastExpr, MimirConstExpr, MimirExpr,
        MimirIndexExpr, MimirPrimitiveTy, MimirTy, MimirTyVar,
    },
    util::{error::ASTError, math},
};

/// This is only really for the Rust to MimirIR compiler
#[inline]
pub fn str_to_ty(ty: &str) -> Result<MimirPrimitiveTy, ASTError> {
    Ok(match ty.replace(" ", "").as_str() {
        "i32" => MimirPrimitiveTy::Int32,
        "usize" => MimirPrimitiveTy::Int32, // we treat usize as an int32
        "f32" => MimirPrimitiveTy::Float32,
        "u32" => MimirPrimitiveTy::Uint32,
        "bool" => MimirPrimitiveTy::Bool,
        _ => return Err(ASTError::Validation(format!("`{ty}` is an invalid type"))),
    })
}

#[inline]
pub fn type_importance(types: &[MimirTy]) -> MimirTy {
    let first_ty = &types[0];
    if types.iter().all(|ty| ty == first_ty) {
        return first_ty.clone();
    }

    if types
        .iter()
        .any(|ty| matches!(*ty, MimirTy::GlobalArray { .. }))
    {
        // this is the morally wrong thing to do
        panic!("Aww shucks! Try not finding the type importance of a pointer to global memory!\nDonkey")
    } else if types.contains(&MimirTy::Primitive(MimirPrimitiveTy::Float32)) {
        MimirTy::Primitive(MimirPrimitiveTy::Float32)
    } else if types.contains(&MimirTy::Primitive(MimirPrimitiveTy::Int32)) {
        MimirTy::Primitive(MimirPrimitiveTy::Int32)
    } else {
        // when in doubt cast to an unsigned 32-bit integer
        MimirTy::Primitive(MimirPrimitiveTy::Uint32)
    }
}

impl MimirExpr {
    /// ONLY USE AFTER CHECKING TO ENSURE ITS SAFE
    pub(crate) fn to_ty(&self, var_map: &BTreeMap<u64, MimirTyVar>) -> MimirPrimitiveTy {
        match self {
            MimirExpr::BinOp(MimirBinOpExpr { lhs, op, .. }) => {
                let lhs_ty = lhs.to_ty(var_map);

                match op {
                    MimirBinOp::Lt
                    | MimirBinOp::Lte
                    | MimirBinOp::Gt
                    | MimirBinOp::Gte
                    | MimirBinOp::Eq
                    | MimirBinOp::Ne => MimirPrimitiveTy::Bool,
                    _ => lhs_ty,
                }
            }
            MimirExpr::Index(MimirIndexExpr { var, .. }) => {
                match &var_map.get(var).unwrap().ty {
                    // this lowk shouldn't happen
                    MimirTy::Primitive(_) => {
                        panic!()
                    }
                    MimirTy::GlobalArray { element_type } => element_type.clone(),
                    MimirTy::SharedMemArray { .. } => MimirPrimitiveTy::Float32,
                }
            }
            MimirExpr::BuiltinFieldAccess { .. } => MimirPrimitiveTy::Uint32,
            MimirExpr::Unary(mimir_un_op_expr) => mimir_un_op_expr.expr.to_ty(var_map),
            MimirExpr::Literal(mimir_lit) => mimir_lit.to_ty(),
            MimirExpr::Var(id) => match &var_map.get(id).unwrap().ty {
                MimirTy::Primitive(mimir_primitive_ty) => mimir_primitive_ty.clone(),
                MimirTy::GlobalArray { .. } => panic!(),
                MimirTy::SharedMemArray { .. } => panic!(),
            },
            MimirExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => math::get_output_ty(
                func,
                &args.iter().map(|a| a.to_ty(var_map)).collect::<Vec<_>>(),
            )
            .unwrap(),
            MimirExpr::Cast(MimirCastExpr { to, .. }) => to.clone(),
            MimirExpr::ConstExpr(mimir_const_expr) => match mimir_const_expr {
                MimirConstExpr::BinOp(mimir_bin_op_expr) => {
                    let lhs_ty = mimir_bin_op_expr.lhs.to_norm_expr().to_ty(var_map);

                    match mimir_bin_op_expr.op {
                        MimirBinOp::Lt
                        | MimirBinOp::Lte
                        | MimirBinOp::Gt
                        | MimirBinOp::Gte
                        | MimirBinOp::Eq
                        | MimirBinOp::Ne => MimirPrimitiveTy::Bool,
                        _ => lhs_ty,
                    }
                }
                MimirConstExpr::Unary(_) => mimir_const_expr.to_norm_expr().to_ty(var_map),
                MimirConstExpr::Literal(mimir_lit) => mimir_lit.to_ty(),
                MimirConstExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
                    math::get_output_ty(
                        func,
                        &args
                            .iter()
                            .map(|a| a.to_norm_expr().to_ty(var_map))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                }
                MimirConstExpr::Cast(MimirCastExpr { to, .. }) => to.clone(),
                MimirConstExpr::ConstGeneric { .. } => MimirPrimitiveTy::Uint32,
            },
        }
    }
}
