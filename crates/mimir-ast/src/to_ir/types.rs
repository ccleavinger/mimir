use super::*;
pub(super) fn scalar(t: PrimitiveTy) -> ir::ScalarType {
    match t {
        PrimitiveTy::Int32 => ir::ScalarType::I32,
        PrimitiveTy::Uint32 => ir::ScalarType::U32,
        PrimitiveTy::Float32 => ir::ScalarType::F32,
        PrimitiveTy::Bool => ir::ScalarType::Bool,
    }
}
pub(super) fn literal(t: Literal) -> ir::Literal {
    match t {
        Literal::Int32(v) => ir::Literal::I32(v),
        Literal::Uint32(v) => ir::Literal::U32(v),
        Literal::Float32(v) => ir::Literal::F32(v),
        Literal::Bool(v) => ir::Literal::Bool(v),
    }
}
pub(super) fn binary(t: &MimirBinOp) -> ir::BinaryOp {
    match t {
        MimirBinOp::Add => ir::BinaryOp::Add,
        MimirBinOp::Sub => ir::BinaryOp::Sub,
        MimirBinOp::Mul => ir::BinaryOp::Mul,
        MimirBinOp::Div => ir::BinaryOp::Div,
        MimirBinOp::Mod => ir::BinaryOp::Rem,
        MimirBinOp::And => ir::BinaryOp::And,
        MimirBinOp::Or => ir::BinaryOp::Or,
        MimirBinOp::Lt => ir::BinaryOp::Lt,
        MimirBinOp::Lte => ir::BinaryOp::Le,
        MimirBinOp::Gt => ir::BinaryOp::Gt,
        MimirBinOp::Gte => ir::BinaryOp::Ge,
        MimirBinOp::Eq => ir::BinaryOp::Eq,
        MimirBinOp::Ne => ir::BinaryOp::Ne,
    }
}
