use crate::{
    PrimitiveTy, RichMimirExpr, Span,
    func::{stmt::MimirFuncBlock, ty::FuncParamTy},
    util::LocalId,
};

mod stmt;
mod ty;

pub struct FuncParamater {
    pub id: LocalId,
    pub name: String,
    pub ty: FuncParamTy,
    pub span: Span,
}

pub struct MimirFuncAst {
    pub name: String,

    pub params: Vec<FuncParamater>,
    pub body: MimirFuncBlock,
}
