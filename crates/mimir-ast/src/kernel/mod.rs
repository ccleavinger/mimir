use crate::{Span, util::LocalId};

mod stmt;
mod ty;

pub use stmt::{AssignTarget, MimirKernelBlock, MimirKernelStmt, RichKernelStmt};
pub use ty::KernelParamTy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstGenericId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelParameter {
    pub id: LocalId,
    pub name: String,
    pub ty: KernelParamTy,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub id: LocalId,
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimirKernelAst {
    pub name: String,
    pub parameters: Vec<KernelParameter>,
    /// Const generic index is its position; all are u32 for the current kernel language.
    pub const_generics: Vec<String>,
    pub body: MimirKernelBlock,
}
