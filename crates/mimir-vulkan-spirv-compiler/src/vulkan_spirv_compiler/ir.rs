use mimir_ir::ir::MimirTy;
use rspirv::spirv::{StorageClass, Word};

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirVariable {
    pub ty: MimirPtrType,
    pub word: Word,
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirPtrType {
    pub base: MimirTy,
    pub storage_class: StorageClass,
}
