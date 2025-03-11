use rspirv::spirv::{StorageClass, Word};
use ordered_float::OrderedFloat;

pub struct MimirVariable {
    pub ty: MimirPtrType,
    pub word: Option<Word>, // if empty then uninitialized
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirType {
    Int32,
    Int64,
    Float32,
    Uint32Vec3,
    Float32Vec3,
    RuntimeArray(Box<MimirType>),
    Bool,
    Void,
    Unknown // error type, must be resolved before compiling. i.e later intialization
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirPtrType {
    pub base: MimirType,
    pub storage_class: StorageClass
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirBuiltIn {
    // CUDA style
    BlockIdx,
    BlockDim,
    ThreadIdx,
    // GLSL style
    GlobalInvocationId
}

#[derive(Debug, Clone, PartialEq)]
pub enum MimirExprIR {
    Local(String, Option<Box<MimirExprIR>>), // var_name, init
    Assign(String, Box<MimirExprIR>), // var_name, value
    BinAssign(String, MimirBinOp, Box<MimirExprIR>), // var_name, bin_op (excluding comparison ops), value
    BinOp(Box<MimirExprIR>, MimirBinOp, Box<MimirExprIR>, bool), // lhs, bin_op, rhs, is_parenthesized 
    Index(String, Box<MimirExprIR>), // array var, index
    Field(String, String), // struct var, field name (compiles to OpAccessChain)
    For(String, i64, i64, Vec<MimirExprIR>), // currently just range-based for. `for i in 0..10` would be For("i", 0, 10)
    If(Box<MimirExprIR>, Box<MimirExprIR>, Option<Box<MimirExprIR>>), // condition, then, else
    Unary(MimirUnOp, Box<MimirExprIR>), // unary op, expr
    Literal(MimirLit),
    Var(String), // variable name; used for referencing variables
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirLit {
    Int32(i32),
    Int64(i64),
    Float32(OrderedFloat<f32>),
    Bool(bool)
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirUnOp {
    Not,
    Neg
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Lt, // <
    Lte, // <=
    Gt, // >
    Gte, // >=
    Eq, // ==
    Ne // !=
}
