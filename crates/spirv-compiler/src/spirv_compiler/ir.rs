use rspirv::spirv::{StorageClass, Word};

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirVariable {
    pub ty: MimirPtrType,
    pub word: Option<Word>, // if empty then uninitialized
}

// TODO: get shared variables actually working. This will require rewriting a *LOT* of internal stuff and I don't wanna do that rn
// #[derive(Debug, Clone, PartialEq, Hash, Eq)]
// pub struct SharedVariable {
//     pub size: i32, // length of arr
//     pub ty: MimirPtrType, // StorageClass willl always be `WorkGroup`
//     pub word: Option<Word>,
// }

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirType {
    Int32,
    Int64,
    Uint32,
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
    Assign(Box<MimirExprIR>, Box<MimirExprIR>), // (index, field, or var), value
    BinAssign(Box<MimirExprIR>, MimirBinOp, Box<MimirExprIR>), // var_name, bin_op (excluding comparison ops), value
    BinOp(Box<MimirExprIR>, MimirBinOp, Box<MimirExprIR>, bool), // lhs, bin_op, rhs, is_parenthesized 
    Index(String, Box<MimirExprIR>), // arr var, index
    Field(String, String), // struct var, field name (compiles to OpAccessChain)
    For(String, Box<MimirExprIR>, Box<MimirExprIR>, Option<Box<MimirExprIR>>, Vec<MimirExprIR>), // range-based for with optional step. `for i in (0..10).step_by(2)` would be For("i", 0, 10, Some(2))
    If(Box<MimirExprIR>, Vec<MimirExprIR>, Option<Vec<MimirExprIR>>), // condition, then, else
    Unary(MimirUnOp, Box<MimirExprIR>), // unary op, expr
    Literal(MimirLit),
    Var(String), // variable name; used for referencing variables
    Return, // always return void for kernel
    ExtInstFunc(ExtInstFunc, Vec<MimirExprIR>),
    Syncthreads, // barrier
}

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirLit {
    Int32(i32),
    Int64(i64),
    Float32(u32), // f32::to_bits()
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


#[derive(Debug, Clone, PartialEq)]
pub enum ExtInstFunc {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Atan2,
    Pow,
    Exp,
    Log,
    Exp2,
    Log2,
    Sqrt,
    Isqrt,
    Max,
    Min,
    Floor,
    Ceil,
    Clamp,
    Mix,
}