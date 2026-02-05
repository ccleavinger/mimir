use std::{
    collections::{BTreeMap, HashMap},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

use crate::{kernel_ir::MimirKernelIR, util::math::intrinsic_param_validity};

//
// WARNING: THIS IS NOT A FINALIZED INTERMEDIATE REPRESENTATION.
// ANYTHIN IS SUBJECT TO CHANGE.
// NOTHING IS STANDARDIZED NOR FINALIZED.
// TODO: standardize and document the IR (file format, memory layout, etc.)

// for kernels and functions (planned feature)
pub trait IRFunctionality {
    fn var_map(&self) -> BTreeMap<u64, MimirTyVar>;

    fn get_body(&self) -> Vec<MimirStmt>;

    fn as_any(&self) -> &dyn std::any::Any;

    fn var_name_map(&self) -> BTreeMap<u64, String>;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirIRKind {
    // TODO: Implement struct ir and function ir
    Kernel(MimirKernelIR),
}

// TODO: better name for this struct
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MimirIRData {
    pub irs: HashMap<String, MimirIRKind>,
    pub source_hashes: HashMap<String, u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirTyVar {
    pub ty: MimirTy,         // type of the variable
    pub scope: MimirTyScope, // scope of the variable (param or local)
    pub annotation: MimirTyAnnotation,
}

impl MimirTyVar {
    pub fn is_mutable(&self) -> bool {
        matches!(self.annotation, MimirTyAnnotation::Mutable)
    }

    pub fn is_const(&self) -> bool {
        matches!(self.annotation, MimirTyAnnotation::Constant)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirTyAnnotation {
    Mutable,
    Constant,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirTyScope {
    Param,
    Local(u8), // u8: scoping for variable acces (stricter checking)
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirPrimitiveTy {
    Int32,
    Uint32,
    Float32,
    Bool,
}

impl MimirPrimitiveTy {
    #[inline]
    pub fn is_numeric(&self) -> bool {
        !matches!(self, Self::Bool)
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Int32 | Self::Uint32)
    }

    #[inline]
    pub fn is_floating_point(&self) -> bool {
        matches!(self, Self::Float32)
    }

    #[inline]
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }
}

impl MimirPrimitiveTy {
    #[inline]
    pub fn to_mimir_ty(&self) -> MimirTy {
        MimirTy::Primitive(self.clone())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirTy {
    Primitive(MimirPrimitiveTy),
    GlobalArray {
        element_type: MimirPrimitiveTy, // type of the elements in the array, size is unkown b/c it is a global array (i.e. a buffer in Vulkan)
    },
    // length must be inlinable
    SharedMemArray {
        length: Box<MimirConstExpr>,
    }, /*
       TODO: Implement structs and sized local arrays
       LocalArray {
           element_type: Box<MimirTy>, // type of the elements in the array
           size: Option<usize>, // optional size, if known at compile time
       },
       Struct {
           fields: HashMap<String, MimirTy>, // field name to type mapping
       }
       */
}

impl MimirTy {
    pub fn is_shared_mem(&self) -> bool {
        matches!(self, MimirTy::SharedMemArray { .. })
    }

    pub fn is_numeric(&self) -> bool {
        if let MimirTy::Primitive(prim) = self {
            prim.is_numeric()
        } else {
            false
        }
    }

    pub fn is_floating_point(&self) -> bool {
        if let MimirTy::Primitive(prim) = self {
            prim.is_floating_point()
        } else {
            false
        }
    }

    pub fn is_bool(&self) -> bool {
        if let MimirTy::Primitive(prim) = self {
            prim.is_bool()
        } else {
            false
        }
    }

    pub fn is_integer(&self) -> bool {
        if let MimirTy::Primitive(prim) = self {
            matches!(prim, MimirPrimitiveTy::Int32 | MimirPrimitiveTy::Uint32)
        } else {
            false
        }
    }

    pub fn is_global_array(&self) -> bool {
        matches!(self, MimirTy::GlobalArray { .. })
    }
}

pub type MimirBlock = Vec<MimirStmt>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirStmtAssignLeft {
    Index(MimirIndexExpr),
    Var(u64),
}

impl MimirStmtAssignLeft {
    pub fn to_norm_expr(&self) -> MimirExpr {
        match self {
            MimirStmtAssignLeft::Index(mimir_index_expr) => {
                MimirExpr::Index(mimir_index_expr.clone())
            }
            MimirStmtAssignLeft::Var(id) => MimirExpr::Var(*id),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirStmt {
    Assign {
        lhs: MimirStmtAssignLeft,
        rhs: MimirExpr,
    },
    // future For loops may support more complex iteration, but for now we only support simple range-based for loops
    RangeFor {
        var: u64,                // var uuid
        start: MimirExpr,        // start of the range
        end: MimirExpr,          // end of the range
        step: Option<MimirExpr>, // optional step value
        body: Vec<MimirStmt>,    // body of the loop
    },
    // else ifs are handled as an if within the else branch
    If {
        condition: MimirExpr,            // condition expression
        then_branch: MimirBlock,         // then branch
        else_branch: Option<MimirBlock>, // optional else branch
    },
    Return(Option<MimirExpr>), // return value for future standard functions, None for kernels/void return
    Syncthreads,               // barrier synchronization
                               // TODO: implement a way to end lifecycles of variables. i.e. iterators in loops or variabes created in if/else statements.
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirUnOp {
    Not,
    Neg,
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum MimirBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Lt,  // <
    Lte, // <=
    Gt,  // >
    Gte, // >=
    Eq,  // ==
    Ne,  // !=
}

// r[ast.struct.indexexpr]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirIndexExpr {
    pub var: u64,
    pub index: Box<MimirExpr>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirBinOpExpr<T> {
    pub lhs: Box<T>,
    pub op: MimirBinOp,
    pub rhs: Box<T>,
    pub is_parenthesized: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirUnOpExpr<T> {
    pub un_op: MimirUnOp,
    pub expr: Box<T>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MathIntrinsicExpr<T, I>
where
    I: IntoIterator<Item = T>,
{
    pub func: MathIntrinsicFunc,
    pub args: I, // T **MUST** be a Vec<T>
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
pub struct MimirCastExpr<T> {
    pub from: Box<T>,
    pub to: MimirPrimitiveTy, // type to cast to
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirExpr {
    BinOp(MimirBinOpExpr<Self>),
    Index(MimirIndexExpr),
    // TODO: come up with a better name than ts
    BuiltinFieldAccess {
        built_in: MimirBuiltIn,
        field: MimirBuiltInField,
    },
    // imma leave this unimplemented for now, but it is planned to be used for structs
    // Field {
    //     var: u32, // struct variable index
    //     field: String, // field name
    // },
    Unary(MimirUnOpExpr<Self>),
    Literal(MimirLit),
    Var(u64), // variable name; used for referencing variables
    MathIntrinsic(MathIntrinsicExpr<Self, Vec<Self>>),
    Cast(MimirCastExpr<Self>), // i.e. foo as f32
    // ConstGeneric {
    //     index: usize, // Index in the generics
    // },
    ConstExpr(MimirConstExpr),
}

impl MimirExpr {
    // lower = higher precedence
    pub fn precedence(&self) -> u16 {
        match self {
            MimirExpr::BinOp(MimirBinOpExpr {
                op,
                is_parenthesized,
                ..
            }) => {
                if *is_parenthesized {
                    0
                } else if matches!(op, MimirBinOp::Mod | MimirBinOp::Div | MimirBinOp::Mul) {
                    6
                } else if matches!(op, MimirBinOp::Add | MimirBinOp::Sub) {
                    7
                } else if matches!(
                    op,
                    MimirBinOp::Eq
                        | MimirBinOp::Ne
                        | MimirBinOp::Gt
                        | MimirBinOp::Gte
                        | MimirBinOp::Lt
                        | MimirBinOp::Lte
                ) {
                    8
                } else if matches!(op, MimirBinOp::And) {
                    9
                } else if matches!(op, MimirBinOp::Or) {
                    10
                } else {
                    // lowk shouldn't get to this one
                    u16::MAX
                }
            }
            MimirExpr::Index { .. } => 3,
            MimirExpr::BuiltinFieldAccess { .. } => 2,
            MimirExpr::Unary { .. } => 4,
            MimirExpr::Literal(_) => 1,
            MimirExpr::Var(_) => 1,
            MimirExpr::MathIntrinsic { .. } => 3,
            MimirExpr::Cast { .. } => 5,
            MimirExpr::ConstExpr(mimir_const_expr) => mimir_const_expr.precedence(),
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            MimirExpr::BinOp(MimirBinOpExpr { lhs, rhs, .. }) => lhs.is_const() && rhs.is_const(),
            MimirExpr::Index { .. } => false,
            MimirExpr::BuiltinFieldAccess { .. } => false,
            MimirExpr::Unary(MimirUnOpExpr { expr, .. }) => expr.is_const(),
            MimirExpr::Literal(_) => true,
            MimirExpr::Var(_) => false,
            MimirExpr::MathIntrinsic { .. } => false,
            MimirExpr::Cast { .. } => false,
            MimirExpr::ConstExpr(_) => true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirConstExpr {
    BinOp(MimirBinOpExpr<Self>),
    Unary(MimirUnOpExpr<Self>),
    Literal(MimirLit),
    MathIntrinsic(MathIntrinsicExpr<Self, Vec<Self>>),
    Cast(MimirCastExpr<Self>), // i.e. foo as f32
    ConstGeneric {
        index: usize, // Index in the generics
    },
}

impl MimirConstExpr {
    pub fn const_expr_to_ty(&self) -> MimirPrimitiveTy {
        match self {
            MimirConstExpr::BinOp(mimir_bin_op_expr) => mimir_bin_op_expr.lhs.const_expr_to_ty(),
            MimirConstExpr::Unary(mimir_un_op_expr) => mimir_un_op_expr.expr.const_expr_to_ty(),
            MimirConstExpr::Literal(mimir_lit) => mimir_lit.to_ty(),
            MimirConstExpr::MathIntrinsic(math_intrinsic_expr) => math_intrinsic_expr.func.ret_ty(
                math_intrinsic_expr
                    .args
                    .iter()
                    .map(|e| e.const_expr_to_ty())
                    .collect::<Vec<MimirPrimitiveTy>>()
                    .as_ref(),
            ),
            MimirConstExpr::Cast(cast_expr) => cast_expr.to.clone(),
            MimirConstExpr::ConstGeneric { .. } => MimirPrimitiveTy::Uint32,
        }
    }

    pub fn to_norm_expr(&self) -> MimirExpr {
        MimirExpr::ConstExpr(self.clone())
    }
}

impl MimirConstExpr {
    pub fn precedence(&self) -> u16 {
        match self {
            MimirConstExpr::BinOp(MimirBinOpExpr {
                op,
                is_parenthesized,
                ..
            }) => {
                if *is_parenthesized {
                    0
                } else if matches!(op, MimirBinOp::Mod | MimirBinOp::Div | MimirBinOp::Mul) {
                    6
                } else if matches!(op, MimirBinOp::Add | MimirBinOp::Sub) {
                    7
                } else if matches!(
                    op,
                    MimirBinOp::Eq
                        | MimirBinOp::Ne
                        | MimirBinOp::Gt
                        | MimirBinOp::Gte
                        | MimirBinOp::Lt
                        | MimirBinOp::Lte
                ) {
                    8
                } else if matches!(op, MimirBinOp::And) {
                    9
                } else if matches!(op, MimirBinOp::Or) {
                    10
                } else {
                    // lowk shouldn't get to this one
                    u16::MAX
                }
            }
            MimirConstExpr::Unary(_) => 4,
            MimirConstExpr::Literal(_) => 1,
            MimirConstExpr::MathIntrinsic(_) => 3,
            MimirConstExpr::Cast(_) => 5,
            MimirConstExpr::ConstGeneric { .. } => 1,
            // MimirExpr::Unary { .. } => 4,
            // MimirExpr::Literal(_) => 1,
            // MimirExpr::Var(_) => 1,
            // MimirExpr::MathIntrinsic { .. } => 3,
            // MimirExpr::Cast { .. } => 5,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirBuiltIn {
    // CUDA style
    BlockIdx,
    BlockDim,
    ThreadIdx,
    // GLSL style
    GlobalInvocationId,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirBuiltInField {
    X,
    Y,
    Z,
}

impl MimirBuiltInField {
    pub fn to_num(&self) -> u32 {
        match self {
            MimirBuiltInField::X => 0,
            MimirBuiltInField::Y => 1,
            MimirBuiltInField::Z => 2,
        }
    }
}

// these are currently based upon SPIR-V & GLSL math extensions, but is still subject to change
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MathIntrinsicFunc {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Atan2,
    Pow,
    Exp,
    // This computes the natural logarithm not log10, this is following in accordance w/ spir-v and CUDA math libraries
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
    Fma, // thought it'd be useful
}

impl FromStr for MathIntrinsicFunc {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asing" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "asinh" => Self::Asinh,
            "acosh" => Self::Acosh,
            "atanh" => Self::Atanh,
            "atan2" => Self::Atan2,
            "pow" => Self::Pow,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "exp2" => Self::Exp2,
            "log2" => Self::Log2,
            "sqrt" => Self::Sqrt,
            "isqrt" => Self::Isqrt,
            "max" => Self::Max,
            "min" => Self::Min,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "clamp" => Self::Clamp,
            "mix" => Self::Mix,
            "fma" => Self::Fma,
            _ => return Err(format!("Invalid input `{s}`")),
        })
    }
}

impl MathIntrinsicFunc {
    pub fn ret_ty(&self, tys: &[MimirPrimitiveTy]) -> MimirPrimitiveTy {
        assert!(intrinsic_param_validity(self, tys));

        if matches!(
            self,
            MathIntrinsicFunc::Max | MathIntrinsicFunc::Min | MathIntrinsicFunc::Clamp
        ) {
            tys[0].clone()
        } else {
            MimirPrimitiveTy::Float32
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq)]
#[repr(u8)]
pub enum MimirLit {
    Int32(i32),
    // Int64(i64),
    // Planned feature
    // Will require a lot of extension work to be properly implemented in Vulkan rn and I don't want to deal with it yet
    //Float16(u16), // f16::to_bits()
    Float32(u32), // f32::to_bits()
    // Float64(u64), // f64::to_bits()
    Bool(bool),
    Uint32(u32),
}

impl MimirLit {
    pub fn to_ty(&self) -> MimirPrimitiveTy {
        match self {
            MimirLit::Int32(_) => MimirPrimitiveTy::Int32,
            MimirLit::Float32(_) => MimirPrimitiveTy::Float32,
            MimirLit::Bool(_) => MimirPrimitiveTy::Bool,
            MimirLit::Uint32(_) => MimirPrimitiveTy::Uint32,
        }
    }

    pub fn to_u32(&self) -> Option<u32> {
        Some(match self {
            MimirLit::Int32(val) => *val as u32,
            MimirLit::Float32(val) => *val,
            MimirLit::Bool(_) => return None,
            MimirLit::Uint32(val) => *val,
        })
    }

    pub fn to_expr(self) -> MimirExpr {
        MimirExpr::Literal(self)
    }
}

impl MimirBinOp {
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            MimirBinOp::Lt
                | MimirBinOp::Lte
                | MimirBinOp::Gt
                | MimirBinOp::Gte
                | MimirBinOp::Eq
                | MimirBinOp::Ne
        )
    }

    pub fn is_logical(&self) -> bool {
        matches!(self, MimirBinOp::And | MimirBinOp::Or)
    }
}
