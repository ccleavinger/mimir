use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

macro_rules! ids {
    ($($name:ident),*) => {$ (
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub u32);
    )*};
}

ids!(
    ParameterId,
    ConstantId,
    ValueId,
    StorageId,
    BlockId,
    FunctionId
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarType {
    Bool,
    I32,
    U32,
    F32,
}

impl ScalarType {
    pub fn is_numeric(self) -> bool {
        self != Self::Bool
    }
    pub fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::U32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AddressSpace {
    Global,
    Local,
    Workgroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub fn can_read(self) -> bool {
        self != Self::Write
    }
    pub fn can_write(self) -> bool {
        self != Self::Read
    }
    pub fn permits(self, required: Self) -> bool {
        (!required.can_read() || self.can_read()) && (!required.can_write() || self.can_write())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Scalar(ScalarType),
    Buffer {
        element: ScalarType,
        space: AddressSpace,
        access: Access,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literal {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(u32),
}
impl Literal {
    pub fn ty(self) -> ScalarType {
        match self {
            Self::Bool(_) => ScalarType::Bool,
            Self::I32(_) => ScalarType::I32,
            Self::U32(_) => ScalarType::U32,
            Self::F32(_) => ScalarType::F32,
        }
    }
    pub fn f32(value: f32) -> Self {
        Self::F32(value.to_bits())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstantOperand {
    Literal(u32),
    Constant(ConstantId),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstantOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedConstant {
    pub id: ConstantId,
    pub op: ConstantOp,
    pub lhs: ConstantOperand,
    pub rhs: ConstantOperand,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parameter {
    pub id: ParameterId,
    pub ty: Type,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Variable {
    Value {
        id: ValueId,
        ty: Type,
    },
    Storage {
        id: StorageId,
        element: ScalarType,
        space: AddressSpace,
        length: ConstantOperand,
    },
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub parameters: Vec<Parameter>,
    /// Positional u32 tuning inputs, distinct from invocation parameters.
    pub constants: Vec<ConstantId>,
    pub derived_constants: Vec<DerivedConstant>,
    pub variables: Vec<Variable>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operand {
    Parameter(ParameterId),
    Constant(ConstantId),
    Value(ValueId),
    Storage(StorageId),
    Literal(Literal),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Builtin {
    GlobalId,
    WorkgroupId,
    LocalId,
    WorkgroupSize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    X,
    Y,
    Z,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryIntrinsic {
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
    Exp,
    Log,
    Exp2,
    Log2,
    Sqrt,
    InverseSqrt,
    Floor,
    Ceil,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryIntrinsic {
    Atan2,
    Pow,
    Min,
    Max,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryIntrinsic {
    Clamp,
    Mix,
    Fma,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueOp {
    Copy(Operand),
    Binary {
        op: BinaryOp,
        lhs: Operand,
        rhs: Operand,
    },
    Unary {
        op: UnaryOp,
        value: Operand,
    },
    Cast {
        value: Operand,
        to: ScalarType,
    },
    Builtin {
        builtin: Builtin,
        axis: Axis,
    },
    Load {
        buffer: Operand,
        index: Operand,
    },
    UnaryIntrinsic {
        op: UnaryIntrinsic,
        value: Operand,
    },
    BinaryIntrinsic {
        op: BinaryIntrinsic,
        args: [Operand; 2],
    },
    TernaryIntrinsic {
        op: TernaryIntrinsic,
        args: [Operand; 3],
    },
}
impl ValueOp {
    fn operands(&self) -> Vec<Operand> {
        match self {
            Self::Copy(a)
            | Self::Unary { value: a, .. }
            | Self::Cast { value: a, .. }
            | Self::UnaryIntrinsic { value: a, .. } => vec![*a],
            Self::Binary { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Load { buffer, index } => vec![*buffer, *index],
            Self::BinaryIntrinsic { args, .. } => args.to_vec(),
            Self::TernaryIntrinsic { args, .. } => args.to_vec(),
            Self::Builtin { .. } => vec![],
        }
    }
    pub(crate) fn operands_mut(&mut self) -> Vec<&mut Operand> {
        match self {
            Self::Copy(a)
            | Self::Unary { value: a, .. }
            | Self::Cast { value: a, .. }
            | Self::UnaryIntrinsic { value: a, .. } => vec![a],
            Self::Binary { lhs, rhs, .. } => vec![lhs, rhs],
            Self::Load { buffer, index } => vec![buffer, index],
            Self::BinaryIntrinsic { args, .. } => args.iter_mut().collect(),
            Self::TernaryIntrinsic { args, .. } => args.iter_mut().collect(),
            Self::Builtin { .. } => vec![],
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Instruction {
    Value {
        result: ValueId,
        op: ValueOp,
    },
    Store {
        buffer: Operand,
        index: Operand,
        value: Operand,
    },
    /// Workgroup execution barrier with workgroup-memory visibility. Must be
    /// encountered uniformly; type/CFG validation does not prove uniformity.
    Barrier,
    Call {
        result: Option<ValueId>,
        function: FunctionId,
        arguments: Vec<Operand>,
        constants: Vec<ConstantOperand>,
    },
}
impl Instruction {
    pub(crate) fn result(&self) -> Option<ValueId> {
        match self {
            Self::Value { result, .. } => Some(*result),
            Self::Call { result, .. } => *result,
            _ => None,
        }
    }
    pub(crate) fn operands_mut(&mut self) -> Vec<&mut Operand> {
        match self {
            Self::Value { op, .. } => op.operands_mut(),
            Self::Store {
                buffer,
                index,
                value,
            } => vec![buffer, index, value],
            Self::Call { arguments, .. } => arguments.iter_mut().collect(),
            Self::Barrier => vec![],
        }
    }
    pub(crate) fn operands(&self) -> Vec<Operand> {
        match self {
            Self::Value { op, .. } => op.operands(),
            Self::Store {
                buffer,
                index,
                value,
            } => vec![*buffer, *index, *value],
            Self::Call { arguments, .. } => arguments.clone(),
            Self::Barrier => vec![],
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub target: BlockId,
    pub arguments: Vec<Operand>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminator {
    Branch(Edge),
    Conditional {
        condition: Operand,
        then_edge: Edge,
        else_edge: Edge,
    },
    Return(Option<Operand>),
    /// Structural-only merge/continue placeholder with no executable incoming
    /// path. Validation rejects reachable uses or nonempty placeholder blocks.
    Unreachable,
}
impl Terminator {
    pub(crate) fn edges(&self) -> Vec<&Edge> {
        match self {
            Self::Branch(e) => vec![e],
            Self::Conditional {
                then_edge,
                else_edge,
                ..
            } => vec![then_edge, else_edge],
            _ => vec![],
        }
    }
    pub(crate) fn edges_mut(&mut self) -> Vec<&mut Edge> {
        match self {
            Self::Branch(e) => vec![e],
            Self::Conditional {
                then_edge,
                else_edge,
                ..
            } => vec![then_edge, else_edge],
            _ => vec![],
        }
    }
    pub(crate) fn operands(&self) -> Vec<Operand> {
        let mut result = match self {
            Self::Conditional { condition, .. } => vec![*condition],
            Self::Return(Some(v)) => vec![*v],
            _ => vec![],
        };
        for e in self.edges() {
            result.extend(&e.arguments);
        }
        result
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Merge {
    Selection { merge: BlockId },
    Loop { merge: BlockId, continuing: BlockId },
}
impl Merge {
    pub(crate) fn target(self) -> BlockId {
        match self {
            Self::Selection { merge } | Self::Loop { merge, .. } => merge,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub arguments: Vec<ValueId>,
    pub instructions: Vec<Instruction>,
    pub merge: Option<Merge>,
    pub terminator: Terminator,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Body {
    pub entry: BlockId,
    pub blocks: Vec<Block>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Routine {
    pub header: Header,
    pub body: Body,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugInfo {
    pub name: String,
    pub value_names: BTreeMap<ValueId, String>,
    pub parameter_names: BTreeMap<ParameterId, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kernel {
    pub routine: Routine,
    pub debug: DebugInfo,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFunction {
    pub id: FunctionId,
    pub routine: Routine,
    pub returns: Option<ScalarType>,
    pub debug: DebugInfo,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Module {
    pub kernels: Vec<Kernel>,
    pub functions: Vec<DeviceFunction>,
}
