use thiserror::Error;

use crate::ir::{MathIntrinsicFunc, MimirBinOp, MimirExpr, MimirLit, MimirPrimitiveTy, MimirTy};

#[derive(Error, Debug)]
pub enum ASTError {
    #[error("AST IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("AST serialization error: {0}")]
    Serialization(String),

    #[error("AST deserialization error: {0}")]
    Deserialization(String),

    #[error("AST validation error: {0}")]
    Validation(String),

    #[error("AST internal error: {0}")]
    Internal(String),

    #[error("Compilation to AST error: {0}")]
    Compiler(String),

    #[error("Inlining error: {0}")]
    Inline(#[from] MimirInlinePassError),

    #[error("Error during multiple IR passes:\n{0}")]
    MultiPass(#[from] MultiPassError),
}

#[derive(Error, Debug, Clone)]
pub enum InvalidCastError {
    #[error("Tried to cast a boolean expression to a numerical value")]
    BoolToNumerical,

    #[error("Tried to cast a numerical expression to a boolean value")]
    NumericalToBool,
}

#[derive(Error, Debug, Clone)]
pub enum PostSafetyPassSpecFailError {
    #[error("CAST ERROR ENCOUNTERED AFTER IR SAFETY & SPEC PASSES: {0}")]
    CastErr(#[from] InvalidCastError),
}

#[derive(Error, Debug, Clone)]
pub enum MimirInlinePassError {
    #[error("Type mismatch, expected `{0:?}` instead got `{1:?}`")]
    TypeMismatch(MimirPrimitiveTy, Box<[MimirLit]>),

    #[error("Generic type mismatch")]
    GenericTyMismatch,

    #[error("The parameters `{0:?}` are invalid for the intrinsic `{1:?}`")]
    InvalidIntrinsicParams(Box<[MimirLit]>, String),

    #[error("Expected `{0}` number of parameters instead got `{1}` when trying to const inline")]
    InvalidNumParams(usize, usize),

    #[error(
        "The following error should be impossible given the IR went through a safety pass:\n {0}"
    )]
    SpecFailErr(#[from] PostSafetyPassSpecFailError),

    #[error("Can't inline a variable access (yet)")]
    AttemptedVarAccess,

    #[error("Can't inline an index access")]
    AttemptedIndex,

    #[error("{0}")]
    Generic(String),
}

#[derive(Error, Debug, Clone)]
pub enum LoopUnrollError {
    #[error("Failed to inline the step expression when unrolling a loop: {0:?}")]
    StepExprInline(MimirInlinePassError),

    #[error("Failed to inline the start expression of a loop: {0:?}")]
    StartExprInline(MimirInlinePassError),

    #[error("Failed to inline the emd expression of a loop: {0:?}")]
    EndExprInline(MimirInlinePassError),
}

#[derive(Error, Debug, Clone)]
pub enum SpecEnforcePassError {
    #[error("Expected two booleans for {0:?} operation, instead recieved a `{1:?}` and a `{2:?}`")]
    InvalidTypesForBooleanBinOp(MimirBinOp, MimirPrimitiveTy, MimirPrimitiveTy),

    #[error("Expected two numeric primitives for `{0:?}` operation, instead recieved a `{1:?}` and a `{2:?}`")]
    InvalidTypesForNumericBinOp(MimirBinOp, MimirPrimitiveTy, MimirPrimitiveTy),

    #[error("Expected two matching numeric primitives for `{0:?}` operation, `{1:?}` ≠ `{2:?}`")]
    NonMatchingTypesForNumericBinOp(MimirBinOp, MimirPrimitiveTy, MimirPrimitiveTy),

    #[error("A var with the ID of `{0}` couldn't be found in the Kernel")]
    CouldNotFindVar(u64),

    #[error("Expected a shared memory array or a global array when indexing, instead indexed into a `{0:?}`")]
    InvalidVarTypeForIndex(MimirTy),

    #[error("Expected the expression in an index to evaluate to be a signed 32-bit integer instead recieved a `{0:?}`")]
    InvalidTypeForIndexExpr(MimirPrimitiveTy),

    #[error(
        "Expected an expression that evaluates to a numeric primitive instead recieved a `{0:?}`"
    )]
    InvalidTypeForNegUnExpr(MimirPrimitiveTy),

    #[error(
        "Expected an expression that evaluates to a boolean primitive instead recieved a `{0:?}`"
    )]
    InvalidTypeForNotUnExpr(MimirPrimitiveTy),

    #[error(
        "Expected {0} parameters for the math intrinsic `{1:?}`, instead recieved {2} parameters"
    )]
    InvalidNumParamsForMathIntrinsic(usize, MathIntrinsicFunc, usize),

    #[error("Recieved invalid types for the math intrinsic `{0:?}`")]
    InvalidParamTypesForMathIntrinsic(MathIntrinsicFunc),

    #[error("Tried to cast from a bool to a `{0:?}`")]
    InvalidCastFromBoolToNonBool(MimirPrimitiveTy),

    #[error("Tried to cast from a `{0:?}` to a bool")]
    InvalidCastFromNonBoolToBool(MimirPrimitiveTy),

    #[error(
        "Const generics in IR are represented as indexes to a runtime submitted list of const generic values. A const generic of `{0}` was found when the largest is `{1}`"
    )]
    OutOfBoundsConstGeneric(usize, usize),

    #[error("Expected matching types for the left hand and right hand side of an assignment statment. Instead encountered trying to assign a `{1:?}` to a `{0:?}`.")]
    MismatchedTypesInAssignStmt(MimirPrimitiveTy, MimirPrimitiveTy),

    #[error("Constant variable cannot be reassigned to")]
    NonMutableLHSReassignment,

    #[error("Expected a left hand side expression that evaluates to a primitive, instead found a `{0:?}`")]
    NonPrimitiveAssignmentLhs(MimirTy),

    #[error("Variables used in an range-based loop must be an Int32 not a `{0:?}`")]
    Non32BitIntPrimInRangeFor(MimirPrimitiveTy),

    #[error("Expected a variable that holds a primitive Int32 type for an iterator in a range for loop, instead found a `{0:?}` which isn't even a primitive type.")]
    NonPrimVarInRangeFor(MimirTy),

    #[error("Expected the variable for an iterator in a range for loop to be a constant, instead it was labeled as mutable")]
    NonConstantVarInRangeFor,

    #[error("Step expression in a iterative for loop evaluates to a `{0:?}`, must evaluate to be an Int32")]
    InvalidStepExpressionTy(MimirPrimitiveTy),

    #[error(
        "The condition in an if statement must evaluate to a boolean. Instead found a `{0:?}`"
    )]
    NonBooleanConditionalExprIf(MimirPrimitiveTy),

    #[error("Expected no return value when returning from a kernel instead found `{0:?}`")]
    InvalidValuedErrorFromKernel(MimirExpr),
}

#[derive(Error, Debug, Clone)]
pub enum MimirTypePassError {
    #[error("Tried calling a type pass for a pass that hasn't implemented it")]
    Unimplemented,
}

#[derive(Error, Debug, Clone)]
pub enum MultiPassError {
    #[error("Failed to inline: {0}")]
    Inline(MimirInlinePassError),

    #[error("Mimir specification was not met:\n{0}")]
    SpecEnforce(SpecEnforcePassError),

    #[error("Failed during the loop unroll pass: {0}")]
    LoopUnroll(LoopUnrollError),

    #[error("Failed during type pass: {0}")]
    Type(MimirTypePassError),
}

#[macro_export]
macro_rules! compiler_err {
    ($($arg:tt)*) => {
        Err(ASTError::Compiler(format!($($arg)*)))
    };
}
