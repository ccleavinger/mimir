use crate::passes::kernel::safety::ValidityPass;
use crate::{KernelAstPass, MimirKernelAst, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub span: Span,
    pub message: String,
}

/// Immutable proof wrapper. Editing an AST requires validating it again.
#[derive(Debug, Clone)]
pub struct ValidatedKernel(MimirKernelAst);
impl ValidatedKernel {
    pub(crate) fn from_function(ast: MimirKernelAst) -> Self {
        Self(ast)
    }
    pub fn ast(&self) -> &MimirKernelAst {
        &self.0
    }
    pub fn into_ast(self) -> MimirKernelAst {
        self.0
    }
}

impl MimirKernelAst {
    pub fn validate(mut self) -> Result<ValidatedKernel, ValidationError> {
        ValidityPass::default().kernel(&mut self)?;
        Ok(ValidatedKernel(self))
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {:?}", self.message, self.span)
    }
}
impl std::error::Error for ValidationError {}

/// A reusable lexical device function AST, with an explicit scalar result.
#[derive(Debug, Clone)]
pub struct MimirFunctionAst {
    pub routine: MimirKernelAst,
    pub returns: crate::PrimitiveTy,
}
#[derive(Debug, Clone)]
pub struct ValidatedFunction(MimirFunctionAst);
impl MimirFunctionAst {
    pub fn validate(mut self) -> Result<ValidatedFunction, ValidationError> {
        ValidityPass::default().function(&mut self.routine, self.returns)?;
        Ok(ValidatedFunction(self))
    }
}
impl ValidatedFunction {
    pub fn ast(&self) -> &MimirFunctionAst {
        &self.0
    }
    pub fn into_ast(self) -> MimirFunctionAst {
        self.0
    }
    /// Calls are resolved when the containing IR module is validated.
    pub fn to_ir(&self, id: u32) -> Result<mimir_ir::flat::DeviceFunction, crate::LoweringError> {
        crate::lower_function(self.0.routine.clone(), id, self.0.returns)
    }
}
