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
