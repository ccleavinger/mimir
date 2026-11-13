//! Rust-oriented kernel AST. Lexical declarations own local scope; validation
//! checks names, types and mutability before subsequent compiler stages.
mod expr;
mod kernel;
mod op;
pub mod passes;
mod to_ir;
mod ty;
mod util;
mod valid;

pub use expr::*;
pub use kernel::*;
pub use op::*;
pub use passes::kernel::KernelAstPass;
pub use ty::*;
pub use util::LocalId;
pub use valid::{ValidatedKernel, ValidationError};

pub use to_ir::LoweringError;

pub use to_ir::lower_function;
pub use valid::{MimirFunctionAst, ValidatedFunction};
