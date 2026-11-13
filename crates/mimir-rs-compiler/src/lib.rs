//! Rust syntax frontend for the reusable Mimir AST.
pub mod frontend;

mod resolve;
mod syntax;

mod source;
pub use source::SourceFunction;
