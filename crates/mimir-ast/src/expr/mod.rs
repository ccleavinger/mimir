use std::ops::Range;
mod const_expr;
mod expression;
mod ops;
mod rich;
pub use const_expr::*;
pub use expression::*;
pub use ops::*;
pub use rich::*;
/// Rust expression precedence: smaller values bind more tightly.
pub trait Expr {
    fn priority(&self) -> u32;
}
pub type Span = Range<usize>;
