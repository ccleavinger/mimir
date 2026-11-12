use crate::{Expr, Span};
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RichExpr<T: Expr> {
    pub expr: T,
    pub is_parenthesized: bool,
    pub span: Span,
}

impl<T: Expr> RichExpr<T> {
    pub fn new(expr: T, span: Span) -> Self {
        Self {
            expr,
            span,
            is_parenthesized: false,
        }
    }

    pub fn parenthesized(self) -> Self {
        self.set_p(true)
    }

    pub fn set_p(mut self, is: bool) -> Self {
        self.is_parenthesized = is;
        self
    }
}

impl<T: Expr> Expr for RichExpr<T> {
    fn priority(&self) -> u32 {
        if self.is_parenthesized {
            0
        } else {
            self.expr.priority()
        }
    }
}
