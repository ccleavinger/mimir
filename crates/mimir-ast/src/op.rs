#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum MimirUnOp {
    Not,
    Neg,
}

impl MimirBinOp {
    pub fn priority(&self) -> u32 {
        match self {
            Self::Mul | Self::Div | Self::Mod => 1 << 7,
            Self::Add | Self::Sub => 1 << 8,
            Self::Lt | Self::Lte | Self::Gt | Self::Gte | Self::Eq | Self::Ne => 1 << 13,
            Self::And => 1 << 14,
            Self::Or => 1 << 15,
        }
    }
}
