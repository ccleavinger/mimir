#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveTy {
    Int32,
    Uint32,
    Float32,
    Bool,
}

impl PrimitiveTy {
    pub fn is_numeric(self) -> bool {
        self != Self::Bool
    }

    pub fn is_integer(self) -> bool {
        matches!(self, Self::Int32 | Self::Uint32)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::Float32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Literal {
    Int32(i32),
    Uint32(u32),
    Float32(u32),
    Bool(bool),
}

impl Literal {
    #[inline]
    pub fn ty(self) -> PrimitiveTy {
        match self {
            Self::Int32(_) => PrimitiveTy::Int32,
            Self::Uint32(_) => PrimitiveTy::Uint32,
            Self::Float32(_) => PrimitiveTy::Float32,
            Self::Bool(_) => PrimitiveTy::Bool,
        }
    }

    #[inline]
    pub fn float(value: f32) -> Self {
        Self::Float32(value.to_bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Mutable,
    Constant,
}
