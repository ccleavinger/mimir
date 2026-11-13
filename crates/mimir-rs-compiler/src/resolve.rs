//! Lexical name resolution and literal typing, separate from grammar parsing.
use crate::syntax::{self, Expression as E, Statement as S};
use mimir_ast::*;
use std::collections::BTreeMap;
use syn::{Ident, Lit};
type Result<T> = syn::Result<T>;
pub(crate) struct Resolver {
    pub(crate) names: BTreeMap<String, LocalId>,
    pub(crate) generics: BTreeMap<String, ConstGenericId>,
    pub(crate) next: u32,
    pub(crate) types: BTreeMap<LocalId, PrimitiveTy>,
    pub(crate) functions: BTreeMap<String, (u32, Vec<KernelParamTy>, PrimitiveTy)>,
    pub(crate) source: crate::syntax::Parsed,
}

impl Resolver {
    pub(crate) fn bind(&mut self, n: &Ident) -> Binding {
        let id = LocalId(self.next);
        self.next += 1;
        self.names.insert(n.to_string(), id);
        Binding {
            id,
            name: n.to_string(),
            span: n.span().byte_range(),
        }
    }

    fn type_of(&self, e: &RichMimirExpr) -> PrimitiveTy {
        match &e.expr {
            MimirExpr::Literal(l) => l.ty(),
            MimirExpr::Var(id) => self.types[id],
            MimirExpr::Index { array, .. } => self.types[array],
            MimirExpr::Cast { to, .. } => *to,
            MimirExpr::BinOp(b) => {
                if matches!(
                    b.op,
                    MimirBinOp::Eq
                        | MimirBinOp::Ne
                        | MimirBinOp::Lt
                        | MimirBinOp::Lte
                        | MimirBinOp::Gt
                        | MimirBinOp::Gte
                        | MimirBinOp::And
                        | MimirBinOp::Or
                ) {
                    PrimitiveTy::Bool
                } else {
                    self.type_of(&b.lhs)
                }
            }
            MimirExpr::Unary(u) => self.type_of(&u.expr),
            MimirExpr::Call { returns, .. } => *returns,
            MimirExpr::MathIntrinsic(i) => match i {
                Intrinsic::Unary(_, e) => self.type_of(e),
                Intrinsic::Binary(_, a) => self.type_of(&a[0]),
                Intrinsic::Ternary(_, a) => self.type_of(&a[0]),
            },
            _ => PrimitiveTy::Uint32,
        }
    }

    fn coerce(e: &mut RichMimirExpr, t: PrimitiveTy) {
        if let MimirExpr::Literal(Literal::Int32(v)) = e.expr
            && t == PrimitiveTy::Uint32
            && v >= 0
        {
            e.expr = MimirExpr::Literal(Literal::Uint32(v as u32));
        }
        if let MimirExpr::Literal(Literal::Uint32(v)) = e.expr
            && t == PrimitiveTy::Int32
        {
            e.expr = MimirExpr::Literal(Literal::Int32(v as i32));
        }
    }

    fn primitive(name: &str, span: proc_macro2::Span) -> Result<PrimitiveTy> {
        match name {
            "u32" => Ok(PrimitiveTy::Uint32),
            "i32" => Ok(PrimitiveTy::Int32),
            "f32" => Ok(PrimitiveTy::Float32),
            "bool" => Ok(PrimitiveTy::Bool),
            _ => Err(syn::Error::new(span, "expected u32, i32, f32 or bool")),
        }
    }
    fn error(&self, span: &std::ops::Range<usize>, message: &str) -> syn::Error {
        syn::Error::new(self.source.source_span(span), message)
    }
    fn named(&mut self, name: &str, span: &std::ops::Range<usize>) -> Binding {
        self.bind(&if let Some(name) = name.strip_prefix("r#") {
            Ident::new_raw(name, self.source.source_span(span))
        } else {
            Ident::new(name, self.source.source_span(span))
        })
    }
    fn local(&self, e: &syntax::Expr) -> Result<LocalId> {
        match &e.kind {
            E::Path(p) if p.len() == 1 => self
                .names
                .get(&p[0])
                .copied()
                .ok_or_else(|| self.error(&e.span, "unknown local")),
            E::Paren(e) => self.local(e),
            _ => Err(self.error(&e.span, "expected local variable")),
        }
    }
    fn binary(op: &str) -> MimirBinOp {
        match op {
            "+" => MimirBinOp::Add,
            "-" => MimirBinOp::Sub,
            "*" => MimirBinOp::Mul,
            "/" => MimirBinOp::Div,
            "%" => MimirBinOp::Mod,
            "==" => MimirBinOp::Eq,
            "!=" => MimirBinOp::Ne,
            "<" => MimirBinOp::Lt,
            "<=" => MimirBinOp::Lte,
            ">" => MimirBinOp::Gt,
            ">=" => MimirBinOp::Gte,
            "&&" => MimirBinOp::And,
            "||" => MimirBinOp::Or,
            _ => unreachable!("grammar operator"),
        }
    }
    fn expr(&mut self, e: &syntax::Expr) -> Result<RichMimirExpr> {
        let span = self.source.source_span(&e.span);
        let source = span.byte_range();
        let kind = match &e.kind {
            E::Literal(text) => {
                let lit: syn::Lit = syn::parse_str(text).map_err(|e| syn::Error::new(span, e))?;
                MimirExpr::Literal(match lit {
                    Lit::Int(v) => match v.suffix() {
                        "i32" | "" => {
                            Literal::Int32(v.base10_parse().map_err(|e| syn::Error::new(span, e))?)
                        }
                        "u32" => {
                            Literal::Uint32(v.base10_parse().map_err(|e| syn::Error::new(span, e))?)
                        }
                        "f32" => {
                            Literal::float(v.base10_parse().map_err(|e| syn::Error::new(span, e))?)
                        }
                        _ => return Err(self.error(&e.span, "unsupported integer type")),
                    },
                    Lit::Bool(v) => Literal::Bool(v.value),
                    Lit::Float(v) => {
                        Literal::float(v.base10_parse().map_err(|e| syn::Error::new(span, e))?)
                    }
                    _ => return Err(self.error(&e.span, "unsupported literal")),
                })
            }
            E::Path(p) => match p.as_slice() {
                [name] if name == "true" || name == "false" => {
                    MimirExpr::Literal(Literal::Bool(name == "true"))
                }
                [name] => {
                    if let Some(id) = self.generics.get(name) {
                        MimirExpr::ConstExpr(RichExpr::new(ConstExpr::Generic(*id), source.clone()))
                    } else {
                        MimirExpr::Var(
                            *self
                                .names
                                .get(name)
                                .ok_or_else(|| self.error(&e.span, "unknown local"))?,
                        )
                    }
                }
                [ty, name] if ty == "f32" => {
                    MimirExpr::Literal(Literal::float(match name.as_str() {
                        "MIN" => f32::MIN,
                        "MAX" => f32::MAX,
                        "INFINITY" => f32::INFINITY,
                        "NEG_INFINITY" => f32::NEG_INFINITY,
                        _ => return Err(self.error(&e.span, "unknown f32 constant")),
                    }))
                }
                _ => return Err(self.error(&e.span, "expected local name or f32 constant")),
            },
            E::Paren(value) => return Ok(self.expr(value)?.parenthesized()),
            E::Unary(op, value) => MimirExpr::Unary(MimirUnaryExpr::new(
                if *op == '-' {
                    MimirUnOp::Neg
                } else {
                    MimirUnOp::Not
                },
                self.expr(value)?,
            )),
            E::Binary(op, lhs, rhs) => {
                let mut lhs = self.expr(lhs)?;
                let mut rhs = self.expr(rhs)?;
                Self::coerce(&mut lhs, self.type_of(&rhs));
                Self::coerce(&mut rhs, self.type_of(&lhs));
                MimirExpr::BinOp(MimirBinOpExpr::new(lhs, Self::binary(op), rhs))
            }
            E::Cast(value, to) => MimirExpr::Cast {
                value: Box::new(self.expr(value)?),
                to: Self::primitive(to, span)?,
            },
            E::Index(array, index) => MimirExpr::Index {
                array: self.local(array)?,
                index: Box::new(self.expr(index)?),
            },
            E::Field(base, field) => {
                let E::Path(path) = &base.kind else {
                    return Err(self.error(&e.span, "expected builtin"));
                };
                let [name] = path.as_slice() else {
                    return Err(self.error(&e.span, "expected builtin"));
                };
                let builtin = match name.as_str() {
                    "global_id" => Builtin::GlobalInvocationId,
                    "thread_idx" => Builtin::ThreadIdx,
                    "block_idx" => Builtin::BlockIdx,
                    "block_dim" => Builtin::BlockDim,
                    _ => return Err(self.error(&base.span, "unknown builtin")),
                };
                let field = match field.as_str() {
                    "x" => Axis::X,
                    "y" => Axis::Y,
                    "z" => Axis::Z,
                    _ => return Err(self.error(&e.span, "expected x, y or z")),
                };
                MimirExpr::Builtin { builtin, field }
            }
            E::Call(callee, args) => {
                let E::Path(path) = &callee.kind else {
                    return Err(self.error(&callee.span, "expected device function name"));
                };
                let [name] = path.as_slice() else {
                    return Err(
                        self.error(&callee.span, "device calls must use module-local names")
                    );
                };
                if let Some(intrinsic) = self.intrinsic(name, args, span)? {
                    MimirExpr::MathIntrinsic(intrinsic)
                } else {
                    let (function, parameters, returns) =
                        self.functions.get(name).cloned().ok_or_else(|| {
                            self.error(&callee.span, "unresolved device function")
                        })?;
                    let mut arguments = args
                        .iter()
                        .map(|a| self.expr(a))
                        .collect::<Result<Vec<_>>>()?;
                    for (arg, param) in arguments.iter_mut().zip(&parameters) {
                        if let KernelParamTy::Scalar(t) = param {
                            Self::coerce(arg, *t)
                        }
                    }
                    MimirExpr::Call {
                        function,
                        parameters,
                        returns,
                        arguments,
                    }
                }
            }
            _ => {
                return Err(self.error(
                    &e.span,
                    "expression is only valid in a range or shared declaration",
                ));
            }
        };
        Ok(RichExpr::new(kind, source))
    }
    fn constant(&self, e: RichMimirExpr) -> Result<RichConstExpr> {
        let expr = match e.expr {
            MimirExpr::ConstExpr(c) => return Ok(c),
            MimirExpr::Literal(l) => ConstExpr::Literal(l),
            MimirExpr::BinOp(b) => ConstExpr::BinOp(MimirBinOpExpr::new(
                self.constant(*b.lhs)?,
                b.op,
                self.constant(*b.rhs)?,
            )),
            _ => {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "shared length must be a tuning expression",
                ));
            }
        };
        Ok(RichExpr::new(expr, e.span))
    }
    fn intrinsic(
        &mut self,
        name: &str,
        args: &[crate::syntax::Expr],
        span: proc_macro2::Span,
    ) -> Result<Option<Intrinsic<RichMimirExpr>>> {
        let arity = match name {
            "min" | "max" | "pow" | "atan2" => 2,
            "clamp" | "mix" | "fma" => 3,
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
            | "asinh" | "acosh" | "atanh" | "exp" | "log" | "exp2" | "log2" | "sqrt" | "floor"
            | "ceil" => 1,
            _ => return Ok(None),
        };
        if args.len() != arity {
            return Err(syn::Error::new(span, "intrinsic argument count mismatch"));
        }
        let mut a = args
            .iter()
            .map(|e| self.expr(e))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(match arity {
            1 => Intrinsic::Unary(
                match name {
                    "sin" => UnaryIntrinsic::Sin,
                    "cos" => UnaryIntrinsic::Cos,
                    "tan" => UnaryIntrinsic::Tan,
                    "asin" => UnaryIntrinsic::Asin,
                    "acos" => UnaryIntrinsic::Acos,
                    "atan" => UnaryIntrinsic::Atan,
                    "sinh" => UnaryIntrinsic::Sinh,
                    "cosh" => UnaryIntrinsic::Cosh,
                    "tanh" => UnaryIntrinsic::Tanh,
                    "asinh" => UnaryIntrinsic::Asinh,
                    "acosh" => UnaryIntrinsic::Acosh,
                    "atanh" => UnaryIntrinsic::Atanh,
                    "exp" => UnaryIntrinsic::Exp,
                    "log" => UnaryIntrinsic::Log,
                    "exp2" => UnaryIntrinsic::Exp2,
                    "log2" => UnaryIntrinsic::Log2,
                    "sqrt" => UnaryIntrinsic::Sqrt,
                    "floor" => UnaryIntrinsic::Floor,
                    "ceil" => UnaryIntrinsic::Ceil,
                    _ => unreachable!(),
                },
                Box::new(a.remove(0)),
            ),
            2 => Intrinsic::Binary(
                match name {
                    "min" => BinaryIntrinsic::Min,
                    "max" => BinaryIntrinsic::Max,
                    "pow" => BinaryIntrinsic::Pow,
                    _ => BinaryIntrinsic::Atan2,
                },
                Box::new([a.remove(0), a.remove(0)]),
            ),
            _ => Intrinsic::Ternary(
                match name {
                    "clamp" => TernaryIntrinsic::Clamp,
                    "mix" => TernaryIntrinsic::Mix,
                    _ => TernaryIntrinsic::Fma,
                },
                Box::new([a.remove(0), a.remove(0), a.remove(0)]),
            ),
        }))
    }

    pub(crate) fn block(&mut self, block: &syntax::Block) -> Result<MimirKernelBlock> {
        let saved = self.names.clone();
        let result = self.block_inner(block);
        self.names = saved;
        result
    }
    fn block_inner(&mut self, block: &syntax::Block) -> Result<MimirKernelBlock> {
        let mut out = vec![];
        for s in block {
            let statement = match &s.kind {
                S::Let {
                    shared,
                    mutable,
                    name,
                    ty,
                    value,
                } => {
                    if *shared {
                        let E::Repeat(initial, length) = &value.kind else {
                            return Err(self.error(
                                &value.span,
                                "shared allocation requires [literal; length]",
                            ));
                        };
                        let initial = self.expr(initial)?;
                        let element = self.type_of(&initial);
                        let length = self.expr(length)?;
                        let length = self.constant(length)?;
                        let binding = self.named(name, &s.span);
                        self.types.insert(binding.id, element);
                        MimirKernelStmt::SharedMem {
                            binding,
                            element,
                            length,
                        }
                    } else {
                        let mut value = self.expr(value)?;
                        let ty = if let Some(ty) = ty {
                            Self::primitive(ty, self.source.source_span(&s.span))?
                        } else {
                            self.type_of(&value)
                        };
                        Self::coerce(&mut value, ty);
                        let binding = self.named(name, &s.span);
                        self.types.insert(binding.id, ty);
                        MimirKernelStmt::Let {
                            binding,
                            ty,
                            mutability: if *mutable {
                                Mutability::Mutable
                            } else {
                                Mutability::Constant
                            },
                            value,
                        }
                    }
                }
                S::Assign { target, op, value } => {
                    let assignment = if let E::Index(array, index) = &target.kind {
                        AssignTarget::Index {
                            array: self.local(array)?,
                            index: self.expr(index)?,
                        }
                    } else {
                        AssignTarget::Var(self.local(target)?)
                    };
                    let mut value = self.expr(value)?;
                    let lhs = self.expr(target)?;
                    Self::coerce(&mut value, self.type_of(&lhs));
                    if op != "=" {
                        value = RichExpr::new(
                            MimirExpr::BinOp(MimirBinOpExpr::new(
                                lhs,
                                Self::binary(&op[..1]),
                                value,
                            )),
                            self.source.source_span(&s.span).byte_range(),
                        )
                    }
                    MimirKernelStmt::Assign {
                        target: assignment,
                        value,
                    }
                }
                S::Return(value) => {
                    if let Some(value) = value {
                        MimirKernelStmt::ReturnValue(self.expr(value)?)
                    } else {
                        MimirKernelStmt::Return
                    }
                }
                S::Block(body) => MimirKernelStmt::Block(self.block(body)?),
                S::If { condition, yes, no } => MimirKernelStmt::If {
                    condition: self.expr(condition)?,
                    then_branch: self.block(yes)?,
                    else_branch: no.as_ref().map(|b| self.block(b)).transpose()?,
                },
                S::For { name, range, body } => {
                    let mut range = range;
                    let mut step = None;
                    if let E::Call(callee, args) = &range.kind
                        && let E::Field(receiver, method) = &callee.kind
                        && method == "step_by"
                        && args.len() == 1
                    {
                        step = Some(self.expr(&args[0])?);
                        range = receiver;
                    }
                    while let E::Paren(inner) = &range.kind {
                        range = inner
                    }
                    let E::Range(start, end) = &range.kind else {
                        return Err(self.error(
                            &range.span,
                            "expected exclusive range, optionally with step_by",
                        ));
                    };
                    let mut start = self.expr(start)?;
                    let end = self.expr(end)?;
                    Self::coerce(&mut start, self.type_of(&end));
                    let saved = self.names.clone();
                    let binding = self.named(name, &s.span);
                    self.types.insert(binding.id, self.type_of(&end));
                    let body = self.block(body);
                    self.names = saved;
                    MimirKernelStmt::RangeFor {
                        binding,
                        start,
                        end,
                        step,
                        body: body?,
                    }
                }
                S::Expression(e) => match &e.kind {
                    E::Call(callee, args)
                        if args.is_empty()
                            && matches!(&callee.kind,E::Path(p) if p.len()==1&&(p[0]=="syncthreads"||p[0]=="__syncthreads")) =>
                    {
                        MimirKernelStmt::Syncthreads
                    }
                    _ => return Err(self.error(&e.span, "unsupported kernel statement")),
                },
            };
            out.push(RichKernelStmt::new(
                statement,
                self.source.source_span(&s.span).byte_range(),
            ));
        }
        Ok(out)
    }
}
