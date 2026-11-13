//! Rust syntax frontend. Produces reusable AST rather than backend instructions.
use mimir_ast::*;
use std::collections::BTreeMap;
use syn::{spanned::Spanned, *};

type Result<T> = std::result::Result<T, syn::Error>;
struct Parser {
    names: BTreeMap<String, LocalId>,
    generics: BTreeMap<String, ConstGenericId>,
    next: u32,
    types: BTreeMap<LocalId, PrimitiveTy>,
    functions: BTreeMap<String, (u32, Vec<KernelParamTy>, PrimitiveTy)>,
}

fn ty(t: &Type) -> Result<PrimitiveTy> {
    if let Type::Path(t) = t {
        match t.path.get_ident().map(|x| x.to_string()).as_deref() {
            Some("u32") => return Ok(PrimitiveTy::Uint32),
            Some("i32") => return Ok(PrimitiveTy::Int32),
            Some("f32") => return Ok(PrimitiveTy::Float32),
            Some("bool") => return Ok(PrimitiveTy::Bool),
            _ => {}
        }
    }
    Err(syn::Error::new(t.span(), "expected u32, i32, f32 or bool"))
}

impl Parser {
    fn bind(&mut self, n: &Ident) -> Binding {
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
        if let MimirExpr::Literal(Literal::Int32(v)) = e.expr {
            if t == PrimitiveTy::Uint32 && v >= 0 {
                e.expr = MimirExpr::Literal(Literal::Uint32(v as u32));
            }
        }
        if let MimirExpr::Literal(Literal::Uint32(v)) = e.expr {
            if t == PrimitiveTy::Int32 {
                e.expr = MimirExpr::Literal(Literal::Int32(v as i32));
            }
        }
    }

    fn expr(&mut self, e: &syn::Expr) -> Result<RichMimirExpr> {
        let x = match e {
            syn::Expr::Lit(l) => MimirExpr::Literal(match &l.lit {
                Lit::Bool(v) => Literal::Bool(v.value),
                Lit::Int(v) => match v.suffix() {
                    "i32" => Literal::Int32(v.base10_parse()?),
                    "u32" => Literal::Uint32(v.base10_parse()?),
                    "" => Literal::Int32(v.base10_parse()?),
                    "f32" => Literal::float(v.base10_parse()?),
                    _ => return Err(syn::Error::new(v.span(), "unsupported integer type")),
                },
                Lit::Float(v) => Literal::float(v.base10_parse()?),
                _ => return Err(syn::Error::new(l.span(), "unsupported literal")),
            }),
            syn::Expr::Path(p)
                if p.path.segments.len() == 2 && p.path.segments[0].ident == "f32" =>
            {
                let v = match p.path.segments[1].ident.to_string().as_str() {
                    "MIN" => f32::MIN,
                    "MAX" => f32::MAX,
                    "INFINITY" => f32::INFINITY,
                    "NEG_INFINITY" => f32::NEG_INFINITY,
                    _ => return Err(syn::Error::new(p.span(), "unknown f32 constant")),
                };
                MimirExpr::Literal(Literal::float(v))
            }
            syn::Expr::Path(p) => {
                let n = p
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(p.span(), "expected local name"))?
                    .to_string();
                if let Some(id) = self.generics.get(&n) {
                    MimirExpr::ConstExpr(RichExpr::new(ConstExpr::Generic(*id), 0..0))
                } else {
                    MimirExpr::Var(
                        *self
                            .names
                            .get(&n)
                            .ok_or_else(|| syn::Error::new(p.span(), "unknown local"))?,
                    )
                }
            }
            syn::Expr::Call(c) => {
                let syn::Expr::Path(path) = c.func.as_ref() else {
                    return Err(syn::Error::new(c.span(), "expected device function name"));
                };
                let name = path
                    .path
                    .get_ident()
                    .ok_or_else(|| {
                        syn::Error::new(path.span(), "device calls must use module-local names")
                    })?
                    .to_string();
                if let Some(intrinsic) = self.intrinsic(&name, &c.args)? {
                    return Ok(RichExpr::new(MimirExpr::MathIntrinsic(intrinsic), 0..0));
                }
                let (function, parameters, returns) =
                    self.functions.get(&name).cloned().ok_or_else(|| {
                        syn::Error::new(path.span(), "unresolved device function")
                    })?;
                let arguments = c
                    .args
                    .iter()
                    .map(|e| self.expr(e))
                    .collect::<Result<Vec<_>>>()?;
                MimirExpr::Call {
                    function,
                    arguments,
                    parameters,
                    returns,
                }
            }
            syn::Expr::Paren(p) => return self.expr(&p.expr),
            syn::Expr::Binary(b) => {
                let mut lhs = self.expr(&b.left)?;
                let mut rhs = self.expr(&b.right)?;
                Self::coerce(&mut lhs, self.type_of(&rhs));
                Self::coerce(&mut rhs, self.type_of(&lhs));
                MimirExpr::BinOp(MimirBinOpExpr::new(
                    lhs,
                    match b.op {
                        BinOp::Add(_) => MimirBinOp::Add,
                        BinOp::Sub(_) => MimirBinOp::Sub,
                        BinOp::Mul(_) => MimirBinOp::Mul,
                        BinOp::Div(_) => MimirBinOp::Div,
                        BinOp::Rem(_) => MimirBinOp::Mod,
                        BinOp::Eq(_) => MimirBinOp::Eq,
                        BinOp::Ne(_) => MimirBinOp::Ne,
                        BinOp::Lt(_) => MimirBinOp::Lt,
                        BinOp::Le(_) => MimirBinOp::Lte,
                        BinOp::Gt(_) => MimirBinOp::Gt,
                        BinOp::Ge(_) => MimirBinOp::Gte,
                        BinOp::And(_) => MimirBinOp::And,
                        BinOp::Or(_) => MimirBinOp::Or,
                        _ => return Err(syn::Error::new(b.span(), "unsupported operator")),
                    },
                    rhs,
                ))
            }
            syn::Expr::Unary(u) => MimirExpr::Unary(MimirUnaryExpr::new(
                match u.op {
                    UnOp::Neg(_) => MimirUnOp::Neg,
                    UnOp::Not(_) => MimirUnOp::Not,
                    _ => return Err(syn::Error::new(u.span(), "unsupported unary operator")),
                },
                self.expr(&u.expr)?,
            )),
            syn::Expr::Cast(c) => MimirExpr::Cast {
                value: Box::new(self.expr(&c.expr)?),
                to: ty(&c.ty)?,
            },
            syn::Expr::Index(i) => {
                let array = self.local(&i.expr)?;
                MimirExpr::Index {
                    array,
                    index: Box::new(self.expr(&i.index)?),
                }
            }
            syn::Expr::Field(f) => {
                let syn::Expr::Path(p) = f.base.as_ref() else {
                    return Err(syn::Error::new(f.span(), "expected builtin"));
                };
                let name = p
                    .path
                    .get_ident()
                    .map(|x| x.to_string())
                    .unwrap_or_default();
                let builtin = match name.as_str() {
                    "global_id" => Builtin::GlobalInvocationId,
                    "thread_idx" => Builtin::ThreadIdx,
                    "block_idx" => Builtin::BlockIdx,
                    "block_dim" => Builtin::BlockDim,
                    _ => return Err(syn::Error::new(f.span(), "unknown builtin")),
                };
                let field = match &f.member {
                    Member::Named(i) => match i.to_string().as_str() {
                        "x" => Axis::X,
                        "y" => Axis::Y,
                        "z" => Axis::Z,
                        _ => return Err(syn::Error::new(f.span(), "expected x, y or z")),
                    },
                    _ => return Err(syn::Error::new(f.span(), "expected builtin axis")),
                };
                MimirExpr::Builtin { builtin, field }
            }
            _ => return Err(syn::Error::new(e.span(), "unsupported kernel expression")),
        };
        Ok(RichExpr::new(x, e.span().byte_range()))
    }

    fn constant(e: RichMimirExpr) -> Result<RichConstExpr> {
        let expr = match e.expr {
            MimirExpr::ConstExpr(c) => return Ok(c),
            MimirExpr::Literal(l) => ConstExpr::Literal(l),
            MimirExpr::BinOp(b) => ConstExpr::BinOp(MimirBinOpExpr::new(
                Self::constant(*b.lhs)?,
                b.op,
                Self::constant(*b.rhs)?,
            )),
            _ => {
                return Err(syn::Error::new(
                    {
                        let f: ItemFn = syn::parse_quote!(
                            fn error() {}
                        );
                        f.span()
                    },
                    "shared length must be a tuning expression",
                ));
            }
        };
        Ok(RichExpr::new(expr, e.span))
    }

    fn intrinsic(
        &mut self,
        name: &str,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
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
            return Err(syn::Error::new(
                args.span(),
                "intrinsic argument count mismatch",
            ));
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
    fn local(&self, e: &syn::Expr) -> Result<LocalId> {
        if let syn::Expr::Path(p) = e {
            if let Some(i) = p.path.get_ident() {
                if let Some(id) = self.names.get(&i.to_string()) {
                    return Ok(*id);
                }
            }
        }
        Err(syn::Error::new(e.span(), "expected local variable"))
    }
    fn block(&mut self, b: &syn::Block) -> Result<MimirKernelBlock> {
        let saved = self.names.clone();
        let result = self.block_inner(b);
        self.names = saved;
        result
    }
    fn block_inner(&mut self, b: &syn::Block) -> Result<MimirKernelBlock> {
        let mut out = vec![];
        for s in &b.stmts {
            let stmt = match s {
                Stmt::Local(l) if l.attrs.iter().any(|a| a.path().is_ident("shared")) => {
                    let Pat::Ident(n) = &l.pat else {
                        return Err(syn::Error::new(l.span(), "expected shared binding"));
                    };
                    let syn::Expr::Repeat(a) = l
                        .init
                        .as_ref()
                        .ok_or_else(|| syn::Error::new(l.span(), "shared initializer required"))?
                        .expr
                        .as_ref()
                    else {
                        return Err(syn::Error::new(
                            l.span(),
                            "shared allocation requires [literal; length]",
                        ));
                    };
                    let initial = self.expr(&a.expr)?;
                    let element = self.type_of(&initial);
                    let length = self.expr(&a.len)?;
                    let length = Self::constant(length)?;
                    let binding = self.bind(&n.ident);
                    self.types.insert(binding.id, element);
                    MimirKernelStmt::SharedMem {
                        binding,
                        element,
                        length,
                    }
                }
                Stmt::Local(l) => {
                    let (pat, annotation) = match &l.pat {
                        Pat::Type(p) => (p.pat.as_ref(), Some(ty(&p.ty)?)),
                        p => (p, None),
                    };
                    let Pat::Ident(n) = pat else {
                        return Err(syn::Error::new(pat.span(), "expected named binding"));
                    };
                    let mut value = self.expr(
                        &l.init
                            .as_ref()
                            .ok_or_else(|| syn::Error::new(l.span(), "initializer required"))?
                            .expr,
                    )?;
                    let t = annotation.unwrap_or_else(|| self.type_of(&value));
                    Self::coerce(&mut value, t);
                    let binding = self.bind(&n.ident);
                    self.types.insert(binding.id, t);
                    MimirKernelStmt::Let {
                        binding,
                        ty: t,
                        mutability: if n.mutability.is_some() {
                            Mutability::Mutable
                        } else {
                            Mutability::Constant
                        },
                        value,
                    }
                }
                Stmt::Expr(e, _) => {
                    match e {
                        syn::Expr::Binary(b)
                            if matches!(
                                b.op,
                                BinOp::AddAssign(_)
                                    | BinOp::SubAssign(_)
                                    | BinOp::MulAssign(_)
                                    | BinOp::DivAssign(_)
                            ) =>
                        {
                            let target = if let syn::Expr::Index(i) = b.left.as_ref() {
                                AssignTarget::Index {
                                    array: self.local(&i.expr)?,
                                    index: self.expr(&i.index)?,
                                }
                            } else {
                                AssignTarget::Var(self.local(&b.left)?)
                            };
                            let lhs = self.expr(&b.left)?;
                            let mut rhs = self.expr(&b.right)?;
                            Self::coerce(&mut rhs, self.type_of(&lhs));
                            let op = match b.op {
                                BinOp::AddAssign(_) => MimirBinOp::Add,
                                BinOp::SubAssign(_) => MimirBinOp::Sub,
                                BinOp::MulAssign(_) => MimirBinOp::Mul,
                                _ => MimirBinOp::Div,
                            };
                            MimirKernelStmt::Assign {
                                target,
                                value: RichExpr::new(
                                    MimirExpr::BinOp(MimirBinOpExpr::new(lhs, op, rhs)),
                                    0..0,
                                ),
                            }
                        }
                        syn::Expr::Assign(a) => {
                            let target = if let syn::Expr::Index(i) = a.left.as_ref() {
                                AssignTarget::Index {
                                    array: self.local(&i.expr)?,
                                    index: self.expr(&i.index)?,
                                }
                            } else {
                                AssignTarget::Var(self.local(&a.left)?)
                            };
                            MimirKernelStmt::Assign {
                                target,
                                value: self.expr(&a.right)?,
                            }
                        }
                        syn::Expr::Return(r) => {
                            if let Some(value) = &r.expr {
                                MimirKernelStmt::ReturnValue(self.expr(value)?)
                            } else {
                                MimirKernelStmt::Return
                            }
                        }
                        syn::Expr::Block(b) => MimirKernelStmt::Block(self.block(&b.block)?),
                        syn::Expr::If(i) => {
                            let condition = self.expr(&i.cond)?;
                            let then_branch = self.block(&i.then_branch)?;
                            let else_branch = match &i.else_branch {
                                None => None,
                                Some((_, e)) => {
                                    if let syn::Expr::Block(b) = e.as_ref() {
                                        Some(self.block(&b.block)?)
                                    } else {
                                        return Err(syn::Error::new(
                                            e.span(),
                                            "else must contain a block",
                                        ));
                                    }
                                }
                            };
                            MimirKernelStmt::If {
                                condition,
                                then_branch,
                                else_branch,
                            }
                        }
                        syn::Expr::ForLoop(f) => {
                            let Pat::Ident(n) = f.pat.as_ref() else {
                                return Err(syn::Error::new(f.span(), "expected loop binding"));
                            };
                            let (range, step) = if let syn::Expr::MethodCall(m) = f.expr.as_ref() {
                                if m.method != "step_by" || m.args.len() != 1 {
                                    return Err(syn::Error::new(
                                        m.span(),
                                        "only step_by is supported for ranges",
                                    ));
                                }
                                let receiver = if let syn::Expr::Paren(p) = m.receiver.as_ref() {
                                    p.expr.as_ref()
                                } else {
                                    m.receiver.as_ref()
                                };
                                (receiver, Some(self.expr(&m.args[0])?))
                            } else {
                                (f.expr.as_ref(), None)
                            };
                            let syn::Expr::Range(r) = range else {
                                return Err(syn::Error::new(f.span(), "expected exclusive range"));
                            };
                            if !matches!(r.limits, RangeLimits::HalfOpen(_)) {
                                return Err(syn::Error::new(r.span(), "expected exclusive range"));
                            }
                            let mut start = self.expr(r.start.as_ref().ok_or_else(|| {
                                syn::Error::new(r.span(), "range start required")
                            })?)?;
                            let end = self.expr(r.end.as_ref().ok_or_else(|| {
                                syn::Error::new(r.span(), "range end required")
                            })?)?;
                            Self::coerce(&mut start, self.type_of(&end));
                            let saved = self.names.clone();
                            let binding = self.bind(&n.ident);
                            self.types.insert(binding.id, self.type_of(&end));
                            let body = self.block(&f.body)?;
                            self.names = saved;
                            MimirKernelStmt::RangeFor {
                                binding,
                                start,
                                end,
                                step,
                                body,
                            }
                        }
                        syn::Expr::Call(c)
                            if matches!(c.func.as_ref(),syn::Expr::Path(p) if (p.path.is_ident("syncthreads") || p.path.is_ident("__syncthreads")))
                                && c.args.is_empty() =>
                        {
                            MimirKernelStmt::Syncthreads
                        }
                        _ => return Err(syn::Error::new(e.span(), "unsupported kernel statement")),
                    }
                }
                _ => return Err(syn::Error::new(s.span(), "unsupported kernel statement")),
            };
            out.push(RichKernelStmt::new(stmt, s.span().byte_range()));
        }
        Ok(out)
    }
}
pub fn parse_kernel(f: &ItemFn) -> Result<MimirKernelAst> {
    parse_with(f, BTreeMap::new())
}
fn parse_with(
    f: &ItemFn,
    functions: BTreeMap<String, (u32, Vec<KernelParamTy>, PrimitiveTy)>,
) -> Result<MimirKernelAst> {
    let mut p = Parser {
        names: BTreeMap::new(),
        generics: BTreeMap::new(),
        next: 0,
        types: BTreeMap::new(),
        functions,
    };
    let mut const_generics = vec![];
    for g in &f.sig.generics.params {
        let GenericParam::Const(c) = g else {
            return Err(syn::Error::new(
                g.span(),
                "only u32 const generics are supported",
            ));
        };
        if ty(&c.ty)? != PrimitiveTy::Uint32 {
            return Err(syn::Error::new(c.span(), "const generics must be u32"));
        }
        p.generics.insert(
            c.ident.to_string(),
            ConstGenericId(const_generics.len() as u32),
        );
        const_generics.push(c.ident.to_string());
    }
    let mut parameters = vec![];
    for arg in &f.sig.inputs {
        let FnArg::Typed(a) = arg else {
            return Err(syn::Error::new(arg.span(), "methods are unsupported"));
        };
        let Pat::Ident(n) = a.pat.as_ref() else {
            return Err(syn::Error::new(a.span(), "expected parameter name"));
        };
        let binding = p.bind(&n.ident);
        let param_ty = if let Type::Reference(r) = a.ty.as_ref() {
            let Type::Slice(s) = r.elem.as_ref() else {
                return Err(syn::Error::new(r.span(), "expected slice"));
            };
            KernelParamTy::Buffer {
                element: ty(&s.elem)?,
                access: if r.mutability.is_some() {
                    Mutability::Mutable
                } else {
                    Mutability::Constant
                },
            }
        } else {
            KernelParamTy::Scalar(ty(&a.ty)?)
        };
        p.types.insert(
            binding.id,
            match param_ty {
                KernelParamTy::Scalar(t) => t,
                KernelParamTy::Buffer { element, .. } => element,
            },
        );
        parameters.push(KernelParameter {
            id: binding.id,
            name: binding.name,
            ty: param_ty,
            span: a.span().byte_range(),
        });
    }

    Ok(MimirKernelAst {
        name: f.sig.ident.to_string(),
        parameters,
        const_generics,
        body: p.block(&f.block)?,
    })
}

/// Compile a self-contained Rust GPU module with module-local device functions.
pub fn compile_module(items: &[(ItemFn, bool)]) -> Result<mimir_ir::flat::Module> {
    let mut signatures = BTreeMap::new();
    for (f, kernel) in items {
        if !kernel {
            let ReturnType::Type(_, result) = &f.sig.output else {
                return Err(syn::Error::new(
                    f.sig.span(),
                    "device functions require a scalar return type",
                ));
            };
            if !f.sig.generics.params.is_empty() {
                return Err(syn::Error::new(
                    f.sig.span(),
                    "generic device functions are not yet supported",
                ));
            }
            let mut signature = f.clone();
            signature.block = Box::new(syn::parse_quote!({}));
            let ast = parse_kernel(&signature)?;
            let id = signatures.len() as u32;
            if signatures
                .insert(
                    f.sig.ident.to_string(),
                    (
                        id,
                        ast.parameters.iter().map(|p| p.ty).collect(),
                        ty(result)?,
                    ),
                )
                .is_some()
            {
                return Err(syn::Error::new(
                    f.sig.ident.span(),
                    "duplicate device function",
                ));
            }
        }
    }
    let mut module = mimir_ir::flat::Module::default();
    for (f, kernel) in items {
        let ast = parse_with(f, signatures.clone())?;
        if *kernel {
            if !matches!(f.sig.output, ReturnType::Default) {
                return Err(syn::Error::new(
                    f.sig.output.span(),
                    "kernels cannot return values",
                ));
            }
            let ast = ast
                .validate()
                .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?;
            module.kernels.extend(
                ast.to_ir()
                    .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?
                    .kernels,
            );
        } else {
            let (id, _, returns) = signatures[&f.sig.ident.to_string()].clone();
            let function = MimirFunctionAst {
                routine: ast,
                returns,
            }
            .validate()
            .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?;
            module.functions.push(
                function
                    .to_ir(id)
                    .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?,
            );
        }
    }
    module.clone().validate().map_err(|e| {
        syn::Error::new(
            items.first().map(|(f, _)| f.sig.span()).unwrap_or_else(|| {
                let f: ItemFn = syn::parse_quote!(
                    fn empty() {}
                );
                f.span()
            }),
            e.to_string(),
        )
    })?;
    Ok(module)
}
