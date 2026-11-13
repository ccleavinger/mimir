//! Lower validated lexical AST into structured flat IR.
use crate::*;
use mimir_ir::flat as ir;
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
#[error("{message} at {span:?}")]
pub struct LoweringError {
    pub span: Span,
    pub message: String,
}

struct Lower {
    header: ir::Header,
    blocks: Vec<ir::Block>,
    current: usize,
    locals: BTreeMap<LocalId, (ir::Operand, ir::Type)>,
    next_value: u32,
    next_storage: u32,
}
fn scalar(t: PrimitiveTy) -> ir::ScalarType {
    match t {
        PrimitiveTy::Int32 => ir::ScalarType::I32,
        PrimitiveTy::Uint32 => ir::ScalarType::U32,
        PrimitiveTy::Float32 => ir::ScalarType::F32,
        PrimitiveTy::Bool => ir::ScalarType::Bool,
    }
}
fn literal(t: Literal) -> ir::Literal {
    match t {
        Literal::Int32(v) => ir::Literal::I32(v),
        Literal::Uint32(v) => ir::Literal::U32(v),
        Literal::Float32(v) => ir::Literal::F32(v),
        Literal::Bool(v) => ir::Literal::Bool(v),
    }
}
fn binary(t: &MimirBinOp) -> ir::BinaryOp {
    match t {
        MimirBinOp::Add => ir::BinaryOp::Add,
        MimirBinOp::Sub => ir::BinaryOp::Sub,
        MimirBinOp::Mul => ir::BinaryOp::Mul,
        MimirBinOp::Div => ir::BinaryOp::Div,
        MimirBinOp::Mod => ir::BinaryOp::Rem,
        MimirBinOp::And => ir::BinaryOp::And,
        MimirBinOp::Or => ir::BinaryOp::Or,
        MimirBinOp::Lt => ir::BinaryOp::Lt,
        MimirBinOp::Lte => ir::BinaryOp::Le,
        MimirBinOp::Gt => ir::BinaryOp::Gt,
        MimirBinOp::Gte => ir::BinaryOp::Ge,
        MimirBinOp::Eq => ir::BinaryOp::Eq,
        MimirBinOp::Ne => ir::BinaryOp::Ne,
    }
}
impl Lower {
    fn new() -> Self {
        let mut s = Self {
            header: ir::Header::default(),
            blocks: vec![],
            current: 0,
            locals: BTreeMap::new(),
            next_value: 0,
            next_storage: 0,
        };
        s.block();
        s
    }
    fn block(&mut self) -> usize {
        let i = self.blocks.len();
        self.blocks.push(ir::Block {
            id: ir::BlockId(i as u32),
            arguments: vec![],
            instructions: vec![],
            merge: None,
            terminator: ir::Terminator::Unreachable,
        });
        i
    }
    fn edge(i: usize) -> ir::Edge {
        ir::Edge {
            target: ir::BlockId(i as u32),
            arguments: vec![],
        }
    }
    fn branch(&mut self, i: usize) {
        self.blocks[self.current].terminator = ir::Terminator::Branch(Self::edge(i));
    }
    fn emit(&mut self, i: ir::Instruction) {
        self.blocks[self.current].instructions.push(i);
    }
    fn value(&mut self, ty: ir::Type, op: ir::ValueOp) -> ir::Operand {
        let id = ir::ValueId(self.next_value);
        self.next_value += 1;
        self.header.variables.push(ir::Variable::Value { id, ty });
        self.emit(ir::Instruction::Value { result: id, op });
        ir::Operand::Value(id)
    }
    fn storage(
        &mut self,
        t: ir::ScalarType,
        space: ir::AddressSpace,
        length: ir::ConstantOperand,
    ) -> ir::Operand {
        let id = ir::StorageId(self.next_storage);
        self.next_storage += 1;
        self.header.variables.push(ir::Variable::Storage {
            id,
            element: t,
            space,
            length,
        });
        ir::Operand::Storage(id)
    }
    fn fail<T>(&self, span: &Span, message: &str) -> Result<T, LoweringError> {
        Err(LoweringError {
            span: span.clone(),
            message: message.into(),
        })
    }
    fn constant(&mut self, e: &RichConstExpr) -> Result<ir::ConstantOperand, LoweringError> {
        match &e.expr {
            ConstExpr::Generic(id) => Ok(ir::ConstantOperand::Constant(ir::ConstantId(id.0))),
            ConstExpr::Literal(Literal::Uint32(v)) => Ok(ir::ConstantOperand::Literal(*v)),
            ConstExpr::BinOp(b) => {
                let lhs = self.constant(&b.lhs)?;
                let rhs = self.constant(&b.rhs)?;
                let op = match b.op {
                    MimirBinOp::Add => ir::ConstantOp::Add,
                    MimirBinOp::Sub => ir::ConstantOp::Sub,
                    MimirBinOp::Mul => ir::ConstantOp::Mul,
                    MimirBinOp::Div => ir::ConstantOp::Div,
                    MimirBinOp::Mod => ir::ConstantOp::Rem,
                    _ => return self.fail(&e.span, "unsupported tuning expression"),
                };
                let id = ir::ConstantId(
                    (self.header.constants.len() + self.header.derived_constants.len()) as u32,
                );
                self.header
                    .derived_constants
                    .push(ir::DerivedConstant { id, op, lhs, rhs });
                Ok(ir::ConstantOperand::Constant(id))
            }
            _ => self.fail(
                &e.span,
                "tuning expressions require u32 literals, generics and checked arithmetic",
            ),
        }
    }
    fn expr(&mut self, e: &RichMimirExpr) -> Result<(ir::Operand, ir::Type), LoweringError> {
        let (op, t) = match &e.expr {
            MimirExpr::Call {
                function,
                arguments,
                returns,
                ..
            } => {
                let mut args = vec![];
                for arg in arguments {
                    args.push(self.expr(arg)?.0)
                }
                let id = ir::ValueId(self.next_value);
                self.next_value += 1;
                let ty = ir::Type::Scalar(scalar(*returns));
                self.header.variables.push(ir::Variable::Value { id, ty });
                self.emit(ir::Instruction::Call {
                    result: Some(id),
                    function: ir::FunctionId(*function),
                    arguments: args,
                    constants: vec![],
                });
                return Ok((ir::Operand::Value(id), ty));
            }
            MimirExpr::Literal(v) => {
                return Ok((
                    ir::Operand::Literal(literal(*v)),
                    ir::Type::Scalar(scalar(v.ty())),
                ));
            }
            MimirExpr::Var(id) => {
                let (v, t) = self.locals[id];
                if matches!(v, ir::Operand::Storage(_)) && matches!(t, ir::Type::Scalar(_)) {
                    (
                        ir::ValueOp::Load {
                            buffer: v,
                            index: ir::Operand::Literal(ir::Literal::U32(0)),
                        },
                        t,
                    )
                } else {
                    return Ok((v, t));
                }
            }
            MimirExpr::Index { array, index } => {
                let (buffer, t) = self.locals[array];
                let (index, _) = self.expr(index)?;
                let ir::Type::Buffer { element, .. } = t else {
                    return self.fail(&e.span, "expected buffer");
                };
                (
                    ir::ValueOp::Load { buffer, index },
                    ir::Type::Scalar(element),
                )
            }
            MimirExpr::BinOp(b) => {
                let (lhs, t) = self.expr(&b.lhs)?;
                let (rhs, _) = self.expr(&b.rhs)?;
                let op = binary(&b.op);
                let t = if matches!(
                    op,
                    ir::BinaryOp::Eq
                        | ir::BinaryOp::Ne
                        | ir::BinaryOp::Lt
                        | ir::BinaryOp::Le
                        | ir::BinaryOp::Gt
                        | ir::BinaryOp::Ge
                ) {
                    ir::Type::Scalar(ir::ScalarType::Bool)
                } else {
                    t
                };
                (ir::ValueOp::Binary { op, lhs, rhs }, t)
            }
            MimirExpr::Unary(u) => {
                let (value, t) = self.expr(&u.expr)?;
                (
                    ir::ValueOp::Unary {
                        op: match u.op {
                            MimirUnOp::Neg => ir::UnaryOp::Neg,
                            MimirUnOp::Not => ir::UnaryOp::Not,
                        },
                        value,
                    },
                    t,
                )
            }
            MimirExpr::Cast { value, to } => {
                let (value, _) = self.expr(value)?;
                let to = scalar(*to);
                (ir::ValueOp::Cast { value, to }, ir::Type::Scalar(to))
            }
            MimirExpr::Builtin { builtin, field } => (
                ir::ValueOp::Builtin {
                    builtin: match builtin {
                        Builtin::BlockIdx => ir::Builtin::WorkgroupId,
                        Builtin::BlockDim => ir::Builtin::WorkgroupSize,
                        Builtin::ThreadIdx => ir::Builtin::LocalId,
                        Builtin::GlobalInvocationId => ir::Builtin::GlobalId,
                    },
                    axis: match field {
                        Axis::X => ir::Axis::X,
                        Axis::Y => ir::Axis::Y,
                        Axis::Z => ir::Axis::Z,
                    },
                },
                ir::Type::Scalar(ir::ScalarType::U32),
            ),
            MimirExpr::ConstExpr(c) => {
                let c = self.constant(c)?;
                return Ok((
                    match c {
                        ir::ConstantOperand::Literal(v) => {
                            ir::Operand::Literal(ir::Literal::U32(v))
                        }
                        ir::ConstantOperand::Constant(id) => ir::Operand::Constant(id),
                    },
                    ir::Type::Scalar(ir::ScalarType::U32),
                ));
            }
            MimirExpr::MathIntrinsic(i) => self.intrinsic(i)?,
        };
        Ok((self.value(t, op), t))
    }
    fn intrinsic(
        &mut self,
        i: &Intrinsic<RichMimirExpr>,
    ) -> Result<(ir::ValueOp, ir::Type), LoweringError> {
        Ok(match i {
            Intrinsic::Unary(op, e) => {
                let (value, t) = self.expr(e)?;
                let op = match op {
                    UnaryIntrinsic::Sin => ir::UnaryIntrinsic::Sin,
                    UnaryIntrinsic::Cos => ir::UnaryIntrinsic::Cos,
                    UnaryIntrinsic::Tan => ir::UnaryIntrinsic::Tan,
                    UnaryIntrinsic::Asin => ir::UnaryIntrinsic::Asin,
                    UnaryIntrinsic::Acos => ir::UnaryIntrinsic::Acos,
                    UnaryIntrinsic::Atan => ir::UnaryIntrinsic::Atan,
                    UnaryIntrinsic::Sinh => ir::UnaryIntrinsic::Sinh,
                    UnaryIntrinsic::Cosh => ir::UnaryIntrinsic::Cosh,
                    UnaryIntrinsic::Tanh => ir::UnaryIntrinsic::Tanh,
                    UnaryIntrinsic::Asinh => ir::UnaryIntrinsic::Asinh,
                    UnaryIntrinsic::Acosh => ir::UnaryIntrinsic::Acosh,
                    UnaryIntrinsic::Atanh => ir::UnaryIntrinsic::Atanh,
                    UnaryIntrinsic::Exp => ir::UnaryIntrinsic::Exp,
                    UnaryIntrinsic::Log => ir::UnaryIntrinsic::Log,
                    UnaryIntrinsic::Exp2 => ir::UnaryIntrinsic::Exp2,
                    UnaryIntrinsic::Log2 => ir::UnaryIntrinsic::Log2,
                    UnaryIntrinsic::Sqrt => ir::UnaryIntrinsic::Sqrt,
                    UnaryIntrinsic::Isqrt => ir::UnaryIntrinsic::InverseSqrt,
                    UnaryIntrinsic::Floor => ir::UnaryIntrinsic::Floor,
                    UnaryIntrinsic::Ceil => ir::UnaryIntrinsic::Ceil,
                };
                (ir::ValueOp::UnaryIntrinsic { op, value }, t)
            }
            Intrinsic::Binary(op, a) => {
                let (x, t) = self.expr(&a[0])?;
                let (y, _) = self.expr(&a[1])?;
                let op = match op {
                    BinaryIntrinsic::Atan2 => ir::BinaryIntrinsic::Atan2,
                    BinaryIntrinsic::Pow => ir::BinaryIntrinsic::Pow,
                    BinaryIntrinsic::Min => ir::BinaryIntrinsic::Min,
                    BinaryIntrinsic::Max => ir::BinaryIntrinsic::Max,
                };
                (ir::ValueOp::BinaryIntrinsic { op, args: [x, y] }, t)
            }
            Intrinsic::Ternary(op, a) => {
                let (x, t) = self.expr(&a[0])?;
                let (y, _) = self.expr(&a[1])?;
                let (z, _) = self.expr(&a[2])?;
                let op = match op {
                    TernaryIntrinsic::Clamp => ir::TernaryIntrinsic::Clamp,
                    TernaryIntrinsic::Mix => ir::TernaryIntrinsic::Mix,
                    TernaryIntrinsic::Fma => ir::TernaryIntrinsic::Fma,
                };
                (
                    ir::ValueOp::TernaryIntrinsic {
                        op,
                        args: [x, y, z],
                    },
                    t,
                )
            }
        })
    }
    fn statements(&mut self, stmts: &MimirKernelBlock) -> Result<bool, LoweringError> {
        for s in stmts {
            match &s.stmt {
                MimirKernelStmt::Let {
                    binding, ty, value, ..
                } => {
                    let (value, _) = self.expr(value)?;
                    let t = scalar(*ty);
                    let buffer =
                        self.storage(t, ir::AddressSpace::Local, ir::ConstantOperand::Literal(1));
                    self.emit(ir::Instruction::Store {
                        buffer,
                        index: ir::Operand::Literal(ir::Literal::U32(0)),
                        value,
                    });
                    self.locals
                        .insert(binding.id, (buffer, ir::Type::Scalar(t)));
                }
                MimirKernelStmt::SharedMem {
                    binding,
                    element,
                    length,
                } => {
                    let length = self.constant(length)?;
                    let element = scalar(*element);
                    let buffer = self.storage(element, ir::AddressSpace::Workgroup, length);
                    self.locals.insert(
                        binding.id,
                        (
                            buffer,
                            ir::Type::Buffer {
                                element,
                                space: ir::AddressSpace::Workgroup,
                                access: ir::Access::ReadWrite,
                            },
                        ),
                    );
                }
                MimirKernelStmt::Assign { target, value } => {
                    let (value, _) = self.expr(value)?;
                    let (buffer, index) = match target {
                        AssignTarget::Var(id) => {
                            (self.locals[id].0, ir::Operand::Literal(ir::Literal::U32(0)))
                        }
                        AssignTarget::Index { array, index } => {
                            let (index, _) = self.expr(index)?;
                            (self.locals[array].0, index)
                        }
                    };
                    self.emit(ir::Instruction::Store {
                        buffer,
                        index,
                        value,
                    });
                }
                MimirKernelStmt::Syncthreads => self.emit(ir::Instruction::Barrier),
                MimirKernelStmt::ReturnValue(v) => {
                    let (value, _) = self.expr(v)?;
                    self.blocks[self.current].terminator = ir::Terminator::Return(Some(value));
                    return Ok(false);
                }
                MimirKernelStmt::Return => {
                    self.blocks[self.current].terminator = ir::Terminator::Return(None);
                    return Ok(false);
                }
                MimirKernelStmt::Block(body) => {
                    if !self.statements(body)? {
                        return Ok(false);
                    }
                }
                MimirKernelStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    let (condition, _) = self.expr(condition)?;
                    let header = self.current;
                    let yes = self.block();
                    let no = self.block();
                    let merge = self.block();
                    self.blocks[header].merge = Some(ir::Merge::Selection {
                        merge: ir::BlockId(merge as u32),
                    });
                    self.blocks[header].terminator = ir::Terminator::Conditional {
                        condition,
                        then_edge: Self::edge(yes),
                        else_edge: Self::edge(no),
                    };
                    self.current = yes;
                    let a = self.statements(then_branch)?;
                    if a {
                        self.branch(merge)
                    }
                    self.current = no;
                    let b = if let Some(body) = else_branch {
                        self.statements(body)?
                    } else {
                        true
                    };
                    if b {
                        self.branch(merge)
                    }
                    self.current = merge;
                    if !a && !b {
                        return Ok(false);
                    }
                }
                MimirKernelStmt::RangeFor {
                    binding,
                    start,
                    end,
                    step,
                    body,
                } => {
                    let (start, t) = self.expr(start)?;
                    let (end, _) = self.expr(end)?;
                    let step = if let Some(step) = step {
                        self.expr(step)?.0
                    } else {
                        ir::Operand::Literal(match t {
                            ir::Type::Scalar(ir::ScalarType::I32) => ir::Literal::I32(1),
                            _ => ir::Literal::U32(1),
                        })
                    };
                    let ir::Type::Scalar(element) = t else {
                        return self.fail(&s.span, "loop index must be scalar");
                    };
                    let buffer = self.storage(
                        element,
                        ir::AddressSpace::Local,
                        ir::ConstantOperand::Literal(1),
                    );
                    let zero = ir::Operand::Literal(ir::Literal::U32(0));
                    self.emit(ir::Instruction::Store {
                        buffer,
                        index: zero,
                        value: start,
                    });
                    self.locals.insert(binding.id, (buffer, t));
                    let header = self.block();
                    let run = self.block();
                    let continuing = self.block();
                    let merge = self.block();
                    self.branch(header);
                    self.current = header;
                    let index = self.value(
                        t,
                        ir::ValueOp::Load {
                            buffer,
                            index: zero,
                        },
                    );
                    let condition = self.value(
                        ir::Type::Scalar(ir::ScalarType::Bool),
                        ir::ValueOp::Binary {
                            op: ir::BinaryOp::Lt,
                            lhs: index,
                            rhs: end,
                        },
                    );
                    self.blocks[header].merge = Some(ir::Merge::Loop {
                        merge: ir::BlockId(merge as u32),
                        continuing: ir::BlockId(continuing as u32),
                    });
                    self.blocks[header].terminator = ir::Terminator::Conditional {
                        condition,
                        then_edge: Self::edge(run),
                        else_edge: Self::edge(merge),
                    };
                    self.current = run;
                    let live = self.statements(body)?;
                    if live {
                        self.branch(continuing);
                        self.current = continuing;
                        let next = self.value(
                            t,
                            ir::ValueOp::Binary {
                                op: ir::BinaryOp::Add,
                                lhs: index,
                                rhs: step,
                            },
                        );
                        self.emit(ir::Instruction::Store {
                            buffer,
                            index: zero,
                            value: next,
                        });
                        self.branch(header)
                    }
                    self.current = merge;
                }
            }
        }
        Ok(true)
    }
}
impl ValidatedKernel {
    pub fn to_ir(&self) -> Result<ir::Module, LoweringError> {
        let ast = self.ast();
        let mut l = Lower::new();
        l.header.constants = (0..ast.const_generics.len())
            .map(|i| ir::ConstantId(i as u32))
            .collect();
        for (i, p) in ast.parameters.iter().enumerate() {
            let id = ir::ParameterId(i as u32);
            let ty = match p.ty {
                KernelParamTy::Scalar(t) => ir::Type::Scalar(scalar(t)),
                KernelParamTy::Buffer { element, access } => ir::Type::Buffer {
                    element: scalar(element),
                    space: ir::AddressSpace::Global,
                    access: match access {
                        Mutability::Mutable => ir::Access::ReadWrite,
                        Mutability::Constant => ir::Access::Read,
                    },
                },
            };
            l.header.parameters.push(ir::Parameter { id, ty });
            l.locals.insert(p.id, (ir::Operand::Parameter(id), ty));
        }
        if l.statements(&ast.body)? {
            l.blocks[l.current].terminator = ir::Terminator::Return(None)
        }
        let module = ir::Module {
            kernels: vec![ir::Kernel {
                routine: ir::Routine {
                    header: l.header,
                    body: ir::Body {
                        entry: ir::BlockId(0),
                        blocks: l.blocks,
                    },
                },
                debug: ir::DebugInfo {
                    name: ast.name.clone(),
                    ..Default::default()
                },
            }],
            functions: vec![],
        };
        Ok(module)
    }
}

/// A device function uses the same lexical body grammar as a kernel and has an
/// explicit scalar return type. Whole-module IR validation resolves call IDs.
pub fn lower_function(
    mut ast: MimirKernelAst,
    id: u32,
    returns: PrimitiveTy,
) -> Result<ir::DeviceFunction, LoweringError> {
    crate::passes::kernel::safety::ValidityPass::default()
        .function(&mut ast, returns)
        .map_err(|e| LoweringError {
            span: e.span,
            message: e.message,
        })?;
    let validated = crate::valid::ValidatedKernel::from_function(ast);
    let mut module = validated.to_ir()?;
    let kernel = module.kernels.remove(0);
    Ok(ir::DeviceFunction {
        id: ir::FunctionId(id),
        routine: kernel.routine,
        returns: Some(scalar(returns)),
        debug: kernel.debug,
    })
}
