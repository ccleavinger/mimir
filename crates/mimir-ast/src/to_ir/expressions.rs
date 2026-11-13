use super::*;
impl RoutineBuilder {
    pub(super) fn constant(
        &mut self,
        e: &RichConstExpr,
    ) -> Result<ir::ConstantOperand, LoweringError> {
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
    pub(super) fn expr(
        &mut self,
        e: &RichMimirExpr,
    ) -> Result<(ir::Operand, ir::Type), LoweringError> {
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
    pub(super) fn intrinsic(
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
}
