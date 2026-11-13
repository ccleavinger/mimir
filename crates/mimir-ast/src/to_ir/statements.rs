use super::*;
// Explicit continuations preserve lexical control flow without recursive
// statement lowering. A return terminates its sequence but still executes the
// structural continuation responsible for merge/continue placeholders.
enum Work<'a> {
    Next(&'a [RichKernelStmt]),
    Then {
        otherwise: &'a [RichKernelStmt],
        no: usize,
        merge: usize,
    },
    Else {
        then_live: bool,
        merge: usize,
    },
    Continue {
        header: usize,
        continuing: usize,
        merge: usize,
        index: ir::Operand,
        buffer: ir::Operand,
        step: ir::Operand,
        ty: ir::Type,
    },
}
impl RoutineBuilder {
    pub(super) fn statements(&mut self, stmts: &MimirKernelBlock) -> Result<bool, LoweringError> {
        let mut work = vec![Work::Next(stmts)];
        let mut live = true;
        while let Some(next) = work.pop() {
            match next {
                Work::Next(stmts) => {
                    if !live {
                        continue;
                    }
                    let Some((s, rest)) = stmts.split_first() else {
                        continue;
                    };
                    work.push(Work::Next(rest));
                    match &s.stmt {
                        MimirKernelStmt::Let {
                            binding, ty, value, ..
                        } => {
                            let (value, _) = self.expr(value)?;
                            let t = scalar(*ty);
                            let buffer = self.storage(
                                t,
                                ir::AddressSpace::Local,
                                ir::ConstantOperand::Literal(1),
                            );
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
                            self.blocks[self.current].terminator =
                                ir::Terminator::Return(Some(value));
                            live = false;
                        }
                        MimirKernelStmt::Return => {
                            self.blocks[self.current].terminator = ir::Terminator::Return(None);
                            live = false;
                        }
                        MimirKernelStmt::Block(body) => work.push(Work::Next(body)),
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
                            work.push(Work::Then {
                                otherwise: else_branch.as_deref().unwrap_or(&[]),
                                no,
                                merge,
                            });
                            work.push(Work::Next(then_branch));
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
                            work.push(Work::Continue {
                                header,
                                continuing,
                                merge,
                                index,
                                buffer,
                                step,
                                ty: t,
                            });
                            work.push(Work::Next(body));
                        }
                    }
                }
                Work::Then {
                    otherwise,
                    no,
                    merge,
                } => {
                    let then_live = live;
                    if live {
                        self.branch(merge)
                    }
                    self.current = no;
                    live = true;
                    work.push(Work::Else { then_live, merge });
                    work.push(Work::Next(otherwise));
                }
                Work::Else { then_live, merge } => {
                    if live {
                        self.branch(merge)
                    }
                    self.current = merge;
                    live = then_live || live;
                }
                Work::Continue {
                    header,
                    continuing,
                    merge,
                    index,
                    buffer,
                    step,
                    ty,
                } => {
                    if live {
                        self.branch(continuing);
                        self.current = continuing;
                        let next = self.value(
                            ty,
                            ir::ValueOp::Binary {
                                op: ir::BinaryOp::Add,
                                lhs: index,
                                rhs: step,
                            },
                        );
                        self.emit(ir::Instruction::Store {
                            buffer,
                            index: ir::Operand::Literal(ir::Literal::U32(0)),
                            value: next,
                        });
                        self.branch(header);
                    }
                    self.current = merge;
                    live = true;
                }
            }
        }
        Ok(live)
    }
}
