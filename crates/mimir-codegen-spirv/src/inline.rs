//! Specialize and inline device routines before SPIR-V generation.
use crate::{CompileError, Result};
use mimir_ir::flat::*;
use std::collections::BTreeMap;

struct Inliner<'a> {
    module: &'a ValidatedModule,
    spec: &'a SpecializedKernel,
    header: Header,
    blocks: Vec<Block>,
    value: u32,
    storage: u32,
}

impl Inliner<'_> {
    fn value(&mut self, t: Type) -> ValueId {
        let id = ValueId(self.value);
        self.value += 1;
        self.header.variables.push(Variable::Value { id, ty: t });
        id
    }

    fn block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(Block {
            id,
            arguments: vec![],
            instructions: vec![],
            merge: None,
            terminator: Terminator::Unreachable,
        });
        id
    }

    fn routine(
        &mut self,
        r: &Routine,
        s: &SpecializedRoutine,
        args: &[Operand],
        exit: Option<BlockId>,
    ) -> Result<BlockId> {
        let params = r
            .header
            .parameters
            .iter()
            .zip(args)
            .map(|(p, a)| (p.id, *a))
            .collect::<BTreeMap<_, _>>();

        let mut values = BTreeMap::new();
        let mut storage = BTreeMap::new();
        let mut blocks = BTreeMap::new();
        for v in &r.header.variables {
            match v {
                Variable::Value { id, ty } => {
                    let new = self.value(*ty);
                    values.insert(*id, new);
                }
                Variable::Storage {
                    id, element, space, ..
                } => {
                    let allocation = s.allocations.iter().find(|a| a.id == *id).unwrap();
                    let new = StorageId(self.storage);
                    self.storage += 1;
                    self.header.variables.push(Variable::Storage {
                        id: new,
                        element: *element,
                        space: *space,
                        length: ConstantOperand::Literal(allocation.length),
                    });
                    storage.insert(*id, new);
                }
            }
        }

        for b in &r.body.blocks {
            let new = self.block();
            blocks.insert(b.id, new);
        }

        let operand = |o: Operand| match o {
            Operand::Parameter(id) => params[&id],
            Operand::Constant(id) => Operand::Literal(Literal::U32(s.constants[&id])),
            Operand::Value(id) => Operand::Value(values[&id]),
            Operand::Storage(id) => Operand::Storage(storage[&id]),
            o => o,
        };

        let edge = |e: &Edge| Edge {
            target: blocks[&e.target],
            arguments: e.arguments.iter().map(|a| operand(*a)).collect(),
        };

        for b in &r.body.blocks {
            let mut current = blocks[&b.id];
            self.blocks[current.0 as usize].arguments =
                b.arguments.iter().map(|v| values[v]).collect();
            for (index, i) in b.instructions.iter().enumerate() {
                if let Instruction::Call {
                    result,
                    function,
                    arguments,
                    ..
                } = i
                {
                    if matches!(b.merge, Some(Merge::Loop { .. })) {
                        return Err(CompileError("device calls in loop headers require moving the call into a preceding block".into()));
                    }
                    let instance = &s.calls[&(b.id, index)];
                    let function = self
                        .module
                        .module()
                        .functions
                        .iter()
                        .find(|f| f.id == *function)
                        .unwrap()
                        .clone();
                    let spec = self.spec.functions[instance].clone();
                    let args = arguments.iter().map(|a| operand(*a)).collect::<Vec<_>>();
                    let continuation = self.block();
                    let entry =
                        self.routine(&function.routine, &spec, &args, Some(continuation))?;
                    let mut fallback = vec![];
                    if let Some(result) = result {
                        let id = values[result];
                        self.blocks[continuation.0 as usize].arguments.push(id);
                        let t = match self
                            .header
                            .variables
                            .iter()
                            .find(|v| matches!(v,Variable::Value{id:x,..} if *x==id))
                            .unwrap()
                        {
                            Variable::Value {
                                ty: Type::Scalar(t),
                                ..
                            } => *t,
                            _ => {
                                return Err(CompileError(
                                    "device functions must return scalars".into(),
                                ));
                            }
                        };
                        fallback.push(Operand::Literal(match t {
                            ScalarType::Bool => Literal::Bool(false),
                            ScalarType::I32 => Literal::I32(0),
                            ScalarType::U32 => Literal::U32(0),
                            ScalarType::F32 => Literal::F32(0),
                        }));
                    }
                    let continuing = self.block();
                    let header = &mut self.blocks[current.0 as usize];
                    header.merge = Some(Merge::Loop {
                        merge: continuation,
                        continuing,
                    });
                    header.terminator = Terminator::Conditional {
                        condition: Operand::Literal(Literal::Bool(true)),
                        then_edge: Edge {
                            target: entry,
                            arguments: vec![],
                        },
                        else_edge: Edge {
                            target: continuation,
                            arguments: fallback,
                        },
                    };
                    current = continuation;
                } else {
                    let mapped = match i {
                        Instruction::Value { result, op } => {
                            let mut op = op.clone();
                            match &mut op {
                                ValueOp::Copy(a)
                                | ValueOp::Unary { value: a, .. }
                                | ValueOp::Cast { value: a, .. }
                                | ValueOp::UnaryIntrinsic { value: a, .. } => *a = operand(*a),
                                ValueOp::Binary { lhs, rhs, .. } => {
                                    *lhs = operand(*lhs);
                                    *rhs = operand(*rhs)
                                }
                                ValueOp::Load { buffer, index } => {
                                    *buffer = operand(*buffer);
                                    *index = operand(*index)
                                }
                                ValueOp::BinaryIntrinsic { args, .. } => {
                                    for a in args {
                                        *a = operand(*a)
                                    }
                                }
                                ValueOp::TernaryIntrinsic { args, .. } => {
                                    for a in args {
                                        *a = operand(*a)
                                    }
                                }
                                ValueOp::Builtin { .. } => {}
                            }
                            Instruction::Value {
                                result: values[result],
                                op,
                            }
                        }
                        Instruction::Store {
                            buffer,
                            index,
                            value,
                        } => Instruction::Store {
                            buffer: operand(*buffer),
                            index: operand(*index),
                            value: operand(*value),
                        },
                        Instruction::Barrier => Instruction::Barrier,
                        _ => unreachable!(),
                    };
                    self.blocks[current.0 as usize].instructions.push(mapped);
                }
            }

            let dst = &mut self.blocks[current.0 as usize];
            dst.merge = b.merge.map(|m| match m {
                Merge::Selection { merge } => Merge::Selection {
                    merge: blocks[&merge],
                },
                Merge::Loop { merge, continuing } => Merge::Loop {
                    merge: blocks[&merge],
                    continuing: blocks[&continuing],
                },
            });

            dst.terminator = match &b.terminator {
                Terminator::Branch(e) => Terminator::Branch(edge(e)),
                Terminator::Conditional {
                    condition,
                    then_edge,
                    else_edge,
                } => Terminator::Conditional {
                    condition: operand(*condition),
                    then_edge: edge(then_edge),
                    else_edge: edge(else_edge),
                },
                Terminator::Return(v) => {
                    if let Some(exit) = exit {
                        Terminator::Branch(Edge {
                            target: exit,
                            arguments: v.iter().map(|a| operand(*a)).collect(),
                        })
                    } else {
                        Terminator::Return(v.map(operand))
                    }
                }
                Terminator::Unreachable => Terminator::Unreachable,
            };
        }

        Ok(blocks[&r.body.entry])
    }
}

pub fn inline(module: &ValidatedModule, spec: &SpecializedKernel) -> Result<ValidatedModule> {
    let kernel = &module.module().kernels[spec.kernel_index];
    let mut i = Inliner {
        module,
        spec,
        header: Header::default(),
        blocks: vec![],
        value: 0,
        storage: 0,
    };
    i.header.parameters = kernel.routine.header.parameters.clone();

    let args = kernel
        .routine
        .header
        .parameters
        .iter()
        .map(|p| Operand::Parameter(p.id))
        .collect::<Vec<_>>();

    let target = i.routine(&kernel.routine, &spec.kernel, &args, None)?;
    let entry = i.block();

    i.blocks[entry.0 as usize].terminator = Terminator::Branch(Edge {
        target,
        arguments: vec![],
    });

    Module {
        kernels: vec![Kernel {
            routine: Routine {
                header: i.header,
                body: Body {
                    entry,
                    blocks: i.blocks,
                },
            },
            debug: kernel.debug.clone(),
        }],
        functions: vec![],
    }
    .validate()
    .map_err(|e| CompileError(e.to_string()))
}
