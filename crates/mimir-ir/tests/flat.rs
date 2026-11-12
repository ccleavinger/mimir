use mimir_ir::flat::*;
fn u(v: u32) -> Operand {
    Operand::Literal(Literal::U32(v))
}
fn v(id: u32) -> Operand {
    Operand::Value(ValueId(id))
}
fn c(id: u32) -> ConstantOperand {
    ConstantOperand::Constant(ConstantId(id))
}
fn arg(id: u32) -> Operand {
    Operand::Parameter(ParameterId(id))
}
fn val(id: u32, ty: ScalarType) -> Variable {
    Variable::Value {
        id: ValueId(id),
        ty: Type::Scalar(ty),
    }
}
fn eval(id: u32, op: ValueOp) -> Instruction {
    Instruction::Value {
        result: ValueId(id),
        op,
    }
}
fn edge(target: u32, arguments: Vec<Operand>) -> Edge {
    Edge {
        target: BlockId(target),
        arguments,
    }
}
fn block(id: u32, instructions: Vec<Instruction>, terminator: Terminator) -> Block {
    Block {
        id: BlockId(id),
        arguments: vec![],
        instructions,
        merge: None,
        terminator,
    }
}
fn buffer(access: Access, space: AddressSpace) -> Type {
    Type::Buffer {
        element: ScalarType::F32,
        space,
        access,
    }
}
fn base() -> Module {
    Module {
        kernels: vec![Kernel {
            debug: DebugInfo::default(),
            routine: Routine {
                header: Header::default(),
                body: Body {
                    entry: BlockId(0),
                    blocks: vec![block(0, vec![], Terminator::Return(None))],
                },
            },
        }],
        functions: vec![],
    }
}
fn r(m: &mut Module) -> &mut Routine {
    &mut m.kernels[0].routine
}
fn tuning() -> Module {
    let mut m = base();
    let h = &mut r(&mut m).header;
    h.constants = vec![ConstantId(7)];
    h.derived_constants = vec![DerivedConstant {
        id: ConstantId(8),
        op: ConstantOp::Mul,
        lhs: c(7),
        rhs: c(7),
    }];
    h.variables.push(Variable::Storage {
        id: StorageId(9),
        element: ScalarType::F32,
        space: AddressSpace::Workgroup,
        length: c(8),
    });
    m
}
#[test]
fn specializes_allocations_and_separates_variant_identity() {
    let m = tuning().validate().unwrap();
    let small = m.specialize_kernel(0, &[8]).unwrap();
    let large = m.specialize_kernel(0, &[16]).unwrap();
    assert_eq!(small.kernel.allocations[0].length, 64);
    assert_eq!(small.kernel.allocations[0].bytes, 256);
    assert_eq!(small.identity, large.identity);
    assert_ne!(
        small.variant_key(b"gpu", b"fast"),
        large.variant_key(b"gpu", b"fast")
    );
    assert_ne!(
        small.variant_key(b"gpu", b"fast"),
        small.variant_key(b"cpu", b"fast")
    );
    assert_ne!(
        small.variant_key(b"gpu", b"fast"),
        small.variant_key(b"gpu", b"precise")
    );
    assert_eq!(
        small.variant_key(b"gpu", b"fast"),
        m.specialize_kernel(0, &[8])
            .unwrap()
            .variant_key(b"gpu", b"fast")
    );
    small.kernel.check_allocation_limits(0, 256).unwrap();
    assert!(small.kernel.check_allocation_limits(0, 255).is_err());
}
#[test]
fn invalid_constant_arithmetic_and_allocations_fail_specialization() {
    let m = tuning().validate().unwrap();
    assert!(m.specialize_kernel(0, &[]).is_err());
    assert!(m.specialize_kernel(0, &[0]).is_err());
    assert!(m.specialize_kernel(0, &[u32::MAX]).is_err());
    for op in [ConstantOp::Div, ConstantOp::Rem, ConstantOp::Sub] {
        let mut m = tuning();
        let d = &mut r(&mut m).header.derived_constants[0];
        d.op = op;
        d.lhs = ConstantOperand::Literal(0);
        d.rhs = if op == ConstantOp::Sub {
            ConstantOperand::Literal(1)
        } else {
            c(7)
        };
        assert!(m.validate().unwrap().specialize_kernel(0, &[0]).is_err());
    }
}
#[test]
fn rejects_constant_cycles_and_undeclared_dependencies() {
    let mut m = tuning();
    r(&mut m).header.derived_constants[0].rhs = c(8);
    assert!(m.validate().is_err());
    let mut m = tuning();
    r(&mut m).header.derived_constants[0].rhs = c(99);
    assert!(m.validate().is_err());
}
fn diamond() -> Module {
    let mut m = base();
    let r = r(&mut m);
    r.header.variables = vec![
        val(0, ScalarType::Bool),
        val(1, ScalarType::U32),
        val(2, ScalarType::U32),
        val(3, ScalarType::U32),
    ];
    let mut head = block(
        0,
        vec![eval(
            0,
            ValueOp::Copy(Operand::Literal(Literal::Bool(true))),
        )],
        Terminator::Conditional {
            condition: v(0),
            then_edge: edge(1, vec![]),
            else_edge: edge(2, vec![]),
        },
    );
    head.merge = Some(Merge::Selection { merge: BlockId(3) });
    let mut merge = block(3, vec![], Terminator::Return(None));
    merge.arguments = vec![ValueId(3)];
    r.body.blocks = vec![
        head,
        block(
            1,
            vec![eval(1, ValueOp::Copy(u(1)))],
            Terminator::Branch(edge(3, vec![v(1)])),
        ),
        block(
            2,
            vec![eval(2, ValueOp::Copy(u(2)))],
            Terminator::Branch(edge(3, vec![v(2)])),
        ),
        merge,
    ];
    m
}
#[test]
fn branch_arguments_merge_values() {
    diamond().validate().unwrap();
}
#[test]
fn rejects_nondominating_uses_and_argument_mismatches() {
    let mut m = diamond();
    r(&mut m).body.blocks[2].terminator = Terminator::Branch(edge(3, vec![v(1)]));
    assert!(m.validate().unwrap_err().0.contains("dominated"));
    let mut m = diamond();
    r(&mut m).body.blocks[2].terminator = Terminator::Branch(edge(3, vec![]));
    assert!(m.validate().unwrap_err().0.contains("count"));
    let mut m = diamond();
    r(&mut m).body.blocks[2].terminator =
        Terminator::Branch(edge(3, vec![Operand::Literal(Literal::Bool(false))]));
    assert!(m.validate().unwrap_err().0.contains("type"));
}
#[test]
fn rejects_duplicate_missing_and_forward_ssa_definitions() {
    let mut m = base();
    r(&mut m).header.variables.push(val(0, ScalarType::U32));
    assert!(m.validate().is_err());
    let mut m = base();
    r(&mut m).header.variables.push(val(0, ScalarType::U32));
    r(&mut m).body.blocks[0].instructions =
        vec![eval(0, ValueOp::Copy(u(1))), eval(0, ValueOp::Copy(u(2)))];
    assert!(m.validate().is_err());
    let mut m = base();
    r(&mut m).header.variables = vec![val(0, ScalarType::U32), val(1, ScalarType::U32)];
    r(&mut m).body.blocks[0].instructions =
        vec![eval(0, ValueOp::Copy(v(1))), eval(1, ValueOp::Copy(u(2)))];
    assert!(m.validate().is_err());
}
fn looping() -> Module {
    let mut m = base();
    let r = r(&mut m);
    r.header.variables = vec![
        val(0, ScalarType::U32),
        val(1, ScalarType::Bool),
        val(2, ScalarType::U32),
    ];
    let mut head = block(
        1,
        vec![eval(
            1,
            ValueOp::Binary {
                op: BinaryOp::Lt,
                lhs: v(0),
                rhs: u(10),
            },
        )],
        Terminator::Conditional {
            condition: v(1),
            then_edge: edge(2, vec![]),
            else_edge: edge(4, vec![]),
        },
    );
    head.arguments = vec![ValueId(0)];
    head.merge = Some(Merge::Loop {
        merge: BlockId(4),
        continuing: BlockId(3),
    });
    r.body.blocks = vec![
        block(0, vec![], Terminator::Branch(edge(1, vec![u(0)]))),
        head,
        block(2, vec![], Terminator::Branch(edge(3, vec![]))),
        block(
            3,
            vec![eval(
                2,
                ValueOp::Binary {
                    op: BinaryOp::Add,
                    lhs: v(0),
                    rhs: u(1),
                },
            )],
            Terminator::Branch(edge(1, vec![v(2)])),
        ),
        block(4, vec![], Terminator::Return(None)),
    ];
    m
}
#[test]
fn loop_carried_ssa_and_continue_structure() {
    looping().validate().unwrap();
    let mut m = looping();
    r(&mut m).body.blocks[2].terminator = Terminator::Branch(edge(1, vec![v(0)]));
    assert!(m.validate().is_err());
}
#[test]
fn rejects_unstructured_and_unknown_control_flow() {
    let mut m = diamond();
    r(&mut m).body.blocks[0].merge = None;
    assert!(m.validate().is_err());
    let mut m = base();
    r(&mut m).body.blocks[0].terminator = Terminator::Branch(edge(99, vec![]));
    assert!(m.validate().is_err());
    let mut m = base();
    r(&mut m)
        .body
        .blocks
        .push(block(99, vec![], Terminator::Return(None)));
    assert!(m.validate().is_err());
}
fn called() -> Module {
    let mut m = tuning();
    let r = r(&mut m);
    r.header.parameters = vec![Parameter {
        id: ParameterId(1),
        ty: buffer(Access::ReadWrite, AddressSpace::Global),
    }];
    r.header.variables.push(val(1, ScalarType::F32));
    r.body.blocks[0].instructions = vec![Instruction::Call {
        result: Some(ValueId(1)),
        function: FunctionId(4),
        arguments: vec![arg(1), Operand::Storage(StorageId(9))],
        constants: vec![c(8)],
    }];
    m.functions.push(DeviceFunction {
        id: FunctionId(4),
        returns: Some(ScalarType::F32),
        debug: DebugInfo::default(),
        routine: Routine {
            header: Header {
                parameters: vec![
                    Parameter {
                        id: ParameterId(2),
                        ty: buffer(Access::Read, AddressSpace::Global),
                    },
                    Parameter {
                        id: ParameterId(3),
                        ty: buffer(Access::ReadWrite, AddressSpace::Workgroup),
                    },
                ],
                constants: vec![ConstantId(2)],
                variables: vec![val(0, ScalarType::F32)],
                ..Header::default()
            },
            body: Body {
                entry: BlockId(0),
                blocks: vec![block(
                    0,
                    vec![
                        eval(
                            0,
                            ValueOp::Load {
                                buffer: arg(2),
                                index: Operand::Constant(ConstantId(2)),
                            },
                        ),
                        Instruction::Store {
                            buffer: arg(3),
                            index: u(0),
                            value: v(0),
                        },
                    ],
                    Terminator::Return(Some(v(0))),
                )],
            },
        },
    });
    m
}
#[test]
fn functions_receive_resources_and_tuning_values_explicitly() {
    let m = called().validate().unwrap();
    let s = m.specialize_kernel(0, &[4]).unwrap();
    assert_eq!(s.functions.len(), 1);
    let (instance, function) = s.functions.iter().next().unwrap();
    assert_eq!(instance.constants, vec![16]);
    assert_eq!(function.constants[&ConstantId(2)], 16);
}
#[test]
fn rejects_call_access_return_signature_and_storage_ownership_errors() {
    let mut m = called();
    m.functions[0].routine.header.parameters[0].ty = buffer(Access::Write, AddressSpace::Global);
    assert!(m.validate().is_err()); // callee loads from write-only buffer
    let mut m = called();
    m.functions[0].returns = None;
    assert!(m.validate().is_err());
    let mut m = called();
    if let Instruction::Call { arguments, .. } = &mut r(&mut m).body.blocks[0].instructions[0] {
        arguments.pop();
    }
    assert!(m.validate().is_err());
    let mut m = called();
    m.functions[0]
        .routine
        .header
        .variables
        .push(Variable::Storage {
            id: StorageId(0),
            element: ScalarType::F32,
            space: AddressSpace::Workgroup,
            length: ConstantOperand::Literal(4),
        });
    assert!(m.validate().is_err());
}
#[test]
fn rejects_recursive_function_calls() {
    let mut m = base();
    r(&mut m).body.blocks[0]
        .instructions
        .push(Instruction::Call {
            result: None,
            function: FunctionId(0),
            arguments: vec![],
            constants: vec![],
        });
    m.functions.push(DeviceFunction {
        id: FunctionId(0),
        returns: None,
        debug: DebugInfo::default(),
        routine: m.kernels[0].routine.clone(),
    });
    assert!(m.validate().unwrap_err().0.contains("recursive"));
}
#[test]
fn typed_memory_and_numeric_operations_are_checked() {
    let mut m = base();
    r(&mut m).header.parameters.push(Parameter {
        id: ParameterId(0),
        ty: buffer(Access::Read, AddressSpace::Global),
    });
    r(&mut m).body.blocks[0]
        .instructions
        .push(Instruction::Store {
            buffer: arg(0),
            index: u(0),
            value: Operand::Literal(Literal::F32(0)),
        });
    assert!(m.validate().is_err());
    let mut m = base();
    r(&mut m).header.variables.push(val(0, ScalarType::U32));
    r(&mut m).body.blocks[0].instructions.push(eval(
        0,
        ValueOp::Binary {
            op: BinaryOp::Lt,
            lhs: u(0),
            rhs: u(1),
        },
    ));
    assert!(m.validate().is_err());
}
#[test]
fn identity_ignores_names_ids_and_incidental_table_order() {
    let original = diamond().validate().unwrap();
    let mut m = diamond();
    m.kernels[0].debug.name = "renamed".into();
    m.kernels[0]
        .debug
        .value_names
        .insert(ValueId(50), "debug".into());
    let r = r(&mut m);
    r.header.variables.reverse();
    r.body.blocks.reverse();
    for var in &mut r.header.variables {
        if let Variable::Value { id, .. } = var {
            id.0 += 50;
        }
    }
    r.body.entry.0 += 70;
    for b in &mut r.body.blocks {
        b.id.0 += 70;
        for a in &mut b.arguments {
            a.0 += 50;
        }
        for i in &mut b.instructions {
            if let Instruction::Value { result, .. } = i {
                result.0 += 50;
            }
        }
        if let Some(Merge::Selection { merge }) = &mut b.merge {
            merge.0 += 70;
        }
        match &mut b.terminator {
            Terminator::Conditional {
                condition,
                then_edge,
                else_edge,
            } => {
                if let Operand::Value(id) = condition {
                    id.0 += 50;
                }
                then_edge.target.0 += 70;
                else_edge.target.0 += 70;
            }
            Terminator::Branch(e) => {
                e.target.0 += 70;
                for a in &mut e.arguments {
                    if let Operand::Value(id) = a {
                        id.0 += 50;
                    }
                }
            }
            _ => {}
        }
    }
    let renamed = m.validate().unwrap();
    assert_eq!(
        original.kernel_hash(0).unwrap(),
        renamed.kernel_hash(0).unwrap()
    );
    assert_eq!(
        original.canonical_kernel_bytes(0).unwrap(),
        renamed.canonical_kernel_bytes(0).unwrap()
    );
}
#[test]
fn constant_identity_normalizes_ids_dependency_order_and_duplicates() {
    let mut a = tuning();
    r(&mut a).header.derived_constants.push(DerivedConstant {
        id: ConstantId(20),
        op: ConstantOp::Add,
        lhs: c(8),
        rhs: ConstantOperand::Literal(1),
    });
    let mut b = a.clone();
    let h = &mut r(&mut b).header;
    h.derived_constants.reverse();
    h.constants[0] = ConstantId(100);
    for d in &mut h.derived_constants {
        if let ConstantOperand::Constant(id) = &mut d.lhs {
            if id.0 == 7 {
                id.0 = 100;
            }
        }
        if let ConstantOperand::Constant(id) = &mut d.rhs {
            if id.0 == 7 {
                id.0 = 100;
            }
        }
    }
    h.derived_constants.push(DerivedConstant {
        id: ConstantId(30),
        op: ConstantOp::Mul,
        lhs: c(100),
        rhs: c(100),
    });
    assert_eq!(
        a.validate().unwrap().kernel_hash(0).unwrap(),
        b.validate().unwrap().kernel_hash(0).unwrap()
    );
}
#[test]
fn callee_identity_is_transitive_and_independent_of_function_ids() {
    let a = called().validate().unwrap();
    let mut b = called();
    b.functions[0].id = FunctionId(90);
    b.functions[0].debug.name = "other".into();
    if let Instruction::Call { function, .. } = &mut r(&mut b).body.blocks[0].instructions[0] {
        *function = FunctionId(90);
    }
    assert_eq!(
        a.kernel_hash(0).unwrap(),
        b.validate().unwrap().kernel_hash(0).unwrap()
    );
    let mut changed = called();
    if let Instruction::Value {
        op: ValueOp::Load { index, .. },
        ..
    } = &mut changed.functions[0].routine.body.blocks[0].instructions[0]
    {
        *index = u(0);
    }
    assert_ne!(
        a.kernel_hash(0).unwrap(),
        changed.validate().unwrap().kernel_hash(0).unwrap()
    );
}
#[test]
fn serialized_module_requires_revalidation() {
    let original = looping().validate().unwrap();
    let bytes = postcard::to_allocvec(original.module()).unwrap();
    let restored: Module = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(
        original.kernel_hash(0).unwrap(),
        restored.validate().unwrap().kernel_hash(0).unwrap()
    );
}

#[test]
fn canonical_form_is_idempotent() {
    for module in [base(), diamond(), looping(), tuning()] {
        let original = module.validate().unwrap();
        let bytes = original.canonical_kernel_bytes(0).unwrap();
        let (version, routine, dependencies): (u32, Routine, Vec<SemanticHash>) =
            postcard::from_bytes(&bytes).unwrap();
        assert_eq!(version, 1);
        assert!(dependencies.is_empty());
        let canonical = Module {
            kernels: vec![Kernel {
                routine,
                debug: DebugInfo::default(),
            }],
            functions: vec![],
        }
        .validate()
        .unwrap();
        assert_eq!(bytes, canonical.canonical_kernel_bytes(0).unwrap());
        assert_eq!(
            original.kernel_hash(0).unwrap(),
            canonical.kernel_hash(0).unwrap()
        );
    }
}

#[test]
fn function_hash_dependencies_propagate_through_multiple_calls() {
    let mut m = called();
    let mut inner = m.functions[0].clone();
    inner.id = FunctionId(5);
    m.functions[0].routine.body.blocks[0].instructions = vec![Instruction::Call {
        result: Some(ValueId(0)),
        function: FunctionId(5),
        arguments: vec![arg(2), arg(3)],
        constants: vec![c(2)],
    }];
    m.functions.push(inner);
    let original = m.clone().validate().unwrap();
    let specialized = original.specialize_kernel(0, &[4]).unwrap();
    assert_eq!(specialized.functions.len(), 2);
    assert!(specialized
        .functions
        .keys()
        .all(|i| i.constants == vec![16]));
    m.functions[1].routine.body.blocks[0]
        .instructions
        .insert(0, Instruction::Barrier);
    let changed = m.validate().unwrap();
    assert_ne!(
        original.function_hash(FunctionId(4)).unwrap(),
        changed.function_hash(FunctionId(4)).unwrap()
    );
    assert_ne!(
        original.kernel_hash(0).unwrap(),
        changed.kernel_hash(0).unwrap()
    );
}

#[test]
fn unrelated_functions_do_not_change_kernel_identity() {
    let original = base().validate().unwrap();
    let mut module = base();
    module.functions.push(DeviceFunction {
        id: FunctionId(123),
        routine: module.kernels[0].routine.clone(),
        returns: None,
        debug: DebugInfo::default(),
    });
    assert_eq!(
        original.kernel_hash(0).unwrap(),
        module.validate().unwrap().kernel_hash(0).unwrap()
    );
}

#[test]
fn returning_selections_preserve_structural_merge_without_dead_code() {
    let mut m = base();
    let mut header = block(
        0,
        vec![],
        Terminator::Conditional {
            condition: Operand::Literal(Literal::Bool(true)),
            then_edge: edge(1, vec![]),
            else_edge: edge(2, vec![]),
        },
    );
    header.merge = Some(Merge::Selection { merge: BlockId(3) });
    let routine = Routine {
        header: Header::default(),
        body: Body {
            entry: BlockId(0),
            blocks: vec![
                header,
                block(1, vec![], Terminator::Return(Some(u(1)))),
                block(2, vec![], Terminator::Return(Some(u(2)))),
                block(3, vec![], Terminator::Unreachable),
            ],
        },
    };
    m.functions.push(DeviceFunction {
        id: FunctionId(1),
        returns: Some(ScalarType::U32),
        debug: DebugInfo::default(),
        routine,
    });
    let validated = m.validate().unwrap();
    validated.function_hash(FunctionId(1)).unwrap();
    let mut invalid = validated.into_module();
    invalid.functions[0].routine.body.blocks[3]
        .instructions
        .push(Instruction::Barrier);
    assert!(invalid.validate().is_err());
}

#[test]
fn loop_with_returning_body_can_have_unreachable_continue_placeholder() {
    let mut m = looping();
    let r = r(&mut m);
    r.header.variables.pop();
    r.body.blocks[2].terminator = Terminator::Return(None);
    r.body.blocks[3].instructions.clear();
    r.body.blocks[3].terminator = Terminator::Unreachable;
    let validated = m.validate().unwrap();
    validated.kernel_hash(0).unwrap();
    let bytes = validated.canonical_kernel_bytes(0).unwrap();
    let (_, routine, _): (u32, Routine, Vec<SemanticHash>) = postcard::from_bytes(&bytes).unwrap();
    let canonical = Module {
        kernels: vec![Kernel {
            routine,
            debug: DebugInfo::default(),
        }],
        functions: vec![],
    }
    .validate()
    .unwrap();
    assert_eq!(
        validated.kernel_hash(0).unwrap(),
        canonical.kernel_hash(0).unwrap()
    );
}
