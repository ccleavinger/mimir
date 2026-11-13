use mimir_codegen_ptx::compile;
use mimir_ir::flat::*;
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

fn scalar(t: ScalarType) -> Type {
    Type::Scalar(t)
}
fn buffer(t: ScalarType, space: AddressSpace) -> Type {
    Type::Buffer {
        element: t,
        space,
        access: Access::ReadWrite,
    }
}
fn u(n: u32) -> Operand {
    Operand::Literal(Literal::U32(n))
}
fn v(n: u32) -> Operand {
    Operand::Value(ValueId(n))
}
fn edge(n: u32, arguments: Vec<Operand>) -> Edge {
    Edge {
        target: BlockId(n),
        arguments,
    }
}
fn block(
    n: u32,
    arguments: Vec<ValueId>,
    instructions: Vec<Instruction>,
    terminator: Terminator,
) -> Block {
    Block {
        id: BlockId(n),
        arguments,
        instructions,
        merge: None,
        terminator,
    }
}
struct KernelBuilder {
    routine: Routine,
}
impl KernelBuilder {
    fn new() -> Self {
        Self {
            routine: Routine {
                header: Header::default(),
                body: Body {
                    entry: BlockId(0),
                    blocks: vec![block(0, vec![], vec![], Terminator::Return(None))],
                },
            },
        }
    }
    fn parameter(&mut self, ty: Type) -> Operand {
        let id = ParameterId(self.routine.header.parameters.len() as u32);
        self.routine.header.parameters.push(Parameter { id, ty });
        Operand::Parameter(id)
    }
    fn value(&mut self, ty: Type, op: ValueOp) -> Operand {
        let id = ValueId(self.routine.header.variables.len() as u32);
        self.routine
            .header
            .variables
            .push(Variable::Value { id, ty });
        self.routine.body.blocks[0]
            .instructions
            .push(Instruction::Value { result: id, op });
        Operand::Value(id)
    }
    fn finish(self) -> Module {
        Module {
            kernels: vec![Kernel {
                routine: self.routine,
                debug: DebugInfo::default(),
            }],
            functions: vec![],
        }
    }
}

fn assemble(ptx: &str) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let assembler = std::env::var_os("PTXAS").unwrap_or_else(|| "ptxas".into());
    match Command::new(&assembler).arg("--version").output() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("PTX assembly check skipped: set PTXAS or install ptxas");
            return;
        }
        result => assert!(result.unwrap().status.success()),
    }
    let prefix = format!(
        "mimir-ptx-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let source: PathBuf = std::env::temp_dir().join(format!("{prefix}.ptx"));
    let binary = source.with_extension("cubin");
    std::fs::write(&source, ptx).unwrap();
    let result = Command::new(assembler)
        .args(["--gpu-name", "sm_50", "--output-file"])
        .arg(&binary)
        .arg(&source)
        .output()
        .unwrap();
    if !result.status.success() {
        panic!(
            "ptxas rejected {}:\n{}\n{ptx}",
            source.display(),
            String::from_utf8_lossy(&result.stderr)
        );
    }
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(binary).unwrap();
}
fn generate(module: Module, tuning: &[u32]) -> String {
    let module = module.validate().unwrap();
    let spec = module.specialize_kernel(0, tuning).unwrap();
    let ptx = compile(&module, &spec).unwrap();
    assert!(!ptx.contains(".extern"));
    assemble(&ptx);
    ptx
}

#[test]
fn every_scalar_operation_and_intrinsic_assembles() {
    use ScalarType::*;
    let mut k = KernelBuilder::new();
    for t in [Bool, I32, U32, F32] {
        let a = k.parameter(scalar(t));
        let b = k.parameter(scalar(t));
        let output = k.parameter(buffer(t, AddressSpace::Global));
        let mut results = vec![k.value(scalar(t), ValueOp::Copy(a))];
        let bool_output = k.parameter(buffer(Bool, AddressSpace::Global));
        for op in [
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Rem,
            BinaryOp::And,
            BinaryOp::Or,
        ] {
            if t == Bool
                && !matches!(
                    op,
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::And | BinaryOp::Or
                )
            {
                continue;
            }
            if t != Bool && matches!(op, BinaryOp::And | BinaryOp::Or) {
                continue;
            }
            let comparison = matches!(
                op,
                BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or
            );
            let result = k.value(
                scalar(if comparison { Bool } else { t }),
                ValueOp::Binary { op, lhs: a, rhs: b },
            );
            if comparison {
                k.routine.body.blocks[0]
                    .instructions
                    .push(Instruction::Store {
                        buffer: bool_output,
                        index: u(0),
                        value: result,
                    });
            } else {
                results.push(result);
            }
        }
        if t == Bool || t == I32 || t == F32 {
            results.push(k.value(
                scalar(t),
                ValueOp::Unary {
                    op: if t == Bool {
                        UnaryOp::Not
                    } else {
                        UnaryOp::Neg
                    },
                    value: a,
                },
            ));
        }
        if t != Bool {
            for to in [I32, U32, F32] {
                k.value(scalar(to), ValueOp::Cast { value: a, to });
            }
            for op in [BinaryIntrinsic::Min, BinaryIntrinsic::Max] {
                results.push(k.value(scalar(t), ValueOp::BinaryIntrinsic { op, args: [a, b] }));
            }
            results.push(k.value(
                scalar(t),
                ValueOp::TernaryIntrinsic {
                    op: TernaryIntrinsic::Clamp,
                    args: [a, a, b],
                },
            ));
        }
        if t == F32 {
            for op in [
                UnaryIntrinsic::Sin,
                UnaryIntrinsic::Cos,
                UnaryIntrinsic::Tan,
                UnaryIntrinsic::Asin,
                UnaryIntrinsic::Acos,
                UnaryIntrinsic::Atan,
                UnaryIntrinsic::Sinh,
                UnaryIntrinsic::Cosh,
                UnaryIntrinsic::Tanh,
                UnaryIntrinsic::Asinh,
                UnaryIntrinsic::Acosh,
                UnaryIntrinsic::Atanh,
                UnaryIntrinsic::Exp,
                UnaryIntrinsic::Log,
                UnaryIntrinsic::Exp2,
                UnaryIntrinsic::Log2,
                UnaryIntrinsic::Sqrt,
                UnaryIntrinsic::InverseSqrt,
                UnaryIntrinsic::Floor,
                UnaryIntrinsic::Ceil,
            ] {
                results.push(k.value(scalar(t), ValueOp::UnaryIntrinsic { op, value: a }));
            }
            for op in [BinaryIntrinsic::Atan2, BinaryIntrinsic::Pow] {
                results.push(k.value(scalar(t), ValueOp::BinaryIntrinsic { op, args: [a, b] }));
            }
            for op in [TernaryIntrinsic::Mix, TernaryIntrinsic::Fma] {
                results.push(k.value(
                    scalar(t),
                    ValueOp::TernaryIntrinsic {
                        op,
                        args: [a, b, b],
                    },
                ));
            }
        }
        for (i, result) in results.iter().enumerate() {
            k.routine.body.blocks[0]
                .instructions
                .push(Instruction::Store {
                    buffer: output,
                    index: u(i as u32),
                    value: *result,
                });
        }
    }
    let ptx = generate(k.finish(), &[]);
    assert!(ptx.contains("setp.neu.f32"));
    assert!(ptx.contains("div.rn.f32"));
    assert!(ptx.contains("copysign.f32"));
}

#[test]
fn memory_spaces_boolean_slots_builtins_and_specialized_storage() {
    let mut k = KernelBuilder::new();
    k.routine.header.constants.push(ConstantId(7));
    k.routine.header.derived_constants.push(DerivedConstant {
        id: ConstantId(8),
        op: ConstantOp::Mul,
        lhs: ConstantOperand::Constant(ConstantId(7)),
        rhs: ConstantOperand::Literal(2),
    });
    for (i, (ty, space)) in [
        ScalarType::Bool,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::F32,
    ]
    .into_iter()
    .flat_map(|t| {
        [
            AddressSpace::Global,
            AddressSpace::Local,
            AddressSpace::Workgroup,
        ]
        .into_iter()
        .map(move |s| (t, s))
    })
    .enumerate()
    {
        let buf = if space == AddressSpace::Global {
            k.parameter(buffer(ty, space))
        } else {
            let id = StorageId(i as u32);
            k.routine.header.variables.push(Variable::Storage {
                id,
                element: ty,
                space,
                length: ConstantOperand::Constant(ConstantId(8)),
            });
            Operand::Storage(id)
        };
        let alias = k.value(buffer(ty, space), ValueOp::Copy(buf));
        let value = Operand::Literal(match ty {
            ScalarType::Bool => Literal::Bool(true),
            ScalarType::I32 => Literal::I32(-3),
            ScalarType::U32 => Literal::U32(8),
            ScalarType::F32 => Literal::f32(-0.0),
        });
        k.routine.body.blocks[0]
            .instructions
            .push(Instruction::Store {
                buffer: alias,
                index: u(0),
                value,
            });
        let loaded = k.value(
            scalar(ty),
            ValueOp::Load {
                buffer: alias,
                index: u(0),
            },
        );
        k.routine.body.blocks[0]
            .instructions
            .push(Instruction::Store {
                buffer: alias,
                index: u(1),
                value: loaded,
            });
    }
    for builtin in [
        Builtin::GlobalId,
        Builtin::LocalId,
        Builtin::WorkgroupId,
        Builtin::WorkgroupSize,
    ] {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            k.value(scalar(ScalarType::U32), ValueOp::Builtin { builtin, axis });
        }
    }
    k.value(
        scalar(ScalarType::U32),
        ValueOp::Copy(Operand::Constant(ConstantId(8))),
    );
    k.routine.body.blocks[0]
        .instructions
        .push(Instruction::Barrier);
    let ptx = generate(k.finish(), &[16]);
    assert!(ptx.contains(".align 4 .b8 storage1[128]"));
    for space in ["global", "local", "shared"] {
        assert!(ptx.contains(&format!("ld.{space}.u32")));
    }
    assert!(ptx.contains("mad.lo.u32"));
}

#[test]
fn loop_edges_copy_cycles_and_nonphysical_entry_order() {
    let mut k = KernelBuilder::new();
    k.routine.header.variables = vec![
        Variable::Value {
            id: ValueId(0),
            ty: scalar(ScalarType::U32),
        },
        Variable::Value {
            id: ValueId(1),
            ty: scalar(ScalarType::U32),
        },
        Variable::Value {
            id: ValueId(2),
            ty: scalar(ScalarType::Bool),
        },
    ];
    let cond = k.parameter(scalar(ScalarType::Bool));
    let out = k.parameter(buffer(ScalarType::U32, AddressSpace::Global));
    let mut head = block(
        1,
        vec![ValueId(0), ValueId(1)],
        vec![Instruction::Value {
            result: ValueId(2),
            op: ValueOp::Copy(cond),
        }],
        Terminator::Conditional {
            condition: v(2),
            then_edge: edge(2, vec![]),
            else_edge: edge(3, vec![]),
        },
    );
    head.merge = Some(Merge::Loop {
        merge: BlockId(3),
        continuing: BlockId(2),
    });
    k.routine.body.blocks = vec![
        block(
            3,
            vec![],
            vec![
                Instruction::Store {
                    buffer: out,
                    index: u(0),
                    value: v(0),
                },
                Instruction::Store {
                    buffer: out,
                    index: u(1),
                    value: v(1),
                },
            ],
            Terminator::Return(None),
        ),
        block(
            0,
            vec![],
            vec![],
            Terminator::Branch(edge(1, vec![u(7), u(11)])),
        ),
        head,
        block(
            2,
            vec![],
            vec![],
            Terminator::Branch(edge(1, vec![v(1), v(0)])),
        ),
    ];
    let ptx = generate(k.finish(), &[]);
    assert!(ptx.contains("bra block0;\nblock3:"));
}

#[test]
fn specialization_mismatch_is_reported_without_panicking() {
    let module = KernelBuilder::new().finish().validate().unwrap();
    let mut spec = module.specialize_kernel(0, &[]).unwrap();
    spec.kernel.constants.insert(ConstantId(99), 4);
    assert!(compile(&module, &spec).is_err());
    spec.kernel_index = 99;
    assert!(compile(&module, &spec).is_err());
}

#[test]
fn specialized_nested_device_calls_in_loop_headers_and_buffer_parameters() {
    let mut k = KernelBuilder::new();
    let out = k.parameter(buffer(ScalarType::Bool, AddressSpace::Global));
    k.routine.header.variables.push(Variable::Value {
        id: ValueId(0),
        ty: scalar(ScalarType::Bool),
    });
    let mut head = block(
        1,
        vec![],
        vec![Instruction::Call {
            result: Some(ValueId(0)),
            function: FunctionId(0),
            arguments: vec![Operand::Literal(Literal::Bool(false)), out],
            constants: vec![ConstantOperand::Literal(3)],
        }],
        Terminator::Conditional {
            condition: v(0),
            then_edge: edge(2, vec![]),
            else_edge: edge(3, vec![]),
        },
    );
    head.merge = Some(Merge::Loop {
        merge: BlockId(3),
        continuing: BlockId(2),
    });
    k.routine.body.blocks = vec![
        block(0, vec![], vec![], Terminator::Branch(edge(1, vec![]))),
        head,
        block(2, vec![], vec![], Terminator::Branch(edge(1, vec![]))),
        block(3, vec![], vec![], Terminator::Return(None)),
    ];
    let mut module = k.finish();
    let mut f = KernelBuilder::new();
    let flag = f.parameter(scalar(ScalarType::Bool));
    let output = f.parameter(buffer(ScalarType::Bool, AddressSpace::Global));
    f.routine.header.constants = vec![ConstantId(0)];
    f.routine.header.variables = vec![Variable::Value {
        id: ValueId(0),
        ty: scalar(ScalarType::Bool),
    }];
    f.routine.body.blocks[0].instructions = vec![
        Instruction::Call {
            result: Some(ValueId(0)),
            function: FunctionId(1),
            arguments: vec![flag],
            constants: vec![],
        },
        Instruction::Store {
            buffer: output,
            index: Operand::Constant(ConstantId(0)),
            value: v(0),
        },
    ];
    f.routine.body.blocks[0].terminator = Terminator::Return(Some(v(0)));
    module.functions.push(DeviceFunction {
        id: FunctionId(0),
        routine: f.routine,
        returns: Some(ScalarType::Bool),
        debug: DebugInfo::default(),
    });
    let mut leaf = KernelBuilder::new();
    let flag = leaf.parameter(scalar(ScalarType::Bool));
    leaf.routine.body.blocks[0].terminator = Terminator::Return(Some(flag));
    module.functions.push(DeviceFunction {
        id: FunctionId(1),
        routine: leaf.routine,
        returns: Some(ScalarType::Bool),
        debug: DebugInfo::default(),
    });
    let ptx = generate(module, &[]);
    assert!(ptx.contains("call.uni (returned), mimir_function0"));
    assert!(ptx.contains(".func (.param .u32 return_value)"));
}

#[test]
fn device_calls_preserve_local_and_shared_pointers_and_specializations() {
    let mut k = KernelBuilder::new();
    let global = k.parameter(buffer(ScalarType::F32, AddressSpace::Global));
    let mut functions = Vec::new();
    for (n, space) in [
        AddressSpace::Global,
        AddressSpace::Local,
        AddressSpace::Workgroup,
    ]
    .into_iter()
    .enumerate()
    {
        let actual = if space == AddressSpace::Global {
            global
        } else {
            k.routine.header.variables.push(Variable::Storage {
                id: StorageId(n as u32),
                element: ScalarType::F32,
                space,
                length: ConstantOperand::Literal(4),
            });
            Operand::Storage(StorageId(n as u32))
        };
        let mut f = KernelBuilder::new();
        let buf = f.parameter(buffer(ScalarType::F32, space));
        f.routine.header.constants = vec![ConstantId(0)];
        f.routine.body.blocks[0]
            .instructions
            .push(Instruction::Store {
                buffer: buf,
                index: Operand::Constant(ConstantId(0)),
                value: Operand::Literal(Literal::f32(2.5)),
            });
        functions.push(DeviceFunction {
            id: FunctionId(n as u32),
            routine: f.routine,
            returns: None,
            debug: DebugInfo::default(),
        });
        for value in [0, 1] {
            k.routine.body.blocks[0]
                .instructions
                .push(Instruction::Call {
                    result: None,
                    function: FunctionId(n as u32),
                    arguments: vec![actual],
                    constants: vec![ConstantOperand::Literal(value)],
                });
        }
    }
    // Also cover no-parameter void function calls and unreachable placeholders.
    let mut empty = KernelBuilder::new();
    empty.routine.body.blocks[0].merge = Some(Merge::Selection { merge: BlockId(8) });
    empty.routine.body.blocks[0].terminator = Terminator::Conditional {
        condition: Operand::Literal(Literal::Bool(true)),
        then_edge: edge(1, vec![]),
        else_edge: edge(2, vec![]),
    };
    empty.routine.body.blocks.extend([
        block(1, vec![], vec![], Terminator::Return(None)),
        block(2, vec![], vec![], Terminator::Return(None)),
        block(8, vec![], vec![], Terminator::Unreachable),
    ]);
    functions.push(DeviceFunction {
        id: FunctionId(9),
        routine: empty.routine,
        returns: None,
        debug: DebugInfo::default(),
    });
    k.routine.body.blocks[0]
        .instructions
        .push(Instruction::Call {
            result: None,
            function: FunctionId(9),
            arguments: vec![],
            constants: vec![],
        });
    let mut module = k.finish();
    module.functions = functions;
    let ptx = generate(module, &[]);
    assert_eq!(ptx.matches(".func ").count(), 14); // 7 definitions and 7 prototypes
    assert!(ptx.contains("cvta.to.local.u64"));
    assert!(ptx.contains("cvta.to.shared.u64"));
}

#[test]
fn scalar_return_types_and_buffer_block_arguments() {
    let mut k = KernelBuilder::new();
    let mut functions = vec![];
    for (i, ty) in [
        ScalarType::Bool,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::F32,
    ]
    .into_iter()
    .enumerate()
    {
        let actual = k.parameter(scalar(ty));
        let out = k.parameter(buffer(ty, AddressSpace::Global));
        let id = ValueId(k.routine.header.variables.len() as u32);
        k.routine
            .header
            .variables
            .push(Variable::Value { id, ty: scalar(ty) });
        k.routine.body.blocks[0].instructions.extend([
            Instruction::Call {
                result: Some(id),
                function: FunctionId(i as u32),
                arguments: vec![actual],
                constants: vec![],
            },
            Instruction::Store {
                buffer: out,
                index: u(0),
                value: Operand::Value(id),
            },
        ]);
        let mut f = KernelBuilder::new();
        let parameter = f.parameter(scalar(ty));
        f.routine.body.blocks[0].terminator = Terminator::Return(Some(parameter));
        functions.push(DeviceFunction {
            id: FunctionId(i as u32),
            routine: f.routine,
            returns: Some(ty),
            debug: DebugInfo::default(),
        });
    }
    let left = k.parameter(buffer(ScalarType::U32, AddressSpace::Global));
    let right = k.parameter(buffer(ScalarType::U32, AddressSpace::Global));
    let flag = k.parameter(scalar(ScalarType::Bool));
    let id = ValueId(k.routine.header.variables.len() as u32);
    k.routine.header.variables.push(Variable::Value {
        id,
        ty: buffer(ScalarType::U32, AddressSpace::Global),
    });
    k.routine.body.blocks[0].merge = Some(Merge::Selection { merge: BlockId(3) });
    k.routine.body.blocks[0].terminator = Terminator::Conditional {
        condition: flag,
        then_edge: edge(1, vec![]),
        else_edge: edge(2, vec![]),
    };
    k.routine.body.blocks.extend([
        block(1, vec![], vec![], Terminator::Branch(edge(3, vec![left]))),
        block(2, vec![], vec![], Terminator::Branch(edge(3, vec![right]))),
        block(
            3,
            vec![id],
            vec![Instruction::Store {
                buffer: Operand::Value(id),
                index: u(0),
                value: u(42),
            }],
            Terminator::Return(None),
        ),
    ]);
    let mut module = k.finish();
    module.functions = functions;
    generate(module, &[]);
}
