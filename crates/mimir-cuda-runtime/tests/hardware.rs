#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    target_pointer_width = "64"
))]
//! These tests require a real CUDA device/driver. They are intentionally ignored
//! on CPU-only development machines; the FFI state machine has separate mocks.
use mimir_ir::flat::*;
use mimir_runtime_core::{Argument, DeviceBackend, KernelHandle, Launch};
use std::sync::Arc;
fn value(id: u32, ty: ScalarType) -> Variable {
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
fn p(id: u32) -> Operand {
    Operand::Parameter(ParameterId(id))
}
fn v(id: u32) -> Operand {
    Operand::Value(ValueId(id))
}
fn u(n: u32) -> Operand {
    Operand::Literal(Literal::U32(n))
}
fn buffer(element: ScalarType, space: AddressSpace) -> Type {
    Type::Buffer {
        element,
        space,
        access: Access::ReadWrite,
    }
}
fn routine(header: Header, instructions: Vec<Instruction>) -> Routine {
    Routine {
        header,
        body: Body {
            entry: BlockId(0),
            blocks: vec![Block {
                id: BlockId(0),
                arguments: vec![],
                instructions,
                merge: None,
                terminator: Terminator::Return(None),
            }],
        },
    }
}
fn handle(routine: Routine, functions: Vec<DeviceFunction>) -> KernelHandle {
    KernelHandle::new(
        Arc::new(
            Module {
                kernels: vec![Kernel {
                    routine,
                    debug: DebugInfo::default(),
                }],
                functions,
            }
            .validate()
            .unwrap(),
        ),
        0,
    )
    .unwrap()
}
fn device() -> Arc<dyn DeviceBackend> {
    mimir_cuda_runtime::discover()
        .unwrap()
        .into_iter()
        .next()
        .expect("compatible NVIDIA CUDA device")
}

fn vector_kernel() -> KernelHandle {
    handle(
        routine(
            Header {
                parameters: vec![
                    Parameter {
                        id: ParameterId(0),
                        ty: buffer(ScalarType::F32, AddressSpace::Global),
                    },
                    Parameter {
                        id: ParameterId(1),
                        ty: Type::Scalar(ScalarType::F32),
                    },
                    Parameter {
                        id: ParameterId(2),
                        ty: buffer(ScalarType::F32, AddressSpace::Global),
                    },
                ],
                variables: vec![
                    value(0, ScalarType::U32),
                    value(1, ScalarType::F32),
                    value(2, ScalarType::F32),
                ],
                ..Header::default()
            },
            vec![
                eval(
                    0,
                    ValueOp::Builtin {
                        builtin: Builtin::GlobalId,
                        axis: Axis::X,
                    },
                ),
                eval(
                    1,
                    ValueOp::Load {
                        buffer: p(0),
                        index: v(0),
                    },
                ),
                eval(
                    2,
                    ValueOp::Binary {
                        op: BinaryOp::Mul,
                        lhs: v(1),
                        rhs: p(1),
                    },
                ),
                Instruction::Store {
                    buffer: p(2),
                    index: v(0),
                    value: v(2),
                },
            ],
        ),
        vec![],
    )
}

#[test]
#[ignore = "requires an NVIDIA CUDA device and driver"]
fn cuda_executes_vector_kernel_and_reuses_it_across_block_sizes_and_threads() {
    let device = device();
    let kernel = vector_kernel();
    let input = device
        .clone()
        .buffer(
            ScalarType::F32,
            &(1..=8).map(|n| (n as f32).to_bits()).collect::<Vec<_>>(),
        )
        .unwrap();
    let outputs: Vec<_> = (0..4)
        .map(|_| device.clone().buffer(ScalarType::F32, &[0; 8]).unwrap())
        .collect();
    std::thread::scope(|scope| {
        for (i, output) in outputs.iter().enumerate() {
            let device = &device;
            let input = &input;
            let kernel = &kernel;
            scope.spawn(move || {
                let arguments = [
                    Argument::Buffer(input.as_ref()),
                    Argument::Scalar(Literal::f32(2.0)),
                    Argument::Buffer(output.as_ref()),
                ];
                let block = if i % 2 == 0 { 2 } else { 4 };
                let launch = Launch::new(
                    device.as_ref(),
                    kernel,
                    &[],
                    [8 / block, 1, 1],
                    [block, 1, 1],
                    &arguments,
                )
                .unwrap();
                // Eight threads access exactly the eight input and output slots.
                unsafe {
                    device.launch(&launch).unwrap();
                }
            });
        }
    });
    for output in outputs {
        assert_eq!(
            device.read(output.as_ref()).unwrap(),
            (1..=8)
                .map(|n| (n as f32 * 2.0).to_bits())
                .collect::<Vec<_>>()
        );
    }
}

fn shared_kernel() -> KernelHandle {
    let function = DeviceFunction {
        id: FunctionId(0),
        debug: DebugInfo::default(),
        returns: Some(ScalarType::U32),
        routine: {
            let mut r = routine(
                Header {
                    parameters: vec![Parameter {
                        id: ParameterId(0),
                        ty: Type::Scalar(ScalarType::U32),
                    }],
                    constants: vec![ConstantId(0)],
                    variables: vec![value(0, ScalarType::U32)],
                    ..Header::default()
                },
                vec![eval(
                    0,
                    ValueOp::Binary {
                        op: BinaryOp::Mul,
                        lhs: p(0),
                        rhs: Operand::Constant(ConstantId(0)),
                    },
                )],
            );
            r.body.blocks[0].terminator = Terminator::Return(Some(v(0)));
            r
        },
    };
    handle(
        routine(
            Header {
                parameters: vec![Parameter {
                    id: ParameterId(0),
                    ty: buffer(ScalarType::U32, AddressSpace::Global),
                }],
                constants: vec![ConstantId(0)],
                variables: (0..6)
                    .map(|id| value(id, ScalarType::U32))
                    .chain([
                        Variable::Storage {
                            id: StorageId(0),
                            element: ScalarType::U32,
                            space: AddressSpace::Workgroup,
                            length: ConstantOperand::Constant(ConstantId(0)),
                        },
                        Variable::Storage {
                            id: StorageId(1),
                            element: ScalarType::U32,
                            space: AddressSpace::Local,
                            length: ConstantOperand::Literal(1),
                        },
                    ])
                    .collect(),
                ..Header::default()
            },
            vec![
                eval(
                    0,
                    ValueOp::Builtin {
                        builtin: Builtin::LocalId,
                        axis: Axis::X,
                    },
                ),
                Instruction::Call {
                    result: Some(ValueId(1)),
                    function: FunctionId(0),
                    arguments: vec![v(0)],
                    constants: vec![ConstantOperand::Literal(2)],
                },
                Instruction::Store {
                    buffer: Operand::Storage(StorageId(1)),
                    index: u(0),
                    value: v(1),
                },
                eval(
                    2,
                    ValueOp::Load {
                        buffer: Operand::Storage(StorageId(1)),
                        index: u(0),
                    },
                ),
                Instruction::Store {
                    buffer: Operand::Storage(StorageId(0)),
                    index: v(0),
                    value: v(2),
                },
                Instruction::Barrier,
                eval(
                    3,
                    ValueOp::Binary {
                        op: BinaryOp::Add,
                        lhs: v(0),
                        rhs: u(1),
                    },
                ),
                eval(
                    4,
                    ValueOp::Binary {
                        op: BinaryOp::Rem,
                        lhs: v(3),
                        rhs: Operand::Constant(ConstantId(0)),
                    },
                ),
                eval(
                    5,
                    ValueOp::Load {
                        buffer: Operand::Storage(StorageId(0)),
                        index: v(4),
                    },
                ),
                Instruction::Store {
                    buffer: p(0),
                    index: v(0),
                    value: v(5),
                },
            ],
        ),
        vec![function],
    )
}

#[test]
#[ignore = "requires an NVIDIA CUDA device and driver"]
fn cuda_shared_local_storage_barriers_and_device_calls() {
    let device = device();
    let kernel = shared_kernel();
    let output = device.clone().buffer(ScalarType::U32, &[0; 4]).unwrap();
    let arguments = [Argument::Buffer(output.as_ref())];
    let launch = Launch::new(
        device.as_ref(),
        &kernel,
        &[4],
        [1; 3],
        [4, 1, 1],
        &arguments,
    )
    .unwrap();
    // All threads reach the barrier and every memory access stays in bounds.
    unsafe {
        device.launch(&launch).unwrap();
    }
    assert_eq!(device.read(output.as_ref()).unwrap(), vec![2, 4, 6, 0]);
}

#[test]
fn hardware_kernels_compile_and_optionally_assemble_without_a_gpu() {
    for (i, (kernel, tuning)) in [(vector_kernel(), vec![]), (shared_kernel(), vec![4])]
        .into_iter()
        .enumerate()
    {
        let specialized = kernel
            .ir()
            .specialize_kernel(kernel.index(), &tuning)
            .unwrap();
        let ptx = mimir_codegen_ptx::compile(kernel.ir(), &specialized).unwrap();
        assert!(ptx.contains(".visible .entry mimir_kernel"));
        if let Some(assembler) = std::env::var_os("PTXAS") {
            let source = std::env::temp_dir()
                .join(format!("mimir-cuda-kernel-{}-{i}.ptx", std::process::id()));
            let binary = source.with_extension("cubin");
            std::fs::write(&source, &ptx).unwrap();
            let result = std::process::Command::new(assembler)
                .args(["--gpu-name", "sm_50", "--output-file"])
                .arg(&binary)
                .arg(&source)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            std::fs::remove_file(source).unwrap();
            std::fs::remove_file(binary).unwrap();
        }
    }
}
