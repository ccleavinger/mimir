//! Vulkan SPIR-V generation from validated, specialized flat IR.
mod inline;
mod layout;
use mimir_ir::flat::*;
use rspirv::{binary::Assemble, dr::Builder, spirv::*};
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
#[error("SPIR-V compilation: {0}")]
pub struct CompileError(pub String);

type Result<T> = std::result::Result<T, CompileError>;

fn build<T>(r: std::result::Result<T, rspirv::dr::Error>) -> Result<T> {
    r.map_err(|e| CompileError(e.to_string()))
}

struct Compiler {
    b: Builder,
    scalar: BTreeMap<u8, u32>,
    params: BTreeMap<ParameterId, (u32, Type)>,
    values: BTreeMap<ValueId, (u32, Type)>,
    storage: BTreeMap<StorageId, (u32, Type)>,
    constants: BTreeMap<ConstantId, u32>,
    labels: BTreeMap<BlockId, u32>,
    builtins: BTreeMap<u8, u32>,
    workgroup: [u32; 3],
    ext: u32,
}

fn key(t: ScalarType) -> u8 {
    match t {
        ScalarType::Bool => 0,
        ScalarType::I32 => 1,
        ScalarType::U32 => 2,
        ScalarType::F32 => 3,
    }
}

fn storage_class(s: AddressSpace) -> StorageClass {
    match s {
        AddressSpace::Global => StorageClass::StorageBuffer,
        AddressSpace::Local => StorageClass::Function,
        AddressSpace::Workgroup => StorageClass::Workgroup,
    }
}

impl Compiler {
    fn ty(&self, t: ScalarType) -> u32 {
        self.scalar[&key(t)]
    }

    fn literal(&mut self, l: Literal) -> u32 {
        match l {
            Literal::Bool(true) => self.b.constant_true(self.ty(ScalarType::Bool)),
            Literal::Bool(false) => self.b.constant_false(self.ty(ScalarType::Bool)),
            Literal::I32(v) => self.b.constant_bit32(self.ty(ScalarType::I32), v as u32),
            Literal::U32(v) => self.b.constant_bit32(self.ty(ScalarType::U32), v),
            Literal::F32(v) => self.b.constant_bit32(self.ty(ScalarType::F32), v),
        }
    }

    fn operand(&mut self, o: Operand) -> (u32, Type) {
        match o {
            Operand::Parameter(i) => self.params[&i],
            Operand::Value(i) => self.values[&i],
            Operand::Storage(i) => self.storage[&i],
            Operand::Constant(i) => {
                let v = self.constants[&i];
                (self.literal(Literal::U32(v)), Type::Scalar(ScalarType::U32))
            }
            Operand::Literal(l) => (self.literal(l), Type::Scalar(l.ty())),
        }
    }

    fn pointer(&mut self, buffer: Operand, index: Operand) -> Result<(u32, ScalarType)> {
        let (buffer, t) = self.operand(buffer);
        let (index, _) = self.operand(index);

        let Type::Buffer { element, space, .. } = t else {
            return Err(CompileError("expected buffer".into()));
        };

        let mem = if element == ScalarType::Bool {
            ScalarType::U32
        } else {
            element
        };

        let ptr = self
            .b
            .type_pointer(None, storage_class(space), self.ty(mem));

        let indices = if space == AddressSpace::Global {
            vec![self.literal(Literal::U32(0)), index]
        } else {
            vec![index]
        };

        Ok((
            build(self.b.access_chain(ptr, None, buffer, indices))?,
            element,
        ))
    }

    fn binary(
        &mut self,
        id: u32,
        t: ScalarType,
        op: BinaryOp,
        a: u32,
        b: u32,
        input: ScalarType,
    ) -> Result<()> {
        let rt = self.ty(t);
        let f = input == ScalarType::F32;
        let s = input == ScalarType::I32;

        macro_rules! emit {
            ($method:ident) => {
                build(self.b.$method(rt, Some(id), a, b))?
            };
        }

        match op {
            BinaryOp::Add => {
                if f {
                    emit!(f_add);
                } else {
                    emit!(i_add);
                }
            }
            BinaryOp::Sub => {
                if f {
                    emit!(f_sub);
                } else {
                    emit!(i_sub);
                }
            }
            BinaryOp::Mul => {
                if f {
                    emit!(f_mul);
                } else {
                    emit!(i_mul);
                }
            }
            BinaryOp::Div => {
                if f {
                    emit!(f_div);
                } else if s {
                    emit!(s_div);
                } else {
                    emit!(u_div);
                }
            }
            BinaryOp::Rem => {
                if f {
                    emit!(f_rem);
                } else if s {
                    emit!(s_rem);
                } else {
                    emit!(u_mod);
                }
            }
            BinaryOp::Eq => {
                if f {
                    emit!(f_ord_equal);
                } else if input == ScalarType::Bool {
                    emit!(logical_equal);
                } else {
                    emit!(i_equal);
                }
            }
            BinaryOp::Ne => {
                if f {
                    emit!(f_unord_not_equal);
                } else if input == ScalarType::Bool {
                    emit!(logical_not_equal);
                } else {
                    emit!(i_not_equal);
                }
            }
            BinaryOp::Lt => {
                if f {
                    emit!(f_ord_less_than);
                } else if s {
                    emit!(s_less_than);
                } else {
                    emit!(u_less_than);
                }
            }
            BinaryOp::Le => {
                if f {
                    emit!(f_ord_less_than_equal);
                } else if s {
                    emit!(s_less_than_equal);
                } else {
                    emit!(u_less_than_equal);
                }
            }
            BinaryOp::Gt => {
                if f {
                    emit!(f_ord_greater_than);
                } else if s {
                    emit!(s_greater_than);
                } else {
                    emit!(u_greater_than);
                }
            }
            BinaryOp::Ge => {
                if f {
                    emit!(f_ord_greater_than_equal);
                } else if s {
                    emit!(s_greater_than_equal);
                } else {
                    emit!(u_greater_than_equal);
                }
            }
            BinaryOp::And => {
                emit!(logical_and);
            }
            BinaryOp::Or => {
                emit!(logical_or);
            }
        }

        Ok(())
    }

    fn instruction(&mut self, i: &Instruction) -> Result<()> {
        match i {
            Instruction::Value { result, op } => {
                let (id, t) = self.values[result];
                let Type::Scalar(t) = t else {
                    if let ValueOp::Copy(source) = op {
                        let source = self.operand(*source);
                        self.values.insert(*result, source);
                        return Ok(());
                    }
                    return Err(CompileError("unsupported buffer-valued operation".into()));
                };
                let rt = self.ty(t);
                match op {
                    ValueOp::Copy(a) => {
                        let (a, _) = self.operand(*a);
                        build(self.b.copy_object(rt, Some(id), a))?;
                    }
                    ValueOp::Binary { op, lhs, rhs } => {
                        let (a, at) = self.operand(*lhs);
                        let (b, _) = self.operand(*rhs);
                        let Type::Scalar(input) = at else {
                            unreachable!()
                        };
                        self.binary(id, t, *op, a, b, input)?;
                    }
                    ValueOp::Unary { op, value } => {
                        let (a, _) = self.operand(*value);
                        match op {
                            UnaryOp::Not => {
                                build(self.b.logical_not(rt, Some(id), a))?;
                            }
                            UnaryOp::Neg => {
                                if t == ScalarType::F32 {
                                    build(self.b.f_negate(rt, Some(id), a))?;
                                } else {
                                    build(self.b.s_negate(rt, Some(id), a))?;
                                }
                            }
                        }
                    }
                    ValueOp::Load { buffer, index } => {
                        let (ptr, element) = self.pointer(*buffer, *index)?;
                        if element == ScalarType::Bool {
                            let loaded =
                                build(self.b.load(self.ty(ScalarType::U32), None, ptr, None, []))?;
                            let zero = self.literal(Literal::U32(0));
                            build(self.b.i_not_equal(rt, Some(id), loaded, zero))?;
                        } else {
                            build(self.b.load(rt, Some(id), ptr, None, []))?;
                        }
                    }
                    ValueOp::Cast { value, to } => {
                        let (a, from) = self.operand(*value);
                        let Type::Scalar(from) = from else {
                            unreachable!()
                        };
                        if from == *to {
                            build(self.b.copy_object(rt, Some(id), a))?;
                        } else if from == ScalarType::F32 {
                            if *to == ScalarType::I32 {
                                build(self.b.convert_f_to_s(rt, Some(id), a))?;
                            } else {
                                build(self.b.convert_f_to_u(rt, Some(id), a))?;
                            }
                        } else if *to == ScalarType::F32 {
                            if from == ScalarType::I32 {
                                build(self.b.convert_s_to_f(rt, Some(id), a))?;
                            } else {
                                build(self.b.convert_u_to_f(rt, Some(id), a))?;
                            }
                        } else {
                            build(self.b.bitcast(rt, Some(id), a))?;
                        }
                    }
                    ValueOp::Builtin { builtin, axis } => {
                        if *builtin == Builtin::WorkgroupSize {
                            let axis = match axis {
                                Axis::X => 0,
                                Axis::Y => 1,
                                Axis::Z => 2,
                            };
                            let value = self.literal(Literal::U32(self.workgroup[axis]));
                            build(self.b.copy_object(rt, Some(id), value))?;
                            return Ok(());
                        }
                        let k = match builtin {
                            Builtin::GlobalId => 0,
                            Builtin::WorkgroupId => 1,
                            Builtin::LocalId => 2,
                            Builtin::WorkgroupSize => 3,
                        };
                        let vec = self.b.type_vector(self.ty(ScalarType::U32), 3);
                        let v = build(self.b.load(vec, None, self.builtins[&k], None, []))?;
                        let axis = match axis {
                            Axis::X => 0,
                            Axis::Y => 1,
                            Axis::Z => 2,
                        };
                        build(self.b.composite_extract(rt, Some(id), v, [axis]))?;
                    }
                    ValueOp::UnaryIntrinsic { op, value } => {
                        let (a, _) = self.operand(*value);
                        let code = match op {
                            UnaryIntrinsic::Sin => GlslStd450Op::Sin as u32,
                            UnaryIntrinsic::Cos => GlslStd450Op::Cos as u32,
                            UnaryIntrinsic::Tan => GlslStd450Op::Tan as u32,
                            UnaryIntrinsic::Asin => GlslStd450Op::Asin as u32,
                            UnaryIntrinsic::Acos => GlslStd450Op::Acos as u32,
                            UnaryIntrinsic::Atan => GlslStd450Op::Atan as u32,
                            UnaryIntrinsic::Sinh => GlslStd450Op::Sinh as u32,
                            UnaryIntrinsic::Cosh => GlslStd450Op::Cosh as u32,
                            UnaryIntrinsic::Tanh => GlslStd450Op::Tanh as u32,
                            UnaryIntrinsic::Asinh => GlslStd450Op::Asinh as u32,
                            UnaryIntrinsic::Acosh => GlslStd450Op::Acosh as u32,
                            UnaryIntrinsic::Atanh => GlslStd450Op::Atanh as u32,
                            UnaryIntrinsic::Exp => GlslStd450Op::Exp as u32,
                            UnaryIntrinsic::Log => GlslStd450Op::Log as u32,
                            UnaryIntrinsic::Exp2 => GlslStd450Op::Exp2 as u32,
                            UnaryIntrinsic::Log2 => GlslStd450Op::Log2 as u32,
                            UnaryIntrinsic::Sqrt => GlslStd450Op::Sqrt as u32,
                            UnaryIntrinsic::InverseSqrt => GlslStd450Op::InverseSqrt as u32,
                            UnaryIntrinsic::Floor => GlslStd450Op::Floor as u32,
                            UnaryIntrinsic::Ceil => GlslStd450Op::Ceil as u32,
                        };
                        build(self.b.ext_inst(
                            rt,
                            Some(id),
                            self.ext,
                            code,
                            [rspirv::dr::Operand::IdRef(a)],
                        ))?;
                    }
                    ValueOp::BinaryIntrinsic { op, args } => {
                        let (a, _) = self.operand(args[0]);
                        let (b, _) = self.operand(args[1]);
                        let code = match op {
                            BinaryIntrinsic::Atan2 => GlslStd450Op::Atan2,
                            BinaryIntrinsic::Pow => GlslStd450Op::Pow,
                            BinaryIntrinsic::Min => match t {
                                ScalarType::F32 => GlslStd450Op::FMin,
                                ScalarType::I32 => GlslStd450Op::SMin,
                                _ => GlslStd450Op::UMin,
                            },
                            BinaryIntrinsic::Max => match t {
                                ScalarType::F32 => GlslStd450Op::FMax,
                                ScalarType::I32 => GlslStd450Op::SMax,
                                _ => GlslStd450Op::UMax,
                            },
                        };
                        build(self.b.ext_inst(
                            rt,
                            Some(id),
                            self.ext,
                            code as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        ))?;
                    }
                    ValueOp::TernaryIntrinsic { op, args } => {
                        let mut operands = vec![];
                        for a in args {
                            operands.push(rspirv::dr::Operand::IdRef(self.operand(*a).0));
                        }
                        let code = match op {
                            TernaryIntrinsic::Mix => GlslStd450Op::FMix,
                            TernaryIntrinsic::Fma => GlslStd450Op::Fma,
                            TernaryIntrinsic::Clamp => match t {
                                ScalarType::F32 => GlslStd450Op::FClamp,
                                ScalarType::I32 => GlslStd450Op::SClamp,
                                _ => GlslStd450Op::UClamp,
                            },
                        };
                        build(
                            self.b
                                .ext_inst(rt, Some(id), self.ext, code as u32, operands),
                        )?;
                    }
                }
            }
            Instruction::Store {
                buffer,
                index,
                value,
            } => {
                let (ptr, t) = self.pointer(*buffer, *index)?;
                let (mut v, _) = self.operand(*value);
                if t == ScalarType::Bool {
                    let one = self.literal(Literal::U32(1));
                    let zero = self.literal(Literal::U32(0));
                    v = build(self.b.select(self.ty(ScalarType::U32), None, v, one, zero))?;
                }
                build(self.b.store(ptr, v, None, []))?;
            }
            Instruction::Barrier => {
                let scope = self.literal(Literal::U32(Scope::Workgroup as u32));
                let semantics = self.literal(Literal::U32(
                    (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
                ));
                build(self.b.control_barrier(scope, scope, semantics))?;
            }
            Instruction::Call { .. } => {
                return Err(CompileError(
                    "device calls require function code generation".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Workgroup size is a pipeline input and must be included in compilation options.
pub fn compile(
    module: &ValidatedModule,
    specialized: &SpecializedKernel,
    workgroup: [u32; 3],
) -> Result<Vec<u32>> {
    if !specialized.functions.is_empty() {
        let module = inline::inline(module, specialized)?;
        let specialized = module
            .specialize_kernel(0, &[])
            .map_err(|e| CompileError(e.to_string()))?;
        return compile(&module, &specialized, workgroup);
    }

    if workgroup.contains(&0) {
        return Err(CompileError("workgroup dimensions must be positive".into()));
    }

    let kernel = module
        .module()
        .kernels
        .get(specialized.kernel_index)
        .ok_or_else(|| CompileError("kernel index out of range".into()))?;

    if module
        .kernel_hash(specialized.kernel_index)
        .map_err(|e| CompileError(e.to_string()))?
        != specialized.identity
    {
        return Err(CompileError(
            "specialization belongs to another kernel".into(),
        ));
    }

    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let ext = b.ext_inst_import("GLSL.std.450");
    let mut scalar = BTreeMap::new();
    scalar.insert(0, b.type_bool());
    scalar.insert(1, b.type_int(32, 1));
    scalar.insert(2, b.type_int(32, 0));
    scalar.insert(3, b.type_float(32, None));
    let mut c = Compiler {
        b,
        scalar,
        params: BTreeMap::new(),
        values: BTreeMap::new(),
        storage: BTreeMap::new(),
        constants: specialized.kernel.constants.clone(),
        labels: BTreeMap::new(),
        builtins: BTreeMap::new(),
        workgroup,
        ext,
    };

    let mut interface = vec![];
    for (k, builtin) in [
        (0, BuiltIn::GlobalInvocationId),
        (1, BuiltIn::WorkgroupId),
        (2, BuiltIn::LocalInvocationId),
    ] {
        let vec = c.b.type_vector(c.ty(ScalarType::U32), 3);
        let ptr = c.b.type_pointer(None, StorageClass::Input, vec);
        let var = c.b.variable(ptr, None, StorageClass::Input, None);
        c.b.decorate(
            var,
            Decoration::BuiltIn,
            [rspirv::dr::Operand::BuiltIn(builtin)],
        );
        c.builtins.insert(k, var);
        interface.push(var);
    }

    let mut scalar_params = vec![];
    let mut binding = 0;
    for p in &kernel.routine.header.parameters {
        match p.ty {
            Type::Buffer {
                element,
                space: AddressSpace::Global,
                ..
            } => {
                let element = if element == ScalarType::Bool {
                    ScalarType::U32
                } else {
                    element
                };
                let array = c.b.type_runtime_array(c.ty(element));
                c.b.decorate(
                    array,
                    Decoration::ArrayStride,
                    [rspirv::dr::Operand::LiteralBit32(4)],
                );
                let structure = c.b.type_struct([array]);
                c.b.decorate(structure, Decoration::Block, []);
                c.b.member_decorate(
                    structure,
                    0,
                    Decoration::Offset,
                    [rspirv::dr::Operand::LiteralBit32(0)],
                );
                let ptr =
                    c.b.type_pointer(None, StorageClass::StorageBuffer, structure);
                let var = c.b.variable(ptr, None, StorageClass::StorageBuffer, None);
                c.b.decorate(
                    var,
                    Decoration::DescriptorSet,
                    [rspirv::dr::Operand::LiteralBit32(0)],
                );
                c.b.decorate(
                    var,
                    Decoration::Binding,
                    [rspirv::dr::Operand::LiteralBit32(binding)],
                );
                binding += 1;
                c.params.insert(p.id, (var, p.ty));
            }
            Type::Scalar(t) => scalar_params.push((p.id, t)),
            _ => return Err(CompileError("kernel buffer must be global".into())),
        }
    }

    let push = if scalar_params.is_empty() {
        None
    } else {
        let types = scalar_params
            .iter()
            .map(|(_, t)| {
                c.ty(if *t == ScalarType::Bool {
                    ScalarType::U32
                } else {
                    *t
                })
            })
            .collect::<Vec<_>>();
        let structure = c.b.type_struct(types);
        c.b.decorate(structure, Decoration::Block, []);
        for i in 0..scalar_params.len() {
            c.b.member_decorate(
                structure,
                i as u32,
                Decoration::Offset,
                [rspirv::dr::Operand::LiteralBit32(i as u32 * 4)],
            );
        }
        let ptr =
            c.b.type_pointer(None, StorageClass::PushConstant, structure);
        Some(c.b.variable(ptr, None, StorageClass::PushConstant, None))
    };

    for v in &kernel.routine.header.variables {
        if let Variable::Value { id, ty } = v {
            let word = c.b.id();
            c.values.insert(*id, (word, *ty));
        }
    }

    for a in &specialized.kernel.allocations {
        let mem = if a.element == ScalarType::Bool {
            ScalarType::U32
        } else {
            a.element
        };
        let length = c.literal(Literal::U32(a.length));
        let array = c.b.type_array(c.ty(mem), length);
        let ptr = c.b.type_pointer(None, storage_class(a.space), array);
        let id = if a.space == AddressSpace::Workgroup {
            c.b.variable(ptr, None, StorageClass::Workgroup, None)
        } else {
            c.b.id()
        };
        c.storage.insert(
            a.id,
            (
                id,
                Type::Buffer {
                    element: a.element,
                    space: a.space,
                    access: Access::ReadWrite,
                },
            ),
        );
    }

    for block in &kernel.routine.body.blocks {
        let label = c.b.id();
        c.labels.insert(block.id, label);
    }

    let void = c.b.type_void();
    let fn_ty = c.b.type_function(void, []);
    let function = build(c.b.begin_function(void, None, FunctionControl::NONE, fn_ty))?;
    c.b.entry_point(ExecutionModel::GLCompute, function, "main", interface);
    c.b.execution_mode(function, ExecutionMode::LocalSize, workgroup);

    let blocks=layout::structured_order(&kernel.routine.body)?;
    for block in blocks {
        build(c.b.begin_block(Some(c.labels[&block.id])))?;
        if block.id == kernel.routine.body.entry {
            for a in &specialized.kernel.allocations {
                if a.space == AddressSpace::Local {
                    let mem = if a.element == ScalarType::Bool {
                        ScalarType::U32
                    } else {
                        a.element
                    };
                    let length = c.literal(Literal::U32(a.length));
                    let array = c.b.type_array(c.ty(mem), length);
                    let ptr = c.b.type_pointer(None, StorageClass::Function, array);
                    c.b.variable(ptr, Some(c.storage[&a.id].0), StorageClass::Function, None);
                }
            }
            for (i, (id, t)) in scalar_params.iter().enumerate() {
                let mem = if *t == ScalarType::Bool {
                    ScalarType::U32
                } else {
                    *t
                };
                let ptr =
                    c.b.type_pointer(None, StorageClass::PushConstant, c.ty(mem));
                let index = c.literal(Literal::U32(i as u32));
                let ptr = build(c.b.access_chain(ptr, None, push.unwrap(), [index]))?;
                let mut value = build(c.b.load(c.ty(mem), None, ptr, None, []))?;
                if *t == ScalarType::Bool {
                    let zero = c.literal(Literal::U32(0));
                    value = build(c.b.i_not_equal(c.ty(*t), None, value, zero))?;
                }
                c.params.insert(*id, (value, Type::Scalar(*t)));
            }
        }
        for (i, arg) in block.arguments.iter().enumerate() {
            let (id, t) = c.values[arg];
            let Type::Scalar(t) = t else {
                return Err(CompileError("buffer block arguments unsupported".into()));
            };
            let mut incoming = vec![];
            for pred in &kernel.routine.body.blocks {
                let edges: Vec<&Edge> = match &pred.terminator {
                    Terminator::Branch(e) => vec![e],
                    Terminator::Conditional {
                        then_edge,
                        else_edge,
                        ..
                    } => vec![then_edge, else_edge],
                    _ => vec![],
                };
                for edge in edges {
                    if edge.target == block.id {
                        incoming.push((c.operand(edge.arguments[i]).0, c.labels[&pred.id]));
                    }
                }
            }
            build(c.b.phi(c.ty(t), Some(id), incoming))?;
        }
        for instruction in &block.instructions {
            c.instruction(instruction)?;
        }
        if let Some(merge) = block.merge {
            match merge {
                Merge::Selection { merge } => {
                    build(c.b.selection_merge(c.labels[&merge], SelectionControl::NONE))?;
                }
                Merge::Loop { merge, continuing } => {
                    build(c.b.loop_merge(
                        c.labels[&merge],
                        c.labels[&continuing],
                        LoopControl::NONE,
                        [],
                    ))?;
                }
            }
        }
        match &block.terminator {
            Terminator::Branch(e) => {
                build(c.b.branch(c.labels[&e.target]))?;
            }
            Terminator::Conditional {
                condition,
                then_edge,
                else_edge,
            } => {
                let condition = c.operand(*condition).0;
                build(c.b.branch_conditional(
                    condition,
                    c.labels[&then_edge.target],
                    c.labels[&else_edge.target],
                    [],
                ))?;
            }
            Terminator::Return(None) => {
                build(c.b.ret())?;
            }
            Terminator::Unreachable => {
                if let Some(header) = kernel.routine.body.blocks.iter().find(
                    |b| matches!(b.merge,Some(Merge::Loop{continuing,..}) if continuing==block.id),
                ) {
                    build(c.b.branch(c.labels[&header.id]))?;
                } else {
                    build(c.b.unreachable())?;
                }
            }
            _ => return Err(CompileError("kernel cannot return a value".into())),
        }
    }
    build(c.b.end_function())?;
    Ok(c.b.module().assemble())
}
