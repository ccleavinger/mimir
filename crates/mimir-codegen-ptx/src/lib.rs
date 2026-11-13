#![doc = include_str!("../README.md")]
mod math;

use mimir_ir::flat::*;
use std::{collections::BTreeMap, fmt::Write};

/// Stable entry-point name used in emitted modules.
pub const ENTRY_NAME: &str = "mimir_kernel";

#[derive(Debug, thiserror::Error)]
#[error("PTX compilation: {0}")]
pub struct CompileError(pub String);
type Result<T> = std::result::Result<T, CompileError>;

struct Emitter<'a> {
    declarations: String,
    out: String,
    types: BTreeMap<ValueId, Type>,
    parameters: BTreeMap<ParameterId, Type>,
    storage: BTreeMap<StorageId, Type>,
    spec: &'a SpecializedRoutine,
    functions: &'a BTreeMap<FunctionInstance, String>,
    temporary: usize,
}

fn suffix(t: ScalarType) -> &'static str {
    match t {
        ScalarType::Bool => "pred",
        ScalarType::I32 => "s32",
        ScalarType::U32 => "u32",
        ScalarType::F32 => "f32",
    }
}
fn register_type(t: Type) -> &'static str {
    match t {
        Type::Scalar(t) => suffix(t),
        Type::Buffer { .. } => "u64",
    }
}
fn abi_type(t: Type) -> &'static str {
    match t {
        Type::Scalar(ScalarType::Bool) => "u32",
        _ => register_type(t),
    }
}
fn space_name(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Global => "global",
        AddressSpace::Local => "local",
        AddressSpace::Workgroup => "shared",
    }
}
fn float(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

impl<'a> Emitter<'a> {
    fn new(
        spec: &'a SpecializedRoutine,
        functions: &'a BTreeMap<FunctionInstance, String>,
    ) -> Self {
        Self {
            declarations: String::new(),
            out: String::new(),
            types: BTreeMap::new(),
            parameters: BTreeMap::new(),
            storage: BTreeMap::new(),
            spec,
            functions,
            temporary: 0,
        }
    }
    fn temp(&mut self, ty: &str) -> String {
        let name = format!("%t{}", self.temporary);
        self.temporary += 1;
        writeln!(self.declarations, "    .reg .{ty} {name};").unwrap();
        name
    }
    fn label(&mut self) -> String {
        let name = format!("L{}", self.temporary);
        self.temporary += 1;
        name
    }
    fn emit(&mut self, op: &str, dest: &str, args: &[&str]) {
        writeln!(self.out, "    {op} {dest}, {};", args.join(", ")).unwrap();
    }
    fn calc(&mut self, op: &str, ty: &str, args: &[&str]) -> String {
        let result = self.temp(ty);
        self.emit(op, &result, args);
        result
    }
    fn operand(&mut self, o: Operand) -> (String, Type) {
        match o {
            Operand::Value(v) => (format!("%v{}", v.0), self.types[&v]),
            Operand::Parameter(p) => (format!("%p{}", p.0), self.parameters[&p]),
            Operand::Storage(s) => (format!("%s{}", s.0), self.storage[&s]),
            Operand::Constant(c) => (
                self.spec.constants[&c].to_string(),
                Type::Scalar(ScalarType::U32),
            ),
            Operand::Literal(l) => {
                let value = match l {
                    Literal::Bool(b) => {
                        let p = self.temp("pred");
                        self.emit("setp.ne.u32", &p, &[if b { "1" } else { "0" }, "0"]);
                        p
                    }
                    Literal::I32(v) => v.to_string(),
                    Literal::U32(v) => v.to_string(),
                    Literal::F32(bits) => format!("0f{bits:08X}"),
                };
                (value, Type::Scalar(l.ty()))
            }
        }
    }
    fn copy(&mut self, dest: &str, source: Operand) {
        let (src, ty) = self.operand(source);
        self.emit(&format!("mov.{}", register_type(ty)), dest, &[&src]);
    }

    fn edge(&mut self, edge: &Edge, body: &Body) {
        let target = body.blocks.iter().find(|b| b.id == edge.target).unwrap();
        // Snapshot every source before overwriting any destination (parallel assignment).
        let mut staged = Vec::new();
        for (dest, source) in target.arguments.iter().zip(&edge.arguments) {
            let ty = register_type(self.types[dest]);
            let temp = self.temp(ty);
            self.copy(&temp, *source);
            staged.push((*dest, temp, ty));
        }
        for (dest, temp, ty) in staged {
            self.emit(&format!("mov.{ty}"), &format!("%v{}", dest.0), &[&temp]);
        }
        writeln!(self.out, "    bra block{};", edge.target.0).unwrap();
    }

    fn address(&mut self, buffer: Operand, index: Operand) -> (String, ScalarType, AddressSpace) {
        let (base, ty) = self.operand(buffer);
        let Type::Buffer { element, space, .. } = ty else {
            unreachable!("validated buffer")
        };
        let (index, _) = self.operand(index);
        let offset = self.calc("mul.wide.u32", "u64", &[&index, "4"]);
        let address = self.calc("add.u64", "u64", &[&base, &offset]);
        let address = self.calc(
            &format!("cvta.to.{}.u64", space_name(space)),
            "u64",
            &[&address],
        );
        (address, element, space)
    }

    fn load_parameter(&mut self, dest: &str, parameter: &str, ty: Type) {
        if ty == Type::Scalar(ScalarType::Bool) {
            let raw = self.calc("ld.param.u32", "u32", &[&format!("[{parameter}]")]);
            self.emit("setp.ne.u32", dest, &[&raw, "0"]);
        } else {
            self.emit(
                &format!("ld.param.{}", abi_type(ty)),
                dest,
                &[&format!("[{parameter}]")],
            );
        }
    }

    fn store_parameter(&mut self, parameter: &str, value: Operand) {
        let (mut src, ty) = self.operand(value);
        if ty == Type::Scalar(ScalarType::Bool) {
            src = self.calc("selp.u32", "u32", &["1", "0", &src]);
        }
        writeln!(
            self.out,
            "    st.param.{} [{parameter}], {src};",
            abi_type(ty)
        )
        .unwrap();
    }

    fn binary(&mut self, dest: &str, op: BinaryOp, a: Operand, b: Operand) {
        let (a, ty) = self.operand(a);
        let (b, _) = self.operand(b);
        let Type::Scalar(ty) = ty else { unreachable!() };
        if ty == ScalarType::Bool {
            let code = match op {
                BinaryOp::And => "and.pred",
                BinaryOp::Or => "or.pred",
                BinaryOp::Eq | BinaryOp::Ne => "xor.pred",
                _ => unreachable!(),
            };
            self.emit(code, dest, &[&a, &b]);
            if op == BinaryOp::Eq {
                self.emit("not.pred", dest, &[dest]);
            }
            return;
        }
        if op == BinaryOp::Rem && ty == ScalarType::F32 {
            let r = self.remainder(&a, &b);
            self.emit("mov.f32", dest, &[&r]);
            return;
        }
        let code = match op {
            BinaryOp::Eq => "setp.eq",
            BinaryOp::Ne if ty == ScalarType::F32 => "setp.neu",
            BinaryOp::Ne => "setp.ne",
            BinaryOp::Lt => "setp.lt",
            BinaryOp::Le => "setp.le",
            BinaryOp::Gt => "setp.gt",
            BinaryOp::Ge => "setp.ge",
            BinaryOp::Add if ty == ScalarType::F32 => "add.rn",
            BinaryOp::Add => "add",
            BinaryOp::Sub if ty == ScalarType::F32 => "sub.rn",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul if ty == ScalarType::F32 => "mul.rn",
            BinaryOp::Mul => "mul.lo",
            BinaryOp::Div if ty == ScalarType::F32 => "div.rn",
            BinaryOp::Div => "div",
            BinaryOp::Rem => "rem",
            BinaryOp::And | BinaryOp::Or => unreachable!(),
        };
        self.emit(&format!("{code}.{}", suffix(ty)), dest, &[&a, &b]);
    }

    fn instruction(&mut self, instruction: &Instruction, block: BlockId, index: usize) {
        match instruction {
            Instruction::Barrier => self.out.push_str("    bar.sync 0;\n"),
            Instruction::Value { result, op } => {
                let dest = format!("%v{}", result.0);
                match op {
                    ValueOp::Copy(src) => self.copy(&dest, *src),
                    ValueOp::Binary { op, lhs, rhs } => self.binary(&dest, *op, *lhs, *rhs),
                    ValueOp::Unary { op, value } => {
                        let (src, ty) = self.operand(*value);
                        let code = match op {
                            UnaryOp::Not => "not.pred".into(),
                            UnaryOp::Neg => format!("neg.{}", register_type(ty)),
                        };
                        self.emit(&code, &dest, &[&src]);
                    }
                    ValueOp::Cast { value, to } => {
                        let (src, from) = self.operand(*value);
                        let Type::Scalar(from) = from else {
                            unreachable!()
                        };
                        let code = if from == *to || (from.is_integer() && to.is_integer()) {
                            format!("mov.{}", suffix(*to))
                        } else {
                            let rounding = if *to == ScalarType::F32 { "rn" } else { "rzi" };
                            format!("cvt.{rounding}.{}.{}", suffix(*to), suffix(from))
                        };
                        self.emit(&code, &dest, &[&src]);
                    }
                    ValueOp::Builtin { builtin, axis } => {
                        let axis = match axis {
                            Axis::X => "x",
                            Axis::Y => "y",
                            Axis::Z => "z",
                        };
                        let reg = match builtin {
                            Builtin::GlobalId | Builtin::LocalId => "tid",
                            Builtin::WorkgroupId => "ctaid",
                            Builtin::WorkgroupSize => "ntid",
                        };
                        self.emit("mov.u32", &dest, &[&format!("%{reg}.{axis}")]);
                        if *builtin == Builtin::GlobalId {
                            let group = self.calc("mov.u32", "u32", &[&format!("%ctaid.{axis}")]);
                            let size = self.calc("mov.u32", "u32", &[&format!("%ntid.{axis}")]);
                            self.emit("mad.lo.u32", &dest, &[&group, &size, &dest]);
                        }
                    }
                    ValueOp::Load { buffer, index } => {
                        let (ptr, element, space) = self.address(*buffer, *index);
                        if element == ScalarType::Bool {
                            let raw = self.calc(
                                &format!("ld.{}.u32", space_name(space)),
                                "u32",
                                &[&format!("[{ptr}]")],
                            );
                            self.emit("setp.ne.u32", &dest, &[&raw, "0"]);
                        } else {
                            self.emit(
                                &format!("ld.{}.{}", space_name(space), suffix(element)),
                                &dest,
                                &[&format!("[{ptr}]")],
                            );
                        }
                    }
                    ValueOp::UnaryIntrinsic { op, value } => {
                        let (a, _) = self.operand(*value);
                        let r = self.unary_math(*op, &a);
                        self.emit("mov.f32", &dest, &[&r]);
                    }
                    ValueOp::BinaryIntrinsic { op, args } => {
                        let (a, ty) = self.operand(args[0]);
                        let (b, _) = self.operand(args[1]);
                        let r = match op {
                            BinaryIntrinsic::Min | BinaryIntrinsic::Max => self.calc(
                                &format!(
                                    "{}.{}",
                                    if *op == BinaryIntrinsic::Min {
                                        "min"
                                    } else {
                                        "max"
                                    },
                                    register_type(ty)
                                ),
                                register_type(ty),
                                &[&a, &b],
                            ),
                            BinaryIntrinsic::Atan2 => self.atan2(&a, &b),
                            BinaryIntrinsic::Pow => self.pow(&a, &b),
                        };
                        self.emit(&format!("mov.{}", register_type(ty)), &dest, &[&r]);
                    }
                    ValueOp::TernaryIntrinsic { op, args } => {
                        let (a, ty) = self.operand(args[0]);
                        let (b, _) = self.operand(args[1]);
                        let (c, _) = self.operand(args[2]);
                        let ty = register_type(ty);
                        match op {
                            TernaryIntrinsic::Clamp => {
                                let lo = self.calc(&format!("max.{ty}"), ty, &[&a, &b]);
                                self.emit(&format!("min.{ty}"), &dest, &[&lo, &c]);
                            }
                            TernaryIntrinsic::Fma => self.emit("fma.rn.f32", &dest, &[&a, &b, &c]),
                            TernaryIntrinsic::Mix => {
                                let inv = self.calc("sub.rn.f32", "f32", &[&float(1.0), &c]);
                                let left = self.calc("mul.rn.f32", "f32", &[&a, &inv]);
                                let right = self.calc("mul.rn.f32", "f32", &[&b, &c]);
                                self.emit("add.rn.f32", &dest, &[&left, &right]);
                            }
                        }
                    }
                }
            }
            Instruction::Store {
                buffer,
                index,
                value,
            } => {
                let (ptr, element, space) = self.address(*buffer, *index);
                let (mut src, _) = self.operand(*value);
                let ty = if element == ScalarType::Bool {
                    src = self.calc("selp.u32", "u32", &["1", "0", &src]);
                    "u32"
                } else {
                    suffix(element)
                };
                writeln!(
                    self.out,
                    "    st.{}.{ty} [{ptr}], {src};",
                    space_name(space)
                )
                .unwrap();
            }
            Instruction::Call {
                result, arguments, ..
            } => {
                let instance = &self.spec.calls[&(block, index)];
                let name = self.functions[instance].clone();
                // A scope gives each call independent parameter storage.
                self.out.push_str("    {\n");
                let mut names = Vec::new();
                for (i, argument) in arguments.iter().enumerate() {
                    let (_, ty) = self.operand(*argument);
                    let name = format!("arg{i}");
                    writeln!(self.out, "    .param .{} {name};", abi_type(ty)).unwrap();
                    self.store_parameter(&name, *argument);
                    names.push(name);
                }
                if let Some(result) = result {
                    let ty = self.types[result];
                    writeln!(self.out, "    .param .{} returned;", abi_type(ty)).unwrap();
                    writeln!(
                        self.out,
                        "    call.uni (returned), {name}, ({});",
                        names.join(", ")
                    )
                    .unwrap();
                    self.load_parameter(&format!("%v{}", result.0), "returned", ty);
                } else {
                    writeln!(self.out, "    call.uni {name}, ({});", names.join(", ")).unwrap();
                }
                self.out.push_str("    }\n");
            }
        }
    }

    fn routine(mut self, routine: &Routine) -> String {
        for p in &routine.header.parameters {
            self.parameters.insert(p.id, p.ty);
            writeln!(
                self.declarations,
                "    .reg .{} %p{};",
                register_type(p.ty),
                p.id.0
            )
            .unwrap();
            self.load_parameter(&format!("%p{}", p.id.0), &format!("param{}", p.id.0), p.ty);
        }
        for variable in &routine.header.variables {
            match *variable {
                Variable::Value { id, ty } => {
                    self.types.insert(id, ty);
                    writeln!(
                        self.declarations,
                        "    .reg .{} %v{};",
                        register_type(ty),
                        id.0
                    )
                    .unwrap();
                }
                Variable::Storage {
                    id, element, space, ..
                } => {
                    let allocation = self.spec.allocations.iter().find(|a| a.id == id).unwrap();
                    self.storage.insert(
                        id,
                        Type::Buffer {
                            element,
                            space,
                            access: Access::ReadWrite,
                        },
                    );
                    writeln!(
                        self.declarations,
                        "    .{} .align 4 .b8 storage{}[{}];\n    .reg .u64 %s{};",
                        space_name(space),
                        id.0,
                        allocation.bytes,
                        id.0
                    )
                    .unwrap();
                    let raw = self.calc("mov.u64", "u64", &[&format!("storage{}", id.0)]);
                    self.emit(
                        &format!("cvta.{}.u64", space_name(space)),
                        &format!("%s{}", id.0),
                        &[&raw],
                    );
                }
            }
        }
        let body = &routine.body;
        writeln!(self.out, "    bra block{};", body.entry.0).unwrap();
        for block in &body.blocks {
            writeln!(self.out, "block{}:", block.id.0).unwrap();
            for (i, instruction) in block.instructions.iter().enumerate() {
                self.instruction(instruction, block.id, i);
            }
            match &block.terminator {
                Terminator::Branch(edge) => self.edge(edge, body),
                Terminator::Conditional {
                    condition,
                    then_edge,
                    else_edge,
                } => {
                    let (condition, _) = self.operand(*condition);
                    writeln!(self.out, "    @{condition} bra then{};", block.id.0).unwrap();
                    self.edge(else_edge, body);
                    writeln!(self.out, "then{}:", block.id.0).unwrap();
                    self.edge(then_edge, body);
                }
                Terminator::Return(value) => {
                    if let Some(value) = value {
                        self.store_parameter("return_value", *value);
                    }
                    self.out.push_str("    ret;\n");
                }
                Terminator::Unreachable => self.out.push_str("    trap;\n"),
            }
        }
        format!("{{\n{}{} }}\n", self.declarations, self.out)
    }
}

fn signature(name: &str, routine: &Routine, returns: Option<ScalarType>, entry: bool) -> String {
    let parameters = routine
        .header
        .parameters
        .iter()
        .map(|p| format!(".param .{} param{}", abi_type(p.ty), p.id.0))
        .collect::<Vec<_>>()
        .join(", ");
    if entry {
        format!(".visible .entry {name}({parameters})")
    } else {
        let ret = returns
            .map(|t| format!("(.param .{} return_value) ", abi_type(Type::Scalar(t))))
            .unwrap_or_default();
        format!(".func {ret}{name}({parameters})")
    }
}

/// Emit a self-contained PTX 6.0 module targeting `sm_50` with entry `mimir_kernel`.
/// Parameters follow header order: buffers are 64-bit generic pointers;
/// scalars occupy 32 bits (booleans use 0/1). Workgroup dimensions are read from
/// launch-time special registers, so this output can be reused across block sizes.
/// No CUDA toolkit, libdevice, LLVM, or GPU is required to generate PTX.
pub fn compile(module: &ValidatedModule, specialized: &SpecializedKernel) -> Result<String> {
    // Specialized structures are public and mutable: reconstruct the proof before
    // indexing them, including transitive function identities and allocations.
    let expected = module
        .specialize_kernel(specialized.kernel_index, &specialized.tuning_values)
        .map_err(|e| CompileError(e.to_string()))?;
    if expected.identity != specialized.identity
        || !same_specialization(&expected.kernel, &specialized.kernel)
        || expected.functions.len() != specialized.functions.len()
        || expected.functions.iter().any(|(k, v)| {
            specialized
                .functions
                .get(k)
                .is_none_or(|other| !same_specialization(v, other))
        })
    {
        return Err(CompileError(
            "specialization does not match module and tuning values".into(),
        ));
    }
    let names: BTreeMap<_, _> = specialized
        .functions
        .keys()
        .enumerate()
        .map(|(i, instance)| (instance.clone(), format!("mimir_function{i}")))
        .collect();
    let mut out = String::from(".version 6.0\n.target sm_50\n.address_size 64\n\n");
    for (instance, name) in &names {
        let f = module
            .module()
            .functions
            .iter()
            .find(|f| f.id == instance.function)
            .unwrap();
        writeln!(out, "{};", signature(name, &f.routine, f.returns, false)).unwrap();
    }
    for (instance, name) in &names {
        let f = module
            .module()
            .functions
            .iter()
            .find(|f| f.id == instance.function)
            .unwrap();
        writeln!(out, "{}", signature(name, &f.routine, f.returns, false)).unwrap();
        out.push_str(&Emitter::new(&specialized.functions[instance], &names).routine(&f.routine));
    }
    let kernel = &module.module().kernels[specialized.kernel_index];
    writeln!(
        out,
        "{}",
        signature(ENTRY_NAME, &kernel.routine, None, true)
    )
    .unwrap();
    out.push_str(&Emitter::new(&specialized.kernel, &names).routine(&kernel.routine));
    Ok(out)
}

fn same_specialization(a: &SpecializedRoutine, b: &SpecializedRoutine) -> bool {
    a.constants == b.constants && a.allocations == b.allocations && a.calls == b.calls
}

#[cfg(test)]
mod tests;
