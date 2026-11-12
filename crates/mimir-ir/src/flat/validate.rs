use super::*;
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct ValidationError(pub String);
pub(crate) type Check<T> = Result<T, ValidationError>;
pub(crate) fn ensure(ok: bool, message: impl Into<String>) -> Check<()> {
    if ok {
        Ok(())
    } else {
        Err(ValidationError(message.into()))
    }
}

/// Owned proof of structural and type validity. Access is immutable so edits
/// require validation again. This does not prove bounds, races or uniformity.
#[derive(Debug, Clone)]
pub struct ValidatedModule {
    pub(crate) module: Module,
}

impl ValidatedModule {
    pub fn module(&self) -> &Module {
        &self.module
    }
    pub fn into_module(self) -> Module {
        self.module
    }
}

impl Module {
    pub fn validate(self) -> Check<ValidatedModule> {
        let functions: BTreeMap<_, _> = self.functions.iter().map(|f| (f.id, f)).collect();
        ensure(
            functions.len() == self.functions.len(),
            "duplicate function ID",
        )?;
        for (i, k) in self.kernels.iter().enumerate() {
            validate_routine(&k.routine, None, true, &functions)
                .map_err(|e| ValidationError(format!("kernel {i}: {e}")))?;
        }
        for f in &self.functions {
            validate_routine(&f.routine, f.returns, false, &functions)
                .map_err(|e| ValidationError(format!("function {:?}: {e}", f.id)))?;
        }
        fn visit(
            id: FunctionId,
            functions: &BTreeMap<FunctionId, &DeviceFunction>,
            active: &mut BTreeSet<FunctionId>,
            done: &mut BTreeSet<FunctionId>,
        ) -> Check<()> {
            if done.contains(&id) {
                return Ok(());
            }
            ensure(active.insert(id), "recursive device-function calls")?;
            for block in &functions[&id].routine.body.blocks {
                for i in &block.instructions {
                    if let Instruction::Call { function, .. } = i {
                        visit(*function, functions, active, done)?;
                    }
                }
            }
            active.remove(&id);
            done.insert(id);
            Ok(())
        }
        let mut done = BTreeSet::new();
        for id in functions.keys() {
            visit(*id, &functions, &mut BTreeSet::new(), &mut done)?;
        }
        Ok(ValidatedModule { module: self })
    }
}

/// Returns derived definitions in dependency order, independent of table order.
pub(crate) fn constant_order(h: &Header) -> Check<Vec<usize>> {
    let inputs: BTreeSet<_> = h.constants.iter().copied().collect();
    ensure(
        inputs.len() == h.constants.len(),
        "duplicate tuning constant ID",
    )?;
    let defs: BTreeMap<_, _> = h
        .derived_constants
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id, i))
        .collect();
    ensure(
        defs.len() == h.derived_constants.len() && defs.keys().all(|id| !inputs.contains(id)),
        "duplicate constant ID",
    )?;
    fn visit(
        i: usize,
        h: &Header,
        inputs: &BTreeSet<ConstantId>,
        defs: &BTreeMap<ConstantId, usize>,
        state: &mut [u8],
        order: &mut Vec<usize>,
    ) -> Check<()> {
        if state[i] == 2 {
            return Ok(());
        }
        ensure(state[i] != 1, "constant dependency cycle")?;
        state[i] = 1;
        let d = &h.derived_constants[i];
        for a in [d.lhs, d.rhs] {
            if let ConstantOperand::Constant(id) = a {
                if !inputs.contains(&id) {
                    visit(
                        *defs
                            .get(&id)
                            .ok_or_else(|| ValidationError("unknown constant ID".into()))?,
                        h,
                        inputs,
                        defs,
                        state,
                        order,
                    )?;
                }
            }
        }
        state[i] = 2;
        order.push(i);
        Ok(())
    }
    let mut order = vec![];
    let mut state = vec![0; h.derived_constants.len()];
    for i in 0..state.len() {
        visit(i, h, &inputs, &defs, &mut state, &mut order)?;
    }
    Ok(order)
}

struct Context<'a> {
    params: BTreeMap<ParameterId, Type>,
    constants: BTreeSet<ConstantId>,
    values: BTreeMap<ValueId, Type>,
    storage: BTreeMap<StorageId, Type>,
    functions: &'a BTreeMap<FunctionId, &'a DeviceFunction>,
}
impl Context<'_> {
    fn operand(&self, a: Operand) -> Check<Type> {
        let ty = match a {
            Operand::Literal(l) => Some(Type::Scalar(l.ty())),
            Operand::Constant(c) => self
                .constants
                .contains(&c)
                .then_some(Type::Scalar(ScalarType::U32)),
            Operand::Parameter(p) => self.params.get(&p).copied(),
            Operand::Value(v) => self.values.get(&v).copied(),
            Operand::Storage(s) => self.storage.get(&s).copied(),
        };
        ty.ok_or_else(|| ValidationError(format!("undeclared operand {a:?}")))
    }
    fn scalar(&self, a: Operand) -> Check<ScalarType> {
        match self.operand(a)? {
            Type::Scalar(t) => Ok(t),
            _ => Err(ValidationError("expected scalar operand".into())),
        }
    }
    fn constant(&self, a: ConstantOperand) -> Check<()> {
        ensure(
            match a {
                ConstantOperand::Literal(_) => true,
                ConstantOperand::Constant(id) => self.constants.contains(&id),
            },
            "unknown constant ID",
        )
    }
    fn indexed(&self, buffer: Operand, index: Operand, write: bool) -> Check<ScalarType> {
        ensure(self.scalar(index)?.is_integer(), "index must be integer")?;
        match self.operand(buffer)? {
            Type::Buffer {
                element, access, ..
            } => {
                ensure(
                    if write {
                        access.can_write()
                    } else {
                        access.can_read()
                    },
                    "memory access violates permissions",
                )?;
                Ok(element)
            }
            _ => Err(ValidationError("expected buffer reference".into())),
        }
    }
    fn value_op(&self, op: &ValueOp) -> Check<Type> {
        use ScalarType::*;
        let ty = match op {
            ValueOp::Copy(a) => return self.operand(*a),
            ValueOp::Binary { op, lhs, rhs } => {
                let l = self.scalar(*lhs)?;
                ensure(l == self.scalar(*rhs)?, "binary operand type mismatch")?;
                match op {
                    BinaryOp::And | BinaryOp::Or => {
                        ensure(l == Bool, "logical operands must be bool")?;
                        Bool
                    }
                    BinaryOp::Eq | BinaryOp::Ne => Bool,
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        ensure(l.is_numeric(), "ordered comparison requires numbers")?;
                        Bool
                    }
                    _ => {
                        ensure(l.is_numeric(), "arithmetic requires numbers")?;
                        l
                    }
                }
            }
            ValueOp::Unary { op, value } => {
                let t = self.scalar(*value)?;
                ensure(
                    match op {
                        UnaryOp::Not => t == Bool,
                        UnaryOp::Neg => matches!(t, I32 | F32),
                    },
                    "invalid unary operand",
                )?;
                t
            }
            ValueOp::Cast { value, to } => {
                let from = self.scalar(*value)?;
                ensure(
                    from.is_numeric() && to.is_numeric(),
                    "cast requires numeric scalar types",
                )?;
                *to
            }
            ValueOp::Builtin { .. } => U32,
            ValueOp::Load { buffer, index } => self.indexed(*buffer, *index, false)?,
            ValueOp::UnaryIntrinsic { value, .. } => {
                ensure(self.scalar(*value)? == F32, "intrinsic requires f32")?;
                F32
            }
            ValueOp::BinaryIntrinsic { op, args } => self.intrinsic(
                args,
                matches!(op, BinaryIntrinsic::Min | BinaryIntrinsic::Max),
            )?,
            ValueOp::TernaryIntrinsic { op, args } => {
                self.intrinsic(args, matches!(op, TernaryIntrinsic::Clamp))?
            }
        };
        Ok(Type::Scalar(ty))
    }
    fn intrinsic(&self, args: &[Operand], numeric: bool) -> Check<ScalarType> {
        let first = self.scalar(args[0])?;
        ensure(
            if numeric {
                first.is_numeric()
            } else {
                first == ScalarType::F32
            },
            "invalid intrinsic type",
        )?;
        for a in &args[1..] {
            ensure(self.scalar(*a)? == first, "intrinsic operand type mismatch")?;
        }
        Ok(first)
    }
}
fn compatible(actual: Type, expected: Type) -> bool {
    match (actual, expected) {
        (
            Type::Buffer {
                element: a,
                space: b,
                access: c,
            },
            Type::Buffer {
                element: x,
                space: y,
                access: z,
            },
        ) => a == x && b == y && c.permits(z),
        _ => actual == expected,
    }
}
fn validate_routine<'a>(
    r: &Routine,
    returns: Option<ScalarType>,
    kernel: bool,
    functions: &'a BTreeMap<FunctionId, &'a DeviceFunction>,
) -> Check<()> {
    constant_order(&r.header)?;
    let mut ctx = Context {
        params: BTreeMap::new(),
        constants: r
            .header
            .constants
            .iter()
            .copied()
            .chain(r.header.derived_constants.iter().map(|d| d.id))
            .collect(),
        values: BTreeMap::new(),
        storage: BTreeMap::new(),
        functions,
    };
    for p in &r.header.parameters {
        if kernel {
            ensure(
                !matches!(
                    p.ty,
                    Type::Buffer {
                        space: AddressSpace::Local | AddressSpace::Workgroup,
                        ..
                    }
                ),
                "kernel parameters must reference global buffers",
            )?;
        }
        ensure(
            ctx.params.insert(p.id, p.ty).is_none(),
            "duplicate parameter ID",
        )?;
    }
    for v in &r.header.variables {
        match *v {
            Variable::Value { id, ty } => {
                ensure(ctx.values.insert(id, ty).is_none(), "duplicate value ID")?
            }
            Variable::Storage {
                id,
                element,
                space,
                length,
            } => {
                ensure(
                    space != AddressSpace::Global && (kernel || space != AddressSpace::Workgroup),
                    "invalid storage ownership/address space",
                )?;
                ctx.constant(length)?;
                ensure(
                    ctx.storage
                        .insert(
                            id,
                            Type::Buffer {
                                element,
                                space,
                                access: Access::ReadWrite,
                            },
                        )
                        .is_none(),
                    "duplicate storage ID",
                )?;
            }
        }
    }
    let blocks: BTreeMap<_, _> = r.body.blocks.iter().map(|b| (b.id, b)).collect();
    ensure(blocks.len() == r.body.blocks.len(), "duplicate block ID")?;
    ensure(blocks.contains_key(&r.body.entry), "missing entry block")?;
    ensure(
        blocks[&r.body.entry].arguments.is_empty(),
        "entry block cannot have arguments",
    )?;
    let mut defs = BTreeMap::new();
    for b in &r.body.blocks {
        for &id in &b.arguments {
            ensure(
                defs.insert(id, (b.id, 0usize)).is_none(),
                "multiple SSA definitions",
            )?;
        }
        for (i, inst) in b.instructions.iter().enumerate() {
            if let Some(id) = inst.result() {
                ensure(
                    defs.insert(id, (b.id, i + 1)).is_none(),
                    "multiple SSA definitions",
                )?;
            }
        }
    }
    ensure(
        defs.len() == ctx.values.len() && defs.keys().all(|id| ctx.values.contains_key(id)),
        "SSA declarations and definitions differ",
    )?;
    let mut predecessors: BTreeMap<_, BTreeSet<_>> =
        blocks.keys().map(|id| (*id, BTreeSet::new())).collect();
    for b in &r.body.blocks {
        for e in b.terminator.edges() {
            let target = blocks
                .get(&e.target)
                .ok_or_else(|| ValidationError("unknown branch target".into()))?;
            predecessors.get_mut(&e.target).unwrap().insert(b.id);
            ensure(
                e.arguments.len() == target.arguments.len(),
                "block argument count mismatch",
            )?;
            for (&a, &v) in e.arguments.iter().zip(&target.arguments) {
                ensure(
                    ctx.operand(a)? == ctx.values[&v],
                    "block argument type mismatch",
                )?;
            }
        }
    }
    ensure(
        predecessors[&r.body.entry].is_empty(),
        "entry block cannot have predecessors",
    )?;
    let reachable = reachable_from(r.body.entry, &blocks, &BTreeSet::new());
    let structural: BTreeSet<_> = r
        .body
        .blocks
        .iter()
        .filter_map(|b| b.merge)
        .flat_map(|m| match m {
            Merge::Selection { merge } => vec![merge],
            Merge::Loop { merge, continuing } => vec![merge, continuing],
        })
        .collect();
    for b in &r.body.blocks {
        if reachable.contains(&b.id) {
            ensure(
                !matches!(b.terminator, Terminator::Unreachable),
                "reachable unreachable terminator",
            )?;
        } else {
            ensure(
                structural.contains(&b.id)
                    && b.arguments.is_empty()
                    && b.instructions.is_empty()
                    && b.merge.is_none()
                    && matches!(b.terminator, Terminator::Unreachable),
                "unreachable code must be an empty structural placeholder",
            )?;
        }
    }
    let all = reachable.clone();
    let mut dom: BTreeMap<_, _> = blocks
        .keys()
        .map(|&b| {
            (
                b,
                if b == r.body.entry || !reachable.contains(&b) {
                    BTreeSet::from([b])
                } else {
                    all.clone()
                },
            )
        })
        .collect();
    loop {
        let mut changed = false;
        for &b in blocks.keys() {
            if b != r.body.entry && reachable.contains(&b) {
                let mut new = all.clone();
                for p in &predecessors[&b] {
                    new = new.intersection(&dom[p]).copied().collect();
                }
                new.insert(b);
                if new != dom[&b] {
                    dom.insert(b, new);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let check_use = |a: Operand, b: BlockId, pos: usize| -> Check<()> {
        ctx.operand(a)?;
        if let Operand::Value(id) = a {
            let (def_block, def_pos) = defs[&id];
            ensure(
                if def_block == b {
                    def_pos < pos
                } else {
                    dom[&b].contains(&def_block)
                },
                "SSA use is not dominated by its definition",
            )?;
        }
        Ok(())
    };
    for b in &r.body.blocks {
        for (i, inst) in b.instructions.iter().enumerate() {
            for a in inst.operands() {
                check_use(a, b.id, i + 1)?;
            }
            match inst {
                Instruction::Value { result, op } => ensure(
                    ctx.values[result] == ctx.value_op(op)?,
                    "instruction result type mismatch",
                )?,
                Instruction::Store {
                    buffer,
                    index,
                    value,
                } => ensure(
                    ctx.scalar(*value)? == ctx.indexed(*buffer, *index, true)?,
                    "store element type mismatch",
                )?,
                Instruction::Barrier => {}
                Instruction::Call {
                    result,
                    function,
                    arguments,
                    constants,
                } => {
                    let f = ctx
                        .functions
                        .get(function)
                        .ok_or_else(|| ValidationError("unknown device function".into()))?;
                    ensure(
                        arguments.len() == f.routine.header.parameters.len(),
                        "function argument count mismatch",
                    )?;
                    for (&a, p) in arguments.iter().zip(&f.routine.header.parameters) {
                        ensure(
                            compatible(ctx.operand(a)?, p.ty),
                            "function argument type/access mismatch",
                        )?;
                    }
                    ensure(
                        constants.len() == f.routine.header.constants.len(),
                        "function tuning constant count mismatch",
                    )?;
                    for &c in constants {
                        ctx.constant(c)?;
                    }
                    match (result, f.returns) {
                        (Some(v), Some(t)) => ensure(
                            ctx.values[v] == Type::Scalar(t),
                            "call result type mismatch",
                        )?,
                        (None, None) => {}
                        _ => return Err(ValidationError("call result presence mismatch".into())),
                    }
                }
            }
        }
        for a in b.terminator.operands() {
            check_use(a, b.id, b.instructions.len() + 1)?;
        }
        match &b.terminator {
            Terminator::Conditional { condition, .. } => ensure(
                ctx.scalar(*condition)? == ScalarType::Bool,
                "branch condition must be bool",
            )?,
            Terminator::Return(v) => match (v, returns) {
                (Some(v), Some(t)) => ensure(ctx.scalar(*v)? == t, "return type mismatch")?,
                (None, None) => {}
                _ => return Err(ValidationError("return value presence mismatch".into())),
            },
            _ => {}
        }
    }
    validate_structure(&r.body, &blocks, &predecessors, &dom)
}
fn reachable_from(
    start: BlockId,
    blocks: &BTreeMap<BlockId, &Block>,
    stops: &BTreeSet<BlockId>,
) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![start];
    while let Some(id) = pending.pop() {
        if !stops.contains(&id) && seen.insert(id) {
            pending.extend(blocks[&id].terminator.edges().iter().map(|e| e.target));
        }
    }
    seen
}
fn validate_structure(
    body: &Body,
    blocks: &BTreeMap<BlockId, &Block>,
    predecessors: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    dom: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> Check<()> {
    for b in &body.blocks {
        if matches!(b.terminator, Terminator::Conditional { .. }) {
            ensure(b.merge.is_some(), "conditional header needs merge metadata")?;
        }
        if let Some(m) = b.merge {
            ensure(
                blocks.contains_key(&m.target()) && m.target() != b.id,
                "invalid merge block",
            )?;
            if matches!(m, Merge::Selection { .. }) {
                ensure(
                    matches!(b.terminator, Terminator::Conditional { .. }),
                    "selection header must branch conditionally",
                )?;
            }
            let mut stops = BTreeSet::from([m.target()]);
            // Nested constructs may exit to enclosing merge/continue targets.
            for &outer in &dom[&b.id] {
                if outer != b.id {
                    if let Some(om) = blocks[&outer].merge {
                        stops.insert(om.target());
                        if let Merge::Loop { continuing, .. } = om {
                            stops.insert(continuing);
                        }
                    }
                }
            }
            let region = reachable_from(b.id, blocks, &stops);
            for id in &region {
                ensure(
                    dom[id].contains(&b.id),
                    "construct escapes its structured region",
                )?;
                if *id != b.id {
                    ensure(
                        predecessors[id].iter().all(|p| region.contains(p)),
                        "branch enters a construct outside its header",
                    )?;
                }
            }
            if let Merge::Loop { continuing, .. } = m {
                ensure(
                    continuing != b.id
                        && continuing != m.target()
                        && blocks.contains_key(&continuing)
                        && (region.contains(&continuing)
                            || matches!(blocks[&continuing].terminator, Terminator::Unreachable)),
                    "invalid loop continue target",
                )?;
                let back: Vec<_> = predecessors[&b.id]
                    .iter()
                    .filter(|p| dom[p].contains(&b.id))
                    .collect();
                ensure(
                    back.iter().all(|p| dom[p].contains(&continuing)),
                    "loop backedge must pass through continue target",
                )?;
            } else if let Terminator::Conditional {
                then_edge,
                else_edge,
                ..
            } = &b.terminator
            {
                let left = reachable_from(then_edge.target, blocks, &stops);
                let right = reachable_from(else_edge.target, blocks, &stops);
                ensure(
                    left.is_disjoint(&right),
                    "selection branches reconverge before their merge",
                )?;
            }
        }
    }
    // Every DFS backedge must target a declared loop header. This also rejects
    // irreducible cycles whose target does not dominate their source.
    fn visit(
        id: BlockId,
        blocks: &BTreeMap<BlockId, &Block>,
        dom: &BTreeMap<BlockId, BTreeSet<BlockId>>,
        active: &mut BTreeSet<BlockId>,
        done: &mut BTreeSet<BlockId>,
    ) -> Check<()> {
        if done.contains(&id) {
            return Ok(());
        }
        active.insert(id);
        for e in blocks[&id].terminator.edges() {
            if active.contains(&e.target) {
                ensure(
                    dom[&id].contains(&e.target)
                        && matches!(blocks[&e.target].merge, Some(Merge::Loop { .. })),
                    "unstructured cycle",
                )?;
            } else {
                visit(e.target, blocks, dom, active, done)?;
            }
        }
        active.remove(&id);
        done.insert(id);
        Ok(())
    }
    visit(
        body.entry,
        blocks,
        dom,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
    )
}
