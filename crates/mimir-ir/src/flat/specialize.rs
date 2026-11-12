use super::*;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("specialization failed: {0}")]
pub struct SpecializationError(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub id: StorageId,
    pub element: ScalarType,
    pub space: AddressSpace,
    pub length: u32,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FunctionInstance {
    pub function: FunctionId,
    pub constants: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct SpecializedRoutine {
    pub constants: BTreeMap<ConstantId, u32>,
    pub allocations: Vec<Allocation>,
    /// Call-site to specialized callee. Locations use original block IDs and
    /// zero-based instruction indices, so they align with the validated IR.
    pub calls: BTreeMap<(BlockId, usize), FunctionInstance>,
}

impl SpecializedRoutine {
    /// Backend-provided byte limits, checked against total declared storage.
    /// A bool occupies a standard 32-bit memory slot, like the other scalars.
    pub fn check_allocation_limits(
        &self,
        local_bytes: u64,
        workgroup_bytes: u64,
    ) -> Result<(), SpecializationError> {
        for (space, limit) in [
            (AddressSpace::Local, local_bytes),
            (AddressSpace::Workgroup, workgroup_bytes),
        ] {
            let total = self
                .allocations
                .iter()
                .filter(|a| a.space == space)
                .try_fold(0u64, |sum, a| sum.checked_add(a.bytes));

            if !matches!(total,Some(bytes) if bytes<=limit) {
                return Err(SpecializationError(format!(
                    "{space:?} allocation limit exceeded"
                )));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SpecializedKernel {
    pub kernel_index: usize,
    pub identity: SemanticHash,
    pub tuning_values: Vec<u32>,
    pub kernel: SpecializedRoutine,
    pub functions: BTreeMap<FunctionInstance, SpecializedRoutine>,
}

impl SpecializedKernel {
    /// Opaque backend fingerprints must encode the target/device capabilities,
    /// compiler version and all code-generating options. Launch dimensions are
    /// not an intrinsic IR field or an implicit part of this key.
    pub fn variant_key(&self, target: &[u8], options: &[u8]) -> VariantKey {
        super::identity::variant_key(self.identity, &self.tuning_values, target, options)
    }
}

pub(crate) fn resolve(a: ConstantOperand, values: &BTreeMap<ConstantId, u32>) -> u32 {
    match a {
        ConstantOperand::Literal(v) => v,
        ConstantOperand::Constant(id) => values[&id],
    }
}

fn specialize(r: &Routine, inputs: &[u32]) -> Result<SpecializedRoutine, SpecializationError> {
    let h = &r.header;
    if inputs.len() != h.constants.len() {
        return Err(SpecializationError("tuning constant count mismatch".into()));
    }

    let mut constants: BTreeMap<_, _> = h
        .constants
        .iter()
        .copied()
        .zip(inputs.iter().copied())
        .collect();

    for i in super::validate::constant_order(h).map_err(|e| SpecializationError(e.to_string()))? {
        let d = &h.derived_constants[i];
        let (a, b) = (resolve(d.lhs, &constants), resolve(d.rhs, &constants));
        let value = match d.op {
            ConstantOp::Add => a.checked_add(b),
            ConstantOp::Sub => a.checked_sub(b),
            ConstantOp::Mul => a.checked_mul(b),
            ConstantOp::Div => a.checked_div(b),
            ConstantOp::Rem => a.checked_rem(b),
            ConstantOp::Min => Some(a.min(b)),
            ConstantOp::Max => Some(a.max(b)),
        }
        .ok_or_else(|| SpecializationError(format!("invalid arithmetic in constant {:?}", d.id)))?;
        constants.insert(d.id, value);
    }

    let mut allocations = vec![];
    for v in &h.variables {
        if let Variable::Storage {
            id,
            element,
            space,
            length,
        } = *v
        {
            let length = resolve(length, &constants);
            if length == 0 {
                return Err(SpecializationError(format!(
                    "zero-length allocation {id:?}"
                )));
            }
            allocations.push(Allocation {
                id,
                element,
                space,
                length,
                bytes: u64::from(length) * 4,
            });
        }
    }

    let mut calls = BTreeMap::new();
    for b in &r.body.blocks {
        for (i, inst) in b.instructions.iter().enumerate() {
            if let Instruction::Call {
                function,
                constants: bindings,
                ..
            } = inst
            {
                calls.insert(
                    (b.id, i),
                    FunctionInstance {
                        function: *function,
                        constants: bindings.iter().map(|a| resolve(*a, &constants)).collect(),
                    },
                );
            }
        }
    }

    Ok(SpecializedRoutine {
        constants,
        allocations,
        calls,
    })
}

impl ValidatedModule {
    pub fn specialize_kernel(
        &self,
        index: usize,
        constants: &[u32],
    ) -> Result<SpecializedKernel, SpecializationError> {
        let k = self
            .module
            .kernels
            .get(index)
            .ok_or_else(|| SpecializationError("unknown kernel index".into()))?;
        let kernel = specialize(&k.routine, constants)?;
        let mut pending: Vec<_> = kernel.calls.values().cloned().collect();
        let mut functions = BTreeMap::new();
        while let Some(instance) = pending.pop() {
            if functions.contains_key(&instance) {
                continue;
            }
            let f = self
                .module
                .functions
                .iter()
                .find(|f| f.id == instance.function)
                .expect("validated call target");
            let specialized = specialize(&f.routine, &instance.constants)
                .map_err(|e| SpecializationError(format!("function {:?}: {e}", f.id)))?;
            pending.extend(specialized.calls.values().cloned());
            functions.insert(instance, specialized);
        }
        Ok(SpecializedKernel {
            kernel_index: index,
            identity: self
                .kernel_hash(index)
                .map_err(|e| SpecializationError(e.to_string()))?,
            tuning_values: constants.to_vec(),
            kernel,
            functions,
        })
    }
}
