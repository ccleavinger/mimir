use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// SHA-256 of versioned canonical semantic content, excluding debug information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SemanticHash(pub [u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariantKey(pub [u8; 32]);

fn bytes<T: Serialize>(v: &T) -> Vec<u8> {
    postcard::to_allocvec(v).expect("serializable flat IR")
}

fn digest(domain: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(payload);
    h.finalize().into()
}

pub(crate) fn variant_key(
    hash: SemanticHash,
    constants: &[u32],
    target: &[u8],
    options: &[u8],
) -> VariantKey {
    VariantKey(digest(
        b"mimir.flat.variant.v1\0",
        &bytes(&(hash, constants, target, options)),
    ))
}

fn remap_constant(a: &mut ConstantOperand, map: &BTreeMap<ConstantId, ConstantId>) {
    if let ConstantOperand::Constant(id) = a {
        *id = map[id];
    }
}

fn constants(h: &Header) -> (Vec<DerivedConstant>, BTreeMap<ConstantId, ConstantId>) {
    let mut map: BTreeMap<_, _> = h
        .constants
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, ConstantId(i as u32)))
        .collect();

    let mut shapes: BTreeMap<ConstantId, (u32, Vec<u8>)> = h
        .constants
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, (0, bytes(&(0u8, i as u32)))))
        .collect();

    let mut groups: BTreeMap<(u32, Vec<u8>), Vec<&DerivedConstant>> = BTreeMap::new();
    for i in super::validate::constant_order(h).expect("validated constants") {
        let d = &h.derived_constants[i];
        let key = |a: ConstantOperand| match a {
            ConstantOperand::Literal(v) => (0, bytes(&(1u8, v))),
            ConstantOperand::Constant(id) => shapes[&id].clone(),
        };
        let (da, a) = key(d.lhs);
        let (db, b) = key(d.rhs);
        let shape = (
            da.max(db) + 1,
            digest(b"mimir.flat.constant.v1\0", &bytes(&(d.op, a, b))).to_vec(),
        );
        shapes.insert(d.id, shape.clone());
        groups.entry(shape).or_default().push(d);
    }

    let mut definitions = vec![];
    for group in groups.values() {
        let id = ConstantId((h.constants.len() + definitions.len()) as u32);
        for d in group {
            map.insert(d.id, id);
        }
        let mut d = group[0].clone();
        d.id = id;
        remap_constant(&mut d.lhs, &map);
        remap_constant(&mut d.rhs, &map);
        definitions.push(d);
    }

    (definitions, map)
}

fn canonical(
    r: &Routine,
    function_hashes: &BTreeMap<FunctionId, SemanticHash>,
) -> (Routine, Vec<SemanticHash>) {
    let blocks: BTreeMap<_, _> = r.body.blocks.iter().map(|b| (b.id, b)).collect();
    let mut order = vec![];
    let mut seen = BTreeSet::new();
    let mut pending = vec![r.body.entry];

    while let Some(id) = pending.pop() {
        if seen.insert(id) {
            order.push(id);
            if let Some(m) = blocks[&id].merge {
                if let Merge::Loop { continuing, .. } = m {
                    pending.push(continuing);
                }
                pending.push(m.target());
            }
            pending.extend(
                blocks[&id]
                    .terminator
                    .edges()
                    .iter()
                    .rev()
                    .map(|e| e.target),
            );
        }
    }

    let block_ids: BTreeMap<_, _> = order
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, BlockId(i as u32)))
        .collect();

    let param_ids: BTreeMap<_, _> = r
        .header
        .parameters
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, ParameterId(i as u32)))
        .collect();

    let (derived_constants, constant_ids) = constants(&r.header);
    let mut values = BTreeMap::new();

    for id in &order {
        let b = blocks[id];
        for v in b
            .arguments
            .iter()
            .copied()
            .chain(b.instructions.iter().filter_map(Instruction::result))
        {
            let next = ValueId(values.len() as u32);
            values.insert(v, next);
        }
    }

    let mut storage_ids = BTreeMap::new();
    for id in &order {
        let b = blocks[id];
        for a in b
            .instructions
            .iter()
            .flat_map(Instruction::operands)
            .chain(b.terminator.operands())
        {
            if let Operand::Storage(id) = a {
                let next = StorageId(storage_ids.len() as u32);
                storage_ids.entry(id).or_insert(next);
            }
        }
    }

    let mut unused = vec![];
    for v in &r.header.variables {
        if let Variable::Storage { id, .. } = v {
            if !storage_ids.contains_key(id) {
                let mut normalized = v.clone();
                if let Variable::Storage { id, length, .. } = &mut normalized {
                    *id = StorageId(0);
                    remap_constant(length, &constant_ids);
                }
                unused.push((bytes(&normalized), *id));
            }
        }
    }

    unused.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, id) in unused {
        let next = StorageId(storage_ids.len() as u32);
        storage_ids.insert(id, next);
    }

    let operand = |a: &mut Operand| match a {
        Operand::Value(id) => *id = values[id],
        Operand::Parameter(id) => *id = param_ids[id],
        Operand::Constant(id) => *id = constant_ids[id],
        Operand::Storage(id) => *id = storage_ids[id],
        _ => {}
    };

    let mut dependencies = vec![];
    let mut dependency_ids = BTreeMap::new();
    let mut canonical_blocks = vec![];
    for id in order {
        let mut b = blocks[&id].clone();
        b.id = block_ids[&id];
        for arg in &mut b.arguments {
            *arg = values[arg];
        }
        for inst in &mut b.instructions {
            for a in inst.operands_mut() {
                operand(a);
            }
            match inst {
                Instruction::Value { result, .. } => *result = values[result],
                Instruction::Call {
                    result,
                    function,
                    constants,
                    ..
                } => {
                    if let Some(v) = result {
                        *v = values[v];
                    }
                    let hash = function_hashes[function];
                    let next = FunctionId(dependencies.len() as u32);
                    let f = *dependency_ids.entry(hash).or_insert_with(|| {
                        dependencies.push(hash);
                        next
                    });
                    *function = f;
                    for c in constants {
                        remap_constant(c, &constant_ids);
                    }
                }
                _ => {}
            }
        }

        for e in b.terminator.edges_mut() {
            e.target = block_ids[&e.target];
            for a in &mut e.arguments {
                operand(a);
            }
        }

        match &mut b.terminator {
            Terminator::Conditional { condition, .. } => operand(condition),
            Terminator::Return(Some(a)) => operand(a),
            _ => {}
        }

        if let Some(m) = &mut b.merge {
            match m {
                Merge::Selection { merge } => *merge = block_ids[merge],
                Merge::Loop { merge, continuing } => {
                    *merge = block_ids[merge];
                    *continuing = block_ids[continuing];
                }
            }
        }

        canonical_blocks.push(b);
    }

    let mut value_vars = vec![];
    let mut storage_vars = vec![];

    for v in &r.header.variables {
        match *v {
            Variable::Value { id, ty } => value_vars.push(Variable::Value {
                id: values[&id],
                ty,
            }),
            Variable::Storage {
                id,
                element,
                space,
                mut length,
            } => {
                remap_constant(&mut length, &constant_ids);
                storage_vars.push(Variable::Storage {
                    id: storage_ids[&id],
                    element,
                    space,
                    length,
                });
            }
        }
    }

    value_vars.sort_by_key(|v| match v {
        Variable::Value { id, .. } => id.0,
        _ => unreachable!(),
    });

    storage_vars.sort_by_key(|v| match v {
        Variable::Storage { id, .. } => id.0,
        _ => unreachable!(),
    });

    value_vars.extend(storage_vars);

    (
        Routine {
            header: Header {
                parameters: r
                    .header
                    .parameters
                    .iter()
                    .map(|p| Parameter {
                        id: param_ids[&p.id],
                        ty: p.ty,
                    })
                    .collect(),
                constants: (0..r.header.constants.len())
                    .map(|i| ConstantId(i as u32))
                    .collect(),
                derived_constants,
                variables: value_vars,
            },
            body: Body {
                entry: BlockId(0),
                blocks: canonical_blocks,
            },
        },
        dependencies,
    )
}

impl ValidatedModule {
    fn function_hashes(&self) -> BTreeMap<FunctionId, SemanticHash> {
        fn visit(id: FunctionId, m: &Module, hashes: &mut BTreeMap<FunctionId, SemanticHash>) {
            if hashes.contains_key(&id) {
                return;
            }

            let f = m
                .functions
                .iter()
                .find(|f| f.id == id)
                .expect("validated function");

            for b in &f.routine.body.blocks {
                for i in &b.instructions {
                    if let Instruction::Call { function, .. } = i {
                        visit(*function, m, hashes);
                    }
                }
            }

            let (r, deps) = canonical(&f.routine, hashes);

            hashes.insert(
                id,
                SemanticHash(digest(
                    b"mimir.flat.function.v1\0",
                    &bytes(&(f.returns, r, deps)),
                )),
            );
        }

        let mut hashes = BTreeMap::new();
        for f in &self.module.functions {
            visit(f.id, &self.module, &mut hashes);
        }

        hashes
    }
    pub fn function_hash(&self, id: FunctionId) -> Result<SemanticHash, ValidationError> {
        self.function_hashes()
            .get(&id)
            .copied()
            .ok_or_else(|| ValidationError("unknown function ID".into()))
    }

    pub fn kernel_hash(&self, index: usize) -> Result<SemanticHash, ValidationError> {
        let k = self
            .module
            .kernels
            .get(index)
            .ok_or_else(|| ValidationError("unknown kernel index".into()))?;
        let (r, deps) = canonical(&k.routine, &self.function_hashes());

        Ok(SemanticHash(digest(
            b"mimir.flat.kernel.v1\0",
            &bytes(&(r, deps)),
        )))
    }
    /// Versioned canonical bytes for inspection and artifact generation. Device
    /// call targets index the accompanying ordered semantic hash table.
    pub fn canonical_kernel_bytes(&self, index: usize) -> Result<Vec<u8>, ValidationError> {
        let k = self
            .module
            .kernels
            .get(index)
            .ok_or_else(|| ValidationError("unknown kernel index".into()))?;
        let (r, deps) = canonical(&k.routine, &self.function_hashes());
        Ok(bytes(&(1u32, r, deps)))
    }
}
