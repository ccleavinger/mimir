use super::*;
use std::collections::BTreeMap;
pub(super) struct RoutineBuilder {
    pub(super) header: ir::Header,
    pub(super) blocks: Vec<ir::Block>,
    pub(super) current: usize,
    pub(super) locals: BTreeMap<LocalId, (ir::Operand, ir::Type)>,
    pub(super) next_value: u32,
    pub(super) next_storage: u32,
}

impl RoutineBuilder {
    pub(super) fn new() -> Self {
        let mut s = Self {
            header: ir::Header::default(),
            blocks: vec![],
            current: 0,
            locals: BTreeMap::new(),
            next_value: 0,
            next_storage: 0,
        };
        s.block();
        s
    }
    pub(super) fn block(&mut self) -> usize {
        let i = self.blocks.len();
        self.blocks.push(ir::Block {
            id: ir::BlockId(i as u32),
            arguments: vec![],
            instructions: vec![],
            merge: None,
            terminator: ir::Terminator::Unreachable,
        });
        i
    }
    pub(super) fn edge(i: usize) -> ir::Edge {
        ir::Edge {
            target: ir::BlockId(i as u32),
            arguments: vec![],
        }
    }
    pub(super) fn branch(&mut self, i: usize) {
        self.blocks[self.current].terminator = ir::Terminator::Branch(Self::edge(i));
    }
    pub(super) fn emit(&mut self, i: ir::Instruction) {
        self.blocks[self.current].instructions.push(i);
    }
    pub(super) fn value(&mut self, ty: ir::Type, op: ir::ValueOp) -> ir::Operand {
        let id = ir::ValueId(self.next_value);
        self.next_value += 1;
        self.header.variables.push(ir::Variable::Value { id, ty });
        self.emit(ir::Instruction::Value { result: id, op });
        ir::Operand::Value(id)
    }
    pub(super) fn storage(
        &mut self,
        t: ir::ScalarType,
        space: ir::AddressSpace,
        length: ir::ConstantOperand,
    ) -> ir::Operand {
        let id = ir::StorageId(self.next_storage);
        self.next_storage += 1;
        self.header.variables.push(ir::Variable::Storage {
            id,
            element: t,
            space,
            length,
        });
        ir::Operand::Storage(id)
    }
    pub(super) fn fail<T>(&self, span: &Span, message: &str) -> Result<T, LoweringError> {
        Err(LoweringError {
            span: span.clone(),
            message: message.into(),
        })
    }
}
