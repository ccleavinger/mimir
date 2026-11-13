//! Structured CFG layout, independent of SPIR-V emission and physical table order.
use crate::{CompileError, Result};
use mimir_ir::flat::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn structured_order(body: &Body) -> Result<Vec<&Block>> {
    enum Frame {
        Enter(BlockId),
        Leave([Option<BlockId>; 2]),
    }
    let blocks = body
        .blocks
        .iter()
        .map(|b| (b.id, b))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut deferred = BTreeMap::<BlockId, usize>::new();
    let mut work = vec![Frame::Enter(body.entry)];
    let mut ordered = Vec::with_capacity(blocks.len());
    while let Some(frame) = work.pop() {
        match frame {
            Frame::Enter(id) => {
                if deferred.contains_key(&id) || !seen.insert(id) {
                    continue;
                }
                let block = *blocks
                    .get(&id)
                    .ok_or_else(|| CompileError(format!("missing CFG block {id:?}")))?;
                ordered.push(block);
                let targets = match block.merge {
                    Some(Merge::Selection { merge }) => [Some(merge), None],
                    Some(Merge::Loop { merge, continuing }) => [Some(continuing), Some(merge)],
                    None => [None, None],
                };
                for target in targets.into_iter().flatten() {
                    *deferred.entry(target).or_default() += 1;
                }
                work.push(Frame::Leave(targets));
                match &block.terminator {
                    Terminator::Branch(edge) => work.push(Frame::Enter(edge.target)),
                    Terminator::Conditional {
                        then_edge,
                        else_edge,
                        ..
                    } => {
                        work.push(Frame::Enter(else_edge.target));
                        work.push(Frame::Enter(then_edge.target));
                    }
                    _ => {}
                }
            }
            Frame::Leave(targets) => {
                for target in targets.into_iter().flatten() {
                    let count = deferred.get_mut(&target).expect("balanced layout frame");
                    *count -= 1;
                    if *count == 0 {
                        deferred.remove(&target);
                    }
                }
                for target in targets.into_iter().rev().flatten() {
                    work.push(Frame::Enter(target));
                }
            }
        }
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn long_cfg_layout_uses_an_explicit_stack() {
        let n = 20000;
        let mut blocks = (0..n)
            .map(|i| Block {
                id: BlockId(i),
                arguments: vec![],
                instructions: vec![],
                merge: None,
                terminator: if i + 1 < n {
                    Terminator::Branch(Edge {
                        target: BlockId(i + 1),
                        arguments: vec![],
                    })
                } else {
                    Terminator::Return(None)
                },
            })
            .collect::<Vec<_>>();
        blocks.reverse();
        let body = Body {
            entry: BlockId(0),
            blocks,
        };
        let order = structured_order(&body).unwrap();
        assert_eq!(order.len(), n as usize);
        for (i, block) in order.iter().enumerate() {
            assert_eq!(block.id, BlockId(i as u32));
        }
    }
}

#[cfg(test)]
mod structured_tests {
    use super::*;
    fn edge(id: u32) -> Edge {
        Edge {
            target: BlockId(id),
            arguments: vec![],
        }
    }
    fn block(id: u32, merge: Option<Merge>, terminator: Terminator) -> Block {
        Block {
            id: BlockId(id),
            arguments: vec![],
            instructions: vec![],
            merge,
            terminator,
        }
    }
    #[test]
    fn nested_selection_defers_its_enclosing_continue_and_merge() {
        let condition = Operand::Literal(Literal::Bool(true));
        let mut blocks = vec![
            block(
                0,
                Some(Merge::Loop {
                    merge: BlockId(3),
                    continuing: BlockId(2),
                }),
                Terminator::Conditional {
                    condition,
                    then_edge: edge(1),
                    else_edge: edge(3),
                },
            ),
            block(
                1,
                Some(Merge::Selection { merge: BlockId(2) }),
                Terminator::Conditional {
                    condition,
                    then_edge: edge(4),
                    else_edge: edge(5),
                },
            ),
            block(2, None, Terminator::Branch(edge(0))),
            block(3, None, Terminator::Return(None)),
            block(4, None, Terminator::Branch(edge(2))),
            block(5, None, Terminator::Branch(edge(2))),
        ];
        blocks.reverse();
        let body = Body {
            entry: BlockId(0),
            blocks,
        };
        assert_eq!(
            structured_order(&body)
                .unwrap()
                .iter()
                .map(|b| b.id.0)
                .collect::<Vec<_>>(),
            [0, 1, 4, 5, 2, 3]
        );
    }
}
