//! Typed lowering passes: declarations and storage, expression values, and
//! structured control flow. Parsing and name resolution belong to the frontend.
use crate::*;
use mimir_ir::flat as ir;
mod builder;
mod expressions;
mod statements;
mod types;
use builder::RoutineBuilder;
use types::{binary, literal, scalar};

#[derive(Debug, thiserror::Error)]
#[error("{message} at {span:?}")]
pub struct LoweringError {
    pub span: Span,
    pub message: String,
}

impl ValidatedKernel {
    pub fn to_ir(&self) -> Result<ir::Module, LoweringError> {
        let ast = self.ast();
        let mut l = RoutineBuilder::new();
        l.header.constants = (0..ast.const_generics.len())
            .map(|i| ir::ConstantId(i as u32))
            .collect();
        for (i, p) in ast.parameters.iter().enumerate() {
            let id = ir::ParameterId(i as u32);
            let ty = match p.ty {
                KernelParamTy::Scalar(t) => ir::Type::Scalar(scalar(t)),
                KernelParamTy::Buffer { element, access } => ir::Type::Buffer {
                    element: scalar(element),
                    space: ir::AddressSpace::Global,
                    access: match access {
                        Mutability::Mutable => ir::Access::ReadWrite,
                        Mutability::Constant => ir::Access::Read,
                    },
                },
            };
            l.header.parameters.push(ir::Parameter { id, ty });
            l.locals.insert(p.id, (ir::Operand::Parameter(id), ty));
        }
        if l.statements(&ast.body)? {
            l.blocks[l.current].terminator = ir::Terminator::Return(None)
        }
        let module = ir::Module {
            kernels: vec![ir::Kernel {
                routine: ir::Routine {
                    header: l.header,
                    body: ir::Body {
                        entry: ir::BlockId(0),
                        blocks: l.blocks,
                    },
                },
                debug: ir::DebugInfo {
                    name: ast.name.clone(),
                    ..Default::default()
                },
            }],
            functions: vec![],
        };
        Ok(module)
    }
}

/// A device function uses the same lexical body grammar as a kernel and has an
/// explicit scalar return type. Whole-module IR validation resolves call IDs.
pub fn lower_function(
    mut ast: MimirKernelAst,
    id: u32,
    returns: PrimitiveTy,
) -> Result<ir::DeviceFunction, LoweringError> {
    crate::passes::kernel::safety::ValidityPass::default()
        .function(&mut ast, returns)
        .map_err(|e| LoweringError {
            span: e.span,
            message: e.message,
        })?;
    let validated = crate::valid::ValidatedKernel::from_function(ast);
    let mut module = validated.to_ir()?;
    let kernel = module.kernels.remove(0);
    Ok(ir::DeviceFunction {
        id: ir::FunctionId(id),
        routine: kernel.routine,
        returns: Some(scalar(returns)),
        debug: kernel.debug,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deeply_nested_statement_lowering_uses_continuations() {
        let mut body = vec![RichKernelStmt::new(MimirKernelStmt::Return, 0..1)];
        for _ in 0..10000 {
            body = vec![RichKernelStmt::new(MimirKernelStmt::Block(body), 0..1)];
        }
        let mut builder = RoutineBuilder::new();
        assert!(!builder.statements(&body).unwrap());
        assert!(matches!(
            builder.blocks[0].terminator,
            ir::Terminator::Return(None)
        ));
        // Dismantle the test tree iteratively as well; recursive AST drop is
        // independent of the lowering algorithm under test.
        while let Some(statement) = body.pop() {
            if let MimirKernelStmt::Block(inner) = statement.stmt {
                body.extend(inner);
            }
        }
    }
}
