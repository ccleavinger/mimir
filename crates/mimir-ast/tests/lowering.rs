use mimir_ast::*;
fn expr(e: MimirExpr) -> RichMimirExpr {
    RichExpr::new(e, 0..1)
}
fn lit(v: u32) -> RichMimirExpr {
    expr(MimirExpr::Literal(Literal::Uint32(v)))
}
fn stmt(s: MimirKernelStmt) -> RichKernelStmt {
    RichKernelStmt::new(s, 0..1)
}
fn binding(id: u32) -> Binding {
    Binding {
        id: LocalId(id),
        name: format!("v{id}"),
        span: 0..1,
    }
}
#[test]
fn lowers_mutation_nested_selection_and_loop() {
    let body = vec![
        stmt(MimirKernelStmt::Let {
            binding: binding(0),
            ty: PrimitiveTy::Uint32,
            mutability: Mutability::Mutable,
            value: lit(0),
        }),
        stmt(MimirKernelStmt::RangeFor {
            binding: binding(1),
            start: lit(0),
            end: lit(4),
            step: None,
            body: vec![stmt(MimirKernelStmt::If {
                condition: expr(MimirExpr::BinOp(MimirBinOpExpr::new(
                    expr(MimirExpr::Var(LocalId(1))),
                    MimirBinOp::Lt,
                    lit(2),
                ))),
                then_branch: vec![stmt(MimirKernelStmt::Assign {
                    target: AssignTarget::Var(LocalId(0)),
                    value: expr(MimirExpr::Var(LocalId(1))),
                })],
                else_branch: None,
            })],
        }),
    ];
    let kernel = MimirKernelAst {
        name: "loop".into(),
        parameters: vec![],
        const_generics: vec![],
        body,
    };
    let module = kernel.validate().unwrap().to_ir().unwrap();
    module
        .validate()
        .unwrap()
        .specialize_kernel(0, &[])
        .unwrap();
}
#[test]
fn lowers_early_returns_with_structural_placeholders() {
    let kernel = MimirKernelAst {
        name: "returns".into(),
        parameters: vec![],
        const_generics: vec![],
        body: vec![stmt(MimirKernelStmt::If {
            condition: expr(MimirExpr::Literal(Literal::Bool(true))),
            then_branch: vec![stmt(MimirKernelStmt::Return)],
            else_branch: Some(vec![stmt(MimirKernelStmt::Return)]),
        })],
    };
    kernel
        .validate()
        .unwrap()
        .to_ir()
        .unwrap()
        .validate()
        .unwrap();
}
