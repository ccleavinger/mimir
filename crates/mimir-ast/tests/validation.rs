use mimir_ast::{
    AssignTarget, Binding, ConstExpr, ConstGenericId, Expr, Literal, LocalId, MimirBinOp,
    MimirBinOpExpr, MimirExpr, MimirKernelAst, MimirKernelBlock, MimirKernelStmt, Mutability,
    PrimitiveTy, RichExpr, RichKernelStmt, RichMimirExpr,
};

fn expr(e: MimirExpr) -> RichMimirExpr {
    RichExpr::new(e, 2..4)
}
fn stmt(s: MimirKernelStmt) -> RichKernelStmt {
    RichKernelStmt::new(s, 1..5)
}
fn binding(id: u32) -> Binding {
    Binding {
        id: LocalId(id),
        name: "x".into(),
        span: 1..2,
    }
}
fn kernel(body: MimirKernelBlock) -> MimirKernelAst {
    MimirKernelAst {
        name: "example".into(),
        parameters: vec![],
        const_generics: vec![],
        body,
    }
}
fn declaration(id: u32, value: RichMimirExpr, ty: PrimitiveTy) -> RichKernelStmt {
    stmt(MimirKernelStmt::Let {
        binding: binding(id),
        ty,
        mutability: Mutability::Constant,
        value,
    })
}
fn literal() -> RichMimirExpr {
    expr(MimirExpr::Literal(Literal::Uint32(1)))
}
#[test]
fn comparison_initializes_boolean() {
    let comparison = expr(MimirExpr::BinOp(MimirBinOpExpr::new(
        literal(),
        MimirBinOp::Lt,
        literal(),
    )));
    assert!(
        kernel(vec![declaration(0, comparison, PrimitiveTy::Bool)])
            .validate()
            .is_ok()
    );
}
#[test]
fn nested_binding_cannot_escape_scope() {
    let result = kernel(vec![
        stmt(MimirKernelStmt::Block(vec![declaration(
            0,
            literal(),
            PrimitiveTy::Uint32,
        )])),
        declaration(1, expr(MimirExpr::Var(LocalId(0))), PrimitiveTy::Uint32),
    ])
    .validate()
    .unwrap_err();
    assert_eq!(result.span, 2..4);
    assert!(result.message.contains("scope"));
}
#[test]
fn immutable_assignment_fails() {
    let result = kernel(vec![
        declaration(0, literal(), PrimitiveTy::Uint32),
        stmt(MimirKernelStmt::Assign {
            target: AssignTarget::Var(LocalId(0)),
            value: literal(),
        }),
    ])
    .validate()
    .unwrap_err();
    assert!(result.message.contains("immutable"));
}
#[test]
fn shared_memory_keeps_element_type_and_checks_generic() {
    let shared = stmt(MimirKernelStmt::SharedMem {
        binding: binding(0),
        element: PrimitiveTy::Float32,
        length: RichExpr::new(ConstExpr::Generic(ConstGenericId(0)), 0..1),
    });
    assert!(kernel(vec![shared.clone()]).validate().is_err());
    let mut k = kernel(vec![
        shared,
        declaration(
            1,
            expr(MimirExpr::Index {
                array: LocalId(0),
                index: Box::new(literal()),
            }),
            PrimitiveTy::Float32,
        ),
    ]);
    k.const_generics.push("N".into());
    assert!(k.validate().is_ok());
}
#[test]
fn loop_index_is_scoped_and_shadowing_uses_distinct_ids() {
    let loop_stmt = stmt(MimirKernelStmt::RangeFor {
        binding: binding(1),
        start: literal(),
        end: literal(),
        step: None,
        body: vec![declaration(
            2,
            expr(MimirExpr::Var(LocalId(1))),
            PrimitiveTy::Uint32,
        )],
    });
    assert!(
        kernel(vec![
            declaration(0, literal(), PrimitiveTy::Uint32),
            loop_stmt.clone()
        ])
        .validate()
        .is_ok()
    );
    assert!(
        kernel(vec![
            loop_stmt,
            declaration(3, expr(MimirExpr::Var(LocalId(1))), PrimitiveTy::Uint32)
        ])
        .validate()
        .is_err()
    );
}
#[test]
fn precedence_preserves_parentheses() {
    let e = expr(MimirExpr::BinOp(MimirBinOpExpr::new(
        literal(),
        MimirBinOp::Add,
        literal(),
    )));
    assert_eq!(e.priority(), 1 << 8);
    assert_eq!(e.parenthesized().priority(), 0);
}
