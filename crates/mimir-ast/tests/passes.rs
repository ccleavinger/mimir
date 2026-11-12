use mimir_ast::{
    AssignTarget, BinaryIntrinsic, Binding, ConstExpr, Intrinsic, KernelAstPass, KernelParamTy,
    KernelParameter, Literal, LocalId, MimirBinOp, MimirBinOpExpr, MimirExpr, MimirKernelAst,
    MimirKernelBlock, MimirKernelStmt, MimirUnOp, MimirUnaryExpr, Mutability, PrimitiveTy,
    RichConstExpr, RichExpr, RichKernelStmt, RichMimirExpr, TernaryIntrinsic,
    passes::kernel::{safety::ValidityPass, walk_const_expr, walk_expr},
};

fn kernel(body: MimirKernelBlock) -> MimirKernelAst {
    MimirKernelAst {
        name: "example".into(),
        parameters: vec![],
        const_generics: vec![],
        body,
    }
}

fn literal(value: u32) -> RichMimirExpr {
    RichExpr::new(
        MimirExpr::Literal(Literal::Uint32(value)),
        value as usize..value as usize + 1,
    )
}

fn constant(value: u32) -> RichConstExpr {
    RichExpr::new(
        ConstExpr::Literal(Literal::Uint32(value)),
        value as usize..value as usize + 1,
    )
}

fn stmt(stmt: MimirKernelStmt) -> RichKernelStmt {
    RichKernelStmt::new(stmt, 0..1)
}
fn declaration(id: u32) -> RichKernelStmt {
    stmt(MimirKernelStmt::Let {
        binding: Binding {
            id: LocalId(id),
            name: "x".into(),
            span: 0..1,
        },
        ty: PrimitiveTy::Uint32,
        mutability: Mutability::Constant,
        value: literal(1),
    })
}

#[test]
fn validity_pass_matches_validate_and_preserves_ast() {
    let mut ast = kernel(vec![declaration(0)]);
    let original = ast.clone();
    let mut pass = ValidityPass::default();
    pass.kernel(&mut ast).unwrap();
    assert_eq!(ast, original);
    assert_eq!(ast.clone().validate().unwrap().into_ast(), original);
    // The trait supports type inference without a separate expression walk.
    assert_eq!(pass.expr(&mut literal(1)).unwrap(), PrimitiveTy::Uint32);
    let mut invalid = kernel(vec![stmt(MimirKernelStmt::Assign {
        target: AssignTarget::Var(LocalId(99)),
        value: literal(1),
    })]);
    let expected = invalid.clone().validate().unwrap_err();
    assert_eq!(pass.kernel(&mut invalid).unwrap_err(), expected);
}

#[test]
fn validity_pass_can_be_reused_after_success_and_failure() {
    let mut pass = ValidityPass::default();
    let mut first = kernel(vec![declaration(0)]);
    pass.kernel(&mut first).unwrap();
    pass.kernel(&mut first).unwrap();
    let mut invalid = kernel(vec![declaration(0), declaration(0)]);
    assert!(pass.kernel(&mut invalid).is_err());
    pass.kernel(&mut first).unwrap();
    // An input in one kernel must not become visible to the next one.
    let mut parameterized = kernel(vec![]);
    parameterized.parameters.push(KernelParameter {
        id: LocalId(9),
        name: "input".into(),
        ty: KernelParamTy::Scalar(PrimitiveTy::Uint32),
        span: 0..1,
    });
    pass.kernel(&mut parameterized).unwrap();
    assert!(
        pass.expr(&mut RichExpr::new(MimirExpr::Var(LocalId(9)), 0..1))
            .is_err()
    );
}

#[derive(Default)]
struct Literals {
    runtime: Vec<u32>,
    constants: Vec<u32>,
    fail_on: Option<u32>,
}

impl KernelAstPass for Literals {
    type Error = u32;
    type ExprOutput = ();
    fn expr(&mut self, expr: &mut RichMimirExpr) -> Result<(), u32> {
        if let MimirExpr::Literal(Literal::Uint32(value)) = &mut expr.expr {
            self.runtime.push(*value);
            if self.fail_on == Some(*value) {
                return Err(*value);
            }
            *value += 1;
        }
        walk_expr(self, expr)
    }
    fn const_expr(&mut self, expr: &mut RichConstExpr) -> Result<(), u32> {
        if let ConstExpr::Literal(Literal::Uint32(value)) = &mut expr.expr {
            self.constants.push(*value);
            *value += 1;
        }
        walk_const_expr(self, expr)
    }
}

#[test]
fn default_statement_walk_visits_targets_constants_and_nested_blocks() {
    let mut ast = kernel(vec![
        stmt(MimirKernelStmt::Assign {
            target: AssignTarget::Index {
                array: LocalId(0),
                index: literal(1),
            },
            value: literal(2),
        }),
        stmt(MimirKernelStmt::SharedMem {
            binding: Binding {
                id: LocalId(0),
                name: "shared".into(),
                span: 0..1,
            },
            element: PrimitiveTy::Uint32,
            length: constant(3),
        }),
        stmt(MimirKernelStmt::RangeFor {
            binding: Binding {
                id: LocalId(1),
                name: "i".into(),
                span: 0..1,
            },
            start: literal(4),
            end: literal(5),
            step: Some(literal(6)),
            body: vec![stmt(MimirKernelStmt::If {
                condition: literal(7),
                then_branch: vec![stmt(MimirKernelStmt::Block(vec![declaration(8)]))],
                else_branch: Some(vec![stmt(MimirKernelStmt::Assign {
                    target: AssignTarget::Var(LocalId(0)),
                    value: literal(9),
                })]),
            })],
        }),
        stmt(MimirKernelStmt::If {
            condition: literal(10),
            then_branch: vec![],
            else_branch: None,
        }),
        stmt(MimirKernelStmt::Return),
        stmt(MimirKernelStmt::Syncthreads),
    ]);
    let mut pass = Literals::default();
    pass.kernel(&mut ast).unwrap();
    assert_eq!(pass.runtime, vec![1, 2, 4, 5, 6, 7, 1, 9, 10]);
    assert_eq!(pass.constants, vec![3]);
    if let MimirKernelStmt::Assign {
        target: AssignTarget::Index { index, .. },
        value,
    } = &ast.body[0].stmt
    {
        assert_eq!(index.expr, MimirExpr::Literal(Literal::Uint32(2)));
        assert_eq!(index.span, 1..2);
        assert_eq!(value.expr, MimirExpr::Literal(Literal::Uint32(3)));
    } else {
        panic!("expected indexed assignment");
    }
}

#[test]
fn expression_walk_reaches_intrinsic_arguments_and_embedded_constants() {
    let runtime = RichExpr::new(
        MimirExpr::MathIntrinsic(Intrinsic::Ternary(
            TernaryIntrinsic::Clamp,
            Box::new([
                RichExpr::new(
                    MimirExpr::BinOp(MimirBinOpExpr::new(literal(1), MimirBinOp::Add, literal(2))),
                    0..1,
                ),
                RichExpr::new(
                    MimirExpr::Unary(MimirUnaryExpr::new(MimirUnOp::Neg, literal(3))),
                    0..1,
                ),
                RichExpr::new(
                    MimirExpr::Cast {
                        value: Box::new(literal(4)),
                        to: PrimitiveTy::Float32,
                    },
                    0..1,
                ),
            ]),
        )),
        0..1,
    );

    let mut ast = kernel(vec![
        stmt(MimirKernelStmt::Let {
            binding: Binding {
                id: LocalId(0),
                name: "x".into(),
                span: 0..1,
            },
            ty: PrimitiveTy::Uint32,
            mutability: Mutability::Constant,
            value: runtime,
        }),
        stmt(MimirKernelStmt::Assign {
            target: AssignTarget::Var(LocalId(0)),
            value: RichExpr::new(
                MimirExpr::ConstExpr(RichExpr::new(
                    ConstExpr::MathIntrinsic(Intrinsic::Binary(
                        BinaryIntrinsic::Max,
                        Box::new([
                            RichExpr::new(
                                ConstExpr::BinOp(MimirBinOpExpr::new(
                                    constant(5),
                                    MimirBinOp::Add,
                                    constant(6),
                                )),
                                0..1,
                            ),
                            RichExpr::new(
                                ConstExpr::Cast {
                                    value: Box::new(RichExpr::new(
                                        ConstExpr::UnOp(MimirUnaryExpr::new(
                                            MimirUnOp::Neg,
                                            constant(7),
                                        )),
                                        0..1,
                                    )),
                                    to: PrimitiveTy::Uint32,
                                },
                                0..1,
                            ),
                        ]),
                    )),
                    0..1,
                )),
                0..1,
            ),
        }),
    ]);
    let mut pass = Literals::default();
    pass.kernel(&mut ast).unwrap();
    assert_eq!(pass.runtime, vec![1, 2, 3, 4]);
    assert_eq!(pass.constants, vec![5, 6, 7]);
}

#[test]
fn default_walk_propagates_typed_errors_and_stops() {
    let mut ast = kernel(vec![
        stmt(MimirKernelStmt::Assign {
            target: AssignTarget::Index {
                array: LocalId(0),
                index: literal(5),
            },
            value: literal(6),
        }),
        declaration(0),
    ]);
    let mut pass = Literals {
        fail_on: Some(5),
        ..Literals::default()
    };
    assert_eq!(pass.kernel(&mut ast), Err(5));
    assert_eq!(pass.runtime, vec![5]);
}
