use mimir_rs_compiler::frontend::{compile_module, parse_kernel};
fn function(source: &str) -> syn::ItemFn {
    syn::parse_str(source).unwrap()
}
#[test]
fn rejects_recursive_and_mistyped_device_calls() {
    let f = function("fn recursive(x:u32)->u32 { return recursive(x); }");
    assert!(
        compile_module(&[(f, false)])
            .unwrap_err()
            .to_string()
            .contains("recursive")
    );
    let f = function("fn helper(x:u32)->u32 { return x; }");
    let k = function("fn kernel(out:&mut [u32]) { out[0]=helper(true); }");
    assert!(
        compile_module(&[(f, false), (k, true)])
            .unwrap_err()
            .to_string()
            .contains("type mismatch")
    );
}
#[test]
fn preserves_source_ranges_and_lexical_scope() {
    let ast = parse_kernel(&function("fn example(x:u32){ let y:u32=x+1; } ")).unwrap();
    assert!(ast.parameters[0].span.end > ast.parameters[0].span.start);
    assert!(ast.body[0].span.end > ast.body[0].span.start);
}

fn source(source: &str) -> syn::Result<mimir_ast::MimirKernelAst> {
    let function: syn::Result<mimir_rs_compiler::SourceFunction> = syn::parse_str(source);
    mimir_rs_compiler::frontend::parse_kernel_source(&function?)
}
#[test]
fn source_grammar_preserves_precedence() {
    use mimir_ast::*;
    let ast = source("fn example(){let x:i32=1+2*3-4/2;let b:bool=1<2 && !false || false;}")
        .unwrap()
        .validate()
        .unwrap();
    let MimirKernelStmt::Let { value, .. } = &ast.ast().body[0].stmt else {
        panic!()
    };
    let MimirExpr::BinOp(root) = &value.expr else {
        panic!()
    };
    assert_eq!(root.op, MimirBinOp::Sub);
    let MimirExpr::BinOp(left) = &root.lhs.expr else {
        panic!()
    };
    assert_eq!(left.op, MimirBinOp::Add);
    let MimirExpr::BinOp(product) = &left.rhs.expr else {
        panic!()
    };
    assert_eq!(product.op, MimirBinOp::Mul);
    let MimirExpr::BinOp(right) = &root.rhs.expr else {
        panic!()
    };
    assert_eq!(right.op, MimirBinOp::Div);
    let MimirKernelStmt::Let { value, .. } = &ast.ast().body[1].stmt else {
        panic!()
    };
    let MimirExpr::BinOp(or) = &value.expr else {
        panic!()
    };
    assert_eq!(or.op, MimirBinOp::Or);
    let MimirExpr::BinOp(and) = &or.lhs.expr else {
        panic!()
    };
    assert_eq!(and.op, MimirBinOp::And);
}
#[test]
fn source_body_errors_come_from_chumsky_and_have_source_spans() {
    let error = source("fn broken(){let x:u32=;}").unwrap_err();
    assert!(error.to_string().contains("invalid GPU syntax"));
    assert!(error.span().byte_range().start > 0);
    assert!(source("fn broken(){let x:u32=1 let y:u32=2;}").is_err());
}
#[test]
fn source_resolution_preserves_scopes_and_reports_unknown_names() {
    assert!(
        source("fn bad(){{let x:u32=1;}let y:u32=x;}")
            .unwrap_err()
            .to_string()
            .contains("unknown local")
    );
    source("fn ok(x:i32){let y:i32=1;if x<0 {let y:i32=2;}else if x<2 {let y:i32=3;}let z:i32=y;}")
        .unwrap()
        .validate()
        .unwrap();
}
#[test]
fn final_assignment_without_semicolon_and_shared_step_ranges_still_work() {
    source("fn example<const N:u32>(out:&mut [u32]){#[shared] let mut scratch=[0u32;N];for i in (0u32..N).step_by(1u32){scratch[i]=i;}out[0u32]=scratch[0u32]}").unwrap().validate().unwrap().to_ir().unwrap().validate().unwrap();
}
#[test]
fn raw_identifiers_are_preserved_and_keywords_are_not_bindings() {
    source("fn ok(){let r#type:u32=1u32;let value:u32=r#type;}")
        .unwrap()
        .validate()
        .unwrap();
    assert!(source("fn bad(){let if:u32=1u32;}").is_err());
}
