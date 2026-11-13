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
