use mimir_ast::MimirGlobalAST;
use spirv_compiler::compiler::SpirVCompiler;
use syn::Block;
pub mod spirv_compiler;




pub fn compile_to_spirv(ast: &MimirGlobalAST) -> Result<Vec<u32>, String> {
    let mut compiler = SpirVCompiler::new();
    
    let body: Block = syn::parse_str(&ast.body)
        .map_err(|e| format!("Failed to parse body: {}", e))?;

    let assembled = compiler.compile_kernel(&body, &ast.parameters.clone())?;
    Ok(assembled)
}