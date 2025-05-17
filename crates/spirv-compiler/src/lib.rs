use anyhow::Result;
use mimir_ast::MimirGlobalAST;
use rspirv::binary::Assemble;
use spirv_compiler::compiler::SpirVCompiler;
use syn::Block;
pub mod spirv_compiler;


pub fn compile_to_spirv(ast: &MimirGlobalAST) -> Result<Vec<u32>> {
    let mut compiler = SpirVCompiler::new();
    
    let body: Block = syn::parse_str(&ast.body)?;

    let assembled_result = compiler.compile_kernel(&body, &ast.parameters.clone());

    if assembled_result.is_err() {
        let info_dump = format!(
            "Vars: {:?}\n\nTypes: {:?}\n\nLiterals: {:?}\n\n IR: {:?}\n\n",
            compiler.vars,
            compiler.types,
            compiler.literals,
            compiler.ir
        );

        println!("Compilation failed with the following info dump:\n{}", info_dump);
    }

    let assembled = assembled_result?.assemble();

    Ok(assembled)
}