use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
use mimir_ast::MimirGlobalAST;

#[proc_macro_attribute]
pub fn mimir_global(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let filepath = "debug.mimir.json";
    
    // Create AST representation
    let ast = MimirGlobalAST::from_syn_fn(&input_fn);
    
    // Load existing ASTs from file
    let mut asts = MimirGlobalAST::load_from_file(filepath).unwrap_or_default();

    // Remove any existing AST with the same name
    asts.retain(|i_ast| i_ast.name != ast.name);

    // Append the new AST
    asts.push(ast);
    
    // Save the updated list to file
    if let Err(e) = MimirGlobalAST::save_to_file(&asts, filepath) {
        panic!("Failed to save ASTs to file: {}", e);
    }

    // return no function to avoid rust compiler errors
    quote! {
        
    }.into()
}