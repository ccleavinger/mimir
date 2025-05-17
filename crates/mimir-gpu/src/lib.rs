use proc_macro::TokenStream;
use proc_macro_error2::{abort, proc_macro_error};
use quote::quote;
use syn::{parenthesized, parse_macro_input, Expr, ItemFn, Token};
use mimir_ast::MimirGlobalAST;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use regex::Regex; // Import Regex

// Define custom punctuation for <<< and >>>
mod kw {
    syn::custom_punctuation!(LtLtLt, <<<);
    syn::custom_punctuation!(GtGtGt, >>>);
}

#[proc_macro_attribute]
#[proc_macro_error]
pub fn mimir_global(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    cfg_if::cfg_if! {
        if #[cfg(feature = "json")] {
            let filepath = "mimir.json";
        } else {
            let filepath = "mimir.bin";
        }
    }

    // Create AST representation
    let ast = MimirGlobalAST::from_syn_fn(&input_fn);

    // Load existing ASTs from file
    let mut asts = MimirGlobalAST::load_from_file(filepath).unwrap_or_else(|_| {
        vec![]
    });

    // Remove any existing AST with the same name
    asts.retain(|i_ast| i_ast.name != ast.name);

    // Append the new AST
    asts.push(ast);

    // Save the updated list to file
    if let Err(e) = MimirGlobalAST::save_to_file(&asts, filepath) {
        abort!("Failed to save ASTs to file: {}", e);
    }

    // return no function to avoid rust compiler errors
    quote! {
        
    }.into()
}

struct Launch {
    name: String,
    grid_dim: Expr,
    block_dim: Expr,
    args: Punctuated<syn::Expr, Token![,]>, // Changed from Vec<String>
}

impl Parse for Launch {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse::<syn::Ident>()?.to_string();

        input.parse::<kw::LtLtLt>()?; // Parse <<<

        let grid_dim = input.parse::<syn::Expr>()?;

        input.parse::<Token![,]>()?;

        let block_dim = if input.peek(syn::token::Bracket) {
            let array = input.parse::<syn::ExprArray>()?;
            Expr::Array(array)
        } else {
            let path = input.parse::<syn::ExprPath>()?;
            Expr::Path(path)
        };

        input.parse::<kw::GtGtGt>()?; // Parse >>>

        let args_content;
        parenthesized!(args_content in input);
        let args = Punctuated::<syn::Expr, Token![,]>::parse_terminated(&args_content)?;

        input.parse::<Token![;]>()?;

        Ok(Launch {
            name,
            grid_dim,
            block_dim,
            args,
        })
    }
}

#[proc_macro]
#[proc_macro_error]
pub fn launch(input: TokenStream) -> TokenStream {
    let Launch {
        name,
        grid_dim,
        block_dim,
        args, // Now args is Punctuated<syn::Expr, Token![,]>
    } = parse_macro_input!(input as Launch);

    cfg_if::cfg_if! {
        if #[cfg(feature = "json")] {
            let filepath = "mimir.json";
        } else {
            let filepath = "mimir.bin";
        }
    }

    let asts = MimirGlobalAST::load_from_file(filepath).unwrap_or_default();

    let ast = asts.iter().find(|ast| ast.name == name).unwrap_or_else(|| {
        abort!("Kernel with name {} not found in ASTs", name);
    });

    // Create Regex for buffer types like [T]
    let buffer_regex = Regex::new(r"\[(.*?)\]").unwrap();

    // Create a vector of wrapped parameters
    let wrapped_args = args.iter().enumerate().map(|(i, arg_expr)| { // Iterate over syn::Expr
        if i < ast.parameters.len() {
            let param = &ast.parameters[i];
            let param_type = &param.type_name;

            // Check if the type matches the buffer pattern using regex
            if let Some(captures) = buffer_regex.captures(param_type) {
                // Extract the inner type name (captured group 1)
                let inner_type = captures.get(1).map_or("", |m| m.as_str());
                
                // Use #arg_expr directly for buffer types
                match inner_type {
                    "i32" => {
                        quote! { Param::Buffer(MimirBuffer::Int32(Box::new(#arg_expr))) }
                    },
                    "i64" => {
                        quote! { Param::Buffer(MimirBuffer::Int64(Box::new(#arg_expr))) }
                    },
                    "f32" => {
                        quote! { Param::Buffer(MimirBuffer::Float32(Box::new(#arg_expr))) }
                    },
                    "bool" => {
                        quote! { Param::Buffer(MimirBuffer::Bool(Box::new(#arg_expr))) }
                    },
                    _ => abort!(
                        proc_macro2::Span::call_site(),
                        "Unsupported buffer type: {}",
                        inner_type
                    ),
                }

            } else {
                // Type is not a buffer, handle as push constant
                // Use #arg_expr directly for push constants
                match param_type.as_str() {
                    "i32" => {
                        quote! { Param::PushConst(MimirPushConst::Int32(#arg_expr)) }
                    },
                    "i64" => {
                        quote! { Param::PushConst(MimirPushConst::Int64(#arg_expr)) }
                    },
                    "f32" => {
                        quote! { Param::PushConst(MimirPushConst::Float32(#arg_expr)) }
                    },
                    "bool" => {
                        quote! { Param::PushConst(MimirPushConst::Bool(#arg_expr)) }
                    },
                    _ => abort!(
                        proc_macro2::Span::call_site(),
                        "Unsupported parameter type: {}",
                        param_type
                    ),
                }
            }
        } else {
            abort!(
                arg_expr.span(), // Use arg_expr.span()
                "Argument count mismatch: expected {} arguments, but got {}",
                ast.parameters.len(),
                args.len()
            );
        }
    }).collect::<Vec<_>>();

    // Convert to a comma-separated list of args
    let arg_tokens = quote! {
        vec![#(#wrapped_args),*].as_slice()
    };

    let grid_dim_path = grid_dim;
    let block_dim_path = block_dim;

    quote! {
        MimirKernel::launch_kernel_name(
            #name,
            &#grid_dim_path,
            &#block_dim_path,
            #arg_tokens,
        )
    }.into()
}