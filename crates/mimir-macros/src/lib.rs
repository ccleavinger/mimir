// super duper cool library that holds the macros that turn Rust into CUDA but better
// :)

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use mimir_ir::ir::{MimirIRData, MimirIRKind, MimirPrimitiveTy, MimirTy, MimirTyScope};
use proc_macro::TokenStream;
use proc_macro_error2::{abort, proc_macro_error};
use quote::quote;
use regex::Regex;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Expr, ItemFn, Token};

use mimir_rs_compiler::compiler::kernel_compiler;

// Define custom punctuation for <<< and >>>
mod kw {
    syn::custom_punctuation!(LtLtLt, <<<);
    syn::custom_punctuation!(GtGtGt, >>>);
}

const FILENAME: &str = "mimir.bin";

#[proc_macro_attribute]
#[proc_macro_error]
pub fn mimir_global(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    // Create bytecode from input_fn
    let mimir_kernel_ir = match kernel_compiler::Compiler::global_compiler(&input_fn) {
        Ok(m_k_ir) => m_k_ir,
        Err(err) => panic!("{}", err.to_string()),
    };

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let pkg_name = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME not set");

    // Save to target directory or project root
    let filepath = PathBuf::from(&manifest_dir).join(format!("{}.{}", pkg_name, FILENAME));

    // Load existing ASTs from file
    let mut file: Box<dyn BufRead> = Box::new(BufReader::new(match File::open(&filepath) {
        Ok(file_h) => file_h,
        Err(_) => File::create_new(&filepath).unwrap(),
    }));
    let mut ir_data = match MimirIRData::load(&mut file) {
        Ok(data) => data,
        Err(_) => MimirIRData {
            irs: HashMap::new(),
            source_hashes: HashMap::new(),
        },
    };

    // Replace any existing AST with the same name
    let _ = ir_data
        .irs
        .entry(mimir_kernel_ir.name.clone())
        .insert_entry(MimirIRKind::Kernel(mimir_kernel_ir.clone()));

    let write_file = OpenOptions::new().write(true).open(&filepath).unwrap();
    let mut output_file: Box<dyn Write> = Box::new(write_file);

    if let Err(err) = MimirIRData::save(&ir_data, &mut output_file) {
        panic!("{}", err.to_string())
    };

    // return no function to avoid rust compiler errors
    quote! {}.into()
}

pub(crate) struct Launch {
    pub name: String,
    pub grid_dim: Expr,
    pub block_dim: Expr,
    pub cgs: Vec<Expr>,
    pub args: Punctuated<Expr, Token![,]>,
}

impl Parse for Launch {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Get the raw input string and normalize whitespace
        let input_str = input.to_string();
        let normalized = input_str.split_whitespace().collect::<Vec<_>>().join(" ");

        // Regex to match: name<cgs><<<grid, block>>>(args);
        // or: name<<<grid, block>>>(args);
        // Uses lookahead to find >>> without consuming content that might contain commas
        let re = Regex::new(
            r"(?x)
            ^(\w+)                          # Capture group 1: kernel name
            \s*                             # Optional whitespace
            (?:<([^>]+)>)?                  # Capture group 2: optional const generics
            \s*                             # Optional whitespace
            <<<                             # Literal <<<
            \s*                             # Optional whitespace
            (.+?)                           # Capture group 3: grid_dim (non-greedy)
            \s*,\s*                         # Comma with optional whitespace
            (.+?)                           # Capture group 4: block_dim (non-greedy)
            \s*                             # Optional whitespace
            >>>                             # Literal >>>
            \s*                             # Optional whitespace
            \((.+)\)                        # Capture group 5: arguments
            \s*;                            # Semicolon with optional whitespace
            ",
        )
        .unwrap();

        let caps = re.captures(&normalized).ok_or_else(|| {
            input.error(format!("Invalid launch syntax.\nKernel str: {}", input_str))
        })?;

        // Extract kernel name
        let name = caps
            .get(1)
            .ok_or_else(|| input.error("Failed to parse kernel name"))?
            .as_str()
            .to_string();

        // Extract const generics if present
        let mut cgs = Vec::new();
        if let Some(cgs_match) = caps.get(2) {
            let cgs_str = cgs_match.as_str();
            // Parse each const generic expression
            for cg_str in cgs_str.split(',') {
                let cg_str = cg_str.trim();
                let cg_expr: Expr = syn::parse_str(cg_str).map_err(|e| {
                    input.error(format!("Failed to parse const generic '{}': {}", cg_str, e))
                })?;
                cgs.push(cg_expr);
            }
        }

        // Extract and parse grid_dim
        let grid_dim_str = caps
            .get(3)
            .ok_or_else(|| input.error("Failed to parse grid dimension"))?
            .as_str()
            .trim();
        let grid_dim: Expr = syn::parse_str(grid_dim_str)
            .map_err(|e| input.error(format!("Failed to parse grid dimension: {}", e)))?;

        // Extract and parse block_dim
        let block_dim_str = caps
            .get(4)
            .ok_or_else(|| input.error("Failed to parse block dimension"))?
            .as_str()
            .trim();
        let block_dim: Expr = syn::parse_str(block_dim_str)
            .map_err(|e| input.error(format!("Failed to parse block dimension: {}", e)))?;

        // Extract and parse arguments
        let args_str = caps
            .get(5)
            .ok_or_else(|| input.error("Failed to parse arguments"))?
            .as_str()
            .trim();

        // Parse the arguments as a comma-separated list
        let args: Punctuated<Expr, Token![,]> = syn::parse_str(&format!("({})", args_str))
            .map(|expr: syn::ExprTuple| {
                let mut punctuated: Punctuated<Expr, syn::token::Comma> = Punctuated::new();
                for elem in expr.elems {
                    punctuated.push(elem);
                }
                punctuated
            })
            .map_err(|e| input.error(format!("Failed to parse arguments: {}", e)))?;

        // Consume all tokens from the input stream so syn doesn't complain
        input.step(|cursor| {
            let mut rest = *cursor;
            while let Some((_, next)) = rest.token_tree() {
                rest = next;
            }
            Ok(((), rest))
        })?;

        Ok(Launch {
            name,
            grid_dim,
            block_dim,
            cgs,
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
        cgs,
    } = parse_macro_input!(input as Launch);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let pkg_name = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME not set");

    // Save to target directory or project root
    let filepath = PathBuf::from(&manifest_dir).join(format!("{}.{}", pkg_name, FILENAME));

    let mut file: Box<dyn BufRead> = Box::new(BufReader::new(File::open(&filepath).unwrap()));
    let ir_data = MimirIRData::load(&mut file).unwrap();

    let kernel_ir = {
        match ir_data.irs.get(&name) {
            Some(ir_kind) => match ir_kind {
                MimirIRKind::Kernel(mimir_kernel_ir) => mimir_kernel_ir,
            },
            None => panic!("Couldn't find a kernel with the name `{name}`"),
        }
    };

    let params_ty_var = kernel_ir
        .var_map
        .iter()
        .filter(|(_, ty_var)| matches!(ty_var.scope, MimirTyScope::Param))
        // .map(|(_, ty_var)| ty_var)
        .collect::<Vec<_>>();

    let params = kernel_ir
        .param_order
        .iter()
        .map(|id| {
            let mut ret = None;
            for (param_id, param) in &params_ty_var {
                if **param_id == *id {
                    ret = Some(*param);
                }
            }
            match ret {
                Some(param) => param,
                None => {
                    panic!(
                        "ID `{}` was marked as a parameter but not found in the param order",
                        id
                    );
                }
            }
        })
        .collect::<Vec<_>>();

    let wrapped_args = args
        .iter()
        .enumerate()
        .map(|(i, arg_expr)| {
            if i < params.len() {
                let param = params[i];

                match &param.ty {
                    MimirTy::Primitive(MimirPrimitiveTy::Int32) => {
                        quote! { Param::Var(MimirVar::Int32(#arg_expr)) }
                    }
                    MimirTy::Primitive(MimirPrimitiveTy::Uint32) => {
                        quote! { Param::Var(MimirVar::Uint32(#arg_expr)) }
                    }
                    // MimirTy::Int64 => {
                    //     quote! { Param::Var(MimirVar::Int64(Box::new(#arg_expr))) }
                    // }
                    MimirTy::Primitive(MimirPrimitiveTy::Float32) => {
                        quote! { Param::Var(MimirVar::Float32(#arg_expr)) }
                    }
                    // MimirTy::Float64 => {
                    //     quote! { Param::Var(MimirVar::Float64(Box::new(#arg_expr))) }
                    // }
                    MimirTy::Primitive(MimirPrimitiveTy::Bool) => {
                        quote! { Param::Var(MimirVar::Bool(#arg_expr)) }
                    }
                    MimirTy::GlobalArray { element_type } => match &element_type {
                        MimirPrimitiveTy::Int32 => {
                            quote! { Param::Buffer(MimirBuffer::Int32(((#arg_expr)))) }
                        },
                        MimirPrimitiveTy::Uint32 => {
                            quote! { Param::Buffer(MimirBuffer::Uint32(((#arg_expr)))) }
                        },
                        // MimirTy::Int64 => {
                        //     quote! { Param::Buffer(MimirBuffer::Int32(Box::new(#arg_expr))) }
                        // }
                        MimirPrimitiveTy::Float32 => {
                            quote! { Param::Buffer(MimirBuffer::Float32(((#arg_expr)))) }
                        },
                        // MimirTy::Float64 => {
                        //     quote! { Param::Buffer(MimirBuffer::Int32(Box::new(#arg_expr))) }
                        // }
                        MimirPrimitiveTy::Bool => {
                            quote! { Param::Buffer(MimirBuffer::Bool(((#arg_expr)))) }
                        },
                    },
                    MimirTy::SharedMemArray { .. } => {
                        abort!(
                            arg_expr.span(),
                            "Argument type mismatch, expecting invalid Shared memory array as a parameter, this should be impossible"
                        )
                    }
                }
            } else {
                abort!(
                    arg_expr.span(),
                    "Argument count mismatch: expected {} arguments, but got {}",
                    params.len(),
                    args.len(),
                )
            }
        })
        .collect::<Vec<_>>();

    // Convert to a comma-separated list of args
    let arg_tokens = quote! {
        vec![#(#wrapped_args),*].as_slice()
    };

    let grid_dim_path = grid_dim;
    let block_dim_path = block_dim;

    quote! {
        mimir_kernel::launch_kernel_name(
            (#name).to_owned(),
            &#grid_dim_path,
            &#block_dim_path,
            #arg_tokens,
            &[#(#cgs),*]
        )
    }
    .into()
}
