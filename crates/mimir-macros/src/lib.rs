//! Compile Rust kernels into embedded, versioned flat IR artifacts.
use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

#[proc_macro_attribute]
pub fn mimir_global(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let result = (|| {
        let ast = mimir_rs_compiler::frontend::parse_kernel(&input)?;
        let validated = ast
            .validate()
            .map_err(|e| syn::Error::new(input.sig.ident.span(), e.to_string()))?;
        let module = validated
            .to_ir()
            .map_err(|e| syn::Error::new(input.sig.ident.span(), e.to_string()))?;
        let validated = module
            .clone()
            .validate()
            .map_err(|e| syn::Error::new(input.sig.ident.span(), e.to_string()))?;
        let hash = validated
            .kernel_hash(0)
            .map_err(|e| syn::Error::new(input.sig.ident.span(), e.to_string()))?;
        let bytes = module
            .to_bytes()
            .map_err(|e| syn::Error::new(input.sig.ident.span(), e.to_string()))?;
        Ok::<_, syn::Error>((bytes, hash.0))
    })();
    match result {
        Err(e) => e.to_compile_error().into(),
        Ok((bytes, hash)) => {
            let name = input.sig.ident;
            let vis = input.vis;
            quote! {
                #[allow(non_upper_case_globals)]
                #vis const #name: ::mimir::KernelHash = ::mimir::KernelHash([#(#hash),*]);
                ::mimir::embed_ir!(&[#name], &[#(#bytes),*]);
            }
            .into()
        }
    }
}

/// An inline GPU module. Mark entry points with `#[kernel]` and helpers with
/// `#[device]`; helper calls are resolved and validated as one flat module.
#[proc_macro_attribute]
pub fn mimir_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemMod);
    let result = (|| -> syn::Result<proc_macro2::TokenStream> {
        let (_, items) = input
            .content
            .ok_or_else(|| syn::Error::new(input.ident.span(), "GPU modules must be inline"))?;
        let mut functions = vec![];
        for item in items {
            let syn::Item::Fn(mut f) = item else {
                return Err(syn::Error::new_spanned(
                    item,
                    "GPU modules contain only kernels and device functions",
                ));
            };
            let kernel = f.attrs.iter().any(|a| a.path().is_ident("kernel"));
            let device = f.attrs.iter().any(|a| a.path().is_ident("device"));
            if kernel == device {
                return Err(syn::Error::new_spanned(
                    f,
                    "mark function with exactly one of #[kernel] or #[device]",
                ));
            }
            f.attrs.clear();
            functions.push((f, kernel));
        }
        let module = mimir_rs_compiler::frontend::compile_module(&functions)?;
        let bytes = module
            .to_bytes()
            .map_err(|e| syn::Error::new(input.ident.span(), e.to_string()))?;
        let validated = module
            .clone()
            .validate()
            .map_err(|e| syn::Error::new(input.ident.span(), e.to_string()))?;
        let name = input.ident;
        let vis = input.vis;
        let mut definitions = vec![];
        let mut index = 0usize;
        let mut hashes = vec![];
        for (f, kernel) in functions {
            if kernel {
                let ident = f.sig.ident;
                let vis = f.vis;
                let hash = validated
                    .kernel_hash(index)
                    .map_err(|e| syn::Error::new(ident.span(), e.to_string()))?
                    .0;
                definitions.push(quote!{#[allow(non_upper_case_globals)] #vis const #ident: ::mimir::KernelHash=::mimir::KernelHash([#(#hash),*]);});
                hashes.push(quote! {::mimir::KernelHash([#(#hash),*])});
                index += 1;
            }
        }
        Ok(
            quote! {#vis mod #name{#(#definitions)* ::mimir::embed_ir!(&[#(#hashes),*],&[#(#bytes),*]);}},
        )
    })();
    result.unwrap_or_else(|e| e.to_compile_error()).into()
}

mod launch;
#[proc_macro]
pub fn launch(input: TokenStream) -> TokenStream {
    launch::expand(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
