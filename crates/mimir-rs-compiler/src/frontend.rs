//! Rust item signatures, source grammar, and semantic resolution entry points.
use crate::SourceFunction;
use mimir_ast::*;
use std::collections::BTreeMap;
use syn::{spanned::Spanned, *};
type Result<T> = syn::Result<T>;
fn ty(t: &Type) -> Result<PrimitiveTy> {
    if let Type::Path(t) = t {
        match t.path.get_ident().map(|x| x.to_string()).as_deref() {
            Some("u32") => return Ok(PrimitiveTy::Uint32),
            Some("i32") => return Ok(PrimitiveTy::Int32),
            Some("f32") => return Ok(PrimitiveTy::Float32),
            Some("bool") => return Ok(PrimitiveTy::Bool),
            _ => {}
        }
    }
    Err(syn::Error::new(t.span(), "expected u32, i32, f32 or bool"))
}

pub fn parse_kernel(f: &ItemFn) -> Result<MimirKernelAst> {
    parse_kernel_source(&SourceFunction::from(f))
}
pub fn parse_kernel_source(f: &SourceFunction) -> Result<MimirKernelAst> {
    parse_with(f, BTreeMap::new())
}
fn parse_with(
    f: &SourceFunction,
    functions: BTreeMap<String, (u32, Vec<KernelParamTy>, PrimitiveTy)>,
) -> Result<MimirKernelAst> {
    let source = crate::syntax::parse(f.body.clone())?;
    let mut p = crate::resolve::Resolver {
        source,
        names: BTreeMap::new(),
        generics: BTreeMap::new(),
        next: 0,
        types: BTreeMap::new(),
        functions,
    };
    let mut const_generics = vec![];
    for g in &f.sig.generics.params {
        let GenericParam::Const(c) = g else {
            return Err(syn::Error::new(
                g.span(),
                "only u32 const generics are supported",
            ));
        };
        if ty(&c.ty)? != PrimitiveTy::Uint32 {
            return Err(syn::Error::new(c.span(), "const generics must be u32"));
        }
        p.generics.insert(
            c.ident.to_string(),
            ConstGenericId(const_generics.len() as u32),
        );
        const_generics.push(c.ident.to_string());
    }
    let mut parameters = vec![];
    for arg in &f.sig.inputs {
        let FnArg::Typed(a) = arg else {
            return Err(syn::Error::new(arg.span(), "methods are unsupported"));
        };
        let Pat::Ident(n) = a.pat.as_ref() else {
            return Err(syn::Error::new(a.span(), "expected parameter name"));
        };
        let binding = p.bind(&n.ident);
        let param_ty = if let Type::Reference(r) = a.ty.as_ref() {
            let Type::Slice(s) = r.elem.as_ref() else {
                return Err(syn::Error::new(r.span(), "expected slice"));
            };
            KernelParamTy::Buffer {
                element: ty(&s.elem)?,
                access: if r.mutability.is_some() {
                    Mutability::Mutable
                } else {
                    Mutability::Constant
                },
            }
        } else {
            KernelParamTy::Scalar(ty(&a.ty)?)
        };
        p.types.insert(
            binding.id,
            match param_ty {
                KernelParamTy::Scalar(t) => t,
                KernelParamTy::Buffer { element, .. } => element,
            },
        );
        parameters.push(KernelParameter {
            id: binding.id,
            name: binding.name,
            ty: param_ty,
            span: a.span().byte_range(),
        });
    }

    Ok(MimirKernelAst {
        name: f.sig.ident.to_string(),
        parameters,
        const_generics,
        body: {
            let body = std::mem::take(&mut p.source.body);
            p.block(&body)?
        },
    })
}

/// Compile a self-contained Rust GPU module with module-local device functions.
pub fn compile_module(items: &[(ItemFn, bool)]) -> Result<mimir_ir::flat::Module> {
    let source = items
        .iter()
        .map(|(f, k)| (SourceFunction::from(f), *k))
        .collect::<Vec<_>>();
    compile_module_sources(&source)
}
pub fn compile_module_sources(items: &[(SourceFunction, bool)]) -> Result<mimir_ir::flat::Module> {
    let mut signatures = BTreeMap::new();
    for (f, kernel) in items {
        if !kernel {
            let ReturnType::Type(_, result) = &f.sig.output else {
                return Err(syn::Error::new(
                    f.sig.span(),
                    "device functions require a scalar return type",
                ));
            };
            if !f.sig.generics.params.is_empty() {
                return Err(syn::Error::new(
                    f.sig.span(),
                    "generic device functions are not yet supported",
                ));
            }
            let mut signature = f.clone();
            signature.body = quote::quote!({});
            let ast = parse_kernel_source(&signature)?;
            let id = signatures.len() as u32;
            if signatures
                .insert(
                    f.sig.ident.to_string(),
                    (
                        id,
                        ast.parameters.iter().map(|p| p.ty).collect(),
                        ty(result)?,
                    ),
                )
                .is_some()
            {
                return Err(syn::Error::new(
                    f.sig.ident.span(),
                    "duplicate device function",
                ));
            }
        }
    }
    let mut module = mimir_ir::flat::Module::default();
    for (f, kernel) in items {
        let ast = parse_with(f, signatures.clone())?;
        if *kernel {
            if !matches!(f.sig.output, ReturnType::Default) {
                return Err(syn::Error::new(
                    f.sig.output.span(),
                    "kernels cannot return values",
                ));
            }
            let ast = ast
                .validate()
                .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?;
            module.kernels.extend(
                ast.to_ir()
                    .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?
                    .kernels,
            );
        } else {
            let (id, _, returns) = signatures[&f.sig.ident.to_string()].clone();
            let function = MimirFunctionAst {
                routine: ast,
                returns,
            }
            .validate()
            .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?;
            module.functions.push(
                function
                    .to_ir(id)
                    .map_err(|e| syn::Error::new(f.sig.span(), e.to_string()))?,
            );
        }
    }
    module.clone().validate().map_err(|e| {
        syn::Error::new(
            items.first().map(|(f, _)| f.sig.span()).unwrap_or_else(|| {
                let f: ItemFn = syn::parse_quote!(
                    fn empty() {}
                );
                f.span()
            }),
            e.to_string(),
        )
    })?;
    Ok(module)
}
