//! CUDA-shaped envelope parsing. Rust token trees are carried through the
//! string/Logos/chumsky pipeline so caller spans and macro hygiene survive.
use chumsky::prelude::*;
use logos::Logos;
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::quote;
use syn::{Expr, Token, parse::Parser as _, punctuated::Punctuated};

#[derive(Logos, Clone, Debug, PartialEq, Eq)]
#[logos(skip r"[ \t\r\n]+")]
enum Kind {
    #[regex(r"\$[0-9]+")]
    Atom,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[regex(r"[^\s$<>:;]")]
    Punct,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Lexeme {
    kind: Kind,
    index: usize,
}

fn reconstruct(tokens: &[Lexeme], original: &[TokenTree]) -> TokenStream {
    tokens.iter().map(|t| original[t.index].clone()).collect()
}
fn expressions(tokens: TokenStream) -> syn::Result<Vec<Expr>> {
    Ok(Punctuated::<Expr, Token![,]>::parse_terminated
        .parse2(tokens)?
        .into_iter()
        .collect())
}
fn error(message: impl std::fmt::Display) -> syn::Error {
    syn::Error::new(proc_macro2::Span::call_site(), message.to_string())
}

pub fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let original = input.into_iter().collect::<Vec<_>>();
    let mut source = String::new();
    let mut positions = std::collections::HashMap::new();
    for (i, tree) in original.iter().enumerate() {
        positions.insert(source.len(), i);
        match tree {
            TokenTree::Punct(p) => source.push(p.as_char()),
            _ => source.push_str(&format!("${i}")),
        }
        source.push(' ');
    }
    let tokens = Kind::lexer(&source)
        .spanned()
        .map(|(kind, span)| {
            Ok(Lexeme {
                kind: kind
                    .map_err(|_| error(format!("invalid launch token at byte {}", span.start)))?,
                index: positions[&span.start],
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let lt = any::<&[Lexeme], extra::Err<Rich<Lexeme>>>().filter(|t: &Lexeme| t.kind == Kind::Lt);
    let gt = any().filter(|t: &Lexeme| t.kind == Kind::Gt);
    let atom = any().filter(|t: &Lexeme| t.kind == Kind::Atom);
    let colon = any().filter(|t: &Lexeme| t.kind == Kind::Colon);
    let semi = any().filter(|t: &Lexeme| t.kind == Kind::Semi);
    let path = colon
        .then(colon)
        .or_not()
        .then(atom)
        .then(colon.then(colon).then(atom).repeated().collect::<Vec<_>>())
        .map(|((prefix, first), tail)| {
            let mut result = vec![];
            if let Some((a, b)) = prefix {
                result.extend([a, b])
            }
            result.push(first);
            for ((a, b), name) in tail {
                result.extend([a, b, name])
            }
            result
        });
    // Nested Rust generics are balanced here; parenthesized/braced expressions
    // remain opaque original token trees and are parsed by syn below.
    let generic = recursive(|nested| {
        any()
            .filter(|t: &Lexeme| !matches!(t.kind, Kind::Lt | Kind::Gt | Kind::Semi))
            .map(|t| vec![t])
            .or(lt
                .then(nested.repeated().collect::<Vec<_>>())
                .then(gt)
                .map(|((a, middle), b)| {
                    let mut out = vec![a];
                    out.extend(middle.into_iter().flatten());
                    out.push(b);
                    out
                }))
    });
    let tuning = lt
        .ignore_then(generic.clone().repeated().collect::<Vec<_>>())
        .then_ignore(gt)
        .map(|v| v.into_iter().flatten().collect::<Vec<_>>());
    let open = lt.then(lt).then(lt).ignored();
    let close = gt.then(gt).then(gt).ignored();
    let angles = lt
        .then(generic.repeated().collect::<Vec<_>>())
        .then(gt)
        .map(|((a, middle), b)| {
            let mut out = vec![a];
            out.extend(middle.into_iter().flatten());
            out.push(b);
            out
        });
    let config = angles
        .or(close.not().ignore_then(any()).map(|t| vec![t]))
        .repeated()
        .collect::<Vec<_>>()
        .map(|v| v.into_iter().flatten().collect::<Vec<_>>());
    let suffix = open.ignore_then(config).then_ignore(close).then(atom);
    let launch = path
        .then(tuning)
        .then(suffix.clone())
        .map(|((path, tuning), (config, args))| (path, tuning, config, args))
        .or(path
            .then(suffix)
            .map(|(path, (config, args))| (path, vec![], config, args)));
    let parser = semi
        .not()
        .ignore_then(any())
        .repeated()
        .collect::<Vec<_>>()
        .then_ignore(semi)
        .then(launch.clone())
        .or(launch.map(|launch| (vec![], launch)))
        .then_ignore(semi.or_not())
        .then_ignore(end());
    let (context, (path, tuning, config, args)) = parser
        .parse(tokens.as_slice())
        .into_result()
        .map_err(|errors| {
            let span = errors
                .first()
                .and_then(|e| tokens.get(e.span().start))
                .map(|t| original[t.index].span())
                .unwrap_or_else(proc_macro2::Span::call_site);
            syn::Error::new(
                span,
                "expected launch!([runtime, device;] kernel<tuning><<<grid, block>>>(arguments...))",
            )
        })?;
    let context = expressions(reconstruct(&context, &original))?;
    let kernel: syn::Path = syn::parse2(reconstruct(&path, &original))?;
    let config = expressions(reconstruct(&config, &original))?;
    let [grid, block] = config.as_slice() else {
        return Err(error(
            "launch configuration requires grid and block dimensions",
        ));
    };
    let tuning = expressions(reconstruct(&tuning, &original))?;
    let TokenTree::Group(args) = &original[args.index] else {
        return Err(error("kernel arguments must be parenthesized"));
    };
    if args.delimiter() != Delimiter::Parenthesis {
        return Err(error("kernel arguments must be parenthesized"));
    }
    let args = expressions(args.stream())?;
    // No generated unsafe block: callers must explicitly acknowledge kernel
    // execution preconditions, just as with Device::launch.
    let runtime_binding =
        proc_macro2::Ident::new("__mimir_runtime", proc_macro2::Span::mixed_site());
    let device_binding = proc_macro2::Ident::new("__mimir_device", proc_macro2::Span::mixed_site());
    let arguments = quote! {&[#((#args).as_argument()),*]};
    match context.as_slice() {
        [] => Ok(
            quote! {{use ::mimir::AsArgument as _;::mimir::launch_default(#kernel,&[#(#tuning),*],#grid,#block,#arguments)}},
        ),
        [device] => Ok(
            quote! {{use ::mimir::AsArgument as _;let #device_binding=&(#device);::mimir::launch_on_default(#device_binding,#kernel,&[#(#tuning),*],#grid,#block,#arguments)}},
        ),
        [runtime, device] => Ok(quote! {{
            use ::mimir::AsArgument as _;let #runtime_binding=&(#runtime);let #device_binding=&(#device);
            #runtime_binding.launch(#device_binding,#kernel,&[#(#tuning),*],#grid,#block,#arguments)
        }}),
        _ => Err(error(
            "launch accepts no context, a device, or runtime and device before ';'",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parses(source: &str) -> bool {
        expand(source.parse().unwrap()).is_ok()
    }
    #[test]
    fn parses_cuda_envelopes_and_nested_rust() {
        for source in [
            "runtime, device; kernel<<<grid, dims::<Foo<Bar<Baz>>>>()>>>()",
            "kernel<<<grid, block>>>()",
            "device; kernel<4u32><<<grid, block>>>()",
            "runtime, device; kernel<<<grid, block>>>()",
            "&runtime, &device; helpers::kernel<N, {N + 1} ><<<[1,1,1], dims()>>>(buffer, value + 1);",
            "runtime(), device; ::kernels::add<{config::<u32, u64>()}> <<<grid, block>>>(make::<u32, u64>(), |x,y| x+y)",
            "runtime, device; kernel<<<grid, block>>>(\"raw,;<>\", r###\"text>>>\"###)",
        ] {
            assert!(parses(source), "{source}");
        }
    }
    #[test]
    fn rejects_malformed_envelopes() {
        for source in [
            "r,d,extra;k<<<g,b>>>()",
            "r,d;k<<g,b>>()",
            "r,d;k<<<g>>>()",
            "r,d;k<<<g,b>>>[]",
            "r,d;k<<<g,b>>>() trailing",
        ] {
            assert!(!parses(source), "{source}");
        }
    }
}
