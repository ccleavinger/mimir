//! Rust item shells. Function bodies stay as original token trees for chumsky;
//! syn handles signatures and attributes, not the GPU body grammar.
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{ToTokens, quote};
use syn::{
    Attribute, Signature, Visibility,
    parse::{Parse, ParseStream},
};
#[derive(Clone)]
pub struct SourceFunction {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub sig: Signature,
    pub body: TokenStream,
}
impl Parse for SourceFunction {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse()?;
        let sig = input.parse()?;
        let content;
        let braces = syn::braced!(content in input);
        let mut group = Group::new(Delimiter::Brace, content.parse()?);
        group.set_span(braces.span.join());
        Ok(Self {
            attrs,
            vis,
            sig,
            body: std::iter::once(TokenTree::Group(group)).collect(),
        })
    }
}
impl From<&syn::ItemFn> for SourceFunction {
    fn from(f: &syn::ItemFn) -> Self {
        Self {
            attrs: f.attrs.clone(),
            vis: f.vis.clone(),
            sig: f.sig.clone(),
            body: f.block.to_token_stream(),
        }
    }
}
impl ToTokens for SourceFunction {
    fn to_tokens(&self, out: &mut TokenStream) {
        let Self {
            attrs,
            vis,
            sig,
            body,
        } = self;
        out.extend(quote! {#(#attrs)* #vis #sig #body});
    }
}
