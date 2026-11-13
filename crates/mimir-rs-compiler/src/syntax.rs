//! Side-effect-free source grammar. Resolution runs only after parsing succeeds.
use chumsky::prelude::*;
use proc_macro2::{Delimiter, Span, TokenStream, TokenTree};
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Word(String),
    Literal(String),
    Symbol(char),
}
impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Word(s) | Self::Literal(s) => f.write_str(s),
            Self::Symbol(c) => write!(f, "{c}"),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Node<T> {
    pub kind: T,
    pub span: Range<usize>,
}
pub type Expr = Node<Expression>;
pub type Stmt = Node<Statement>;
pub type Block = Vec<Stmt>;
#[derive(Clone, Debug)]
pub enum Expression {
    Literal(String),
    Path(Vec<String>),
    Paren(Box<Expr>),
    Unary(char, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, String),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Field(Box<Expr>, String),
    Repeat(Box<Expr>, Box<Expr>),
    Range(Box<Expr>, Box<Expr>),
}
#[derive(Clone, Debug)]
pub enum Statement {
    Let {
        shared: bool,
        mutable: bool,
        name: String,
        ty: Option<String>,
        value: Expr,
    },
    Assign {
        target: Expr,
        op: String,
        value: Expr,
    },
    Expression(Expr),
    Return(Option<Expr>),
    Block(Block),
    If {
        condition: Expr,
        yes: Block,
        no: Option<Block>,
    },
    For {
        name: String,
        range: Expr,
        body: Block,
    },
}
pub struct Parsed {
    pub body: Block,
    pub spans: Vec<Span>,
}
impl Parsed {
    pub fn source_span(&self, range: &Range<usize>) -> Span {
        let Some(first) = self.spans.get(range.start).copied() else {
            return Span::call_site();
        };
        self.spans
            .get(range.end.saturating_sub(1))
            .and_then(|last| first.join(*last))
            .unwrap_or(first)
    }
}

fn grammar<'a>() -> impl Parser<'a, &'a [Token], Block, extra::Err<Rich<'a, Token>>> {
    let symbol = |c| just(Token::Symbol(c)).ignored();
    let word = |w: &str| just(Token::Word(w.into())).ignored();
    let ident = select! {Token::Word(w) if syn::parse_str::<syn::Ident>(&w).is_ok()=>w};
    let op = |s: &str| just(s.chars().map(Token::Symbol).collect::<Vec<_>>()).to(s.to_string());
    let expression=recursive::<&[Token],Expr,extra::Err<Rich<Token>>,_,_>(|expr| {
        let literal=select!{Token::Literal(l)=>Expression::Literal(l),Token::Word(w) if w=="true"||w=="false"=>Expression::Literal(w)};
        let path=ident.separated_by(symbol(':').then(symbol(':'))).at_least(1).collect::<Vec<_>>().map(Expression::Path);
        let paren=expr.clone().delimited_by(symbol('('),symbol(')')).map(|e|Expression::Paren(Box::new(e)));
        let repeat=expr.clone().then_ignore(symbol(';')).then(expr.clone()).delimited_by(symbol('['),symbol(']')).map(|(v,n)|Expression::Repeat(Box::new(v),Box::new(n)));
        let atom=choice((literal,path,paren,repeat)).map_with(|kind,e|Node{kind,span:e.span().into_range()}).boxed();
        #[derive(Clone)]enum Suffix{Call(Vec<Expr>),Index(Expr),Field(String)}
        let suffix=choice((
            expr.clone().separated_by(symbol(',')).allow_trailing().collect::<Vec<_>>().delimited_by(symbol('('),symbol(')')).map(Suffix::Call),
            expr.clone().delimited_by(symbol('['),symbol(']')).map(Suffix::Index),
            symbol('.').ignore_then(ident).map(Suffix::Field),
        ));
        let postfix=atom.foldl_with(suffix.repeated(),|base,suffix,e|Node{kind:match suffix{Suffix::Call(args)=>Expression::Call(Box::new(base),args),Suffix::Index(index)=>Expression::Index(Box::new(base),Box::new(index)),Suffix::Field(field)=>Expression::Field(Box::new(base),field)},span:e.span().into_range()}).boxed();
        let unary=one_of([Token::Symbol('-'),Token::Symbol('!')]).repeated().foldr_with(postfix,|op,value,e|Node{kind:Expression::Unary(match op{Token::Symbol(c)=>c,_=>unreachable!()},Box::new(value)),span:e.span().into_range()}).boxed();
        let cast=unary.foldl_with(word("as").ignore_then(ident).repeated(),|value,ty,e|Node{kind:Expression::Cast(Box::new(value),ty),span:e.span().into_range()}).boxed();
        let binary=|left:Expr,(op,right):(String,Expr),span:SimpleSpan|Node{kind:Expression::Binary(op,Box::new(left),Box::new(right)),span:span.into_range()};
        let product=cast.clone().foldl_with(choice((op("*"),op("/"),op("%"))).then(cast).repeated(),move|a,b,e|binary(a,b,e.span())).boxed();
        let sum=product.clone().foldl_with(choice((op("+"),op("-"))).then(product).repeated(),move|a,b,e|binary(a,b,e.span())).boxed();
        let comparison=sum.clone().foldl_with(choice((op("<="),op(">="),op("=="),op("!="),op("<"),op(">"))).then(sum).repeated(),move|a,b,e|binary(a,b,e.span())).boxed();
        let and=comparison.clone().foldl_with(op("&&").then(comparison).repeated(),move|a,b,e|binary(a,b,e.span())).boxed();
        let or=and.clone().foldl_with(op("||").then(and).repeated(),move|a,b,e|binary(a,b,e.span())).boxed();
        or.clone().then(op("..").ignore_then(or).or_not()).map_with(|(start,end),e|if let Some(end)=end{Node{kind:Expression::Range(Box::new(start),Box::new(end)),span:e.span().into_range()}}else{start})
    }).boxed();
    recursive(|stmt| {
        let block = stmt
            .clone()
            .repeated()
            .collect::<Vec<_>>()
            .delimited_by(symbol('{'), symbol('}'))
            .boxed();
        let shared = symbol('#')
            .ignore_then(word("shared").delimited_by(symbol('['), symbol(']')))
            .or_not();
        let declaration = shared
            .then_ignore(word("let"))
            .then(word("mut").or_not())
            .then(ident)
            .then(symbol(':').ignore_then(ident).or_not())
            .then_ignore(symbol('='))
            .then(expression.clone())
            .then_ignore(symbol(';'))
            .map(|((((shared, mutable), name), ty), value)| Statement::Let {
                shared: shared.is_some(),
                mutable: mutable.is_some(),
                name,
                ty,
                value,
            })
            .boxed();
        let returning = word("return")
            .ignore_then(expression.clone().or_not())
            .then_ignore(symbol(';').or(symbol('}').rewind()))
            .map(Statement::Return)
            .boxed();
        let conditional = word("if")
            .ignore_then(expression.clone())
            .then(block.clone())
            .then(
                word("else")
                    .ignore_then(block.clone().or(stmt.clone().map(|stmt| vec![stmt])))
                    .or_not(),
            )
            .map(|((condition, yes), no)| Statement::If { condition, yes, no })
            .boxed();
        let looping = word("for")
            .ignore_then(ident)
            .then_ignore(word("in"))
            .then(expression.clone())
            .then(block.clone())
            .map(|((name, range), body)| Statement::For { name, range, body })
            .boxed();
        let assignment = expression
            .clone()
            .then(
                choice((op("+="), op("-="), op("*="), op("/="), op("=")))
                    .then(expression.clone())
                    .or_not(),
            )
            .then_ignore(symbol(';').or(symbol('}').rewind()))
            .map(|(target, rhs)| {
                if let Some((op, value)) = rhs {
                    Statement::Assign { target, op, value }
                } else {
                    Statement::Expression(target)
                }
            })
            .boxed();
        choice((
            declaration,
            returning,
            conditional.then_ignore(symbol(';').or_not()),
            looping.then_ignore(symbol(';').or_not()),
            block
                .map(Statement::Block)
                .then_ignore(symbol(';').or_not()),
            assignment,
        ))
        .map_with(|kind, e| Node {
            kind,
            span: e.span().into_range(),
        })
    })
    .repeated()
    .collect::<Vec<_>>()
    .delimited_by(symbol('{'), symbol('}'))
    .then_ignore(end())
}

pub fn parse(stream: TokenStream) -> syn::Result<Parsed> {
    let mut tokens = vec![];
    let mut spans = vec![];
    enum Item {
        Tree(TokenTree),
        Close(char, Span),
    }
    let mut work = stream
        .into_iter()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(Item::Tree)
        .collect::<Vec<_>>();
    while let Some(item) = work.pop() {
        let (token, span) = match item {
            Item::Close(c, s) => (Token::Symbol(c), s),
            Item::Tree(TokenTree::Ident(i)) => (Token::Word(i.to_string()), i.span()),
            Item::Tree(TokenTree::Literal(l)) => (Token::Literal(l.to_string()), l.span()),
            Item::Tree(TokenTree::Punct(p)) => (Token::Symbol(p.as_char()), p.span()),
            Item::Tree(TokenTree::Group(g)) => {
                let delimiters = match g.delimiter() {
                    Delimiter::Parenthesis => Some(('(', ')')),
                    Delimiter::Brace => Some(('{', '}')),
                    Delimiter::Bracket => Some(('[', ']')),
                    Delimiter::None => None,
                };
                if let Some((open, close)) = delimiters {
                    tokens.push(Token::Symbol(open));
                    spans.push(g.span_open());
                    work.push(Item::Close(close, g.span_close()));
                }
                work.extend(
                    g.stream()
                        .into_iter()
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .map(Item::Tree),
                );
                continue;
            }
        };
        tokens.push(token);
        spans.push(span);
    }
    let parsed = grammar().parse(tokens.as_slice()).into_result();
    match parsed {
        Ok(body) => Ok(Parsed { body, spans }),
        Err(errors) => {
            let mut diagnostics = None;
            for error in errors {
                let index = error.span().start;
                let span = spans
                    .get(index)
                    .copied()
                    .or_else(|| spans.last().copied())
                    .unwrap_or_else(Span::call_site);
                let diagnostic = syn::Error::new(span, format!("invalid GPU syntax: {error}"));
                if let Some(combined) = &mut diagnostics {
                    syn::Error::combine(combined, diagnostic)
                } else {
                    diagnostics = Some(diagnostic)
                }
            }
            Err(diagnostics
                .unwrap_or_else(|| syn::Error::new(Span::call_site(), "invalid GPU syntax")))
        }
    }
}
