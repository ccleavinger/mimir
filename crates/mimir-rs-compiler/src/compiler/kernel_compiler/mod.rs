use core::f32;
use mimir_ir::{
    compiler_err,
    ir::{
        MathIntrinsicExpr, MathIntrinsicFunc, MimirBinOp, MimirBinOpExpr, MimirBlock,
        MimirCastExpr, MimirConstExpr, MimirExpr, MimirIndexExpr, MimirLit, MimirPrimitiveTy,
        MimirStmt, MimirStmtAssignLeft, MimirTy, MimirTyAnnotation, MimirTyScope, MimirTyVar,
        MimirUnOp, MimirUnOpExpr,
    },
    kernel_ir::MimirKernelIR,
    passes::{KernelIRPrePassSettings, kernel_ir_pre_pass},
    util::{
        builtin::{str_to_builtin, str_to_builtin_field},
        cast,
        error::ASTError,
        math::{self},
        ty,
    },
};
use quote::ToTokens;
use std::{
    collections::{BTreeMap, HashMap},
    str::FromStr,
    vec,
};
use syn::{FnArg, ItemFn, ReturnType, Signature, Stmt};

use crate::compiler::kernel_compiler::util::calculate_hash;

pub mod for_loop;
pub mod if_else;

#[macro_use]
pub mod util;
mod binop;

pub struct Compiler {
    map: BTreeMap<u64, MimirTyVar>,
    hash_to_uuid: HashMap<u64, u64>,
    uuid_to_name: BTreeMap<u64, String>,
    body: MimirBlock,
    // order of param uuids
    param_order: Vec<u64>,
    const_generics: HashMap<u64, usize>,
    // manage scoping. (probably wrong term but basically whenever you enter a new block)
    // (we limit to 255 max nested scopes, which is more than enough for virtually all kernels)
    scope_stack: Vec<u8>, // last value is current scope
    // scope_stack[0] will always be 0, whenever nest ends the last values is popped
    curr_max_scope: u8, // always increased whenever enters new scope
}

impl Compiler {
    fn parse_param(&mut self, param: &FnArg) -> Result<(), ASTError> {
        match param {
            syn::FnArg::Receiver(_) => {
                return compiler_err!("`self` statements aren't allowed in Kernels, please remove");
            }
            syn::FnArg::Typed(pat_type) => {
                match *pat_type.ty.clone() {
                    syn::Type::Array(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a fixed size array is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::BareFn(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a bare function is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Group(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a grouped type is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::ImplTrait(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing an impl trait is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Infer(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, type inference is not allowed, please specify a type",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Macro(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a macro as a type is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Never(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, the never type is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Paren(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a parenthesized type is not allowed, please remove the parentheses",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Path(type_path) => {
                        let name = pat_type.pat.to_token_stream().to_string();

                        let is_mut = match *pat_type.pat {
                            syn::Pat::Ident(ref pat_ident) => pat_ident.mutability.is_some(),
                            _ => false,
                        };

                        let ty = match type_path.path.segments.last() {
                            Some(seg) => ty::str_to_ty(seg.ident.to_string().as_str()),
                            None => {
                                return Err(ASTError::Compiler(format!(
                                    "{} is invalid, could not parse type",
                                    pat_type.to_token_stream()
                                )));
                            }
                        }?
                        .to_mimir_ty();

                        let annotation = if is_mut {
                            MimirTyAnnotation::Mutable
                        } else {
                            MimirTyAnnotation::Constant
                        };

                        self.add_ty_var(
                            &name,
                            MimirTyVar {
                                ty,
                                scope: MimirTyScope::Param,
                                annotation,
                            },
                        )?;

                        let uuid = self.get_uuid(&name).unwrap();
                        self.param_order.push(*uuid);
                    }
                    syn::Type::Ptr(_) => {
                        return Err(ASTError::Compiler(format!(
                            "{} is invalid, passing a raw pointer is not allowed",
                            pat_type.to_token_stream()
                        )));
                    }
                    syn::Type::Reference(type_reference) => {
                        if let syn::Type::Slice(ty_slice) = *type_reference.clone().elem {
                            let name = pat_type.pat.to_token_stream().to_string();

                            let ty = MimirTy::GlobalArray {
                                element_type: (ty::str_to_ty(
                                    ty_slice.elem.to_token_stream().to_string().as_str(),
                                )?),
                            };

                            let annotation = MimirTyAnnotation::Mutable;

                            self.add_ty_var(
                                &name,
                                MimirTyVar {
                                    ty,
                                    scope: MimirTyScope::Param,
                                    annotation,
                                },
                            )?;

                            let uuid = self.get_uuid(&name).unwrap();
                            self.param_order.push(*uuid);

                            return Ok(());
                        } else {
                            return compiler_err!(
                                "`{}` is an invalid parameter",
                                type_reference.elem.to_token_stream().to_string()
                            );
                        }
                    }
                    syn::Type::Slice(ty_slice) => {
                        let name = pat_type.pat.to_token_stream().to_string();

                        let ty = {
                            let ty = MimirTy::GlobalArray {
                                element_type: (ty::str_to_ty(
                                    ty_slice.elem.to_token_stream().to_string().as_str(),
                                )?),
                            };

                            // all arrays in global mem are mutable. we do not care whether
                            let annotation = MimirTyAnnotation::Mutable;

                            MimirTyVar {
                                ty,
                                scope: MimirTyScope::Param,
                                annotation,
                            }
                        };

                        self.add_ty_var(&name, ty)?;

                        let uuid = self.get_uuid(&name).unwrap();
                        self.param_order.push(*uuid);
                    }
                    syn::Type::TraitObject(_) => {
                        return compiler_err!(
                            "{} is invalid, passing a trait object is not allowed",
                            pat_type.to_token_stream().to_string()
                        );
                    }
                    syn::Type::Tuple(_) => {
                        return compiler_err!(
                            "{} is invalid, passing a tuple is not allowed",
                            pat_type.to_token_stream().to_string()
                        );
                    }
                    syn::Type::Verbatim(_) => {
                        return compiler_err!(
                            "{} is invalid, this is an unkown pattern",
                            pat_type.to_token_stream().to_string()
                        );
                    }
                    _ => return compiler_err!("Unkown expression in parameter"),
                }
            }
        }
        Ok(())
    }

    pub fn global_compiler(item_fn: &ItemFn) -> Result<MimirKernelIR, ASTError> {
        Self::validate_kernel_sig(&item_fn.sig)?;

        let name = &item_fn.sig.ident.to_token_stream().to_string();

        let (const_gens, cg_map) = {
            let gens = &item_fn.sig.generics.params;
            let mut cgs = vec![];
            let mut cg_map = HashMap::new();
            for (i, g) in gens.iter().enumerate() {
                match g {
                    syn::GenericParam::Lifetime(_) => continue,
                    syn::GenericParam::Type(_) => continue,
                    syn::GenericParam::Const(const_param) => {
                        let ty = const_param.ty.to_token_stream().to_string();

                        if ty != "usize" && ty != "u32" {
                            return compiler_err!(
                                "Invalid const param of type `{}`, must be either usize or u32 right now",
                                ty
                            );
                        }
                        cgs.push(i);
                        cg_map.insert(calculate_hash(&const_param.ident.to_string()), i);
                    }
                }
            }
            (cgs, cg_map)
        };

        let mut compiler = Self {
            map: BTreeMap::new(),
            hash_to_uuid: HashMap::new(),
            uuid_to_name: BTreeMap::new(),
            body: vec![],
            // name: name.clone(),
            param_order: vec![],
            scope_stack: vec![0],
            curr_max_scope: 0,
            const_generics: cg_map.clone(),
        };

        // do param stuf
        for param in item_fn.sig.inputs.iter() {
            compiler.parse_param(param)?;
        }

        for (i, stmt) in item_fn.block.stmts.clone().iter().enumerate() {
            let error = compiler.parse_stmt(stmt);

            if let Err(err) = error {
                return Err(ASTError::Compiler(format!(
                    "Line {} of kernel `{}` for `{}`\n{}",
                    i + 1,
                    name,
                    stmt.to_token_stream(),
                    err
                )));
            }
        }

        /*
        name: name.clone(),
            var_map: compiler.map.clone(),
            name_map: compiler.uuid_to_name.clone(),
            body: compiler.body.clone(),
            param_order: compiler.param_order.clone(),
            const_generics: const_gens.len(),
         */
        let kernel_ir = MimirKernelIR::new(
            name.clone(),
            compiler.map.clone(),
            compiler.param_order.clone(),
            compiler.uuid_to_name.clone(),
            compiler.body.clone(),
            const_gens.len(),
        );

        let final_ir = kernel_ir_pre_pass(
            &kernel_ir,
            KernelIRPrePassSettings {
                const_expr_pass: true,
            },
        )?;

        Ok(final_ir)
    }

    fn parse_stmt(&mut self, stmt: &Stmt) -> Result<(), ASTError> {
        let stmt_ret = self.parse_stmt_ret_ir(stmt)?;

        if let Some(mimir_ir) = stmt_ret {
            self.body.push(mimir_ir.clone())
        }

        Ok(())
    }

    fn parse_stmt_ret_ir(&mut self, stmt: &Stmt) -> Result<Option<MimirStmt>, ASTError> {
        match stmt {
            syn::Stmt::Local(local) => {
                if local
                    .clone()
                    .attrs
                    .into_iter()
                    .any(|x| x.meta.to_token_stream().to_string() == "shared")
                {
                    if let Some(init) = local.clone().init.as_ref() {
                        if let syn::Expr::Repeat(repeat) = init.expr.as_ref() {
                            let mimir_repeat_len = self.parse_expr(&repeat.len)?.to_const_expr()?;

                            let ty = MimirTy::SharedMemArray {
                                length: Box::new(mimir_repeat_len),
                            };
                            let local_name = local
                                .pat
                                .to_token_stream()
                                .to_string()
                                .replace("mut", "")
                                .replace(" ", "");
                            let ty_var = MimirTyVar {
                                ty,
                                scope: MimirTyScope::Local(*self.scope_stack.last().unwrap()),
                                annotation: MimirTyAnnotation::Mutable,
                            };

                            let uuid = self.create_uuid(&local_name)?;

                            self.map.insert(uuid, ty_var);

                            return Ok(None);
                        } else {
                            return compiler_err!(
                                "Cannot initialize a shared memory array without using a repeat expression, i.e.: `[0f32; 256];`."
                            );
                        }
                    } else {
                        return compiler_err!(
                            "Cannot have an uninitialized shared memory array. Size and type must be specified"
                        );
                    }
                }

                let (local_name, annotation) = if let syn::Pat::Ident(ident) = local.clone().pat {
                    (
                        ident
                            .to_token_stream()
                            .to_string()
                            .replace("mut", "")
                            .replace(" ", ""),
                        match ident.mutability {
                            Some(_) => MimirTyAnnotation::Mutable,
                            None => MimirTyAnnotation::Constant,
                        },
                    )
                } else if let syn::Pat::Type(ty) = &local.pat {
                    if let syn::Pat::Ident(ident) = ty.pat.as_ref() {
                        (
                            ident
                                .to_token_stream()
                                .to_string()
                                .replace("mut", "")
                                .replace(" ", ""),
                            match ident.mutability {
                                Some(_) => MimirTyAnnotation::Mutable,
                                None => MimirTyAnnotation::Constant,
                            },
                        )
                    } else {
                        return compiler_err!(
                            "Cannot parse `{}` in `let` statment",
                            &local.pat.to_token_stream().to_string()
                        );
                    }
                } else {
                    return compiler_err!(
                        "Cannot parse `{}` in `let` statment",
                        &local.pat.to_token_stream().to_string()
                    );
                };

                let expr = if let Some(init) = local.clone().init.as_ref() {
                    let bind_ref = init.expr.as_ref();
                    let bind = self.parse_expr(bind_ref)?;
                    Some(bind)
                } else {
                    None
                };

                let ty: Option<MimirTy> = {
                    if let Some(expr_) = &expr {
                        Some(self.expr_ty(expr_)?.to_mimir_ty())
                    } else if let syn::Pat::Type(typ) = local.clone().pat {
                        Some(MimirTy::Primitive(ty::str_to_ty(
                            &typ.ty.to_token_stream().to_string(),
                        )?))
                    } else {
                        None
                    }
                };

                // if &local_name == "max_val" {
                //     return compiler_err!(
                //         "max_val info:\nMimir type annotation {:?}\nRight hand side expression: {:?}\nRight hand side mimir expr: {:?}\nRHS ty: {:?}",
                //         annotation,
                //         local.clone().init.as_ref().map(|x| x.expr.to_token_stream()),
                //         expr,
                //         ty
                //     );
                // }

                if let Some(ty_) = ty {
                    let ty_var = MimirTyVar {
                        ty: ty_,
                        scope: MimirTyScope::Local(*self.scope_stack.last().unwrap()),
                        annotation,
                    };

                    let uuid = self.create_uuid(&local_name)?;

                    self.map.insert(uuid, ty_var);

                    if let Some(mimir_expr) = &expr {
                        let ir = MimirStmt::Assign {
                            lhs: MimirStmtAssignLeft::Var(uuid),
                            rhs: mimir_expr.clone(),
                        };

                        return Ok(Some(ir));
                    }
                } else {
                    // if we don't know the type then we just get the uuid set and pray that everything else works right *-*
                    let _ = self.create_uuid(&local_name)?;

                    // return compiler_err!(
                    //     "Variable assignment failed.\nType: {ty:?}\nLHS: {}\nRHS: {:?}",
                    //     local.pat.to_token_stream(),
                    //     local.init.clone().map(|init| init.expr.to_token_stream())
                    // )
                }
            }
            syn::Stmt::Item(item) => {
                // at least rn this will always error but some macros may get implemented
                // idk
                let mimir_ir = self.parse_item(item)?.clone();
                return Ok(Some(mimir_ir));
            }
            syn::Stmt::Expr(expr, _) => {
                let mimir_ir = self.parse_body_expr(expr)?;
                return Ok(Some(mimir_ir));
            }
            syn::Stmt::Macro(_) => return compiler_err!("Macros currently aren't supported"),
        }

        Ok(None)
    }

    fn parse_body_expr(
        &mut self,
        expr: &syn::Expr,
        // map: &mut BTreeMap<String, MimirTyVar>,
        // names: &mut Vec<String>,
    ) -> Result<MimirStmt, ASTError> {
        match expr {
            // syn::Expr::Array(expr_array) => todo!(),
            syn::Expr::Assign(expr_assign) => {
                let lhs = match self.parse_expr(&expr_assign.left)? {
                    MimirExpr::Index(index_expr) => MimirStmtAssignLeft::Index(index_expr),
                    MimirExpr::Var(id) => MimirStmtAssignLeft::Var(id),
                    _ => {
                        return compiler_err!(
                            "The left hand side of an assign statement must be either an index expression or a variable access expression."
                        );
                    }
                };

                let rhs = self.parse_expr(&expr_assign.right)?;

                // handle if a variable exists but its type isn't declared with it.
                if let MimirStmtAssignLeft::Var(var) = lhs {
                    let option_name = self.uuid_to_name.get(&var);

                    if !self.map.contains_key(&var) {
                        if let Some(some_name) = option_name {
                            let name = some_name.replace(" mut", "");
                            // get the type variable
                            let ty = self.expr_ty(&rhs)?.to_mimir_ty();

                            let idx;
                            {
                                let mut opt_idx = None;
                                {
                                    let hash = calculate_hash(&name);

                                    if let Some(uuid) = self.hash_to_uuid.get(&hash)
                                        && *uuid == var
                                    {
                                        opt_idx = Some(0)
                                    }
                                }

                                for scope in self.scope_stack[1..].iter() {
                                    let scoped_name = format!("{name}\"{scope}\"");
                                    let hash = calculate_hash(&scoped_name);

                                    if let Some(uuid) = self.hash_to_uuid.get(&hash)
                                        && *uuid == var
                                    {
                                        opt_idx = Some(*scope);
                                        break;
                                    }
                                }

                                if let Some(val) = opt_idx {
                                    idx = val;
                                } else {
                                    return compiler_err!(
                                        "Couldn't find a variable named `{name}` in the current scope.\nUUID -> Var Name map: {:?}\nScope stack: {:?}",
                                        self.uuid_to_name,
                                        self.scope_stack
                                    );
                                }
                            }
                            self.map.insert(
                                var,
                                MimirTyVar {
                                    ty,
                                    scope: MimirTyScope::Local(idx),
                                    annotation: MimirTyAnnotation::Mutable,
                                },
                            );
                        } else {
                            return compiler_err!(
                                "Variable `{}` has a UUID but no name or associated typing!",
                                expr_assign.left.to_token_stream(),
                            );
                        }
                    }
                }

                let ir = MimirStmt::Assign { lhs, rhs };

                Ok(ir)
            }
            // syn::Expr::Async(expr_async) => todo!(),
            // syn::Expr::Await(expr_await) => todo!(),
            syn::Expr::Binary(expr_binary) => {
                let op = match expr_binary.op {
                    syn::BinOp::AddAssign(_) => MimirBinOp::Add,
                    syn::BinOp::SubAssign(_) => MimirBinOp::Sub,
                    syn::BinOp::MulAssign(_) => MimirBinOp::Mul,
                    syn::BinOp::DivAssign(_) => MimirBinOp::Div,
                    syn::BinOp::RemAssign(_) => MimirBinOp::Mod,
                    _ => {
                        return compiler_err!(
                            "Invalid binary operation in statement tried to use `{:?}`",
                            expr_binary.op.to_token_stream()
                        );
                    }
                };

                let lhs = match self.parse_expr(&expr_binary.left)? {
                    MimirExpr::Index(index_expr) => MimirStmtAssignLeft::Index(index_expr),
                    MimirExpr::Var(id) => MimirStmtAssignLeft::Var(id),
                    _ => {
                        return compiler_err!(
                            "The left hand side of an assign statement must be either an index expression or a variable access expression."
                        );
                    }
                };
                let rhs = self.parse_expr(&expr_binary.right)?;

                let expr = self.compile_binop_from_expr(&lhs.to_norm_expr(), &op, &rhs)?;

                Ok(MimirStmt::Assign { lhs, rhs: expr })
            }
            // syn::Expr::Block(expr_block) => todo!(),
            // syn::Expr::Break(expr_break) => todo!(),
            syn::Expr::Call(expr_call) => {
                let func_sig = expr_call.func.to_token_stream().to_string();

                match func_sig.as_str() {
                    "syncthreads" | "__syncthreads" => Ok(MimirStmt::Syncthreads),
                    _ => Err(ASTError::Compiler(format!(
                        "Unknown function: `{func_sig}`, Mimir currently only supports `synchthreads()` as a standalone function."
                    ))),
                }
            }
            // syn::Expr::Cast(expr_cast) => todo!(),
            // syn::Expr::Closure(expr_closure) => todo!(),
            // syn::Expr::Const(expr_const) => todo!(),
            // syn::Expr::Continue(expr_continue) => todo!(),
            // syn::Expr::Field(expr_field) => todo!(),
            syn::Expr::ForLoop(expr_for_loop) => self.compile_for_loop(expr_for_loop),
            // syn::Expr::Group(expr_group) => todo!(),
            syn::Expr::If(expr_if) => self.compile_if_else(expr_if),
            // syn::Expr::Index(expr_index) => {},
            // syn::Expr::Infer(expr_infer) => todo!(),
            // syn::Expr::Let(expr_let) => todo!(),
            // syn::Expr::Lit(expr_lit) => todo!(),
            // syn::Expr::Loop(expr_loop) => todo!(),
            // syn::Expr::Macro(expr_macro) => todo!(),
            // syn::Expr::Match(expr_match) => todo!(),
            // syn::Expr::MethodCall(expr_method_call) => todo!(),
            // syn::Expr::Paren(expr_paren) => todo!(),
            // syn::Expr::Path(expr_path) => todo!(),
            // syn::Expr::Range(expr_range) => todo!(),
            // syn::Expr::RawAddr(expr_raw_addr) => todo!(),
            // syn::Expr::Reference(expr_reference) => todo!(),
            // syn::Expr::Repeat(expr_repeat) => todo!(),
            // syn::Expr::Return(expr_return) => todo!(),
            // syn::Expr::Struct(expr_struct) => todo!(),
            // syn::Expr::Try(expr_try) => todo!(),
            // syn::Expr::TryBlock(expr_try_block) => todo!(),
            // syn::Expr::Tuple(expr_tuple) => todo!(),
            // syn::Expr::Unary(expr_unary) => todo!(),
            // syn::Expr::Unsafe(expr_unsafe) => todo!(),
            // syn::Expr::Verbatim(token_stream) => todo!(),
            // syn::Expr::While(expr_while) => todo!(),
            // syn::Expr::Yield(expr_yield) => todo!(),
            _ => {
                compiler_err!(
                    "Unsuported expression `{}`",
                    expr.to_token_stream().to_string()
                )
            }
        }
    }

    fn parse_item(&self, item: &syn::Item) -> Result<MimirStmt, ASTError> {
        match item {
            syn::Item::Const(_) => Err(ASTError::Compiler(
                "Constants are not implmented yet".to_owned(),
            )),
            syn::Item::Enum(_) => Err(ASTError::Compiler(
                "Creating enums inside kernels isn't permitted".to_owned(),
            )),
            syn::Item::ExternCrate(_) => Err(ASTError::Compiler(
                "Externing crates inside kernels isn't permitted".to_owned(),
            )),
            syn::Item::Fn(_) => Err(ASTError::Compiler(
                "Declaring functions within kernels isn't permitted".to_owned(),
            )),
            syn::Item::ForeignMod(_) => Err(ASTError::Compiler(
                "Foriegn modules within kernels isn't permitted".to_owned(),
            )),
            syn::Item::Impl(_) => Err(ASTError::Compiler(
                "Implementing traites within kernels isn't permitted".to_owned(),
            )),
            syn::Item::Macro(_) => Err(ASTError::Compiler(
                "Invoking macros within kernels isn't permitted (yet)".to_owned(),
            )),
            syn::Item::Mod(_) => Err(ASTError::Compiler(
                "Module declerations within kernels isn't permitted".to_owned(),
            )),
            syn::Item::Static(_) => Err(ASTError::Compiler(
                "Static items aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Struct(_) => Err(ASTError::Compiler(
                "Struct declerations aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Trait(_) => Err(ASTError::Compiler(
                "Trait declerations aren't permitted within kernels".to_owned(),
            )),
            syn::Item::TraitAlias(_) => Err(ASTError::Compiler(
                "Trait aliases aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Type(_) => Err(ASTError::Compiler(
                "Type aliases aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Union(_) => Err(ASTError::Compiler(
                "Union delcerations aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Use(_) => Err(ASTError::Compiler(
                "Use declerations aren't permitted within kernels".to_owned(),
            )),
            syn::Item::Verbatim(token_stream) => {
                Err(ASTError::Compiler(format!("Unkown item: {}", token_stream)))
            }
            _ => Err(ASTError::Compiler("Unkown item".to_owned())),
        }
    }

    // recursive func to evaluate IR
    // only parses expressions used inside statements (syn expresses this a little weird)
    fn parse_expr(&self, expr: &syn::Expr) -> Result<MimirExpr, ASTError> {
        match expr {
            syn::Expr::Array(_) => Err(ASTError::Compiler(
                "Array literals aren't allowed".to_owned(),
            )),
            syn::Expr::Assign(_) => Err(ASTError::Compiler(
                "Assignments aren't allowed within expressions".to_owned(),
            )),
            syn::Expr::Async(_) => Err(ASTError::Compiler(
                "Async operations aren't allowed within kernels".to_owned(),
            )),
            syn::Expr::Await(_) => Err(ASTError::Compiler(
                "Await operations aren't allowed within kernels".to_owned(),
            )),
            syn::Expr::Binary(expr_binary) => self.compile_binop(expr_binary),
            syn::Expr::Block(_) => Err(ASTError::Compiler(
                "Blocks aren't currently allowed inside expressions".to_owned(),
            )),
            syn::Expr::Break(_) => Err(ASTError::Compiler(
                "`break` statements aren't allowed inside expressions".to_owned(),
            )),
            syn::Expr::Call(expr_call) => {
                let func_str = &expr_call.func.to_token_stream().to_string();
                if let Ok(math_intrinsic) = MathIntrinsicFunc::from_str(func_str) {
                    let params = expr_call
                        .args
                        .iter()
                        .map(|expr| self.parse_expr(expr))
                        .collect::<Result<Vec<_>, _>>()?;

                    let tys = params
                        .iter()
                        .map(|param| self.expr_ty(param))
                        .collect::<Result<Vec<_>, _>>()?;

                    if !math::intrinsic_param_validity(&math_intrinsic, &tys) {
                        return Err(ASTError::Compiler("Invalid math intrinsic!".to_owned()));
                    }

                    return Ok(MimirExpr::MathIntrinsic(MathIntrinsicExpr {
                        func: math_intrinsic,
                        args: params,
                    }));
                }

                compiler_err!(
                    "Unkown function: {func_str}!\nMimir has limited function support {func_str} is not supported"
                )
            }
            syn::Expr::Cast(expr_cast) => {
                let expr = self.parse_expr(&expr_cast.expr)?;

                let expr_ty = self.expr_ty(&expr)?.to_mimir_ty();

                let to = match expr_cast.ty.as_ref().clone() {
                    syn::Type::Paren(paren) => {
                        if let syn::Type::Path(path) = *paren.elem {
                            ty::str_to_ty(&path.to_token_stream().to_string())?
                        } else {
                            return Err(ASTError::Compiler(format!(
                                "Only primitive casting is permitted, `{:?}` can't be casted to",
                                expr_cast.ty.to_token_stream().to_string()
                            )));
                        }
                    }
                    syn::Type::Path(path) => ty::str_to_ty(&path.to_token_stream().to_string())?,
                    _ => {
                        return Err(ASTError::Compiler(format!(
                            "Only primitive casting is permitted, `{:?}` can't be casted to",
                            expr_cast.ty.to_token_stream().to_string()
                        )));
                    }
                };

                if cast::is_legal(&expr_ty, &to) {
                    Ok(MimirExpr::Cast(MimirCastExpr {
                        from: Box::new(expr.clone()),
                        to: to.clone(),
                    }))
                } else {
                    Err(ASTError::Compiler(format!(
                        "Casting from a {:?} to a {:?} isn't allowed",
                        expr_ty, to
                    )))
                }
            }
            syn::Expr::Closure(_) => Err(ASTError::Compiler(
                "Closures aren't supported in Mimir".to_owned(),
            )),
            syn::Expr::Const(_) => Err(ASTError::Compiler(
                "Const expressions/blocks aren't supported in Mimir".to_owned(),
            )),
            syn::Expr::Continue(_) => Err(ASTError::Compiler(
                "`continue` isn't allowed inside expressions".to_owned(),
            )),
            syn::Expr::Field(expr_field) => {
                let base_str = expr_field.base.to_token_stream().to_string();

                let mem_str = match &expr_field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => {
                        return Err(ASTError::Compiler("A number cannot be a member".to_owned()));
                    }
                };

                match str_to_builtin(&base_str) {
                    Ok(builtin) => {
                        let builtin_field = str_to_builtin_field(&mem_str)?;
                        Ok(MimirExpr::BuiltinFieldAccess {
                            built_in: builtin,
                            field: builtin_field,
                        })
                    }
                    Err(err) => Err(err),
                }
            }
            syn::Expr::ForLoop(_) => Err(ASTError::Compiler(
                "for loops aren't allowed inside expressions".to_owned(),
            )),
            syn::Expr::Group(_) => Err(ASTError::Compiler(
                "groups aren't allowed inside expressions".to_owned(),
            )),
            syn::Expr::If(_) => Err(ASTError::Compiler(
                "if statements aren't currently allowed inside expressions".to_owned(),
            )),
            syn::Expr::Index(expr_index) => {
                if let syn::Expr::Path(path) = expr_index.expr.as_ref() {
                    let str = path.to_token_stream().to_string();

                    let var_idx = match self.get_uuid(&str) {
                        Some(uid) => uid,
                        None => {
                            return compiler_err!(
                                "Couldn't find a variable named `{str}` in the current scope.\nUUID -> Var Name map: {:?}\nScope stack: {:?}",
                                self.uuid_to_name,
                                self.scope_stack
                            );
                        }
                    };

                    // check if variable is valid
                    {
                        match self.get_ty_var(&str) {
                            Some(ty) => {
                                if !ty.ty.is_global_array() && !ty.ty.is_shared_mem() {
                                    return Err(ASTError::Compiler(
                                        "Only indexing into a global array or a shared memory array is allowed".to_owned(),
                                    ));
                                }
                            }
                            None => {
                                return compiler_err!(
                                    "Couldn't find a variable named `{str}` in the current scope.\nUUID -> Var Name map: {:?}\nScope stack: {:?}",
                                    self.uuid_to_name,
                                    self.scope_stack
                                );
                            }
                        }
                    }

                    let expr = self.parse_expr(&expr_index.index)?;

                    let ty = self.expr_ty(&expr)?.to_mimir_ty();

                    if !ty.is_integer() {
                        return Err(ASTError::Compiler(
                            "Indexing is only allowed via integers".to_owned(),
                        ));
                    }

                    Ok(MimirExpr::Index(MimirIndexExpr {
                        var: *var_idx,
                        index: Box::new(expr),
                    }))
                } else {
                    Err(ASTError::Compiler(
                        "Indexing is only allowed into variable global array".to_owned(),
                    ))
                }
            }
            syn::Expr::Infer(_) => Err(ASTError::Compiler(
                "Inferenced types aren't allowed within expressions".to_owned(),
            )),
            syn::Expr::Let(_) => Err(ASTError::Compiler(
                "Let gaurds aren't allowed within expressions".to_owned(),
            )),
            syn::Expr::Lit(expr_lit) => Ok(match &expr_lit.lit {
                syn::Lit::Int(lit_int) => {
                    let lit = match lit_int.suffix() {
                        "u32" => MimirLit::Uint32(lit_int.base10_parse::<u32>().map_err(|e| {
                            ASTError::Compiler(format!("Failed to parse integer literal: {e}"))
                        })?),
                        "i32" | "" => {
                            MimirLit::Int32(lit_int.base10_parse::<i32>().map_err(|e| {
                                ASTError::Compiler(format!("Failed to parse integer literal: {e}"))
                            })?)
                        }
                        // "i64" => MimirLit::Int64(lit_int.base10_parse::<i64>().map_err(|e| {
                        //     ASTError::Compiler(format!("Failed to parse integer literal: {e}"))
                        // })?),
                        _ => {
                            return compiler_err!(
                                "Unkown or unsupported suffix for integer literal: {}",
                                lit_int.suffix()
                            );
                        }
                    };

                    MimirExpr::Literal(lit)
                }
                syn::Lit::Float(lit_float) => {
                    let lit = match lit_float.suffix() {
                        "f32" | "" => MimirLit::Float32(
                            lit_float
                                .base10_parse::<f32>()
                                .map_err(|e| {
                                    ASTError::Compiler(format!(
                                        "Failed to parse float literal: {e}"
                                    ))
                                })?
                                .to_bits(),
                        ),
                        _ => {
                            return Err(ASTError::Compiler(format!(
                                "Unkown suffix for float literal: {}",
                                lit_float.suffix()
                            )));
                        }
                    };

                    MimirExpr::Literal(lit)
                }
                syn::Lit::Bool(lit_bool) => MimirExpr::Literal(MimirLit::Bool(lit_bool.value)),
                _ => {
                    return Err(ASTError::Compiler(
                        "Unsupported literal in mimir".to_owned(),
                    ));
                }
            }),
            syn::Expr::Loop(_) => Err(ASTError::Compiler(
                "Loops aren't allowed within expressions.".to_owned(),
            )),
            syn::Expr::Macro(_) => Err(ASTError::Compiler(
                "Macros aren't currently allowed within expressions in Mimir.".to_owned(),
            )),
            syn::Expr::Match(_) => Err(ASTError::Compiler(
                "Match statements and expressions aren't allowed within expressions.".to_owned(),
            )),
            syn::Expr::MethodCall(_) => Err(ASTError::Compiler(
                "Match statements and expressions aren't allowed within expressions.".to_owned(),
            )),
            syn::Expr::Paren(expr_paren) => {
                let mimir_expr = self.parse_expr(&expr_paren.expr)?;

                if let MimirExpr::BinOp(MimirBinOpExpr { lhs, op, rhs, .. }) = mimir_expr {
                    Ok(MimirExpr::BinOp(MimirBinOpExpr {
                        lhs: lhs.clone(),
                        op,
                        rhs: rhs.clone(),
                        is_parenthesized: true,
                    }))
                } else {
                    Ok(mimir_expr)
                }
            }
            syn::Expr::Path(expr_path) => {
                let len = expr_path.path.segments.len();
                if len == 1 {
                    // assume var
                    let var_name = expr_path.to_token_stream().to_string();

                    let uuid = match self.get_uuid(&var_name) {
                        Some(uid) => uid,
                        None => {
                            if let Some(idx) = self.const_generics.get(&calculate_hash(&var_name)) {
                                return Ok(MimirExpr::ConstExpr(MimirConstExpr::ConstGeneric {
                                    index: *idx,
                                }));
                            }

                            return compiler_err!(
                                "Couldn't find a variable named `{var_name}` in the current scope.\nUUID -> Var Name map: {:?}\nScope stack: {:?}",
                                self.uuid_to_name,
                                self.scope_stack
                            );
                        }
                    };

                    Ok(MimirExpr::Var(*uuid))
                } else if len == 2 {
                    // try for min/max constants and return a matching literal

                    let constant = expr_path.path.segments[1].to_token_stream().to_string();

                    // match first segment
                    let first = expr_path.path.segments[0].to_token_stream().to_string();
                    let lit = match first.as_str() {
                        "u32" => match constant.as_str() {
                            "MAX" => MimirLit::Uint32(u32::MAX),
                            "MIN" => MimirLit::Uint32(u32::MIN),
                            _ => {
                                return Err(ASTError::Compiler(format!(
                                    "Unkown constant `{constant}` for a u32"
                                )));
                            }
                        },
                        "i32" => match constant.as_str() {
                            "MAX" => MimirLit::Int32(i32::MAX),
                            "MIN" => MimirLit::Int32(i32::MIN),
                            _ => {
                                return Err(ASTError::Compiler(format!(
                                    "Unkown constant `{constant}` for a i32"
                                )));
                            }
                        },
                        // "i64" => match constant.as_str() {
                        //     "MAX" => MimirLit::Int64(i64::MAX),
                        //     "MIN" => MimirLit::Int64(i64::MIN),
                        //     _ => {
                        //         return Err(ASTError::Compiler(format!(
                        //             "Unkown constant `{constant}` for a i64"
                        //         )))
                        //     }
                        // },
                        // this doesn't match official Rust but make sense to me so I'm keeping it
                        "f32" => MimirLit::Float32(
                            match constant.as_str() {
                                "MAX" => f32::MAX,
                                "MIN" => f32::MIN,
                                "MANTISSA_DIGITS" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Uint32(
                                        f32::MANTISSA_DIGITS,
                                    )));
                                }
                                "MAX_10_EXP" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Int32(
                                        f32::MAX_10_EXP,
                                    )));
                                }
                                "INFINITY" => f32::INFINITY,
                                "DIGITS" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Uint32(f32::DIGITS)));
                                }
                                "EPSILON" => f32::EPSILON,
                                "MAX_EXP" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Int32(f32::MAX_EXP)));
                                }
                                "MIN_EXP" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Int32(f32::MIN_EXP)));
                                }
                                "MIN_10_EXP" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Int32(
                                        f32::MIN_10_EXP,
                                    )));
                                }
                                "MIN_POSITIVE" => f32::MIN_POSITIVE,
                                "NAN" => f32::NAN,
                                "NEG_INFINITY" => f32::NEG_INFINITY,
                                "RADIX" => {
                                    return Ok(MimirExpr::Literal(MimirLit::Uint32(f32::RADIX)));
                                }
                                // mathemtical constants may grow more as time goes on
                                "E" => f32::consts::E,
                                "PI" => f32::consts::PI,
                                "TAU" => f32::consts::TAU,
                                _ => {
                                    return Err(ASTError::Compiler(format!(
                                        "Unkown constant `{constant}` for a f32"
                                    )));
                                }
                            }
                            .to_bits(),
                        ),
                        _ => {
                            return Err(ASTError::Compiler(format!(
                                "Unkown identifier, expecting either `u32`, `i32`, `i64`, `f32`, or `f64` instead got `{first}`"
                            )));
                        }
                    };

                    Ok(MimirExpr::Literal(lit))
                } else {
                    Err(ASTError::Compiler(format!(
                        "Unkown path expression `{}`",
                        expr_path.to_token_stream()
                    )))
                }
            }
            syn::Expr::Range(_) => Err(ASTError::Compiler(
                "Range expressions aren't currently allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::RawAddr(_) => Err(ASTError::Compiler(
                "Range address of operations aren't allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::Reference(_) => Err(ASTError::Compiler(
                "Borrowing and references aren't needed in Mimir".to_owned(),
            )),
            syn::Expr::Repeat(_) => Err(ASTError::Compiler(
                "Array literals aren't currently allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::Return(_) => Err(ASTError::Compiler(
                "Return statments aren't allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::Struct(_) => Err(ASTError::Compiler(
                "Instantiating structs isn't currently allowed within expressions in Mimir"
                    .to_owned(),
            )),
            syn::Expr::Try(_) => Err(ASTError::Compiler(
                "Try expressions and error propogations isn't allowed within expressions in Mimir"
                    .to_owned(),
            )),
            syn::Expr::TryBlock(_) => Err(ASTError::Compiler(
                "Try blocks and error propogations isn't allowed within expressions in Mimir"
                    .to_owned(),
            )),
            syn::Expr::Tuple(_) => Err(ASTError::Compiler(
                "Tuple expressions aren't allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::Unary(expr_unary) => {
                let un_op = match expr_unary.op {
                    syn::UnOp::Not(_) => MimirUnOp::Not,
                    syn::UnOp::Neg(_) => MimirUnOp::Neg,
                    _ => {
                        return Err(ASTError::Compiler(format!(
                            "`{}` is an unsupported unary operator",
                            expr_unary.op.to_token_stream()
                        )));
                    }
                };

                let mimir_expr = self.parse_expr(&expr_unary.expr)?;

                Ok(MimirExpr::Unary(MimirUnOpExpr {
                    un_op,
                    expr: Box::new(mimir_expr),
                }))
            }
            syn::Expr::Unsafe(_) => Err(ASTError::Compiler(
                "Try expressions and error propogations isn't allowed within expressions in Mimir"
                    .to_owned(),
            )),
            syn::Expr::Verbatim(tk_strm) => Err(ASTError::Compiler(format!(
                "Unkown expression: `{}`",
                tk_strm
            ))),
            syn::Expr::While(_) => Err(ASTError::Compiler(
                "While statments/expressions aren't allowed within expressions in Mimir".to_owned(),
            )),
            syn::Expr::Yield(_) => Err(ASTError::Compiler(
                "Yield statments/expressions aren't allowed within expressions in Mimir".to_owned(),
            )),
            _ => Err(ASTError::Compiler(format!(
                "Unkown expression: `{}`",
                expr.to_token_stream()
            ))),
        }
    }

    fn expr_ty(&self, expr: &MimirExpr) -> Result<MimirPrimitiveTy, ASTError> {
        match expr {
            MimirExpr::BinOp(MimirBinOpExpr { lhs, op, rhs, .. }) => {
                let lhs_ty = self.expr_ty(lhs.as_ref())?;
                let rhs_ty = self.expr_ty(rhs.as_ref())?;

                // TODO: move all necessary safety asserts to the new pass
                if lhs_ty != rhs_ty {
                    return Err(ASTError::Compiler(format!(
                        "LHS type ({:?}) needs to be the same as RHS type ({:?})",
                        lhs_ty, rhs_ty
                    )));
                }

                match op {
                    MimirBinOp::Add
                    | MimirBinOp::Sub
                    | MimirBinOp::Mul
                    | MimirBinOp::Div
                    | MimirBinOp::Mod => {
                        if lhs_ty.is_numeric() && rhs_ty.is_numeric() {
                            Ok(lhs_ty)
                        } else {
                            compiler_err!(
                                "LHS type ({:?}) and RHS type ({:?}) both need to be numeric for {:?} operation",
                                lhs_ty,
                                rhs_ty,
                                op
                            )
                        }
                    }
                    MimirBinOp::And | MimirBinOp::Or => {
                        if lhs_ty.is_bool() && rhs_ty.is_bool() {
                            Ok(lhs_ty)
                        } else {
                            compiler_err!(
                                "LHS type ({:?}) and RHS type ({:?}) both need to be numeric for {:?} operation",
                                lhs_ty,
                                rhs_ty,
                                op
                            )
                        }
                    }
                    MimirBinOp::Lt
                    | MimirBinOp::Lte
                    | MimirBinOp::Gt
                    | MimirBinOp::Gte
                    | MimirBinOp::Eq
                    | MimirBinOp::Ne => {
                        if lhs_ty.is_numeric() && rhs_ty.is_numeric() {
                            Ok(MimirPrimitiveTy::Bool)
                        } else {
                            compiler_err!(
                                "LHS type ({:?}) and RHS type ({:?}) must both be numeric for {:?} operation",
                                lhs_ty,
                                rhs_ty,
                                op
                            )
                        }
                    }
                }
            }
            MimirExpr::Index(MimirIndexExpr { var, .. }) => {
                let ty = if let Some(ty) = self.map.get(var) {
                    ty
                } else {
                    return compiler_err!(
                        "Variable `{}` doesn't have a type associated",
                        self.uuid_to_name.get(var).map_or("ERR!", |v| v)
                    );
                };

                if let MimirTy::GlobalArray { element_type } = &ty.ty {
                    Ok(element_type.clone())
                } else if let MimirTy::SharedMemArray { .. } = &ty.ty {
                    Ok(MimirPrimitiveTy::Float32)
                } else {
                    compiler_err!("Only global arrays and shared memory arrays can be indexed into")
                }
            }
            MimirExpr::Unary(MimirUnOpExpr { un_op, expr }) => {
                let ty = self.expr_ty(expr)?;

                match un_op {
                    MimirUnOp::Not => {
                        if ty.is_bool() {
                            Ok(MimirPrimitiveTy::Bool)
                        } else {
                            Err(ASTError::Compiler(
                                "Only logical NOT is permitted.".to_owned(),
                            ))
                        }
                    }
                    MimirUnOp::Neg => {
                        if ty.is_numeric() {
                            Ok(ty.clone())
                        } else {
                            Err(ASTError::Compiler(format!("Cannot negate a {:?}", ty)))
                        }
                    }
                }
            }
            MimirExpr::Literal(mimir_lit) => Ok(mimir_lit.to_ty()),
            MimirExpr::Var(uuid) => match self.map.get(uuid) {
                Some(ty_var) => Ok(match ty_var.ty.clone() {
                    MimirTy::Primitive(mimir_primitive_ty) => mimir_primitive_ty,
                    _ => {
                        return compiler_err!(
                            "A var expression must evaluate to a primitive type."
                        );
                    }
                }),
                None => compiler_err!(
                    "Couldn't find a type for `{}`",
                    self.uuid_to_name.get(uuid).unwrap_or(&"ERR".to_string())
                ),
            },
            MimirExpr::MathIntrinsic(MathIntrinsicExpr { func, args }) => {
                let tys = args
                    .iter()
                    .map(|arg| self.expr_ty(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                math::get_output_ty(func, &tys)
            }
            MimirExpr::Cast(MimirCastExpr { from, to }) => {
                let from_ty = self.expr_ty(from)?;
                if cast::is_legal(&from_ty.to_mimir_ty(), to) {
                    Ok(to.clone())
                } else {
                    compiler_err!("Casting from {:?} to {:?} is not legal", from_ty, to)
                }
            }
            MimirExpr::BuiltinFieldAccess { .. } => Ok(MimirPrimitiveTy::Uint32),
            MimirExpr::ConstExpr(mimir_const_expr) => Ok(mimir_const_expr.const_expr_to_ty()),
        }
    }

    fn validate_kernel_sig(sig: &Signature) -> Result<(), ASTError> {
        if sig.asyncness.is_some() {
            Err(ASTError::Compiler(
                "Kernels cannot be `async`, please remove".to_owned(),
            ))
        } else if sig.constness.is_some() {
            Err(ASTError::Compiler(
                "Kernels cannot be `const`, please remove".to_owned(),
            ))
        } else if sig.abi.is_some() {
            Err(ASTError::Compiler(
                "Kernels cannot have abi specifications, please remove".to_owned(),
            ))
        } else if sig.variadic.is_some() {
            Err(ASTError::Compiler(
                "Kernels cannot have variadic parameters, please remove".to_owned(),
            ))
        } else if !matches!(sig.output, ReturnType::Default) {
            Err(ASTError::Compiler(
                "Kernels cannot have return types, please remove".to_owned(),
            ))
        } else {
            Ok(())
        }
    }
}
