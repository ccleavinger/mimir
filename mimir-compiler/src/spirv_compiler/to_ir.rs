use mimir_ast::Parameter;
use quote::ToTokens;
use rspirv::spirv::StorageClass;
use syn::{Block, ExprBinary, Pat};

use crate::SpirVCompiler;

use super::{ir::{MimirBinOp, MimirExprIR, MimirLit, MimirPtrType, MimirType, MimirVariable}, util::expr_to_ty};


impl SpirVCompiler {
    pub fn params_to_ir(&mut self, parameters: &[Parameter]) {

        let re = regex::Regex::new(r"\[(\d+)\]").unwrap();

        for param in parameters {
            let ty = if let Some(ty_) = re.captures(&param.type_name) {
                MimirPtrType {
                    base: match ty_.get(1).unwrap().as_str() {
                        "i32" => MimirType::Int32,
                        "f32" | "f64" => MimirType::Float32,
                        "bool" => MimirType::Bool,
                        _ => MimirType::Unknown,
                    },
                    storage_class: StorageClass::Uniform,
                }
            } else {
                MimirPtrType {
                    base: match param.type_name.as_str() {
                        "i32" => MimirType::Int32,
                        "f32" | "f64" => MimirType::Float32,
                        "bool" => MimirType::Bool,
                        _ => MimirType::Unknown,
                    },
                    storage_class: StorageClass::PushConstant,
                }  
            };

            let var = MimirVariable {
                ty,
                word: None,
            };

            self.vars.insert(param.name.to_string(), var);
        }
    }

    pub fn body_to_ir(&mut self, body: &Block) -> Result<(), String> {
        for stmt in &body.stmts {
            self.stmt_to_ir(stmt)?;
        }
        Ok(())
    }

    pub fn stmt_to_ir(&mut self, stmt: &syn::Stmt) -> Result<(), String> {
        match stmt {
            syn::Stmt::Local(local) => {
                let ty = if local.init.is_none() {
                    MimirPtrType {
                        base: MimirType::Unknown,
                        storage_class: StorageClass::Function,
                    }
                } else {
                    MimirPtrType {
                        base: expr_to_ty(&local.init.as_ref().unwrap().expr, &self.vars)?,
                        storage_class: StorageClass::Function,
                    }
                };

                let var = MimirVariable {
                    ty,
                    word: None,
                };

                self.vars.insert(local.pat.to_token_stream().to_string(), var);

                self.ir.push(MimirExprIR::Local(local.pat.to_token_stream().to_string(), None));

                Ok(())
            },
            syn::Stmt::Item(_) => {
                Err("Item declarations not supported".to_string())
            },
            syn::Stmt::Expr(expr, _) => {
                if let Some(ir) = self.expr_to_ir(expr)? {
                    self.ir.push(ir);
                }
                Ok(())
            },
            _ => Err("Unsupported statement".to_string()),
        }
    }

    pub fn stmt_to_var_ir(&mut self, stmt: &syn::Stmt) -> Result<MimirExprIR, String> {
        match stmt {
            syn::Stmt::Local(local) => {
                let ty = if local.init.is_none() {
                    MimirPtrType {
                        base: MimirType::Unknown,
                        storage_class: StorageClass::Function,
                    }
                } else {
                    MimirPtrType {
                        base: expr_to_ty(&local.init.as_ref().unwrap().expr, &self.vars)?,
                        storage_class: StorageClass::Function,
                    }
                };

                let var = MimirVariable {
                    ty,
                    word: None,
                };

                self.vars.insert(local.pat.to_token_stream().to_string(), var);

                Ok(MimirExprIR::Local(local.pat.to_token_stream().to_string(), None))
            },
            syn::Stmt::Item(_) => {
                Err("Item declarations not supported".to_string())
            },
            syn::Stmt::Expr(expr, _) => {
                if let Some(ir) = self.expr_to_ir(expr)? {
                    Ok(ir)
                } else {
                    Err("Expected expression".to_string())
                }
            },
            _ => Err("Unsupported statement".to_string()),
        }
    }

    pub fn expr_to_ir(&mut self, expr: &syn::Expr) -> Result<Option<MimirExprIR>, String> {
        match expr {
            syn::Expr::Array(_expr_array) => 
                Err("Array literals unsupported as of now. Sorry.".to_string()),
            syn::Expr::Assign(expr_assign) => {
                let var_name = match &*expr_assign.left {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err("Only variable assignment supported".to_string()),
                };

                
                if !self.vars.contains_key(&var_name) {
                    return Err(format!("Variable {} not declared", var_name));
                }

                Ok(Some(MimirExprIR::Assign(var_name, Box::new(self.expr_to_ir(&expr_assign.right)?.unwrap_or_else(|| panic!("Expected expression"))))))
            },
            syn::Expr::Async(_expr_async) => {
                Err("Async expressions not supported".to_string())
            },
            syn::Expr::Await(_expr_await) => {
                Err("Await expressions not supported".to_string())
            },
            syn::Expr::Binary(expr_binary) => {
                Ok(Some(self.binary_expr_to_ir(expr_binary, false)?))
            },
            syn::Expr::Block(_) => {
                Err("Block expressions not supported currently".to_string())
            },
            syn::Expr::Break(_) => {
                Err("Break expressions not supported currently".to_string())
            },
            syn::Expr::Call(_) => {
                Err("Function calls not supported currently".to_string())
            },
            syn::Expr::Cast(_) => {
                Err("Casting not supported currently".to_string())
            },
            syn::Expr::Closure(_) => {
                Err("Closures aren't permitted.".to_string())
            },
            syn::Expr::Const(_) => {
                Err("Const expressions not supported currently".to_string())
            },
            syn::Expr::Continue(_) => {
                Err("Continue expressions not supported currently".to_string())
            },
            syn::Expr::Field(expr_field) => {
                let var_name = match &*expr_field.base {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err("Only variable assignment supported".to_string()),
                };

                self.handle_builtin(&var_name)?;

                let field_name = expr_field.member.to_token_stream().to_string();

                Ok(Some(MimirExprIR::Field(var_name, field_name)))
            },
            syn::Expr::ForLoop(expr_for_loop) => {
                let var_name = match &*expr_for_loop.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => return Err("Only variable assignment supported".to_string()),
                };

                self.vars.insert(var_name.clone(), MimirVariable {
                    ty: MimirPtrType {
                        base: MimirType::Int32,
                        storage_class: StorageClass::Function,
                    },
                    word: None,
                });
                
                let (var1, var2) = match *expr_for_loop.expr.clone() {
                    syn::Expr::Range(expr_range) => {
                        let start = match &*expr_range.start.ok_or_else(|| "Expected start of range".to_string())? {
                            syn::Expr::Lit(expr_lit) => expr_lit.lit.to_token_stream().to_string().parse::<i64>().unwrap(),
                            _ => return Err("Only range-based for loops supported".to_string()),
                        };
                        let end = match &*expr_range.end.ok_or_else(|| "Expected end of range".to_string())? {
                            syn::Expr::Lit(expr_lit) => expr_lit.lit.to_token_stream().to_string().parse::<i64>().unwrap(),
                            _ => return Err("Only range-based for loops supported".to_string()),
                        };
                        (start, end)
                    },
                    _ => return Err("Only range-based for loops supported".to_string()),
                };


                let ir: Vec<_> = expr_for_loop.body.stmts.iter().map(|stmt| self.stmt_to_var_ir(stmt)).collect::<Result<_, _>>()?;

                Ok(Some(MimirExprIR::For(var_name, var1, var2, ir)))
            },
            syn::Expr::Group(_) => {
                Err("Group expressions not supported currently".to_string())
            },
            syn::Expr::If(expr_if) => {
                let condition = self.expr_to_ir(&expr_if.cond)?.ok_or_else(|| "Expected expression for condition".to_string())?;

                let then = self.expr_to_ir(&syn::Expr::Block(syn::ExprBlock {
                    attrs: Vec::new(),
                    label: None,
                    block: expr_if.then_branch.clone(),
                }))?.ok_or_else(|| "Expected expression for then branch".to_string())?;

                let else_ = if let Some((_, else_)) = &expr_if.else_branch {
                    Some(self.expr_to_ir(else_)?)
                } else {
                    None
                };

                Ok(Some(MimirExprIR::If(Box::new(condition), Box::new(then), else_.and_then(|ir| ir.map(Box::new)))))
            },
            syn::Expr::Index(expr_index) => {
                let var_name = match &*expr_index.expr {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err("Only variable assignment supported".to_string()),
                };

                self.handle_builtin(&var_name)?;

                if !self.vars.contains_key(&var_name) {
                    return Err(format!("Variable {} not declared", var_name));
                }

                let index = self.expr_to_ir(&expr_index.index)?.ok_or_else(|| "Expected expression for index".to_string())?;

                Ok(Some(MimirExprIR::Index(var_name, Box::new(index))))
            },
            syn::Expr::Infer(_) => {
                Err("Infer expressions not supported currently".to_string())
            },
            syn::Expr::Let(_) => {
                Err("Let gaurd expressions not supported currently".to_string())
            },
            syn::Expr::Lit(expr_lit) => {
                let lit = match &expr_lit.lit {
                    syn::Lit::Int(lit_int) => MimirLit::Int32(lit_int.base10_parse::<i32>().unwrap()),
                    syn::Lit::Float(lit_float) => MimirLit::Float32(lit_float.base10_parse::<f32>().unwrap()),
                    syn::Lit::Bool(lit_bool) => MimirLit::Bool(lit_bool.value),
                    _ => return Err("Unsupported literal".to_string()),
                };

                Ok(Some(MimirExprIR::Literal(lit)))
            },
            syn::Expr::Loop(_expr_loop) => {
                Err("Loop expressions not supported currently".to_string())
            },
            syn::Expr::Macro(_expr_macro) => {
                Err("Macro expressions not supported currently".to_string())
            },
            syn::Expr::Match(_expr_match) => {
                Err("Match expressions not supported currently".to_string())
            },
            syn::Expr::MethodCall(_expr_method_call) => {
                Err("Method call expressions not supported currently".to_string())
            },
            syn::Expr::Paren(expr_paren) => {
                if let syn::Expr::Binary(bin) = &*expr_paren.expr {
                    Ok(Some(self.binary_expr_to_ir(&bin.clone(), true)?))
                } else {
                    self.expr_to_ir(&expr_paren.expr)
                }
            },
            syn::Expr::Path(expr_path) => {
                let var_name = expr_path.path.to_token_stream().to_string();

                if self.vars.contains_key(&var_name) {
                    Ok(Some(MimirExprIR::Var(var_name)))
                } else {
                    Err(format!("Variable {} not declared", var_name))
                }
            },
            syn::Expr::Range(_) => {
                Err("Range expressions not supported currently".to_string())
            },
            syn::Expr::RawAddr(_) => {
                Err("Raw address expressions not supported currently".to_string())
            },
            syn::Expr::Reference(_) => {
                Err("Reference expressions not supported currently".to_string())
            },
            syn::Expr::Repeat(_) => {
                Err("Repeat expressions not supported currently".to_string())
            },
            syn::Expr::Return(_) => {
                Err("Return expressions not supported currently".to_string())
            },
            syn::Expr::Struct(_) => {
                Err("Struct expressions not supported currently".to_string())
            },
            syn::Expr::Try(_) => {
                Err("Try expressions not supported currently".to_string())
            },
            syn::Expr::TryBlock(_) => {
                Err("Try block expressions not supported currently".to_string())
            },
            syn::Expr::Tuple(_) => {
                Err("Tuple expressions not supported currently".to_string())
            },
            syn::Expr::Unary(_) => {
                Err("Unary expressions not supported currently".to_string())
            },
            syn::Expr::Unsafe(_) => {
                Err("Unsafe expressions not supported currently".to_string())
            },
            syn::Expr::Verbatim(_) => {
                Err("Verbatim expressions not supported currently".to_string())
            },
            syn::Expr::While(_) => {
                Err("While expressions not supported currently".to_string())
            },
            syn::Expr::Yield(_) => {
                Err("Yield expressions not supported currently".to_string())
            },
            _ => todo!(),
        }
    }

    pub fn binary_expr_to_ir(&mut self, expr_bin: &ExprBinary, paren: bool) -> Result<MimirExprIR, String> {
        let lhs = self.expr_to_ir(&expr_bin.left)?.ok_or_else(|| "Expected expression from left hand side".to_string())?;
        let rhs = self.expr_to_ir(&expr_bin.right)?.ok_or_else(|| "Expected expression from right hand side".to_string())?;
    
        let op = match expr_bin.op {
            syn::BinOp::Add(_plus) => 
                MimirBinOp::Add,
            syn::BinOp::Sub(_minus) => 
                MimirBinOp::Sub,
            syn::BinOp::Mul(_star) => 
                MimirBinOp::Mul,
            syn::BinOp::Div(_slash) => 
                MimirBinOp::Div,
            syn::BinOp::Rem(_percent) => 
                MimirBinOp::Mod,
            syn::BinOp::And(_and_and) => 
                MimirBinOp::And,
            syn::BinOp::Or(_or_or) => 
                MimirBinOp::Or,
            syn::BinOp::Eq(_eq_eq) => 
                MimirBinOp::Eq,
            syn::BinOp::Lt(_lt) => 
                MimirBinOp::Lt,
            syn::BinOp::Le(_le) => 
                MimirBinOp::Lte,
            syn::BinOp::Ne(_ne) => 
                MimirBinOp::Ne,
            syn::BinOp::Ge(_ge) => 
                MimirBinOp::Gte,
            syn::BinOp::Gt(_gt) => 
                MimirBinOp::Gt,
            _ => return Err("Unsupported binary operation in the expression".to_string()),
        };
    
        Ok(MimirExprIR::BinOp(Box::new(lhs), op, Box::new(rhs), paren))
    }

}