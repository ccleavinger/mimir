use anyhow::{anyhow, Result};

use mimir_ast::Parameter;
use quote::ToTokens;
use rspirv::spirv::StorageClass;
use syn::{Block, ExprBinary, Pat};
use crate::spirv_compiler::ir::MimirType::RuntimeArray;
use crate::SpirVCompiler;

use super::ir::{ExtInstFunc, MimirBinOp, MimirExprIR, MimirLit, MimirPtrType, MimirType, MimirVariable};

impl SpirVCompiler {
    pub fn params_to_ir(&mut self, parameters: &[Parameter]) -> Result<()> {
        let re = regex::Regex::new(r"\s*\[(\w+)]")?;

        for param in parameters {
            let ty = if let Some(ty_) = re.captures(&param.type_name) {
                self.buffer_order.push(param.name.to_string());
                MimirPtrType {
                    base: RuntimeArray(Box::new(match ty_.get(1).unwrap().as_str() {
                        "i32" => MimirType::Int32,
                        "f32" | "f64" => MimirType::Float32,
                        "bool" => MimirType::Bool,
                        _ => return Err(anyhow!("Failed to parse parameter type: {}", param.type_name)),
                    })),
                    storage_class: StorageClass::Uniform,
                }
            } else {
                self.param_order.push(param.name.to_string());
                MimirPtrType {
                    base: match param.type_name.as_str() {
                        "i32" => MimirType::Int32,
                        "f32" | "f64" => MimirType::Float32,
                        "bool" => MimirType::Bool,
                        _ => return Err(anyhow!("Failed to parse parameter type: {}", param.type_name)),
                    },
                    storage_class: StorageClass::PushConstant,
                }
            };

            let var = MimirVariable { ty, word: None };

            self.vars.insert(param.name.to_string(), var);
        }

        Ok(())
    }

    pub fn body_to_ir(&mut self, body: &Block) -> Result<()> {

        for stmt in &body.stmts {
            self.stmt_to_ir(stmt)?;
        }
        Ok(())
    }

    pub fn stmt_to_ir(&mut self, stmt: &syn::Stmt) -> Result<()> {
        let ir = self.stmt_to_ir_ret(stmt)?;
        if let Some(ir) = ir {
            self.ir.push(ir);
        }
        Ok(())
    }

    pub fn stmt_to_ir_ret(&mut self, stmt: &syn::Stmt) -> Result<Option<MimirExprIR>> {
        match stmt {
            syn::Stmt::Local(local) => {

                // TODO: get shared variables actually working. This will require rewriting a *LOT* of internal stuff and I don't wanna do that rn

                // if !local.attrs.is_empty() && local.attrs.iter().map(|attr| attr.meta.to_token_stream().to_string()).collect::<Vec<_>>().contains(&"shared".to_string()) {
                //     // hoist and treat as a seperate group/type of variable
                //     // (GLSL and SPIR-V treat as an array and is expressed outside shader)
                //     if let syn::Expr::Repeat(repeat) = local.init.clone()
                //         .ok_or(anyhow!("shared variables must be initialized with a repeate expression"))?
                //         .expr.as_ref() {
                        
                //         let ty = self.expr_to_ty(&repeat.expr, &self.vars);

                //         let len = if let syn::Expr::Lit(literal) = repeat.len.as_ref() {
                            
                //         } else {
                //             return Err(anyhow!("Size for a shared var must be a literal as of now"));
                //         }

                //         let shared = SharedVariable {
                //             size: repeat.len,

                //         }
                //     }
                // }

                let ty = if local.init.is_none() {
                    MimirPtrType {
                        base: MimirType::Unknown,
                        storage_class: StorageClass::Function,
                    }
                } else {
                    MimirPtrType {
                        base: self.expr_to_ty(&local.init.as_ref().unwrap().expr, &self.vars.clone())?,
                        storage_class: StorageClass::Function,
                    }
                };

                let var = MimirVariable { ty, word: None,};

                // for some stupid reason it keeps mut in the name
                let name = local.pat.to_token_stream().to_string().replace("mut", "").replace(" ", "");

                self.vars
                    .insert(name.clone(), var);

                let ir = MimirExprIR::Local(
                    name.clone(),  // Use the cleaned name here instead of local.pat.to_token_stream().to_string()
                    if let Some(init) = &local.init {
                        self.expr_to_ir(&init.expr)?
                            .map(Box::new)
                    } else {
                        None
                    },
                );

                if true {
                    self.block_vars.insert(name);
                }

                Ok(Some(ir))
            }
            syn::Stmt::Item(_) => Err(anyhow!("Item declarations not supported")),
            syn::Stmt::Expr(expr, _) => {
                if let Some(ir) = self.expr_to_ir(expr)? {
                    Ok(Some(ir))
                } else {
                    Err(anyhow!("Expected expression {:?}", expr.to_token_stream()))
                }
            }
            _ => Err(anyhow!("Unsupported statement")),
        }
    }

    pub fn stmt_to_var_ir(&mut self, stmt: &syn::Stmt) -> Result<MimirExprIR> {
        match stmt {
            syn::Stmt::Local(local) => {
                let ty = if local.init.is_none() {
                    MimirPtrType {
                        base: MimirType::Unknown,
                        storage_class: StorageClass::Function,
                    }
                } else {
                    MimirPtrType {
                        base: self.expr_to_ty(&local.init.as_ref().unwrap().expr, &self.vars.clone())?,
                        storage_class: StorageClass::Function,
                    }
                };

                let var = MimirVariable { ty, word: None,};

                self.vars
                    .insert(local.pat.to_token_stream().to_string(), var);

                Ok(MimirExprIR::Local(
                    local.pat.to_token_stream().to_string(),
                    None,
                ))
            }
            syn::Stmt::Item(_) => Err(anyhow!("Item declarations not supported in stmt to ir process")),
            syn::Stmt::Expr(expr, _) => {
                let result = self.expr_to_ir(expr)?;

                if let Some(ir) = result {
                    Ok(ir)
                } else {
                    Err(anyhow!("Expected expression {:?}", expr.to_token_stream()))
                }
            }
            _ => Err(anyhow!("Unsupported statement")),
        }
    }

    pub fn expr_to_ir(&mut self, expr: &syn::Expr) -> Result<Option<MimirExprIR>> {
        match expr {
            syn::Expr::Array(_expr_array) => {
                Err(anyhow!("Array literals unsupported as of now. Sorry."))
            }
            syn::Expr::Assign(expr_assign) => {

                // add support for member assignment, i.e 'vec3.x = 0;'
                let mimir_expr = match &*expr_assign.left {
                    syn::Expr::Path(expr_path) => {
                        if expr_path.path.segments.len() == 1 {
                            let var_name = expr_path.path.to_token_stream().to_string();

                            let _var = self.vars.get(&var_name).ok_or(anyhow!("Variable {} not declared", var_name))?;

                            MimirExprIR::Var(var_name)
                        } else {
                            return Err(anyhow!("Only variable assignment supported. Expression: {}", expr.to_token_stream()));
                        }
                    },
                    syn::Expr::Index(expr_index) => {
                        let var_name = match &*expr_index.expr {
                            syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                            _ => return Err(anyhow!("Only variable assignment supported. Expression: {}", expr.to_token_stream())),
                        };
                        self.handle_builtin(&var_name)?;
                        let index = self
                            .expr_to_ir(&expr_index.index)?
                            .ok_or_else(|| anyhow!("Expected expression for index"))?;

                        MimirExprIR::Index(var_name, Box::new(index))
                    },
                    _ => return Err(anyhow!("Only variable assignment with a variable on the left side supported;\n Expression: {}", expr.to_token_stream())),
                };

                // handle if right side is an in place if expression
                // TODO: handle more complex expressions
                if let syn::Expr::If(ref expr_if) = *expr_assign.right {
                    let cond_expr = self.expr_to_ir(&expr_if.cond)?
                        .ok_or_else(|| anyhow!("Invalid if expression condition"))?;

                    // assume simple standalone expression (THIS WILL BREAK MORE COMPLEX CODE)
                    let if_branch_expr = if let syn::Stmt::Expr(if_expr, _) = &expr_if.then_branch.stmts[0] {
                        self.expr_to_ir(if_expr)?.ok_or_else(|| anyhow!("Invalid if statement assignment"))?
                    } else {
                        return Err(anyhow!("If statement assignment is invalid"));
                    };

                    let else_branch_expr = match &expr_if.else_branch {
                        Some((_, else_expr)) => {
                            if let syn::Expr::Block(block) = else_expr.as_ref() {
                                if let syn::Stmt::Expr(else_expr_stmt, _) = &block.block.stmts[0] {
                                    self.expr_to_ir(else_expr_stmt)?.ok_or(anyhow!("Invalid else statement in if assignment"))
                                } else {
                                    return Err(anyhow!("Invalid else statement in if assignment"))
                                }
                            } else {
                                return Err(anyhow!("Invalid else statement in if assignment"))
                            }
                        },
                        None => {
                            return Err(anyhow!("There must be an else for a variable assignment if!"))
                        },
                    }?;

                    return Ok(Some(MimirExprIR::If(
                        Box::new(cond_expr.clone()),
                        vec![MimirExprIR::Assign(
                            Box::new(mimir_expr.clone()),
                            Box::new(if_branch_expr)
                        )],
                        Some(vec![
                            MimirExprIR::Assign(
                                Box::new(mimir_expr.clone()),
                                Box::new(else_branch_expr)
                            )
                        ])
                    )))
                }

                Ok(Some(MimirExprIR::Assign(
                    Box::new(mimir_expr),
                    Box::new(
                        self.expr_to_ir(&expr_assign.right)?
                            .unwrap_or_else(|| panic!("Expected expression")),
                    ),
                )))
            }
            syn::Expr::Async(_expr_async) => Err(anyhow!("Async expressions not supported")),
            syn::Expr::Await(_expr_await) => Err(anyhow!("Await expressions not supported")),
            syn::Expr::Binary(expr_binary) => Ok(Some(self.binary_expr_to_ir(expr_binary, false)?)),
            syn::Expr::Block(expr_block) => {
                let block = expr_block.block.clone();

                for stmt in &block.stmts {
                    self.stmt_to_ir(stmt)?;
                }

                Ok(None)
            },
            syn::Expr::Break(_) => Err(anyhow!("Break expressions not supported currently")),
            syn::Expr::Call(expr_call) => {
                let func_name = match &*expr_call.func {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err(anyhow!("Only function calls supported, Expression: {}", expr_call.func.to_token_stream())),
                };

                let ext_inst = match func_name.as_str() {
                    "sin" => ExtInstFunc::Sin,
                    "cos" => ExtInstFunc::Cos,
                    "tan" => ExtInstFunc::Tan,
                    "asin" => ExtInstFunc::Asin,
                    "acos" => ExtInstFunc::Acos,
                    "sinh" => ExtInstFunc::Sinh,
                    "cosh" => ExtInstFunc::Cosh,
                    "tanh" => ExtInstFunc::Tanh,
                    "asinh" => ExtInstFunc::Asinh,
                    "acosh" => ExtInstFunc::Acosh,
                    "atanh" => ExtInstFunc::Atanh,
                    "atan2" => ExtInstFunc::Atan2,
                    "pow" => ExtInstFunc::Pow,
                    "exp" => ExtInstFunc::Exp,
                    "log" => ExtInstFunc::Log,
                    "exp2" => ExtInstFunc::Exp2,
                    "log2" => ExtInstFunc::Log2,
                    "sqrt" => ExtInstFunc::Sqrt,
                    "i_sqrt" | "isqrt" => ExtInstFunc::Isqrt,
                    "max" => ExtInstFunc::Max,
                    "min" => ExtInstFunc::Min,
                    "floor" => ExtInstFunc::Floor,
                    "ceil" => ExtInstFunc::Ceil,
                    "clamp" => ExtInstFunc::Clamp,
                    "mix" => ExtInstFunc::Mix,
                    "__syncthreads" | "syncthreads" | "barrier" => {
                        return Ok(Some(MimirExprIR::Syncthreads))
                    }
                    _ => return Err(anyhow!("Unsupported function call: {}", func_name)),
                };

                let args: Vec<_> = expr_call
                    .args
                    .iter()
                    .map(|arg| self.expr_to_ir(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                
                let args: Vec<MimirExprIR> = args.into_iter().flatten().collect();

                Ok(Some(MimirExprIR::ExtInstFunc(
                    ext_inst,
                    args,
                )))
            },
            syn::Expr::Cast(_) => Err(anyhow!("Casting not supported currently")),
            syn::Expr::Closure(_) => Err(anyhow!("Closures aren't permitted.")),
            syn::Expr::Const(_) => Err(anyhow!("Const expressions not supported currently")),
            syn::Expr::Continue(_) => {
                Err(anyhow!("Continue expressions not supported currently"))
            }
            syn::Expr::Field(expr_field) => {
                let var_name = match &*expr_field.base {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err(anyhow!("Error in field expression: Only variable assignment supported, Expression: {}", expr_field.base.to_token_stream())),
                };

                self.handle_builtin(&var_name)?;

                let field_name = expr_field.member.to_token_stream().to_string();

                Ok(Some(MimirExprIR::Field(var_name, field_name)))
            }
            syn::Expr::ForLoop(expr_for_loop) => {
                let var_name = match &*expr_for_loop.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => return Err(anyhow!("Error in For loop: Only variable assignment supported, Expression: {}", expr_for_loop.pat.to_token_stream())),
                };

                self.vars.insert(
                    var_name.clone(),
                    MimirVariable {
                        ty: MimirPtrType {
                            base: MimirType::Int32,
                            storage_class: StorageClass::Function,
                        },
                        word: None,
                    },
                );

                self.block_vars.insert(var_name.clone()); // Mark this variable for OpVariable hoisting

                let (var1, var2) = match *expr_for_loop.expr.clone() {
                    syn::Expr::Range(expr_range) => {
                        let start = match &*expr_range
                            .start
                            .ok_or_else(|| anyhow!("Expected start of range"))?
                        {
                            syn::Expr::Lit(expr_lit) => {
                                let lit = expr_lit
                                    .lit
                                    .to_token_stream()
                                    .to_string()
                                    .parse::<i32>()?;
                                
                                MimirExprIR::Literal(MimirLit::Int32(lit)) // For the sake of consistency, we return a literal here for start.
                            },
                            syn::Expr::Path(expr_path) => {
                                // Handle the case where start is a variable, e.g `for i in x..10`
                                let var_name = expr_path.path.to_token_stream().to_string();
                                if self.vars.contains_key(&var_name) {
                                    MimirExprIR::Var(var_name)
                                } else {
                                    return Err(anyhow!("Variable {} not declared", expr_path.path.to_token_stream().to_string()));
                                }
                            },
                            _ => return Err(anyhow!("Only range-based for loops supported")),
                        };
                        let end = match &*expr_range
                            .end
                            .ok_or_else(|| anyhow!("Expected end of range"))?
                        {
                            syn::Expr::Lit(expr_lit) => {
                                let lit = expr_lit
                                    .lit
                                    .to_token_stream()
                                    .to_string()
                                    .parse::<i32>()?;

                                MimirExprIR::Literal(MimirLit::Int32(lit)) // For the sake of consistency, we return a literal here for end.
                            },
                            syn::Expr::Path(expr_path) => {
                                // Handle the case where end is a variable, e.g `for i in 0..y`
                                let var_name = expr_path.path.to_token_stream().to_string();
                                if self.vars.contains_key(&var_name) {
                                    MimirExprIR::Var(var_name)
                                } else {
                                    return Err(anyhow!("Variable {} not declared", expr_path.path.to_token_stream().to_string()));
                                }
                            },
                            _ => return Err(anyhow!("Only range-based for loops supported")),
                        };
                        (start, end)
                    },
                    syn::Expr::MethodCall(expr_method_call) => {

                        let reciever_expr = if let syn::Expr::Paren(expr_paren) = &*expr_method_call.receiver {
                            expr_paren.expr.clone()
                        } else {
                            expr_method_call.receiver.clone()
                        };

                        if let syn::Expr::Range(range_expr) = *reciever_expr {
                            let start = match &*range_expr
                            .clone().start
                            .ok_or_else(|| anyhow!("Expected start of range"))? {
                                syn::Expr::Lit(expr_lit) => {
                                    let lit = expr_lit
                                        .lit
                                        .to_token_stream()
                                        .to_string()
                                        .parse::<i32>()?;
                                    
                                    MimirExprIR::Literal(MimirLit::Int32(lit)) // For the sake of consistency, we return a literal here for start.
                                },
                                syn::Expr::Path(expr_path) => {
                                    // Handle the case where start is a variable, e.g `for i in x..10`
                                    let var_name = expr_path.path.to_token_stream().to_string();
                                    if self.vars.contains_key(&var_name) {
                                        MimirExprIR::Var(var_name)
                                    } else {
                                        return Err(anyhow!("Variable {} not declared", expr_path.path.to_token_stream().to_string()));
                                    }
                                },
                                _ => return Err(anyhow!("Non var or lit start expression found in for loop")),
                            };
                            let end = match &*range_expr.clone().end.ok_or(anyhow!("Expected end of range"))? {
                                syn::Expr::Lit(expr_lit) => {
                                    let lit = expr_lit
                                        .lit
                                        .to_token_stream()
                                        .to_string()
                                        .parse::<i32>()?;

                                    MimirExprIR::Literal(MimirLit::Int32(lit)) // For the sake of consistency, we return a literal here for end
                                },
                                syn::Expr::Path(expr_path) => {
                                    // Handle the case where end is a variable, e.g `for i in 0..y`
                                    let var_name = expr_path.path.to_token_stream().to_string();
                                    if self.vars.contains_key(&var_name) {
                                        MimirExprIR::Var(var_name)
                                    } else {
                                        return Err(anyhow!("Variable {} not declared", expr_path.path.to_token_stream().to_string()));
                                    }
                                },
                                _ => return Err(anyhow!("Non var or lit end expression found in for loop")),
                            };
                            (start, end)
                        } else {
                            return Err(anyhow!("Method call on non-range expression found in for loop"));
                        }
                    },
                    _ => return Err(anyhow!("Only range-based for loops supported")),
                };

                let step = if let syn::Expr::MethodCall(method_call) = &*expr_for_loop.expr {
                    if method_call.method == "step_by" {
                        if let syn::Expr::Range(_) = &*method_call.receiver {
                            let step_expr = method_call.args.first()
                                .ok_or_else(|| anyhow!("Expected an argument for step_by"))?;
                            Some(Box::new(self.expr_to_ir(step_expr)?.ok_or_else(|| anyhow!("Invalid step expression"))?))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let ir: Vec<_> = expr_for_loop
                    .body
                    .stmts
                    .iter()
                    .map(|stmt| self.stmt_to_ir_ret(stmt)) // Produces Result<Option<MimirExprIR>, String>
                    .collect::<Result<Vec<_>, _>>()? // Collect into Vec<Option<MimirExprIR>>
                    .into_iter()
                    .flatten() // Filter out None values
                    .collect::<Vec<_>>();  // Collect into Vec<MimirExprIR>

                Ok(Some(MimirExprIR::For(var_name, Box::new(var1), Box::new(var2), step, ir)))
            }
            syn::Expr::Group(_) => Err(anyhow!("Group expressions not supported currently")),
            syn::Expr::If(expr_if) => {
                let condition = self
                    .expr_to_ir(&expr_if.cond)?
                    .ok_or_else(|| anyhow!("Expected expression for condition"))?;

                let then = expr_if
                    .then_branch
                    .stmts
                    .iter()
                    .map(|stmt| self.stmt_to_ir_ret(stmt)) // Produces Result<Option<MimirExprIR>, String>
                    .collect::<Result<Vec<_>, _>>()? // Collect into Vec<Option<MimirExprIR>>
                    .into_iter()
                    .flatten() // Filter out None values
                    .collect::<Vec<_>>();  // Collect into Vec<MimirExprIR>

                let else_ = if let Some((_, else_)) = &expr_if.else_branch {
                    Some(self.expr_to_ir(else_)?)
                } else {
                    None
                };

                Ok(Some(MimirExprIR::If(
                    Box::new(condition),
                    then,
                    else_.and_then(|ir| ir.map(|expr| vec![expr])),
                )))
            }
            syn::Expr::Index(expr_index) => {
                let var_name = match &*expr_index.expr {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err(anyhow!("Error in index: Only variable assignment supported, Expression: {}", expr_index.expr.to_token_stream())),
                };

                self.handle_builtin(&var_name)?;

                if !self.vars.contains_key(&var_name) {
                    return Err(anyhow!("Variable {} not declared", var_name));
                }

                let index = self
                    .expr_to_ir(&expr_index.index)?
                    .ok_or_else(|| anyhow!("Expected expression for index"))?;

                Ok(Some(MimirExprIR::Index(var_name, Box::new(index))))
            }
            syn::Expr::Infer(_) => Err(anyhow!("Infer expressions not supported currently")),
            syn::Expr::Let(_) => Err(anyhow!("Let guard expressions not supported currently")),
            syn::Expr::Lit(expr_lit) => {
                let lit = match &expr_lit.lit {
                    syn::Lit::Int(lit_int) => {
                        MimirLit::Int32(lit_int.base10_parse::<i32>()?)
                    }
                    syn::Lit::Float(lit_float) => {

                        let float = lit_float.base10_parse::<f32>()?;
                        let float_bits = f32::to_bits(float);

                        MimirLit::Float32(float_bits)
                    }
                    syn::Lit::Bool(lit_bool) => MimirLit::Bool(lit_bool.value),
                    _ => return Err(anyhow!("Unsupported literal")),
                };

                Ok(Some(MimirExprIR::Literal(lit)))
            }
            syn::Expr::Loop(_expr_loop) => {
                Err(anyhow!("Loop expressions not supported currently"))
            }
            syn::Expr::Macro(_expr_macro) => {
                Err(anyhow!("Macro expressions not supported currently"))
            }
            syn::Expr::Match(_expr_match) => {
                Err(anyhow!("Match expressions not supported currently"))
            }
            syn::Expr::MethodCall(_expr_method_call) => {
                Err(anyhow!("Method call expressions not supported currently"))
            }
            syn::Expr::Paren(expr_paren) => {
                if let syn::Expr::Binary(bin) = &*expr_paren.expr {
                    Ok(Some(self.binary_expr_to_ir(&bin.clone(), true)?))
                } else {
                    self.expr_to_ir(&expr_paren.expr)
                }
            }
            syn::Expr::Path(expr_path) => {
                if expr_path.path.segments.len() == 2 {
                    let ty_path = expr_path.path.segments[0].ident.to_string();
                    let field_name = expr_path.path.segments[1].ident.to_string();

                    match ty_path.as_str() {
                        "i32" => {
                            let num = if field_name == "MAX" {
                                i32::MAX
                            } else if field_name == "MIN" {
                                i32::MIN
                            } else {
                                return Err(anyhow!("Unsupported field name for i32: {}", field_name));
                            };

                            return Ok(Some(MimirExprIR::Literal(MimirLit::Int32(num))))
                        },
                        "f32" => {
                            let num = if field_name == "MAX" {
                                f32::MAX
                            } else if field_name == "MIN" {
                                f32::MIN
                            } else {
                                return Err(anyhow!("Unsupported field name for f32: {}", field_name));
                            };

                            return Ok(Some(MimirExprIR::Literal(MimirLit::Float32(f32::to_bits(num)))))
                        },
                        _ => {
                            return Err(anyhow!("Unsupported type for field assignment: {}", ty_path));
                        }
                    }
                }


                let var_name = expr_path.path.to_token_stream().to_string();

                if self.vars.contains_key(&var_name) {
                    Ok(Some(MimirExprIR::Var(var_name)))
                } else {
                    Err(anyhow!("Variable {} not declared", var_name))
                }
            }
            syn::Expr::Range(_) => Err(anyhow!("Range expressions not supported currently")),
            syn::Expr::RawAddr(_) => {
                Err(anyhow!("Raw address expressions are not supported and never will be."))
            }
            syn::Expr::Reference(_) => {
                Err(anyhow!("Reference expressions not supported currently"))
            }
            syn::Expr::Repeat(_) => Err(anyhow!("Repeat expressions not supported currently")),
            syn::Expr::Return(_) => Err(anyhow!("Return expressions not supported currently")),
            syn::Expr::Struct(_) => Err(anyhow!("Struct expressions not supported currently")),
            syn::Expr::Try(_) => Err(anyhow!("Try expressions not supported currently")),
            syn::Expr::TryBlock(_) => {
                Err(anyhow!("Try block expressions not supported currently"))
            }
            syn::Expr::Tuple(_) => Err(anyhow!("Tuple expressions not supported currently")),
            syn::Expr::Unary(unary) => {
                // Optimization: Handle negation of literals directly
                if let syn::Expr::Lit(expr_lit) = &*unary.expr {
                    if let syn::UnOp::Neg(_) = unary.op {
                        match &expr_lit.lit {
                            syn::Lit::Int(lit_int) => {
                                let lit = -lit_int.base10_parse::<i32>()?;
                                return Ok(Some(MimirExprIR::Literal(MimirLit::Int32(lit))));
                            },
                            syn::Lit::Float(lit_float) => {
                                let float_val = -lit_float.base10_parse::<f32>()?;
                                let float_bits = f32::to_bits(float_val);
                                return Ok(Some(MimirExprIR::Literal(MimirLit::Float32(float_bits))));
                            },
                            _ => return Err(anyhow!("Unsupported literal type for unary negation")),
                        }
                    }
                }
                
                // General case for unary operators
                let inner_expr_ir = self.expr_to_ir(&unary.expr)?.ok_or_else(|| anyhow!("Expected expression inside unary operator"))?;

                match unary.op {
                    syn::UnOp::Neg(_) => {
                        // Determine the type of the inner expression
                        let inner_ty = self.expr_to_ty(&unary.expr, &self.vars.clone())?;
                        
                        // Create the appropriate literal for -1 based on the type
                        let neg_one_literal = match inner_ty {
                            MimirType::Int32 => MimirExprIR::Literal(MimirLit::Int32(-1)),
                            MimirType::Int64 => MimirExprIR::Literal(MimirLit::Int64(-1)), // Assuming Int64 support
                            MimirType::Float32 => MimirExprIR::Literal(MimirLit::Float32(f32::to_bits(-1.0))),
                            // Add other numeric types if needed
                            _ => return Err(anyhow!("Unary negation is only supported for numeric types (Int32, Int64, Float32)")),
                        };

                        // Represent negation as multiplication by -1
                        Ok(Some(MimirExprIR::BinOp(
                            Box::new(inner_expr_ir),
                            MimirBinOp::Mul,
                            Box::new(neg_one_literal),
                            false, // Negation doesn't introduce parentheses itself
                        )))
                    }
                    syn::UnOp::Not(_) => {
                        // Handle logical NOT as before
                        Ok(Some(MimirExprIR::Unary(super::ir::MimirUnOp::Not, Box::new(inner_expr_ir))))
                    }
                    _ => Err(anyhow!("Unsupported unary operation")),
                }
            },
            syn::Expr::Unsafe(_) => Err(anyhow!("Unsafe expressions not supported currently")),
            syn::Expr::Verbatim(_) => {
                Err(anyhow!("Verbatim expressions not supported currently"))
            }
            syn::Expr::While(_) => Err(anyhow!("While expressions not supported currently")),
            syn::Expr::Yield(_) => Err(anyhow!("Yield expressions not supported currently")),
            _ => todo!(),
        }
    }

    pub fn binary_expr_to_ir(
        &mut self,
        expr_bin: &ExprBinary,
        paren: bool,
    ) -> Result<MimirExprIR> {
        let lhs = self
            .expr_to_ir(&expr_bin.left)?
            .ok_or_else(|| anyhow!("Expected expression from left hand side"))?;
        let rhs = self
            .expr_to_ir(&expr_bin.right)?
            .ok_or_else(|| anyhow!("Expected expression from right hand side"))?;

        let op = match expr_bin.op {
            syn::BinOp::Add(_plus) => MimirBinOp::Add,
            syn::BinOp::Sub(_minus) => MimirBinOp::Sub,
            syn::BinOp::Mul(_star) => MimirBinOp::Mul,
            syn::BinOp::Div(_slash) => MimirBinOp::Div,
            syn::BinOp::Rem(_percent) => MimirBinOp::Mod,
            syn::BinOp::And(_and_and) => MimirBinOp::And,
            syn::BinOp::Or(_or_or) => MimirBinOp::Or,
            syn::BinOp::Eq(_eq_eq) => MimirBinOp::Eq,
            syn::BinOp::Lt(_lt) => MimirBinOp::Lt,
            syn::BinOp::Le(_le) => MimirBinOp::Lte,
            syn::BinOp::Ne(_ne) => MimirBinOp::Ne,
            syn::BinOp::Ge(_ge) => MimirBinOp::Gte,
            syn::BinOp::Gt(_gt) => MimirBinOp::Gt,
            syn::BinOp::AddAssign(_plus_eq) => {

                return Ok(MimirExprIR::BinAssign(
                    Box::new(lhs),
                    MimirBinOp::Add,
                    Box::new(rhs),
                ));
            },
            syn::BinOp::SubAssign(_minus_eq) => {

                return Ok(MimirExprIR::BinAssign(
                    Box::new(lhs),
                    MimirBinOp::Sub,
                    Box::new(rhs),
                ));
            },
            syn::BinOp::MulAssign(_star_eq) => {

                return Ok(MimirExprIR::BinAssign(
                    Box::new(lhs),
                    MimirBinOp::Mul,
                    Box::new(rhs),
                ));
            },
            syn::BinOp::DivAssign(_slash_eq) => {

                return Ok(MimirExprIR::BinAssign(
                    Box::new(lhs),
                    MimirBinOp::Div,
                    Box::new(rhs),
                ));
            },
            syn::BinOp::RemAssign(_percent_eq) => {

                return Ok(MimirExprIR::BinAssign(
                    Box::new(lhs),
                    MimirBinOp::Mod,
                    Box::new(rhs),
                ));
            },
            _ => return Err(anyhow!("Unsupported binary operation in the expression")),
        };

        Ok(MimirExprIR::BinOp(Box::new(lhs), op, Box::new(rhs), paren))
    }
}
