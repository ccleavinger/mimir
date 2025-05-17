use std::collections::HashMap;
use quote::ToTokens;
use rspirv::spirv::StorageClass;
use anyhow::{anyhow, Result};

use super::{compiler::SpirVCompiler, ir::{MimirBuiltIn, MimirPtrType, MimirType, MimirVariable}};


impl SpirVCompiler {
    pub fn get_builtin_ty(&mut self, builtin: &str) -> Result<MimirBuiltIn, String> {
        match builtin {
            "block_idx" => {
                Ok(MimirBuiltIn::BlockIdx)                
            },
            "block_dim" => {
                Ok(MimirBuiltIn::BlockDim)
            },
            "thread_idx" => {
                Ok(MimirBuiltIn::ThreadIdx)
            },
            "global_invocation_id" => {
                Ok(MimirBuiltIn::GlobalInvocationId)
            },
            _ => Err("Unsupported builtin".to_string()),
        }
    }

    pub fn handle_builtin(&mut self, var_name: &str) -> Result<()> {
        if let Ok(builtin) = self.get_builtin_ty(var_name) {
            if self.builtins.insert(builtin.clone()) {
                self.vars.insert(var_name.to_string(), MimirVariable {
                    ty: MimirPtrType {
                        base: MimirType::Uint32Vec3,
                        storage_class: StorageClass::Input,
                    },
                    word: None,
                });
            }
        }
        Ok(())
    }
}

pub fn type_importance(types: &[MimirType]) -> Result<MimirType> {
    if !types.is_empty() {
        let first_type = &types[0];
        if types.iter().all(|ty| ty == first_type) {
            return Ok(first_type.clone());
        }
    }

    if types.contains(&MimirType::Float32Vec3) || (types.contains(&MimirType::Float32) && types.contains(&MimirType::Uint32Vec3)) {
        Ok(MimirType::Float32Vec3)
    } else if types.contains(&MimirType::Uint32Vec3) {
        Ok(MimirType::Uint32Vec3)
    } else if types.contains(&MimirType::Float32) {
        Ok(MimirType::Float32)
    } else if types.contains(&MimirType::Int64) {
        Ok(MimirType::Int64)
    } else if types.contains(&MimirType::Int32) {
        Ok(MimirType::Int32)
    } else {
        Err(anyhow!("No type importance found from the given types: {:?}", types))
    }
}



/// Converts a vector of Rust statements to a formatted string that shows
/// the actual AST structure rather than just the token stream.
pub fn stmts_to_debug_string(stmts: &[syn::Stmt]) -> String {
    let mut result = String::new();
    result.push_str(&format!("Statement count: {}\n", stmts.len()));
    
    for (i, stmt) in stmts.iter().enumerate() {
        // Add statement header with index
        result.push_str(&format!("\n[Statement {}] ", i));
        
        // Format based on statement type to show structure
        match stmt {
            syn::Stmt::Local(local) => {
                result.push_str("Local Declaration:\n");
                result.push_str(&format!("  Pattern: {}\n", local.pat.to_token_stream()));
                if let Some(expr) = &local.init {
                    result.push_str(&format!("  Init Expression: {}\n", expr.expr.to_token_stream()));
                }
            },
            syn::Stmt::Item(item) => {
                result.push_str(&format!("Item Declaration: {}\n", item.to_token_stream()));
            },
            syn::Stmt::Expr(expr, _) => {
                result.push_str("Expression Statement:\n");
                result.push_str(&format!("  Expression: {}\n", expr.to_token_stream()));
            },
            #[allow(unreachable_patterns)]
            _ => {
                // Handle any future statement types that might be added to the syn crate
                result.push_str(&format!("Unknown Statement Type: {}\n", stmt.to_token_stream()));
            }
        }
    }
    
    result
}

impl SpirVCompiler {
    // deduce the type of initialization, types on LHS or RHS of binary op, or other necessary type deductions
    pub fn expr_to_ty(&mut self, expr: &syn::Expr, vars: &HashMap<String, MimirVariable>) -> Result<MimirType> {
        match expr {
            syn::Expr::Array(_) => {
                Err(anyhow!("Array literals unsupported as of now for type deduction. Sorry."))
            },
            syn::Expr::Assign(_) => {
                Err(anyhow!("Assignment expressions aren't intended for type deduction"))
            },
            syn::Expr::Async(_) => {
                Err(anyhow!("Async expressions aren't intended for type deduction"))
            },
            syn::Expr::Await(_) => {
                Err(anyhow!("Await expressions aren't intended for type deduction"))
            },
            syn::Expr::Binary(expr_binary) => {
                let lhs_ty = self.expr_to_ty(&expr_binary.left, vars)?;
                let rhs_ty = self.expr_to_ty(&expr_binary.right, vars)?;

                type_importance(&[lhs_ty, rhs_ty])
            },
            syn::Expr::Block(_) => {
                Err(anyhow!("Block expressions aren't intended for type deduction"))
            },
            syn::Expr::Break(_) => {
                Err(anyhow!("Break expressions aren't intended for type deduction"))
            },
            syn::Expr::Call(call) => {
                let func_name = match &*call.func {
                    syn::Expr::Path(expr_path) => expr_path.path.to_token_stream().to_string(),
                    _ => return Err(anyhow!("Only function calls supported for type deduction, Expression: {}", call.func.to_token_stream())),
                };

                match func_name.as_str() {
                    // Most GLSL standard library functions return Float32
                    "sin" | "cos" | "tan" | "asin" | "acos" | "sinh" | "cosh" | "tanh" |
                    "asinh" | "acosh" | "atanh" | "atan2" | "pow" | "exp" | "log" |
                    "exp2" | "log2" | "sqrt" | "isqrt" | "floor" | "ceil" | "mix" => Ok(MimirType::Float32),

                    // Functions where return type depends on arguments
                    "max" | "min" | "clamp" => {
                        let arg_types: Result<Vec<MimirType>> = call.args.iter()
                            .map(|arg| self.expr_to_ty(arg, vars))
                            .collect();
                        type_importance(&arg_types?)
                    }

                    // Functions that don't return a value usable in expressions
                    "__syncthreads" | "syncthreads" | "barrier" => {
                        Err(anyhow!("Barrier functions do not return a value and cannot be used in type deduction contexts"))
                    }

                    _ => Err(anyhow!("Unsupported function call for type deduction: {}", func_name)),
                }
            },
            syn::Expr::Cast(expr_cast) => {
                let ty = expr_cast.ty.to_token_stream().to_string();

                match ty.as_str() {
                    "f32" => Ok(MimirType::Float32),
                    "i32" => Ok(MimirType::Int32),
                    "i64" => Ok(MimirType::Int64),
                    _ => Err(anyhow!("Unknown type in type cast!"))
                }
            },
            syn::Expr::Closure(_) =>
                Err(anyhow!("Closure expressions aren't supported as of now. (and probably never will be) Sorry.")),
            syn::Expr::Const(_) => {
                Err(anyhow!("Const expressions aren't intended for type deduction"))
            },
            syn::Expr::Continue(_) => {
                Err(anyhow!("Continue expressions aren't intended for type deduction"))
            },
            syn::Expr::Field(expr_field) => {
                let ty = self.expr_to_ty(&expr_field.base, vars)?;

                match ty {
                    MimirType::Float32Vec3 => Ok(MimirType::Float32),
                    MimirType::Uint32Vec3 => Ok(MimirType::Uint32),
                    _ => Err(anyhow!("Field access on non-struct/non-vector types not supported"))
                }
            },
            syn::Expr::ForLoop(_) => {
                Err(anyhow!("For loop expressions aren't intended for type deduction"))
            },
            syn::Expr::Group(_) => {
                Err(anyhow!("Group expressions aren't intended for type deduction, also this theoretically shouldn't show up. \nYou messed up bad..."))
            },
            syn::Expr::If(_) => {
                Err(anyhow!("Type deduction of if expressions isn't supported as of now. Sorry."))
            },
            syn::Expr::Index(expr_index) => {
                let ty = self.expr_to_ty(&expr_index.expr, vars)?;

                match ty {
                    MimirType::RuntimeArray(ty) => Ok(*ty),
                    _ => Err(anyhow!("Indexing on non-arr types not supported"))
                }
            },
            syn::Expr::Infer(_) => {
                Err(anyhow!("Infer expressions aren't intended for type deduction"))
            },
            syn::Expr::Let(_) => {
                Err(anyhow!("Let expressions aren't intended for type deduction"))
            },
            syn::Expr::Lit(expr_lit) => {
                match expr_lit.lit {
                    syn::Lit::Int(_) => Ok(MimirType::Int32),
                    syn::Lit::Float(_) => Ok(MimirType::Float32),
                    syn::Lit::Str(_) => Err(anyhow!("String literals aren't supported as of now. Sorry.")),
                    syn::Lit::Byte(_) => Err(anyhow!("Byte literals aren't supported as of now. Sorry.")),
                    syn::Lit::Char(_) => Err(anyhow!("Char literals aren't supported as of now. Sorry.")),
                    syn::Lit::Bool(_) => Ok(MimirType::Bool),
                    syn::Lit::Verbatim(_) => Err(anyhow!("Verbatim literals aren't supported as of now. Sorry.")),
                    _ => Err(anyhow!("Unknown literal type"))
                }
            },
            syn::Expr::Loop(_) => {
                Err(anyhow!("Loop expressions aren't intended for type deduction"))
            },
            syn::Expr::Macro(_) => {
                Err(anyhow!("Macro expressions aren't intended for type deduction"))
            },
            syn::Expr::Match(_) => {
                Err(anyhow!("Match expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::MethodCall(_) => {
                Err(anyhow!("Method call expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::Paren(expr_paren) => {
                self.expr_to_ty(&expr_paren.expr, vars)
            },
            syn::Expr::Path(expr_path) => {
                if expr_path.path.segments.len() == 1 {
                    let var_name = expr_path.path.segments[0].ident.to_string();

                    if let Some(var) = vars.get(&var_name) {
                        Ok(var.ty.base.clone())
                    } else {
                        match var_name.as_str() {
                            "block_idx" | "block_dim" | "thread_idx" | "global_invocation_id" => {
                                self.handle_builtin(&var_name)?;
                                return Ok(MimirType::Uint32Vec3)
                            }
                            _=> {}
                        }

                        Err(anyhow!("Variable {} not found", expr_path.to_token_stream()))
                    }
                } else {
                    // support for type literals (MAX, MIN, etc)
                    if expr_path.path.segments[1].ident == "MAX" || 
                        expr_path.path.segments[1].ident == "MIN" {

                        match expr_path.path.segments[0].ident.to_string().as_str() {
                            "i32" => return Ok(MimirType::Int32),
                            "i64" => return Ok(MimirType::Int64),
                            "f32" => return Ok(MimirType::Float32),
                            _ => {}
                        }

                        return Ok(MimirType::Float32);
                    }


                    Err(
                        anyhow!("Path expressions are assumed to be variable names. \nPath expressions with more than one segment not supported")
                    )
                }
            },
            syn::Expr::Range(_) => {
                Err(anyhow!("Range expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::RawAddr(_) => {
                Err(anyhow!("RawAddr expressions aren't supported. Sorry."))
            },
            syn::Expr::Reference(_) => {
                Err(anyhow!("Reference expressions aren't supported. Sorry."))
            },
            syn::Expr::Repeat(_) => {
                Err(anyhow!("Repeat expressions aren't supported. Sorry."))
            },
            syn::Expr::Return(_) => {
                Err(anyhow!("Return expressions aren't intended for type deduction."))
            },
            syn::Expr::Struct(_) => {
                Err(anyhow!("Struct expressions aren't intended for type deduction."))
            },
            syn::Expr::Try(_) => {
                Err(anyhow!("Try expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::TryBlock(_) => {
                Err(anyhow!("TryBlock expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::Tuple(_) => {
                Err(anyhow!("Tuple expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::Unary(expr_unary) => {
                self.expr_to_ty(&expr_unary.expr, vars)
            },
            syn::Expr::Unsafe(_) => {
                Err(anyhow!("Unsafe expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::Verbatim(_) => {
                Err(anyhow!("Verbatim expressions aren't supported as of now. Sorry."))
            },
            syn::Expr::While(_) => {
                Err(anyhow!("While expressions aren't intended for type deduction. Sorry."))
            },
            syn::Expr::Yield(_) => {
                Err(anyhow!("Yield expressions aren't intended for type deduction. Sorry."))
            },
            _ => {
                Err(anyhow!("Unknown expression type"))
            },
        }
    }
}