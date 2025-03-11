use std::collections::HashMap;

use quote::ToTokens;
use rspirv::spirv::StorageClass;

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

    pub fn handle_builtin(&mut self, var_name: &str) -> Result<(), String> {
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

pub fn type_importance(types: &[MimirType]) -> Result<MimirType, String> {
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
        Err("No type found".to_string())
    }
}

// deduce the type of initialization, types on LHS or RHS of binary op, or other necessary type deductions
pub fn expr_to_ty(expr: &syn::Expr, vars: &HashMap<String, MimirVariable>) -> Result<MimirType, String> {
    match expr {
        syn::Expr::Array(_) => {
            Err("Array literals unsupported as of now. Sorry.".to_string())
        },
        syn::Expr::Assign(_) => {
            Err("Assignment expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Async(_) => {
            Err("Async expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Await(_) => {
            Err("Await expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Binary(expr_binary) => {
            let lhs_ty = expr_to_ty(&expr_binary.left, vars)?;
            let rhs_ty = expr_to_ty(&expr_binary.right, vars)?;

            type_importance(&[lhs_ty, rhs_ty])
        },
        syn::Expr::Block(_) => {
            Err("Block expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Break(_) => {
            Err("Break expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Call(_) => {
            Err("Function calls aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Cast(expr_cast) => {
            let ty = expr_cast.ty.to_token_stream().to_string();

            match ty.as_str() {
                "f32" => Ok(MimirType::Float32),
                "i32" => Ok(MimirType::Int32),
                "i64" => Ok(MimirType::Int64),
                _ => Err("Unknown type in type cast!".to_string())
            }
        },
        syn::Expr::Closure(_) =>
            Err("Closure expressions aren't supported as of now. (and probably never will be) Sorry.".to_string()),
        syn::Expr::Const(_) => {
            Err("Const expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Continue(_) => {
            Err("Continue expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Field(expr_field) => {
            let ty = expr_to_ty(&expr_field.base, vars)?;

            match ty {
                MimirType::Float32Vec3 => Ok(MimirType::Float32),
                MimirType::Uint32Vec3 => Ok(MimirType::Int32),
                _ => Err("Field access on non-struct/non-vector types not supported".to_string())
            }
        },
        syn::Expr::ForLoop(_) => {
            Err("For loop expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Group(_) => {
            Err("Group expressions aren't intended for type deduction, also this theoretically shouldn't show up. \nYou messed up bad...".to_string())
        },
        syn::Expr::If(_) => {
            Err("Type deduction of if expressions isn't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Index(expr_index) => {
            let ty = expr_to_ty(&expr_index.expr, vars)?;

            match ty {
                MimirType::RuntimeArray(ty) => Ok(*ty),
                _ => Err("Indexing on non-array types not supported".to_string())
            }
        },
        syn::Expr::Infer(_) => {
            Err("Infer expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Let(_) => {
            Err("Let expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Lit(expr_lit) => {
            match expr_lit.lit {
                syn::Lit::Int(_) => Ok(MimirType::Int32),
                syn::Lit::Float(_) => Ok(MimirType::Float32),
                syn::Lit::Str(_) => Err("String literals aren't supported as of now. Sorry.".to_string()),
                syn::Lit::Byte(_) => Err("Byte literals aren't supported as of now. Sorry.".to_string()),
                syn::Lit::Char(_) => Err("Char literals aren't supported as of now. Sorry.".to_string()),
                syn::Lit::Bool(_) => Err("Bool literals aren't supported as of now. Sorry.".to_string()),
                syn::Lit::Verbatim(_) => Err("Verbatim literals aren't supported as of now. Sorry.".to_string()),  
                _ => Err("Unknown literal type".to_string())              
            }
        },
        syn::Expr::Loop(_) => {
            Err("Loop expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Macro(_) => {
            Err("Macro expressions aren't intended for type deduction".to_string())
        },
        syn::Expr::Match(_) => {
            Err("Match expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::MethodCall(_) => {
            Err("Method call expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Paren(expr_paren) => {
            expr_to_ty(&expr_paren.expr, vars)
        },
        syn::Expr::Path(expr_path) => {
            if expr_path.path.segments.len() == 1 {
                let var_name = expr_path.path.segments[0].ident.to_string();

                if let Some(var) = vars.get(&var_name) {
                    Ok(var.ty.base.clone())
                } else {
                    Err("Variable not found".to_string())
                }
            } else {
                Err(
                    "Path expressions are assumed to be variable names. \nPath expressions with more than one segment not supported".to_string()
                )
            }
        },
        syn::Expr::Range(_) => {
            Err("Range expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::RawAddr(_) => {
            Err("RawAddr expressions aren't supported. Sorry.".to_string())
        },
        syn::Expr::Reference(_) => {
            Err("Reference expressions aren't supported. Sorry.".to_string())
        },
        syn::Expr::Repeat(_) => {
            Err("Repeat expressions aren't supported. Sorry.".to_string())
        },
        syn::Expr::Return(_) => {
            Err("Return expressions aren't intended for type deduction.".to_string())
        },
        syn::Expr::Struct(_) => {
            Err("Struct expressions aren't intended for type deduction.".to_string())
        },
        syn::Expr::Try(_) => {
            Err("Try expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::TryBlock(_) => {
            Err("TryBlock expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Tuple(_) => {
            Err("Tuple expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Unary(expr_unary) => {
            expr_to_ty(&expr_unary.expr, vars)
        },
        syn::Expr::Unsafe(_) => {
            Err("Unsafe expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::Verbatim(_) => {
            Err("Verbatim expressions aren't supported as of now. Sorry.".to_string())
        },
        syn::Expr::While(_) => {
            Err("While expressions aren't intended for type deduction. Sorry.".to_string())
        },
        syn::Expr::Yield(_) => {
            Err("Yield expressions aren't intended for type deduction. Sorry.".to_string())
        },
        _ => {
            Err("Unknown expression type".to_string())
        },
    }
}