use std::vec;

use rspirv::{dr::Operand, spirv::{StorageClass, Word}};
use anyhow::{anyhow, Result};

use crate::spirv_compiler::{compiler::SpirVCompiler, ir::{self, ExtInstFunc, MimirExprIR, MimirType}, util::type_importance};
use crate::spirv_compiler::ir::{MimirLit, MimirPtrType};

impl SpirVCompiler {
    pub fn build_binop(&mut self, lhs: MimirExprIR, op: &ir::MimirBinOp, rhs: MimirExprIR) -> Result<(MimirType, Word)> {
        // Process left-hand side
        let (lhs_type, lhs_word) = self.parse_hand_side(&lhs)?;
        
        // Process right-hand side only once
        let (rhs_type, rhs_word) = self.parse_hand_side(&rhs)?;
        
        // Determine result type based on operand types
        let result_type = type_importance(&[lhs_type.clone(), rhs_type.clone()])?;
        let result_type_word = *self.types.get(&result_type).ok_or(anyhow!("Type not found"))?;
        
        // Apply the operation using the appropriate SPIR-V instruction
        let result_word = match (result_type.clone(), op) {
            (MimirType::Float32, ir::MimirBinOp::Add) => self.spirv_builder.f_add(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Float32, ir::MimirBinOp::Sub) => self.spirv_builder.f_sub(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Float32, ir::MimirBinOp::Mul) => self.spirv_builder.f_mul(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Float32, ir::MimirBinOp::Div) => self.spirv_builder.f_div(result_type_word, None, lhs_word, rhs_word)?,
            
            (MimirType::Int32, ir::MimirBinOp::Add) | (MimirType::Int64, ir::MimirBinOp::Add) => 
                self.spirv_builder.i_add(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Int32, ir::MimirBinOp::Sub) | (MimirType::Int64, ir::MimirBinOp::Sub) => 
                self.spirv_builder.i_sub(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Int32, ir::MimirBinOp::Mul) | (MimirType::Int64, ir::MimirBinOp::Mul) => 
                self.spirv_builder.i_mul(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Int32, ir::MimirBinOp::Div) | (MimirType::Int64, ir::MimirBinOp::Div) => 
                self.spirv_builder.s_div(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Int32, ir::MimirBinOp::Mod) | (MimirType::Int64, ir::MimirBinOp::Mod) => 
                self.spirv_builder.s_mod(result_type_word, None, lhs_word, rhs_word)?,
                
            // Unsigned integer operations
            (MimirType::Uint32, ir::MimirBinOp::Add) => 
                self.spirv_builder.i_add(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Uint32, ir::MimirBinOp::Sub) => 
                self.spirv_builder.i_sub(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Uint32, ir::MimirBinOp::Mul) => 
                self.spirv_builder.i_mul(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Uint32, ir::MimirBinOp::Div) => 
                self.spirv_builder.u_div(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Uint32, ir::MimirBinOp::Mod) => 
                self.spirv_builder.u_mod(result_type_word, None, lhs_word, rhs_word)?,
                
            // Boolean operations
            (MimirType::Bool, ir::MimirBinOp::And) => self.spirv_builder.logical_and(result_type_word, None, lhs_word, rhs_word)?,
            (MimirType::Bool, ir::MimirBinOp::Or) => self.spirv_builder.logical_or(result_type_word, None, lhs_word, rhs_word)?,
            
            // Comparison operations
            (_, ir::MimirBinOp::Lt) => {
                let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                let result = match lhs_type {
                    MimirType::Float32 => self.spirv_builder.f_ord_less_than(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Int32 | MimirType::Int64 => self.spirv_builder.s_less_than(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Uint32 => self.spirv_builder.u_less_than(bool_ty, None, lhs_word, rhs_word)?,
                    _ => return Err(anyhow!("Unsupported type for less than comparison: {:?}", lhs_type)),
                };

                return Ok((MimirType::Bool, result));
            },

            (_, ir::MimirBinOp::Gt) => {
                let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                let result = match lhs_type {
                    MimirType::Float32 => self.spirv_builder.f_ord_greater_than(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Int32 | MimirType::Int64 => self.spirv_builder.s_greater_than(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Uint32 => self.spirv_builder.u_greater_than(bool_ty, None, lhs_word, rhs_word)?,
                    _ => return Err(anyhow!("Unsupported type for greater than comparison: {:?}", lhs_type)),
                };

                return Ok((MimirType::Bool, result));
            },

            (_, ir::MimirBinOp::Eq) => {
                let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                let result = match lhs_type {
                    MimirType::Float32 => self.spirv_builder.f_ord_equal(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Int32 | MimirType::Int64 => self.spirv_builder.i_equal(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Uint32 => self.spirv_builder.i_equal(bool_ty, None, lhs_word, rhs_word)?,
                    _ => return Err(anyhow!("Unsupported type for equality comparison: {:?}", lhs_type)),
                };

                return Ok((MimirType::Bool, result));
            },

            (_, ir::MimirBinOp::Ne) => {
                let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                let result = match lhs_type {
                    MimirType::Float32 => self.spirv_builder.f_ord_not_equal(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Int32 | MimirType::Int64 => self.spirv_builder.i_not_equal(bool_ty, None, lhs_word, rhs_word)?,
                    MimirType::Uint32 => self.spirv_builder.i_not_equal(bool_ty, None, lhs_word, rhs_word)?,
                    _ => return Err(anyhow!("Unsupported type for not equal comparison: {:?}", lhs_type)),
                };

                return Ok((MimirType::Bool, result));
            },

                        
            _ => return Err(anyhow!("Unsupported operation {:?} for types {:?} and {:?}", op, lhs_type, rhs_type)),
        };
        
        Ok((result_type, result_word))
    }

    pub fn parse_hand_side(&mut self, expr: &MimirExprIR) -> Result<(MimirType, Word)> {
        match expr {
            MimirExprIR::Var(name) => {
                let var = self.vars.get(name).unwrap();

                if let Some(var_word) = var.word {

                    let ty = *self.types.get(&var.ty.base).ok_or(anyhow!("Type not found for variable: {:?}\nWith the type: {:?}", name, var.ty.base))?;

                    let word = self.spirv_builder.load(
                        ty,
                        None,
                        var_word,
                        None,
                        vec![]
                    )?;

                    Ok((var.ty.base.clone(), word))
                } else {
                    // push const?
                    if var.ty.storage_class == StorageClass::PushConstant {
                        let idx = self.param_order.iter().position(|x| x == name).ok_or(anyhow!("Parameter not found in param_order"))?;

                        let push_const = self.vars.get("push_const").ok_or(anyhow!("Push constant not found"))?;

                        let push_const_word = push_const.word.ok_or(anyhow!("Push constant word not found"))?;

                        let ty = *self.types.get(&var.ty.base).ok_or(anyhow!("Type not found for variable: {:?}\nWith the type: {:?}", name, var.ty.base))?;
                        
                        let ptr = self.ptr_types.get(
                            &MimirPtrType { 
                                base: var.ty.base.clone(), 
                                storage_class: StorageClass::PushConstant 
                            }
                        ).ok_or(anyhow!("Pointer type for push constant not found for base type: {:?} and storage class: PushConstant", var.ty.base))?;

                        let word = self.spirv_builder.access_chain(
                            *ptr,
                            None,
                            push_const_word,
                            vec![*self.literals.get(&MimirLit::Int32(idx as i32)).ok_or(anyhow!("Literal not found for index: {}", idx))?]
                        )?;

                        let load = self.spirv_builder.load(
                            ty,
                            None,
                            word,
                            None,
                            vec![]
                        )?;

                        Ok((var.ty.base.clone(), load))
                    } else {
                        Err(anyhow!("Variable word not found for var: {:?}", name))
                    }
                }
            },
            MimirExprIR::Literal(lit) => {
                match lit {
                    MimirLit::Int32(val) => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        
                        Ok((MimirType::Int32, self.spirv_builder.constant_bit32(
                            *int32, 
                            *val as u32
                        )))
                    },
                    MimirLit::Int64(val) => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        
                        Ok((MimirType::Int64, self.spirv_builder.constant_bit64(
                            *int64,
                            *val as u64
                        )))
                    },
                    ir::MimirLit::Float32(val) => {
                        let float32 = self.types.get(&MimirType::Float32).unwrap();
                        
                        Ok((MimirType::Float32, self.spirv_builder.constant_bit32(
                            *float32,
                            *val
                        )))
                    },
                    MimirLit::Bool(val) => {
                        let bool_ty = self.types.get(&MimirType::Bool).unwrap();

                        Ok((MimirType::Bool, if *val {
                            self.spirv_builder.constant_true(*bool_ty)
                        } else {
                            self.spirv_builder.constant_false(*bool_ty)
                        }))
                    }
                }
            },
            MimirExprIR::BinOp(lhs, bin, rhs, _is_paren) => {
                self.binop_to_word(lhs, bin, rhs)
            },
            MimirExprIR::Field(var_str, field) => {
                let var = self.vars.get(var_str).ok_or(anyhow!("Variable not found in binop!!!"))?;

                match var.ty.base {
                    MimirType::Uint32Vec3 | MimirType::Float32Vec3 => {
                        if let Some(load_word) = var.word {
                            let (ty, ptr) = match var.ty.base {
                                MimirType::Uint32Vec3 => {
                                    let ty = self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found in binop!!!"))?;

                                    let ptr = *self.ptr_types.get(&MimirPtrType { 
                                        base: MimirType::Uint32, 
                                        storage_class: var.ty.storage_class 
                                    }).ok_or(anyhow!("Uint32 ptr with storage class of {:?} type not found in binop!!!", var.ty.storage_class))?;

                                    (*ty, ptr)
                                },
                                MimirType::Float32Vec3 => {
                                    let ty = *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found in binop!!!"))?;

                                    let ptr = *self.ptr_types.get(&MimirPtrType {
                                        base: MimirType::Float32,
                                        storage_class: var.ty.storage_class
                                    }).ok_or(anyhow!("Float32 ptr with storage class of {:?} not found in binop!!!", var.ty.storage_class))?;

                                    (ty, ptr)
                                },
                                _ => Err(anyhow!("Unsupported type for field access: {:?}", var.ty.base))?,
                            };

                            let idx = match field.as_str() {
                                "x" | "X" => 0,
                                "y" | "Y" => 1,
                                "z" | "Z" => 2,
                                _ => Err(anyhow!("Field not found for var: {:?} and field: {:?}", var_str, field))?,
                            };

                            let idx_literal = *self.literals.get(&MimirLit::Int32(idx)).ok_or(anyhow!("Literal not found for index: {}", idx))?;


                            let access_word = if var_str != "block_dim" {
                                self.spirv_builder.access_chain(
                                    ptr,
                                    None,
                                    load_word,
                                    vec![idx_literal]
                                )?
                            } else {
                                self.spirv_builder.composite_extract(
                                    ty,
                                    None,
                                    load_word,
                                    vec![idx as u32]
                                )?
                            };

                            let load = if *var_str != "block_dim" {
                                self.spirv_builder.load(
                                    ty,
                                    None,
                                    access_word,
                                    None,
                                    vec![]
                                )?
                            } else {
                                access_word // For block_dim, we can just return the composite extract directly as it is already a scalar value.
                            };

                            match var.ty.base {
                                MimirType::Uint32Vec3 => Ok((MimirType::Uint32, load)),
                                MimirType::Float32Vec3 => Ok((MimirType::Float32, load)),
                                _ => Err(anyhow!("Unsupported type for field access: {:?}", var.ty.base)),
                            }

                        } else {
                            Err(anyhow!("Variable word not found for var: {:?}", var_str))
                        }
                    },
                    _ => {
                        Err(anyhow!("Field access not supported for var: {:?} and type: {:?}", var_str, var.ty.base))
                    }
                }

            }
            MimirExprIR::Index(name, index) => {
                let var = self.vars.get(name).ok_or(anyhow!("Couldn't find var {}!", name))?;

                if let MimirType::RuntimeArray(inny_ty) = var.ty.clone().base {
                    let var_word = var.word.ok_or(anyhow!("Variable {} doesn't have a init", name))?;
                    let index_word = self.ir_to_word(index)?;

                    let lit_0 = *self.literals.get(&MimirLit::Int32(0)).ok_or(anyhow!("Couldn't find literal 0"))?;

                    let ptr_type = self.ptr_types.get(
                        &MimirPtrType { base: (*inny_ty).clone(), storage_class: StorageClass::Uniform }
                    ).ok_or(anyhow!("Couldn't find type: {:?}", inny_ty))?;

                    let ty = self.types.get(&inny_ty).ok_or(anyhow!("Couldn't find type: {:?}", inny_ty))?;

                    let ptr =
                        self.spirv_builder
                            .access_chain(*ptr_type, None, var_word, vec![lit_0, index_word])?;


                    let load = self.spirv_builder.load(
                        *ty,
                        None,
                        ptr,
                        None,
                        vec![]
                    )?;
                    
                    Ok((inny_ty.as_ref().clone(), load))
                } else {
                    Err(anyhow!("Index access not supported for type: {:?}. Expected RuntimeArray.", var.ty.base))
                }
            }
            MimirExprIR::ExtInstFunc(func, inner_expr) => {
                let arg_results: Vec<(MimirType, Word)> = inner_expr
                    .iter()
                    .map(|arg| self.parse_hand_side(arg))
                    .collect::<Result<Vec<_>>>()?;

                let arg_types: Vec<MimirType> = arg_results.iter().map(|(ty, _)| ty.clone()).collect();
                let arg_words: Vec<Operand> = arg_results.iter().map(|(_, word)| Operand::IdRef(*word)).collect();

                let (result_type, inst_code) = match func {
                    ExtInstFunc::Sin | ExtInstFunc::Cos | ExtInstFunc::Tan |
                    ExtInstFunc::Asin | ExtInstFunc::Acos | ExtInstFunc::Sinh |
                    ExtInstFunc::Cosh | ExtInstFunc::Tanh | ExtInstFunc::Asinh |
                    ExtInstFunc::Acosh | ExtInstFunc::Atanh | ExtInstFunc::Atan2 |
                    ExtInstFunc::Pow | ExtInstFunc::Exp | ExtInstFunc::Log |
                    ExtInstFunc::Exp2 | ExtInstFunc::Log2 | ExtInstFunc::Sqrt |
                    ExtInstFunc::Isqrt | ExtInstFunc::Floor | ExtInstFunc::Ceil |
                    ExtInstFunc::Mix => {
                        if !arg_types.iter().all(|ty| *ty == MimirType::Float32) {
                            return Err(anyhow!("Function {:?} requires Float32 arguments, got {:?}", func, arg_types));
                        }
                        let code = match func {
                            ExtInstFunc::Sin => 13, ExtInstFunc::Cos => 14, ExtInstFunc::Tan => 15,
                            ExtInstFunc::Asin => 16, ExtInstFunc::Acos => 17, ExtInstFunc::Sinh => 19,
                            ExtInstFunc::Cosh => 20, ExtInstFunc::Tanh => 21, ExtInstFunc::Asinh => 22,
                            ExtInstFunc::Acosh => 23, ExtInstFunc::Atanh => 24, ExtInstFunc::Atan2 => 25,
                            ExtInstFunc::Pow => 26, ExtInstFunc::Exp => 27, ExtInstFunc::Log => 28,
                            ExtInstFunc::Exp2 => 29, ExtInstFunc::Log2 => 30, ExtInstFunc::Sqrt => 31,
                            ExtInstFunc::Isqrt => 32, ExtInstFunc::Floor => 8, ExtInstFunc::Ceil => 9,
                            ExtInstFunc::Mix => 46,
                            _ => unreachable!(), // Should be covered by outer match
                        };
                        (MimirType::Float32, code)
                    }
                    ExtInstFunc::Max | ExtInstFunc::Min | ExtInstFunc::Clamp => {
                        let common_type = type_importance(&arg_types)?;
                        let code = match (func, &common_type) {
                            (ExtInstFunc::Max, MimirType::Float32) => 40,
                            (ExtInstFunc::Max, MimirType::Uint32) => 41,
                            (ExtInstFunc::Max, MimirType::Int32 | MimirType::Int64) => 42,
                            (ExtInstFunc::Min, MimirType::Float32) => 37,
                            (ExtInstFunc::Min, MimirType::Uint32) => 38,
                            (ExtInstFunc::Min, MimirType::Int32 | MimirType::Int64) => 39,
                            (ExtInstFunc::Clamp, MimirType::Float32) => 43,
                            (ExtInstFunc::Clamp, MimirType::Uint32) => 44,
                            (ExtInstFunc::Clamp, MimirType::Int32 | MimirType::Int64) => 45,
                            _ => return Err(anyhow!("Unsupported type {:?} for function {:?}", common_type, func)),
                        };
                        (common_type, code)
                    }
                };

                let result_type_word = *self.types.get(&result_type).ok_or(anyhow!("Type {:?} not found for ext inst result", result_type))?;

                let result_word = self.spirv_builder.ext_inst(
                    result_type_word,
                    None,
                    self.ext_inst,
                    inst_code,
                    arg_words,
                )?;

                Ok((result_type, result_word))
            }
            _ => Err(anyhow!("Unsupported expression {:?}", expr))
        }
    }

    fn binop_to_word(&mut self, lhs: &MimirExprIR, bin: &ir::MimirBinOp, rhs: &MimirExprIR) -> Result<(MimirType, u32)> {
        let lhs = self.parse_hand_side(lhs)?;
        let rhs = self.parse_hand_side(rhs)?;
    
        let lhs_ty = lhs.0.clone();
        let rhs_ty = rhs.0.clone();
    
        let op_ty = if lhs_ty != rhs_ty {
            type_importance(&[lhs_ty.clone(), rhs_ty.clone()])?
        } else {
            lhs_ty.clone()
        }.clone();

        let lhs_word: Word = self.cast_word(lhs.1, &lhs_ty, &op_ty)?;
    
        let rhs_word = self.cast_word(rhs.1, &rhs_ty, &op_ty)?;
    
        match bin {
            ir::MimirBinOp::Add => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_add(
                            *int32,
                            None,
                            lhs_word, 
                            rhs_word
                        )?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_add(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.f_add(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32 => {
                        let uint32 = *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?;
                        let result = self.spirv_builder.i_add(
                            uint32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Uint32, result))
                    },
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 addition not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 addition not supported as of now.")),
                    MimirType::RuntimeArray(mimir_type) => Err(anyhow!("RuntimeArray addition not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err(anyhow!("Bool addition not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to add void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to add unknown types")),
                }
            },
            ir::MimirBinOp::Sub => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_sub(
                            *int32,
                                None,
                                lhs_word,
                                rhs_word
                        )?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_sub(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.f_sub(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 subtraction not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 subtraction not supported as of now.")),
                    MimirType::RuntimeArray(mimir_type) => Err(anyhow!("RuntimeArray subtraction not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err(anyhow!("Bool subtraction not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to subtract void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to subtract unknown types")),
                    MimirType::Uint32 => {
                        let uint32 = *self.types.get(&MimirType::Float32).ok_or(anyhow!("Uint32 type not found!!!"))?;
                        let result = self.spirv_builder.i_sub(
                            uint32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Uint32, result))
                    }
                }
            },
            &ir::MimirBinOp::Mul => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_mul(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_mul(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.f_mul(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 multiplication not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 multiplication not supported as of now.")),
                    MimirType::RuntimeArray(mimir_type) => Err(anyhow!("RuntimeArray multiplication not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err(anyhow!("Bool multiplication not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to multiply void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to multiply unknown types")),
                    MimirType::Uint32 => {
                        let uint32 = *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?;
                        let result = self.spirv_builder.i_mul(
                            uint32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Uint32, result))
                    }
                }
            }
            &ir::MimirBinOp::Div => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_div(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found in binop!!!"))?;
                        let result = self.spirv_builder.s_div(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.f_div(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 division not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 division not supported as of now.")),
                    MimirType::RuntimeArray(mimir_type) => Err(anyhow!("RuntimeArray division not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err(anyhow!("Bool division not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to divide void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to divide unknown types")),
                    MimirType::Uint32 => {
                        let uint32 = *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?;
                        let result = self.spirv_builder.u_div(
                            uint32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Uint32, result))
                    }
                }
            },
            &ir::MimirBinOp::Mod => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found in binop!!!"))?;
                        let result = self.spirv_builder.s_mod(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found in binop!!!"))?;
                        let result = self.spirv_builder.s_mod(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => Err(anyhow!("Float32 modulo not supported.")),
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 modulo not supported.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 modulo not supported.")),
                    MimirType::RuntimeArray(mimir_type) => Err(anyhow!("RuntimeArray modulo not supported. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err(anyhow!("Bool modulus not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to modulo void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to modulo unknown types")),
                    MimirType::Uint32 => {
                        let uint32 = *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?;
                        let result = self.spirv_builder.u_mod(
                            uint32,
                            None,
                            lhs_word,
                            rhs_word
                        )?;
                        Ok((MimirType::Uint32, result))
                    }
                }
            },
            &ir::MimirBinOp::Eq => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.i_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.i_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_ord_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 equality not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 equality not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray equality not supported as of now.")),
                    MimirType::Bool => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.logical_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;
                        let result = self.spirv_builder.i_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                }
            },
            &ir::MimirBinOp::Gt => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_greater_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_greater_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.u_greater_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_unord_greater_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 greater than not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 greater than not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray greater than not supported as of now.")),
                    MimirType::Bool => Err(anyhow!("Bool greater than not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::Lt => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.u_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word,
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_unord_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 less than not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 less than not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray less than not supported as of now.")),
                    MimirType::Bool => Err(anyhow!("Bool less than not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::And => {
                match op_ty.clone() {
                    MimirType::Int32 => Err(anyhow!("Int32 \'&&\' not supported.")),
                    MimirType::Int64 => Err(anyhow!("Int64 \'&&\' not supported.")),
                    MimirType::Uint32 => Err(anyhow!("Uint32 \'&&\' not supported.")),
                    MimirType::Float32 => Err(anyhow!("Float32 \'&&\' not supported.")),
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 \'&&\' not supported.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 \'&&\' not supported.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray \'&&\' not supported.")),
                    MimirType::Bool => {
                        let bool_ty = *self.types
                            .get(&MimirType::Bool)
                            .ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.logical_and(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::Or => {
                match op_ty.clone() {
                    MimirType::Int32 => Err(anyhow!("Int32 \'||\' not supported.")),
                    MimirType::Int64 => Err(anyhow!("Int64 \'||\' not supported.")),
                    MimirType::Uint32 => Err(anyhow!("Uint32 \'||\' not supported.")),
                    MimirType::Float32 => Err(anyhow!("Float32 \'||\' not supported.")),
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 \'||\' not supported.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 \'||\' not supported.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray \'||\' not supported.")),
                    MimirType::Bool => {
                        let bool_ty = *self.types
                            .get(&MimirType::Bool)
                            .ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.logical_or(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::Gte => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_greater_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_greater_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.u_greater_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_unord_greater_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 greater than not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 greater than not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray greater than not supported as of now.")),
                    MimirType::Bool => Err(anyhow!("Bool greater than not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::Lte => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.s_less_than(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.u_less_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_unord_less_than_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 less than not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 less than not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray less than not supported as of now.")),
                    MimirType::Bool => Err(anyhow!("Bool less than not supported.")),
                    MimirType::Void => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare void.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error. Attempted to compare unknown types")),
                }
            }
            &ir::MimirBinOp::Ne => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.i_not_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Int64 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.i_not_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.i_not_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Float32 => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.f_unord_not_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Uint32Vec3 => Err(anyhow!("Uint32Vec3 not equal not supported as of now.")),
                    MimirType::Float32Vec3 => Err(anyhow!("Float32Vec3 not equal not supported as of now.")),
                    MimirType::RuntimeArray(_) => Err(anyhow!("RuntimeArray not equal not supported as of now.")),
                    MimirType::Bool => {
                        let bool_ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found in binop!!!"))?;

                        let result = self.spirv_builder.logical_not_equal(
                            bool_ty,
                            None,
                            lhs_word,
                            rhs_word
                        )?;

                        Ok((MimirType::Bool, result))
                    }
                    MimirType::Void => Err(anyhow!("Void not equal not supported as of now.")),
                    MimirType::Unknown => Err(anyhow!("This should not be possible. Very bad no good error.")),
                }
            }
        }
    }

    pub fn cast_word(&mut self, word: Word, orig_ty: &MimirType, end_ty: &MimirType) -> Result<Word> {
        if orig_ty == end_ty {
            Ok(word)
        } else if orig_ty == &MimirType::Float32 && end_ty == &MimirType::Int32 {
            self.spirv_builder.convert_f_to_s(
                *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Int32 && end_ty == &MimirType::Float32 {
            self.spirv_builder.convert_s_to_f(
                *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!!!"))?, // Use Float32 as result type
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Float32 && end_ty == &MimirType::Uint32 {
            self.spirv_builder.convert_f_to_u(
                *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Uint32 && end_ty == &MimirType::Float32 {
            self.spirv_builder.convert_u_to_f(
                *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Float32 && end_ty == &MimirType::Int64 {
            self.spirv_builder.convert_f_to_s(
                *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Int64 && end_ty == &MimirType::Float32 {
            self.spirv_builder.convert_s_to_f(
                *self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!!!"))?, // Use Float32 as result type
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Int32 && end_ty == &MimirType::Int64 {
            self.spirv_builder.s_convert(
                *self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Int64 && end_ty == &MimirType::Int32 {
            self.spirv_builder.s_convert(
                *self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Int32 && end_ty == &MimirType::Uint32 {
            self.spirv_builder.u_convert(
                *self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        } else if orig_ty == &MimirType::Uint32 && end_ty == &MimirType::Int32 {
            self.spirv_builder.bitcast(
                *self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found!!!"))?,
                None,
                word
            ).map_err(|e| anyhow!(e))
        }
        else {
            Err(anyhow!("Unsupported cast from {:?} to {:?}", orig_ty, end_ty))
        }
    }
}