use rspirv::spirv::Word;

use super::{compiler::SpirVCompiler, ir::{self, MimirExprIR, MimirType}, util::type_importance};

impl SpirVCompiler {
    pub fn build_binop(&mut self, lhs: MimirExprIR, op: &ir::MimirBinOp, rhs: MimirExprIR) -> Result<(MimirType, Word), String> {

        let result = self.binop_to_word(&lhs, &op, &rhs)?;

        Ok(result)
    }

    fn parse_hand_side(&mut self, expr: &MimirExprIR) -> Result<(MimirType, Word), String> {
        match expr {
            MimirExprIR::Var(name) => {
                let var = self.vars.get(name).unwrap();
                Ok((var.ty.base.clone(), var.word.unwrap()))
            },
            MimirExprIR::Literal(lit) => {
                match lit {
                    ir::MimirLit::Int32(val) => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        
                        Ok((MimirType::Int32, self.spirv_builder.constant_bit32(
                            *int32, 
                            *val as u32
                        )))
                    },
                    ir::MimirLit::Int64(val) => {
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
                            val.0 as u32
                        )))
                    },
                    ir::MimirLit::Bool(val) => {
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
            _ => panic!("Unsupported expression")
        }
    }

fn binop_to_word(&mut self, lhs: &MimirExprIR, bin: &ir::MimirBinOp, rhs: &MimirExprIR) -> Result<(MimirType, u32), String> {
        let lhs = self.parse_hand_side(lhs)?;
        let rhs = self.parse_hand_side(rhs)?;
    
        let lhs_ty = lhs.0.clone();
        let rhs_ty = rhs.0.clone();
    
        let op_ty = if lhs_ty != rhs_ty {
            type_importance(&[lhs_ty.clone(), rhs_ty.clone()])?
        } else {
            lhs_ty.clone()
        }.clone();
    
        let lhs_word: Word = if lhs_ty != op_ty  {
            let ty_word = match op_ty.clone() {
                MimirType::Int32 => {
                    Ok(*self.types.get(&MimirType::Int32).unwrap())
                },
                MimirType::Int64 => {
                    Ok(*self.types.get(&MimirType::Int64).unwrap())
                },
                MimirType::Float32 => {
                    Ok(*self.types.get(&MimirType::Float32).unwrap())
                },
                MimirType::Uint32Vec3 => Err("Uint32Vec3 operations not supported as of now.".to_string()),
                MimirType::Float32Vec3 => Err("Float32Vec3 operations not supported as of now.".to_string()),
                MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray operations not supported as of now. Type: {:?}", mimir_type.clone())),
                MimirType::Bool => Err("Bool operations not supported.".to_string()),
                MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to perform op on void.".to_string()),
                MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to perform op on unknown types".to_string()),
            }?;
            let result = self.spirv_builder.bitcast(ty_word, None, lhs.1).map_err(|e| e.to_string())?;
            Ok::<Word, String>(result)
        } else {
            Ok::<Word, String>(lhs.1)
        }?;
    
        let rhs_word = if rhs_ty != op_ty {
            let ty_word = match op_ty.clone() {
                MimirType::Int32 => {
                    Ok(*self.types.get(&MimirType::Int32).unwrap())
                },
                MimirType::Int64 => {
                    Ok(*self.types.get(&MimirType::Int64).unwrap())
                },
                MimirType::Float32 => {
                    Ok(*self.types.get(&MimirType::Float32).unwrap())
                },
                MimirType::Uint32Vec3 => Err("Uint32Vec3 operations not supported as of now.".to_string()),
                MimirType::Float32Vec3 => Err("Float32Vec3 operations not supported as of now.".to_string()),
                MimirType::RuntimeArray(_) => Err("RuntimeArray operations not supported as of now.".to_string()),
                MimirType::Bool => Err("Bool operations not supported.".to_string()),
                MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to perform op on void.".to_string()),
                MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to perform op on unknown types".to_string()),
            }?;
            let result = self.spirv_builder.bitcast(ty_word, None, rhs.1).map_err(|e| e.to_string())?;
            Ok::<Word, String>(result)
        } else {
            Ok::<Word, String>(rhs.1)
        }?;
    
        match bin {
            ir::MimirBinOp::Add => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        let result = self.spirv_builder.i_add(
                            *int32,
                            None,
                            lhs_word, 
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        let result = self.spirv_builder.i_add(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).unwrap();
                        let result = self.spirv_builder.f_add(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err("Uint32Vec3 addition not supported as of now.".to_string()),
                    MimirType::Float32Vec3 => Err("Float32Vec3 addition not supported as of now.".to_string()),
                    MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray addition not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err("Bool addition not supported.".to_string()),
                    MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to add void.".to_string()),
                    MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to add unknown types".to_string()),
                }
            },
            ir::MimirBinOp::Sub => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        let result = self.spirv_builder.i_sub(
                            *int32,
                                None,
                                lhs_word,
                                rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        let result = self.spirv_builder.i_sub(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).unwrap();
                        let result = self.spirv_builder.f_sub(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err("Uint32Vec3 subtraction not supported as of now.".to_string()),
                    MimirType::Float32Vec3 => Err("Float32Vec3 subtraction not supported as of now.".to_string()),
                    MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray subtraction not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err("Bool subtraction not supported.".to_string()),
                    MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to subtract void.".to_string()),
                    MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to subtract unknown types".to_string())
                }
            },
            &ir::MimirBinOp::Mul => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        let result = self.spirv_builder.i_mul(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        let result = self.spirv_builder.i_mul(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).unwrap();
                        let result = self.spirv_builder.f_mul(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err("Uint32Vec3 multiplication not supported as of now.".to_string()),
                    MimirType::Float32Vec3 => Err("Float32Vec3 multiplication not supported as of now.".to_string()),
                    MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray multiplication not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err("Bool multiplication not supported.".to_string()),
                    MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to multiply void.".to_string()),
                    MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to multiply unknown types".to_string()),
                }
            }
            &ir::MimirBinOp::Div => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();

                        let result = self.spirv_builder.s_div(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        let result = self.spirv_builder.s_div(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => {
                        let float32 = self.types.get(&MimirType::Float32).unwrap();
                        let result = self.spirv_builder.f_div(
                            *float32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Float32, result))
                    },
                    MimirType::Uint32Vec3 => Err("Uint32Vec3 division not supported as of now.".to_string()),
                    MimirType::Float32Vec3 => Err("Float32Vec3 division not supported as of now.".to_string()),
                    MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray division not supported as of now. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err("Bool division not supported.".to_string()),
                    MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to divide void.".to_string()),
                    MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to divide unknown types".to_string()),
                }
            },
            &ir::MimirBinOp::Mod => {
                match op_ty.clone() {
                    MimirType::Int32 => {
                        let int32 = self.types.get(&MimirType::Int32).unwrap();
                        let result = self.spirv_builder.s_mod(
                            *int32,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int32, result))
                    },
                    MimirType::Int64 => {
                        let int64 = self.types.get(&MimirType::Int64).unwrap();
                        let result = self.spirv_builder.s_mod(
                            *int64,
                            None,
                            lhs_word,
                            rhs_word
                        ).map_err(|e| e.to_string())?;
                        Ok((MimirType::Int64, result))
                    },
                    MimirType::Float32 => Err("Float32 modulo not supported.".to_string()),
                    MimirType::Uint32Vec3 => Err("Uint32Vec3 modulo not supported.".to_string()),
                    MimirType::Float32Vec3 => Err("Float32Vec3 modulo not supported.".to_string()),
                    MimirType::RuntimeArray(mimir_type) => Err(format!("RuntimeArray modulo not supported. Type: {:?}", mimir_type)),
                    MimirType::Bool => Err("Bool modulo not supported.".to_string()),
                    MimirType::Void => Err("This should not be possible. Very bad no good error. Attempted to modulo void.".to_string()),
                    MimirType::Unknown => Err("This should not be possible. Very bad no good error. Attempted to modulo unknown types".to_string()),
                }
            }
            _ => panic!("Unsupported binary operator")
        }
    }
}