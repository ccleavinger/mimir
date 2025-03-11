use std::collections::HashSet;
use std::hash::Hash;
use crate::spirv_compiler::ir;

use super::{compiler::SpirVCompiler, ir::{MimirBuiltIn, MimirExprIR, MimirPtrType, MimirType}, util::type_importance};
use rspirv::{dr::Operand, spirv::{self, StorageClass, Word}};
use crate::spirv_compiler::ir::{MimirLit, MimirVariable};

impl SpirVCompiler {
    pub fn ir_to_spirv(&mut self) -> Result<(), String> {
        self.init_builder()?;


        Ok(())
    }

    pub fn setup_ptr_ty(&mut self) {
        for ty in self.vars.values().map(|var| var.ty.clone()).collect::<HashSet<MimirPtrType>>() {
            let base = ty.base.clone();
            let storage_class = ty.storage_class;


            if let MimirType::RuntimeArray(arr_ty) = &ty.base {
                let base_word = self.get_raw_type(arr_ty).unwrap();
                match self.ptr_types.entry(ty.clone()) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        let array = self.spirv_builder.type_runtime_array(base_word);
                        e.insert(array);
                    },
                    std::collections::hash_map::Entry::Occupied(_) => {
                        let base_word = self.get_raw_type(&base).unwrap();
                        let ptr = self.spirv_builder.type_pointer(None, storage_class, base_word);
                        self.ptr_types.insert(ty, ptr);
                    }
                }
            }
        }
    }

    pub fn build_ir(&mut self, ir: MimirExprIR) -> Result<(), String> {
        match ir {
            MimirExprIR::Local(name, mimir_expr_ir) => {
                let word = {
                    let var = self.vars.get_mut(&name).unwrap();
                    let ty_word = self.ptr_types.get(&var.ty).unwrap();

                    let word = self.spirv_builder.variable(
                        *ty_word,
                        None,
                        var.ty.storage_class,
                        None,
                    );

                    self.spirv_builder.name(word, &name);

                    var.word = Some(word);
                    word
                };

                if let Some(expr) = mimir_expr_ir {
                    let expr_word = self.ir_to_word(&expr)?;
                    self.spirv_builder.store(word, expr_word, None, vec![]).map_err(|e| e.to_string())?;

                    Ok(())
                } else {
                    Ok(())
                }
            },
            MimirExprIR::Assign(name, mimir_expr_ir) => {
                let var = self.vars.get(&name).unwrap();
                let word = var.word.unwrap();
                let expr_word = self.ir_to_word(&mimir_expr_ir)?;

                self.spirv_builder.store(word, expr_word, None, vec![]).map_err(|e| e.to_string())?;

                Ok(())
            },
            MimirExprIR::BinAssign(name, mimir_bin_op, mimir_expr_ir) => {
                let var = self.vars.get(&name).unwrap();
                let word = var.word.unwrap();
                let expr_word = self.ir_to_word(&mimir_expr_ir)?;

                let binop = self.build_binop(
                    MimirExprIR::Var(name.clone()),
                    &mimir_bin_op,
                    *mimir_expr_ir
                )?;

                self.spirv_builder.store(word, binop.1, None, vec![]).map_err(|e| e.to_string())?;

                Ok(())
            },
            MimirExprIR::BinOp(mimir_expr_ir, mimir_bin_op, mimir_expr_ir1, _) => {
                Err("Standalone BinOp not allowed".to_string())
            },
            MimirExprIR::Index(_, mimir_expr_ir) => {
                Err("Standalone Index not allowed".to_string())
            },
            MimirExprIR::Field(var, member) => {
                Err("Standalone Field access not allowed".to_string())
            },
            MimirExprIR::For(iter_var_name, lhs, rhs, mimir_expr_irs) => {
                if self.vars.contains_key(&iter_var_name) {
                    return Err(format!("Variable {} already exists", iter_var_name))
                }

                let ty = *self.ptr_types.get(&MimirPtrType {
                    base: MimirType::Int32,
                    storage_class: StorageClass::Function
                }).unwrap();

                let iter_var_word = self.spirv_builder.variable(
                    ty,
                    None,
                    StorageClass::Function,
                    None
                );

                self.spirv_builder.store(
                    iter_var_word,
                    self.literals.get(&MimirLit::Int32(lhs as i32)).unwrap().clone(),
                    None,
                    vec![]
                ).map_err(|e| e.to_string())?;

                let iter_var = MimirVariable {
                    ty: MimirPtrType {
                        base: MimirType::Int32,
                        storage_class: StorageClass::Function
                    },
                    word: Some(iter_var_word)
                };

                self.vars.insert(iter_var_name.clone(), iter_var);

                self.spirv_builder.

                Err("".to_string())
            },
            MimirExprIR::If(mimir_expr_ir, mimir_expr_ir1, mimir_expr_ir2) => todo!(),
            MimirExprIR::Unary(mimir_un_op, mimir_expr_ir) => todo!(),
            MimirExprIR::Literal(mimir_lit) => todo!(),
            MimirExprIR::Var(_) => todo!(),
        }
    }

    pub fn init_types(&mut self) {
        let types: Vec<_> = self.vars.iter().map(|(_, var)| var.ty.base.clone()).collect();
        for ty in types {
            if !self.types.contains_key(&ty) {
                let opt_word = self.get_raw_type(&ty);

                if opt_word.is_none() {
                    // handle vector and runtime array types
                    match ty {
                        MimirType::Uint32Vec3 => {
                            let int32 = if self.types.contains_key(&MimirType::Int32) {
                                *self.types.get(&MimirType::Int32).unwrap()
                            } else {
                                let int32 = self.spirv_builder.type_int(32, 1);
                                self.types.insert(MimirType::Int32, int32);
                                int32
                            };

                            let vec3 = self.spirv_builder.type_vector(int32, 3);
                            self.types.insert(ty.clone(), vec3);
                        },
                        MimirType::Float32Vec3 => {
                            let float32 = if let std::collections::hash_map::Entry::Vacant(e) = self.types.entry(MimirType::Float32) {
                                let float32 = self.spirv_builder.type_float(32);
                                e.insert(float32);
                                float32
                            } else {
                                *self.types.get(&MimirType::Float32).unwrap()
                            };

                            let vec3 = self.spirv_builder.type_vector(float32, 3);
                            self.types.insert(ty.clone(), vec3);
                        },
                        MimirType::RuntimeArray(ty) => {
                            let base = if self.types.contains_key(&*ty) {
                                *self.types.get(&*ty).unwrap()
                            } else {
                                let base = self.get_raw_type(&*ty).unwrap();
                                self.types.insert((*ty).clone(), base);
                                base
                            };

                            let array = self.spirv_builder.type_runtime_array(base);
                            self.types.insert((*ty).clone(), array);
                        },
                        _ => {}
                    }
                } else {
                    let word = opt_word.unwrap();
                    self.types.insert(ty, word);
                }
            }
        }
    }

    fn get_raw_type(&mut self, ty: &MimirType) -> Option<Word> {
        match ty {
            MimirType::Int32 => Some(self.spirv_builder.type_int(32, 1)),
            MimirType::Int64 => Some(self.spirv_builder.type_int(64, 1)),
            MimirType::Float32 => Some(self.spirv_builder.type_float(32)),
            MimirType::Bool => Some(self.spirv_builder.type_bool()),
            MimirType::Void => Some(self.spirv_builder.type_void()),
            MimirType::Unknown => panic!("Unknown type found in variable"),
            _ => None
        }
    }

    pub fn init_builder(&mut self) -> Result<(), String> {
        self.spirv_builder.capability(spirv::Capability::Shader);
        self.spirv_builder.memory_model(rspirv::spirv::AddressingModel::Logical, rspirv::spirv::MemoryModel::GLSL450);


        let void = self.spirv_builder.type_void();
        self.types.insert(
            MimirType::Void,
            void
        );

        self.init_types();

        let voidf = self.spirv_builder.type_function(void, vec![void]);

        let main = self.spirv_builder.begin_function(
            void,
            None,
            spirv::FunctionControl::empty(),
            voidf, ).map_err(|e| format!("{:?}", e)
        )?;

        // GPU side builtins (block_idx, block_dim, thread_idx, global_invocation_id)
        let builtins: Vec<_> = self.builtins.iter().map(|builtin| {
            let vec3_word = self.types.get(&MimirType::Uint32Vec3).unwrap();

            let vec3_ptr_word = if let std::collections::hash_map::Entry::Vacant(e) = self.ptr_types.entry(MimirPtrType {
                base: MimirType::Uint32Vec3,
                storage_class: StorageClass::Input
            }) {
                let vec3_ptr_word = self.spirv_builder.type_pointer(
                    None,
                    StorageClass::Input,
                    *vec3_word
                );
                e.insert(vec3_ptr_word);
                vec3_ptr_word
            } else {
                *self.ptr_types.get(&MimirPtrType {
                    base: MimirType::Uint32Vec3,
                    storage_class: StorageClass::Input
                }).unwrap()
            };

            let word = self.spirv_builder.variable(
                vec3_ptr_word,
                None,
                StorageClass::Input,
                None,
            );

            self.spirv_builder.decorate(
                word,
                spirv::Decoration::BuiltIn,
                match builtin {
                    MimirBuiltIn::BlockIdx => vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupId)],
                    MimirBuiltIn::BlockDim => vec![Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)],
                    MimirBuiltIn::ThreadIdx => vec![Operand::BuiltIn(spirv::BuiltIn::LocalInvocationId)],
                    MimirBuiltIn::GlobalInvocationId => vec![Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)],
                }
            );

            word
        }).collect();

        self.spirv_builder.entry_point(
            spirv::ExecutionModel::GLCompute,
            main,
            "main",
            &builtins
        );

        let const1 = self.spirv_builder.constant_bit32(
            *self.types.get(&MimirType::Int32).unwrap(),
            1
        );

        self.spirv_builder.execution_mode(
            main,
            spirv::ExecutionMode::LocalSizeId,
            vec![const1, const1, const1]
        );
        
        let mut literals_set = self.get_literals();
        
        literals_set.insert(MimirLit::Int32(0));
        literals_set.insert(MimirLit::Int32(1));
        literals_set.insert(MimirLit::Int32(2));
        
        let literals = literals_set.into_iter().collect::<Vec<_>>();
        
        for lit in literals {
            let word = match lit {
                MimirLit::Int32(val) => self.spirv_builder.constant_bit32(
                    *self.types.get(&MimirType::Int32).unwrap(),
                    val as u32
                ),
                MimirLit::Int64(val) => self.spirv_builder.constant_bit64(
                    *self.types.get(&MimirType::Int64).unwrap(),
                    val as u64
                ),
                MimirLit::Float32(val) => self.spirv_builder.constant_bit32(
                    *self.types.get(&MimirType::Float32).unwrap(),
                    val.to_bits()
                ),
                MimirLit::Bool(val) => {
                    if val {
                        self.spirv_builder.constant_true(
                            *self.types.get(&MimirType::Bool).unwrap()
                        )
                    } else {
                        self.spirv_builder.constant_false(
                            *self.types.get(&MimirType::Bool).unwrap()
                        )
                    }
                },
            };

            self.literals.insert(lit, word);
        }

        Ok(())
    }

    fn get_literals(&mut self) -> HashSet<MimirLit> {
        let mut literals = HashSet::new();
        for ir_ in self.ir.clone() {
            let mut ir_vec = vec![ir_.clone()];
            let mut flag = true;
            
            while flag {
                let ir_vec_copy = ir_vec.clone();
                ir_vec = vec![];
                
                for ir in ir_vec_copy {
                    match ir {
                        MimirExprIR::Local(_, expr) => {
                            if let Some(val) = expr {
                                if let MimirExprIR::Literal(lit) = *val {
                                    literals.insert(lit);
                                } else {
                                    ir_vec.push(*val);
                                }
                            }
                        },
                        MimirExprIR::Assign(_, expr) => {
                            if let MimirExprIR::Literal(lit) = *expr {
                                literals.insert(lit);
                            } else {
                                ir_vec.push(*expr);
                            }
                        },
                        MimirExprIR::BinAssign(_, _, expr) => {
                            if let MimirExprIR::Literal(lit) = *expr {
                                literals.insert(lit);
                            } else {
                                ir_vec.push(*expr);
                            }
                        },
                        MimirExprIR::BinOp(lhs, _, rhs, _) => {
                            ir_vec.push(*lhs);
                            ir_vec.push(*rhs);
                        },
                        MimirExprIR::Index(_, index) => {
                            ir_vec.push(*index);
                        },
                        MimirExprIR::For(_, l, r, expr) => {
                            literals.insert(MimirLit::Int64(l));
                            literals.insert(MimirLit::Int64(r));
                            
                            for e in expr {
                                ir_vec.push(e);
                            }
                        },
                        MimirExprIR::If(cond, then, els) => {
                            ir_vec.push(*cond);
                            ir_vec.push(*then);
                            if let Some(e) = els {
                                ir_vec.push(*e);
                            }
                        },
                        MimirExprIR::Unary(_, expr) => {
                            ir_vec.push(*expr);
                        },
                        MimirExprIR::Literal(lit) => {
                            literals.insert(lit);
                        },
                        MimirExprIR::Var(_) => {},
                        _ => continue
                    }
                }
                if ir_vec.is_empty() {
                    flag = false;
                }
            }
        }
        literals
    }

    // Converts expressions used in deeply nested expressions to spirv words
    pub fn ir_to_word(&mut self, ir: &MimirExprIR) -> Result<Word, String> {
        match ir {
            MimirExprIR::Local(_, _) => Err("Local not allowed within expression".to_string()),
            MimirExprIR::Assign(_, _) => Err("Assign not allowed within expression".to_string()),
            MimirExprIR::BinAssign(_, _, _) => Err("BinAssign not allowed within expression".to_string()),
            MimirExprIR::BinOp(lhs, op, rhs, _) => {
                Ok(self.build_binop(**lhs, op, **rhs)?.1)
            },
            MimirExprIR::Index(var, index) => {
                let var_word = self.vars.get(var).unwrap().word.unwrap();
                let index_word = self.ir_to_word(index);

                let ptr_type = Ok(
                    self.ptr_types.get(
                        &self.vars.get(var).ok_or(format!("Failed to retrieve var {}", var))?.ty
                    ))?.ok_or(format!("Failed to retrieve var {}", var)
                )?;
                
                let result = self.spirv_builder.access_chain(
                    *ptr_type,
                    None,
                    var_word,
                    vec![index_word?]
                );

                Ok(result.map_err(|e| e.to_string())?)
            },
            MimirExprIR::Field(var, member) => {
                let mimir_var = self.vars.get(var).ok_or(format!("Failed to retrieve var {}", var))?;
                
                if MimirType::Uint32Vec3 == mimir_var.ty.base {
                    let idx = match member {
                        "x" => 
                    }
                } else {}

                Err("".to_string())
            },
            MimirExprIR::For(_, _, _, _) => {},
            MimirExprIR::If(_, _, _) => {},
            MimirExprIR::Unary(_, _) => {},
            MimirExprIR::Literal(_) => {},
            MimirExprIR::Var(_) => {},
        }
    }
}