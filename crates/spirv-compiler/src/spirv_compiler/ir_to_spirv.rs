use anyhow::{anyhow, Ok, Result};
use hash_map::Entry;
use std::collections::{hash_map, HashSet};
use Entry::Vacant;

use super::{
    compiler::SpirVCompiler,
    ir::{self, MimirBinOp, MimirBuiltIn, MimirExprIR, MimirPtrType, MimirType},
};
use crate::spirv_compiler::ir::MimirLit;
use rspirv::{
    dr::Operand,
    spirv::{self, StorageClass, Word},
};

impl SpirVCompiler {
    pub fn ir_to_spirv(&mut self) -> Result<()> {
        self.init_builder()?;
        let result = self.setup_block_vars();

        if result.is_err() {
            // Handle setup_block_vars failure gracefully
            println!("Error coming up, block vars are as follows: {:?}", self.block_vars);

            return Err(result.err().unwrap());
        }
        
        for ir in self.ir.clone() {
            self.build_ir(ir)?;
        }

        self.spirv_builder.ret()?;
        self.spirv_builder.end_function()?;

        Ok(())
    }

    fn setup_block_vars(&mut self) -> Result<()> {
        for name in self.block_vars.iter() {
            let var = match self.vars.get_mut(name) {
                Some(var) => var,
                None => return Err(anyhow!("Block variable {} not found in vars", name)),
            };

            if var.word.is_none() {
                // If the variable is not initialized, initialize it now
                let ty_word = self.ptr_types.get(&var.ty).ok_or(anyhow!("Type {:?} not found", var.ty))?;
                let new_word = self.spirv_builder.variable(*ty_word, None, var.ty.storage_class, None);
                self.spirv_builder.name(new_word, name); // Name the variable for debugging
                var.word = Some(new_word);
            }
        }
        

        Ok(())
    }

    pub fn setup_ptr_ty(&mut self) -> Result<()> {
        // Ensure we have Function storage class pointers for common basic types
        let basic_types = vec![MimirType::Float32, MimirType::Int32, MimirType::Bool];
        for basic_ty in basic_types {
            if self.types.contains_key(&basic_ty) && !self.ptr_types.contains_key(&MimirPtrType {
                base: basic_ty.clone(),
                storage_class: StorageClass::Function,
            }) {
                let ty_word = *self.types.get(&basic_ty).unwrap();
                let ptr_ty = self.spirv_builder.type_pointer(None, StorageClass::Function, ty_word);
                self.ptr_types.insert(
                    MimirPtrType {
                        base: basic_ty,
                        storage_class: StorageClass::Function,
                    },
                    ptr_ty,
                );
            }
        }

        // Process all variable types
        for ty in self.vars
            .values()
            .map(|var| var.ty.clone())
            .collect::<HashSet<MimirPtrType>>() // remove duplicates
        {
            let base = ty.base.clone();
            let storage_class = ty.storage_class;

            if let MimirType::RuntimeArray(arr_ty) = &ty.base {
                // Get or create the base element type
                let base_word = if let Some(word) = self.types.get(&arr_ty.as_ref().clone()) {
                    *word
                } else {
                    let base_word = self.get_raw_type(arr_ty)?;
                    self.types.insert(arr_ty.as_ref().clone(), base_word);
                    base_word
                };

                // Create a pointer type for the element type (needed for OpAccessChain)
                if let std::collections::hash_map::Entry::Vacant(e) = self.ptr_types.entry(MimirPtrType {
                    base: *arr_ty.clone(),
                    storage_class,
                }) {
                    let element_ptr = self.spirv_builder.type_pointer(None, storage_class, base_word);
                    e.insert(element_ptr);
                }

                // Only create new types if they don't already exist
                if !self.ptr_types.contains_key(&ty.clone()) {
                    // Create or get the runtime arr type
                    let array_type = if let Some(word) = self.types.get(&MimirType::RuntimeArray(arr_ty.clone())) {
                        *word
                    } else {
                        let array = self.spirv_builder.type_runtime_array(base_word);
                        self.types.insert(MimirType::RuntimeArray(arr_ty.clone()), array);
                        array
                    };
                    
                    // Create or get the struct containing the runtime arr
                    let struct_type = if let Some(word) = self.types.get(&MimirType::RuntimeArray(
                        Box::new(MimirType::RuntimeArray(arr_ty.clone()))
                    )) {
                        *word
                    } else {
                        let struct_ty = self.spirv_builder.type_struct(vec![array_type]);
                        self.types.insert(
                            MimirType::RuntimeArray(Box::new(MimirType::RuntimeArray(arr_ty.clone()))),
                            struct_ty
                        );
                        struct_ty
                    };
                    
                    // Create a pointer to the struct type
                    let struct_ptr = self.spirv_builder.type_pointer(None, storage_class, struct_type);
                    self.ptr_types.insert(ty.clone(), struct_ptr);
                    
                    // Also store it under the double-nested type key for consistency
                    self.ptr_types.insert(
                        MimirPtrType {
                            base: MimirType::RuntimeArray(Box::new(MimirType::RuntimeArray(arr_ty.clone()))),
                            storage_class,
                        },
                        struct_ptr
                    );
                }
            } else {
                // Handle non-runtime arr types (same as before)
                if !self.ptr_types.contains_key(&ty) {
                    let ty_word = self.types.get(&base).ok_or(anyhow!("Type {:?} not found", base))?;
                    let ptr_ty = self.spirv_builder.type_pointer(None, storage_class, *ty_word);
                    self.ptr_types.insert(ty.clone(), ptr_ty);
                }
            }
        }
        Ok(())
    }

    pub fn build_ir(&mut self, ir: MimirExprIR) -> Result<()> {
        match ir {
            MimirExprIR::Local(name, mimir_expr_ir) => {
                        let word = {
                            let var = self.vars.get_mut(&name).ok_or(anyhow!("Variable {} not found", name))?;
                            let ty_word = self.ptr_types.get(&var.ty).ok_or(anyhow!("Type {:?} not found", var.ty))?;

                            let word = match var.word {
                                Some(w) => w, // already initialized, return the existing word
                                None => {
                                    // Create a new variable in the SPIR-V builder
                                    let new_word = self.spirv_builder.variable(*ty_word, None, var.ty.storage_class, None);
                                    self.spirv_builder.name(new_word, &name); // Name the variable for debugging

                                    new_word
                                },
                            };
                        
                            // This naming might be redundant if the variable already exists and was named
                            self.spirv_builder.name(word, &name); 

                            var.word = Some(word);
                            word
                        };

                        if let Some(expr) = mimir_expr_ir {
                            // Get the type and word of the initializer expression
                            let (expr_ty, expr_word) = self.parse_hand_side(&expr)?; 

                            // Get the type of the variable being initialized
                            let var_ty = self.vars.get(&name)
                                .ok_or_else(|| anyhow!("Variable {} not found during initialization", name))?
                                .ty.base.clone();

                            // Cast the expression word if types don't match
                            let value_to_store = if expr_ty != var_ty {
                                // Ensure cast_word handles all necessary conversions (like uint -> int)
                                self.cast_word(expr_word, &expr_ty, &var_ty)
                                    .map_err(|e| anyhow!("Failed to cast initializer for var {}: {}", name, e))?
                            } else {
                                expr_word // Types match, use the word directly
                            };
                            
                            // Store the correctly typed value
                            self.spirv_builder
                                .store(word, value_to_store, None, vec![])?;

                            Ok(())
                        } else {
                            Ok(())
                        }
                    }
            MimirExprIR::Assign(name_expr, mimir_expr_ir) => {
                // Determine the pointer (lhs_ptr_word) to store into
                let (lhs_ptr_word, lhs_base_ty) = match name_expr.as_ref() {
                    MimirExprIR::Index(name, idx) => {
                        // First get the index word, before borrowing the variable
                        let index_word = self.ir_to_word(idx)?; // Get index value
                        
                        let var = self.vars.get(name).ok_or(anyhow!("Variable {:?} not found", name))?;
                        if let MimirType::RuntimeArray(inny_ty) = &var.ty.base {
                            let var_word = var.word.ok_or(anyhow!("Variable {:?} has no corresponding word", var))?;
                            let lit_0 = *self.literals.get(&MimirLit::Int32(0)).ok_or(anyhow!("Couldn't find literal 0"))?;
                            
                            // Determine the correct pointer type for access chain based on storage class
                            let ptr_storage_class = var.ty.storage_class;
                            let element_ptr_type = self.ptr_types.get(&MimirPtrType { base: *inny_ty.clone(), storage_class: ptr_storage_class })
                                .ok_or(anyhow!("Couldn't find element pointer type for {:?} with storage class {:?}", inny_ty, ptr_storage_class))?;

                            // The access chain needs the pointer to the struct containing the runtime array
                            let ptr = self.spirv_builder.access_chain(*element_ptr_type, None, var_word, vec![lit_0, index_word])?;
                            (ptr, *inny_ty.clone()) // Return pointer and base type
                        } else {
                            return Err(anyhow!("Indexing not allowed for non-runtime arr types: {:?}", var.ty.base));
                        }
                    }
                    MimirExprIR::Var(name) => {
                        let var = self.vars.get(name).ok_or(anyhow!("Variable {:?} not found", name))?;
                        let ptr = var.word.ok_or(anyhow!("Variable {:?} has no corresponding word", var))?;
                        (ptr, var.ty.base.clone()) // Return pointer and base type
                    }
                    _ => return Err(anyhow!("LHS of Assign must be a variable or index, got {:?}", name_expr)),
                };

                // Get the type and value of the RHS expression
                let (rhs_ty, rhs_word) = self.parse_hand_side(&mimir_expr_ir)?;

                // Cast RHS value if its type doesn't match the LHS base type
                let value_to_store = if rhs_ty != lhs_base_ty {
                    self.cast_word(rhs_word, &rhs_ty, &lhs_base_ty)
                        .map_err(|e| anyhow!("Failed to cast RHS for assignment: {}", e))?
                } else {
                    rhs_word
                };

                // Store the correctly typed value into the LHS pointer
                self.spirv_builder
                    .store(lhs_ptr_word, value_to_store, None, vec![])?;

                Ok(())
            }
            MimirExprIR::BinAssign(lhs, mimir_bin_op, rhs) => {
                // Determine the pointer (lhs_ptr_word) and base type (lhs_base_ty) of the LHS
                 let (lhs_ptr_word, lhs_base_ty) = match lhs.as_ref() {
                    MimirExprIR::Index(var_name, mimir_expr_ir) => {
                        // First, extract all necessary information from the variables before making mutable method calls
                        let (var_word, storage_class, inny_ty) = {
                            let var = self.vars.get(var_name).ok_or(anyhow!("Variable {:?} not found", var_name))?;
                            if let MimirType::RuntimeArray(inny_ty) = &var.ty.base {
                                let var_word = var.word.ok_or(anyhow!("Variable {:?} has no corresponding word", var))?;
                                let storage_class = var.ty.storage_class;
                                (var_word, storage_class, inny_ty.clone())
                            } else {
                                return Err(anyhow!("Indexing not allowed for non-runtime arr types in BinAssign: {:?}", var.ty.base));
                            }
                        };
                        
                        // Now we can call ir_to_word without holding a reference to self.vars
                        let index_word = self.ir_to_word(mimir_expr_ir)?;
                        let lit_0 = *self.literals.get(&MimirLit::Int32(0)).ok_or(anyhow!("Couldn't find literal 0"))?;
                        
                        let element_ptr_type = self.ptr_types.get(&MimirPtrType { base: *inny_ty.clone(), storage_class })
                            .ok_or(anyhow!("Couldn't find element pointer type for {:?} with storage class {:?}", inny_ty.clone(), storage_class))?;

                        let ptr = self.spirv_builder.access_chain(*element_ptr_type, None, var_word, vec![lit_0, index_word])?;
                        (ptr, inny_ty.as_ref().clone())
                    }
                    MimirExprIR::Var(var_name) => {
                        let var = self.vars.get(var_name).ok_or(anyhow!("Variable {:?} not found", var_name))?;
                        let ptr = var.word.ok_or(anyhow!("Variable {:?} has no corresponding word", var))?;
                        (ptr, var.ty.base.clone())
                    }
                    _ => return Err(anyhow!("LHS of BinAssign must be a variable or index, got {:?}", lhs)),
                };

                // Get the type word for the LHS base type
                let lhs_ty_word = *self.types.get(&lhs_base_ty)
                    .ok_or(anyhow!("Type {:?} not found for LHS of BinAssign", lhs_base_ty))?;

                // Load the current value from the LHS pointer
                let lhs_current_value = self.spirv_builder.load(lhs_ty_word, None, lhs_ptr_word, None, vec![])?;

                // Get the type and value of the RHS expression
                let (rhs_ty, rhs_value_word) = self.parse_hand_side(&rhs)?;

                // Cast RHS value if its type doesn't match the LHS base type
                let rhs_casted_word = if rhs_ty != lhs_base_ty {
                    self.cast_word(rhs_value_word, &rhs_ty, &lhs_base_ty)
                        .map_err(|e| anyhow!("Failed to cast RHS for BinAssign: {}", e))?
                } else {
                    rhs_value_word
                };

                // Perform the binary operation using the (casted) RHS value and the loaded LHS value
                // The result type should match the LHS type (lhs_ty_word)
                let result_word = match lhs_base_ty {
                    MimirType::Int32 | MimirType::Int64 => {
                        match mimir_bin_op {
                            MimirBinOp::Add => self.spirv_builder.i_add(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Sub => self.spirv_builder.i_sub(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Mul => self.spirv_builder.i_mul(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Div => self.spirv_builder.s_div(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Mod => self.spirv_builder.s_mod(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            _ => return Err(anyhow!("Unsupported binop {:?} for type {:?}", mimir_bin_op, lhs_base_ty)),
                        }
                    },
                    MimirType::Float32 => {
                        match mimir_bin_op {
                            MimirBinOp::Add => self.spirv_builder.f_add(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Sub => self.spirv_builder.f_sub(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Mul => self.spirv_builder.f_mul(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Div => self.spirv_builder.f_div(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            // Modulo typically not supported for floats in SPIR-V core
                            _ => return Err(anyhow!("Unsupported binop {:?} for type {:?}", mimir_bin_op, lhs_base_ty)),
                        }
                    },
                    MimirType::Uint32 => {
                        match mimir_bin_op {
                            MimirBinOp::Add => self.spirv_builder.i_add(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Sub => self.spirv_builder.i_sub(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Mul => self.spirv_builder.i_mul(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Div => self.spirv_builder.u_div(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            MimirBinOp::Mod => self.spirv_builder.u_mod(lhs_ty_word, None, lhs_current_value, rhs_casted_word)?,
                            _ => return Err(anyhow!("Unsupported binop {:?} for type {:?}", mimir_bin_op, lhs_base_ty)),
                        }
                    },
                    _ => return Err(anyhow!("Unsupported LHS type {:?} for BinAssign", lhs_base_ty)),
                };

                // Store the result back into the LHS pointer
                self.spirv_builder.store(lhs_ptr_word, result_word, None, vec![])?;

                Ok(())
            }
            MimirExprIR::BinOp(_, _, _, _) => Err(anyhow!("Standalone BinOp not allowed")),
            MimirExprIR::Index(_, _) => Err(anyhow!("Standalone Index not allowed")),
            MimirExprIR::Field(_, _) => Err(anyhow!("Standalone Field access not allowed")),
            MimirExprIR::For(iter_var_name, lhs, rhs, step, mimir_expr_irs) => {
                        // TODO: Handle step value if necessary
                        self.for_loop(iter_var_name, &lhs, &rhs, mimir_expr_irs, step)
                            .map_err(|e| anyhow!(e))?;
                        Ok(())
                    }
            MimirExprIR::If(cond, then, opt_else) => {
                        self.build_branch(&cond, &then, &opt_else)
                    },
            MimirExprIR::Unary(_, _) => Err(anyhow!("Standalone Unary operation not allowed")),
            MimirExprIR::Literal(_) => Err(anyhow!("Standalone literal not allowed")),
            MimirExprIR::Var(_) => Err(anyhow!("Standalone variable not allowed")),
            MimirExprIR::Return => {
                self.spirv_builder.ret()?;

                Ok(())
            },
            MimirExprIR::ExtInstFunc(_, _) => Err(anyhow!("Standalone ExtInstFunc not allowed")),
            MimirExprIR::Syncthreads => {
                // Get constants for scopes and semantics
                let workgroup_scope = self.spirv_builder.constant_bit32(
                    *self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found"))?,
                    spirv::Scope::Workgroup as u32
                );
                
                let device_scope = self.spirv_builder.constant_bit32(
                    *self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found"))?,
                    spirv::Scope::Device as u32
                );
                
                let semantics = self.spirv_builder.constant_bit32(
                    *self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found"))?,
                    (spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits()
                );
                
                // OpControlBarrier with Workgroup execution scope, Device memory scope, and appropriate semantics
                self.spirv_builder.control_barrier(workgroup_scope, device_scope, semantics)?;
                
                Ok(())
            }
        }
    }

    pub fn init_types(&mut self) -> Result<()> {
        let mut types = self
            .vars.values().map(|var| var.ty.base.clone())
            .collect::<HashSet<MimirType>>();

        // add builtins to the types
        if !self.builtins.is_empty() {
            types.insert(
                MimirType::Uint32
            );
            types.insert(
                MimirType::Uint32Vec3
            );
        }

        // need int32 ty for literals, and work group size specialization
        types.insert(MimirType::Int32);

        for ty in types {
            if !self.types.contains_key(&ty) {
                // handle vector and runtime arr types
                match ty {
                    MimirType::Uint32Vec3 => {
                        let uint32 = if let std::collections::hash_map::Entry::Vacant(e) = self.types.entry(MimirType::Uint32) {
                            let uint32 = self.spirv_builder.type_int(32, 0);
                            e.insert(uint32);
                            uint32
                        } else {
                            *self.types.get(&MimirType::Uint32).unwrap()
                        };

                        let vec3 = self.spirv_builder.type_vector(uint32, 3);
                        self.types.insert(ty.clone(), vec3);
                    }
                    MimirType::Float32Vec3 => {
                        let float32 = if let Vacant(e) =
                            self.types.entry(MimirType::Float32)
                        {
                            let float32 = self.spirv_builder.type_float(32);
                            e.insert(float32);
                            float32
                        } else {
                            *self.types.get(&MimirType::Float32).unwrap()
                        };

                        let vec3 = self.spirv_builder.type_vector(float32, 3);
                        self.types.insert(ty.clone(), vec3);
                    }
                    MimirType::RuntimeArray(inner_ty_) => {
                        let inner_ty = inner_ty_.as_ref().clone();

                        // Get or create the base type for the elements in the runtime arr
                        let base = if self.types.contains_key(&inner_ty) {
                            *self.types.get(&inner_ty).unwrap()
                        } else {
                            let base = self.get_raw_type(&inner_ty)?;
                            self.types.insert(inner_ty.clone(), base);
                            base
                        };

                        // Create runtime arr type
                        let array = self.spirv_builder.type_runtime_array(base);
                        self.types.insert(MimirType::RuntimeArray(Box::new(inner_ty.clone())), array);

                        let num_bytes = match inner_ty {
                            MimirType::Int32 => 4,
                            MimirType::Int64 => 8,
                            MimirType::Float32 => 4,
                            MimirType::Bool => 1,
                            _ => return Err(anyhow!("Unsupported type for runtime array")),
                        } as u32;

                        self.spirv_builder.decorate(
                            array,
                            spirv::Decoration::ArrayStride,
                            vec![Operand::LiteralBit32(num_bytes)],
                        );

                        // Create a struct containing the runtime arr
                        let struct_ty = self.spirv_builder.type_struct(vec![array]);
                        
                        // Store struct type using RuntimeArray(RuntimeArray(inner_ty))
                        // This matches how buffers_to_spirv expects to find it
                        self.types.insert(
                            MimirType::RuntimeArray(Box::new(MimirType::RuntimeArray(Box::new(inner_ty.clone())))),
                            struct_ty
                        );

                        self.spirv_builder.decorate(
                            struct_ty,
                            spirv::Decoration::BufferBlock,
                            vec![],
                        );

                        // Also doesn't work for some reason
                        self.spirv_builder.member_decorate(
                            struct_ty,
                            0,
                            spirv::Decoration::Offset,
                            vec![Operand::LiteralBit32(0)]
                        );


                    }
                    MimirType::Int32 | MimirType::Int64 | MimirType::Float32 | MimirType::Bool => {
                        let raw_ty = self.get_raw_type(&ty)?;
                        self.types.insert(ty.clone(), raw_ty);
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    pub(crate) fn get_raw_type(&mut self, ty: &MimirType) -> Result<Word> {
        match ty {
            MimirType::Int32 => Ok(self.spirv_builder.type_int(32, 1)),
            MimirType::Int64 => Ok(self.spirv_builder.type_int(64, 1)),
            MimirType::Float32 => Ok(self.spirv_builder.type_float(32)),
            MimirType::Bool => Ok(self.spirv_builder.type_bool()),
            MimirType::Void => Ok(self.spirv_builder.type_void()),
            MimirType::Unknown => Err(anyhow!("Unknown type found in variable")),
            _ => Err(anyhow!("Unsupported type found: {:?}", ty)), // handle unsupported types
        }
    }

    pub fn init_builder(&mut self) -> Result<()> {
        // Set SPIR-V version first (requires SPIR-V version 1.2 for OpExecutionMode & LocalSizeId)
        self.spirv_builder.set_version(1, 2);

        // Add Shader capability - required for compute shaders
        self.spirv_builder.capability(spirv::Capability::Shader);
        
        // Set memory model after capabilities
        self.spirv_builder
            .memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        self.ext_inst = self.spirv_builder.ext_inst_import("GLSL.std.450".to_string());
        
        let void = self.spirv_builder.type_void();
        self.types.insert(MimirType::Void, void);

        // init types & corresponding ptr types
        self.init_types()?;
        self.setup_ptr_ty()?;

        self.types.insert(MimirType::Bool, self.spirv_builder.type_bool());
        self.ptr_types.insert(
            MimirPtrType {
                base: MimirType::Bool,
                storage_class: StorageClass::Function,
            },
            self.spirv_builder.type_pointer(
                None,
                StorageClass::Function,
                *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found"))?
            ),
        );

        let voidf = self.spirv_builder.type_function(void, vec![]);


        let (const_x, const_y, const_z) = {
            let uint = *self
                .types
                .get(&MimirType::Uint32)
                .ok_or(anyhow!("Uint32 type not found"))?;

            let const_x = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_x,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(0)]
            );
            let const_y = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_y,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(1)]
            );
            let const_z = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_z,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(2)]
            );

            (const_x, const_y, const_z)
        };

        // GPU side builtins (block_idx, block_dim, thread_idx, global_invocation_id)
        let mut builtins = vec![];
        for builtin in self.builtins.clone() {
            if builtin == MimirBuiltIn::BlockDim {
                let vec3_type = self.types.get(&MimirType::Uint32Vec3).ok_or(anyhow!("Uint32Vec3 type not found"))?;
                
                // Create a constant vector (1,1,1) for the work group size
                let workgroup_size = self.spirv_builder.spec_constant_composite(*vec3_type, vec![const_x, const_y, const_z]);
                
                // Mark it with the WorkgroupSize builtin decoration
                self.spirv_builder.decorate(
                    workgroup_size,
                    spirv::Decoration::BuiltIn,
                    vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupSize)],
                );
                
                // Update the variable in the IR to point to this constant
                let var = self.vars.get_mut("block_dim").ok_or(anyhow!("Variable for builtin {:?} not found", builtin))?;
                var.word = Some(workgroup_size);
                
                continue; // Skip the rest of the builtins for this iteration, since we handled BlockDim separately 
            }

            let vec3_word = self.types.get(&MimirType::Uint32Vec3).ok_or(anyhow!("Uint32Vec3 type not found"))?;

            let vec3_ptr_word = if let Vacant(e) = self
                .ptr_types
                .entry(
                    MimirPtrType {
                        base: MimirType::Uint32Vec3,
                        storage_class: StorageClass::Input,
                    }
                ) {
                    let vec3_ptr_word = self.spirv_builder
                        .type_pointer(None, StorageClass::Input, *vec3_word);

                    e.insert(vec3_ptr_word);
                    vec3_ptr_word
                } else {
                    *self
                        .ptr_types
                        .get(&MimirPtrType {
                            base: MimirType::Uint32Vec3,
                            storage_class: StorageClass::Input,
                        })
                        .unwrap()
                };

                if let Vacant(e) = self.ptr_types.entry(MimirPtrType {
                    base: MimirType::Uint32,
                    storage_class: StorageClass::Input,
                }) {
                    let uint_word = self.types.get(&MimirType::Uint32).ok_or(anyhow!("Uint32 type not found"))?;

                    let ptr_word = self.spirv_builder
                        .type_pointer(None, StorageClass::Input, *uint_word);

                    e.insert(ptr_word);
                }

                let word = self
                    .spirv_builder
                    .variable(vec3_ptr_word, None, StorageClass::Input, None);

                self.spirv_builder.decorate(
                    word,
                    spirv::Decoration::BuiltIn,
                    match builtin {
                        MimirBuiltIn::BlockIdx => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupId)]
                        }
                        MimirBuiltIn::BlockDim => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)]
                        }
                        MimirBuiltIn::ThreadIdx => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::LocalInvocationId)]
                        }
                        MimirBuiltIn::GlobalInvocationId => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)]
                        }
                    },
                );

                builtins.push(word);
                let var = self.vars.get_mut(match builtin {
                    MimirBuiltIn::BlockIdx => "block_idx",
                    MimirBuiltIn::BlockDim => "block_dim",
                    MimirBuiltIn::ThreadIdx => "thread_idx",
                    MimirBuiltIn::GlobalInvocationId => "global_invocation_id",
                }).ok_or(anyhow!("Variable for builtin {:?} not found", builtin))?;

                var.word = Some(word);
        }

        // let const1 = self
        //     .spirv_builder
        //     .constant_bit32(*self.types.get(&MimirType::Int32).unwrap(), 1);

        self.spirv_builder.source(spirv::SourceLanguage::Unknown, 1, None, None as Option<String>);

        // self.spirv_builder.extension("SPV_KHR_storage_buffer_storage_class".to_string());
        // self.spirv_builder.extension("SPV_KHR_variable_pointers".to_string());

        let mut literals_set = self.get_literals();

        literals_set.insert(MimirLit::Int32(0));
        literals_set.insert(MimirLit::Int32(1));
        literals_set.insert(MimirLit::Int32(2));

        let literals = literals_set.into_iter().collect::<Vec<_>>();

        for lit in literals {
            let word = match lit {
                MimirLit::Int32(val) => self
                    .spirv_builder
                    .constant_bit32(*self.types.get(&MimirType::Int32).ok_or(anyhow!("Int32 type not found!"))?, val as u32),
                MimirLit::Int64(val) => self
                    .spirv_builder
                    .constant_bit64(*self.types.get(&MimirType::Int64).ok_or(anyhow!("Int64 type not found!"))?, val as u64),
                MimirLit::Float32(val) => {
                    let bits = val;
                    
                    self.spirv_builder
                        .constant_bit32(*self.types.get(&MimirType::Float32).ok_or(anyhow!("Float32 type not found!"))?, bits)
                },
                MimirLit::Bool(val) => {
                    if val {
                        self.spirv_builder
                            .constant_true(*self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found!"))?)
                    } else {
                        self.spirv_builder
                            .constant_false(*self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found!"))?)
                    }
                }
            };

            self.literals.insert(lit, word);
        }

        self.params_to_spirv()?;        

        let main = self.spirv_builder.begin_function(
                void,
                None,
                spirv::FunctionControl::empty(),
                voidf
        )?;

        self.spirv_builder.begin_block(None)?;

        self.spirv_builder
            .entry_point(spirv::ExecutionModel::GLCompute, main, "main", &builtins);

        // Add ExecutionMode LocalSizeId using the specialization constants
        self.spirv_builder.execution_mode_id(
            main,
            spirv::ExecutionMode::LocalSizeId,
            vec![
                const_x,
                const_y,
                const_z,
            ],
        );

        // self.spirv_builder.execution_mode(
        //     main,
        //     spirv::ExecutionMode::LocalSize,
        //     vec![1, 1, 1],
        // );

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
                        MimirExprIR::Var(name) => {
                            if self.param_order.contains(&name) {
                                let index = self.param_order.iter().position(|x| *x == name);
                                if let Some(idx) = index {
                                    literals.insert(MimirLit::Int32(idx as i32));
                                }
                            }
                        },
                        MimirExprIR::Local(_, expr) => {
                            if let Some(val) = expr {
                                if let MimirExprIR::Literal(lit) = *val {
                                    literals.insert(lit);
                                } else {
                                    ir_vec.push(*val);
                                }
                            }
                        }
                        MimirExprIR::Assign(_, expr) => {
                            if let MimirExprIR::Literal(lit) = *expr {
                                literals.insert(lit);
                            } else {
                                ir_vec.push(*expr);
                            }
                        }
                        MimirExprIR::BinAssign(_, _, expr) => {
                            if let MimirExprIR::Literal(lit) = *expr {
                                literals.insert(lit);
                            } else {
                                ir_vec.push(*expr);
                            }
                        }
                        MimirExprIR::BinOp(lhs, _, rhs, _) => {
                            ir_vec.push(*lhs);
                            ir_vec.push(*rhs);
                        }
                        MimirExprIR::Index(_, index) => {
                            ir_vec.push(*index);
                        }
                        MimirExprIR::For(_, l, r, step, expr) => {
                            if let Some(step) = step {
                                if let MimirExprIR::Literal(lit) = *step {
                                    literals.insert(lit);
                                } else {
                                    ir_vec.push(*step);
                                }
                            }

                            ir_vec.push(*l); // lhs of the for loop (init)
                            ir_vec.push(*r); // rhs of the for loop (end)

                            for e in expr {
                                ir_vec.push(e);
                            }
                        }
                        MimirExprIR::If(cond, then, els) => {
                            ir_vec.push(*cond);
                            ir_vec.extend(then);
                            if let Some(e) = els {
                                ir_vec.extend(e);
                            }
                        }
                        MimirExprIR::Unary(_, expr) => {
                            ir_vec.push(*expr);
                        }
                        MimirExprIR::Literal(lit) => {
                            literals.insert(lit);
                        }
                        MimirExprIR::Field(var, field) => {
                            if let Some(mimir_var) = self.vars.get(&var) {
                                if mimir_var.ty.base == MimirType::Uint32Vec3
                                    || mimir_var.ty.base == MimirType::Float32Vec3
                                {
                                    match field.as_str() {
                                        "x" => {
                                            literals.insert(MimirLit::Int32(0));
                                        }
                                        "y" => {
                                            literals.insert(MimirLit::Int32(1));
                                        }
                                        "z" => {
                                            literals.insert(MimirLit::Int32(2));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        _ => continue,
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
    pub fn ir_to_word(&mut self, ir: &MimirExprIR) -> Result<Word> {
        match ir {
            MimirExprIR::Local(_, _) => Err(anyhow!("Local not allowed within expression")),
            MimirExprIR::Assign(_, _) => Err(anyhow!("Assign not allowed within expression")),
            MimirExprIR::BinAssign(_, _, _) => Err(anyhow!("BinAssign not allowed within expression")),
            MimirExprIR::BinOp(lhs, op, rhs, _) => {
                // build_binop handles parsing sides, casting, and performing the operation
                Ok(self.build_binop(*lhs.clone(), op, *rhs.clone())?.1)
            }
            MimirExprIR::Index(name, index) => {
                // Extract all necessary information from the variable before recursive calls
                let (var_word, inny_ty, ptr_storage_class) = {
                    let var = self.vars.get(name).ok_or(anyhow!("Couldn't find var {}!", name))?;
                    if let MimirType::RuntimeArray(inny_ty) = &var.ty.base {
                        let var_word = var.word.ok_or(anyhow!("Variable {} doesn't have a init", name))?;
                        let ptr_storage_class = var.ty.storage_class;
                        (var_word, inny_ty.clone(), ptr_storage_class)
                    } else {
                        return Err(anyhow!("Indexing not allowed for non-runtime arr types: {:?}", var.ty.base));
                    }
                };
                
                // Now we can make recursive calls without holding references to self.vars
                let index_word = self.ir_to_word(index)?;
                let lit_0 = *self.literals.get(&MimirLit::Int32(0)).ok_or(anyhow!("Couldn't find literal 0"))?;

                let element_ptr_type = self.ptr_types.get(&MimirPtrType { base: *inny_ty.clone(), storage_class: ptr_storage_class })
                    .ok_or(anyhow!("Couldn't find element pointer type for {:?} with storage class {:?}", inny_ty, ptr_storage_class))?;
                
                let ty = *self.types.get(inny_ty.as_ref()).ok_or(anyhow!("Couldn't find type: {:?}", inny_ty))?;

                // Access chain to get pointer to the element
                let ptr = self.spirv_builder.access_chain(*element_ptr_type, None, var_word, vec![lit_0, index_word])?;
                
                // Load the value from the element pointer
                let result = self.spirv_builder.load(ty, None, ptr, None, vec![])?;
                Ok(result)
            }
            MimirExprIR::Field(_, _) => {
                 // Use parse_hand_side as it already implements the logic for field access and returns the loaded value
                 Ok(self.parse_hand_side(ir)?.1)
            }
            MimirExprIR::For(_, _, _, _, _) => Err(anyhow!("For not allowed within expression")),
            MimirExprIR::If(_, _, _) => Err(anyhow!("If not allowed within expression")),
            MimirExprIR::Unary(op, expr) => {
                // Use parse_hand_side to get the inner expression's value and type
                let (inner_ty, inner_word) = self.parse_hand_side(expr)?;
                let ty_word = *self.types.get(&inner_ty).ok_or(anyhow!("Type {:?} not found for unary op", inner_ty))?;

                match op {
                    ir::MimirUnOp::Not => {
                        if inner_ty != MimirType::Bool {
                            return Err(anyhow!("Logical Not requires Bool type, got {:?}", inner_ty));
                        }
                        let bool_ty_word = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Bool type not found"))?;
                        Ok(self.spirv_builder.logical_not(bool_ty_word, None, inner_word)?)
                    }
                    ir::MimirUnOp::Neg => {
                        match inner_ty {
                            MimirType::Int32 | MimirType::Int64 => Ok(self.spirv_builder.s_negate(ty_word, None, inner_word)?),
                            MimirType::Float32 => Ok(self.spirv_builder.f_negate(ty_word, None, inner_word)?),
                            _ => Err(anyhow!("Negation not supported for type {:?}", inner_ty)),
                        }
                    }
                }
            }
            MimirExprIR::Literal(lit) => self
                        .literals
                        .get(lit)
                        .cloned()
                        .ok_or_else(|| anyhow!("Failed to retrieve literal {:?} - might be missing from get_literals scan", lit)),
            MimirExprIR::Var(_) => {
                // Use parse_hand_side as it handles loading variables, push constants, and built-ins
                Ok(self.parse_hand_side(ir)?.1)
            }
            MimirExprIR::Return => Err(anyhow!("Return not allowed within expression")),
            MimirExprIR::ExtInstFunc(_, _) => {
                // Use parse_hand_side as it handles ExtInstFunc
                Ok(self.parse_hand_side(ir)?.1)
            }
            MimirExprIR::Syncthreads => Err(anyhow!("Syncthreads not allowed within expression")),
        }
    }
}
