use std::vec;

use anyhow::{anyhow, Result};

use crate::spirv_compiler::compiler::SpirVCompiler;
use crate::spirv_compiler::ir::{MimirPtrType, MimirType, MimirVariable};
use rspirv::spirv::{StorageClass, Word};

impl SpirVCompiler {
    pub fn params_to_spirv(&mut self) -> Result<()> {
        self.buffers_to_spirv()?;

        self.push_const_to_spirv()?;

        Ok(())
    }

    fn buffers_to_spirv(&mut self) -> Result<()> {
        for (str, ty) in self
            .vars
            .iter()
            .filter(|(_, var)| var.ty.storage_class == StorageClass::Uniform)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>()
        {
            if let MimirType::RuntimeArray(arr_ty) = &ty.clone().base.clone() {
                let base_ty = ty.clone().base;

                let ptr_ty = if !self.types.contains_key(&base_ty)
                    || !self.ptr_types.contains_key(&ty.clone())
                {
                    // get word of the type of the elements in the runtime arr
                    let base_word = self.types.get(arr_ty).ok_or(anyhow!(
                        "failed to retrieve base type of runtime arr: {:?}",
                        arr_ty
                    ))?;

                    // create runtime arr type (pretty simple)
                    let runtime_ty = self.spirv_builder.type_runtime_array(*base_word);

                    // create a struct type
                    let struct_ty = self.spirv_builder.type_struct(vec![runtime_ty]);

                    self.spirv_builder.decorate(
                        struct_ty,
                        rspirv::spirv::Decoration::BufferBlock,
                        vec![], 
                    );

                    // Also doesn't work for some reason
                    self.spirv_builder.member_decorate(
                        struct_ty, 
                        0,
                        rspirv::spirv::Decoration::Offset,
                        vec![rspirv::dr::Operand::LiteralBit32(0)]
                    );

                    // save struct type in the types map (may be unnecessary but helps with debugging)
                    self.types.insert(
                        base_ty.clone(),
                        struct_ty,
                    );

                    // Create a pointer to the struct (only type that should be needed)
                    let ptr_ty = self
                        .spirv_builder
                        .type_pointer(None, ty.storage_class, struct_ty);
                        
                    // This shouldn't be needed OpAccessChains mean we can skip the ptr_runtime_array and jump to the indexed value
                    // let ptr_runtime_array = self
                    //     .spirv_builder
                    //     .type_pointer(None, ty.storage_class, runtime_ty);
                    
                    // Register both pointer types
                    self.ptr_types.insert(ty.clone(), ptr_ty);
                    self.ptr_types.insert(
                        MimirPtrType {
                            base: MimirType::RuntimeArray(arr_ty.clone()),
                            storage_class: StorageClass::Uniform,
                        },
                        ptr_ty
                    );

                    ptr_ty
                } else {
                    *self.ptr_types.get(&ty).unwrap()
                };

                let word = self
                    .spirv_builder
                    .variable(ptr_ty, None, ty.storage_class, None);

                self.spirv_builder.name(word, &str);

                let binding_idx = self
                    .buffer_order
                    .iter()
                    .position(|s| s == &str)
                    .ok_or(anyhow!("Failed to find binding index for buffer: {}", str))?;

                self.spirv_builder.decorate(
                    word,
                    rspirv::spirv::Decoration::Binding,
                    vec![rspirv::dr::Operand::LiteralBit32(binding_idx as u32)],
                );

                self.spirv_builder.decorate(
                    word,
                    rspirv::spirv::Decoration::DescriptorSet,
                    vec![rspirv::dr::Operand::LiteralBit32(0)], // Descriptor set 0 for uniform buffers
                );

                self.vars.insert(
                    str,
                    MimirVariable {
                        ty: ty.clone(),
                        word: Some(word),
                    },
                );

                if let std::collections::hash_map::Entry::Vacant(e) = self.ptr_types.entry(MimirPtrType {
                    base: arr_ty.as_ref().clone(),
                    storage_class: StorageClass::Uniform 
                }) {
                    let base_word = self.types.get(arr_ty).ok_or(anyhow!(
                        "failed to retrieve base type of runtime arr: {:?}",
                        arr_ty
                    ))?;

                    let ptr = self.spirv_builder.type_pointer(
                        None,
                        StorageClass::Uniform,
                        *base_word,
                    );

                    e.insert(ptr);
                }

            }
        }

        Ok(())
    }

    fn push_const_to_spirv(&mut self) -> Result<()> {
        let types = self
            .vars
            .iter()
            .filter(|(_, var)| var.ty.storage_class == StorageClass::PushConstant)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>();

        // exit early if no push constants
        if types.is_empty() {
            return Ok(());
        }

        for (str, ty) in self
            .vars
            .iter()
            .filter(|(_, var)| var.ty.storage_class == StorageClass::PushConstant)
            .map(|(str, var)| (str.clone(), var.ty.clone()))
            .collect::<Vec<(String, MimirPtrType)>>()
        {
            let base_word = self.get_raw_type(&ty.base).unwrap();
            if let std::collections::hash_map::Entry::Vacant(e) = self.ptr_types.entry(ty.clone()) {
                let ptr_ty = self
                    .spirv_builder
                    .type_pointer(None, ty.storage_class, base_word);

                e.insert(ptr_ty);
            }

            // let ptr_word = self.ptr_types.get(&ty).unwrap();

            // let word = self
            //     .spirv_builder
            //     .variable(*ptr_word, None, ty.storage_class, None);

            // self.spirv_builder.name(word, &str);

            self.vars.insert(
                str,
                MimirVariable {
                    ty: ty.clone(),
                    word: None,
                },
            );
        }

        let indices: Vec<_> = types
            .iter()
            .map(|(param, _)| self.param_order.iter().position(|p| *p == *param).unwrap())
            .collect();

        let mut ty_words = types
            .iter()
            .map(|(_, ty)| *self.types.get(&ty.base).unwrap())
            .collect::<Vec<Word>>();

        ty_words = indices.iter().map(|i| ty_words[*i]).collect();

        let push_const_struct_ty = self.spirv_builder.type_struct(ty_words.clone()); // Clone here

        let ptr_push_const_struct_ty =
            self.spirv_builder
                .type_pointer(None, StorageClass::PushConstant, push_const_struct_ty);

        self.spirv_builder.decorate(
            push_const_struct_ty,
            rspirv::spirv::Decoration::Block,
            vec![],
        );
        
        // Add required Offset decorations for each member of the struct
        let mut current_offset = 0;
        for (i, &ty_word) in ty_words.iter().enumerate() { // Iterate over the original ty_words
            // Calculate size based on type
            let size = if let Some(ty_info) = self.types.iter().find(|(_, &word)| word == ty_word) {
                match ty_info.0 {
                    MimirType::Int32 | MimirType::Uint32 | MimirType::Float32 => 4,
                    MimirType::Int64 => 8,
                    MimirType::Bool => 4, // Booleans in SPIR-V are 32-bit
                    _ => 4, // Default to 4 bytes for other types
                }
            } else {
                4 // Default to 4 bytes if type info not found
            };
            
            // Add Offset decoration for this member
            self.spirv_builder.member_decorate(
                push_const_struct_ty,
                i as u32,
                rspirv::spirv::Decoration::Offset,
                vec![rspirv::dr::Operand::LiteralBit32(current_offset)],
            );
            
            // Update the offset for the next member (align to 4-byte boundary)
            current_offset += size;
            // Ensure proper alignment
            if current_offset % 4 != 0 {
                current_offset = ((current_offset / 4) + 1) * 4;
            }
        }

        let push_const_var = self.spirv_builder.variable(
            ptr_push_const_struct_ty,
            None,
            StorageClass::PushConstant,
            None,
        );

        self.vars.insert(
            "push_const".to_string(),
            MimirVariable {
                ty: MimirPtrType {
                    base: MimirType::RuntimeArray(Box::new(MimirType::Float32)),
                    storage_class: StorageClass::PushConstant,
                },
                word: Some(push_const_var),
            },
        );

        Ok(())
    }
}
