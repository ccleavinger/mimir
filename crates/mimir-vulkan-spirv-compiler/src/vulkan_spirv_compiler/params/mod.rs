use std::collections::hash_map::Entry;

use mimir_ir::{
    ir::{MimirPrimitiveTy, MimirTy},
    kernel_ir::MimirKernelIR,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::{
    dr::Operand,
    spirv::{self, StorageClass},
};

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler,
    ir::{MimirPtrType, MimirVariable},
};

impl VulkanSpirVCompiler {
    pub(crate) fn params_to_spirv(
        &mut self,
        kernel: &MimirKernelIR,
    ) -> Result<(), MimirJITCompilationError> {
        self.buffers_to_spirv(kernel)?;

        self.push_const_to_spirv(kernel)?;

        Ok(())
    }

    fn buffers_to_spirv(&mut self, kernel: &MimirKernelIR) -> Result<(), MimirJITCompilationError> {
        let binding = kernel.param_order.clone();
        let buffers = binding
            .iter()
            .map(|uuid| (uuid, kernel.var_map.get(uuid).unwrap().clone()))
            .filter(|(_, ty_var)| ty_var.ty.is_global_array())
            .collect::<Vec<_>>();

        for (idx, (uuid, buffer)) in buffers.iter().enumerate() {
            self.buffer_order.push(**uuid);

            let mimir_ptr_ty = MimirPtrType {
                base: buffer.ty.clone(),
                storage_class: StorageClass::Uniform,
            };

            if let MimirTy::GlobalArray { element_type } = &buffer.ty {
                let ptr_ty_word = match self.ptr_types.entry(mimir_ptr_ty.clone()) {
                    Entry::Occupied(occupied_entry) => *occupied_entry.get(),
                    Entry::Vacant(vacant_entry) => {
                        let base_word = match self
                            .types
                            .get(&element_type.to_mimir_ty())
                            .ok_or_else(|| MimirJITCompilationError::TypeNotFound(buffer.ty.clone()))
                        {
                            Ok(w) => Ok(w),
                            Err(e) => {
                                println!("params.rs1");
                                Err(e)
                            }
                        }?;

                        let runtime_ty = self.spirv_builder.type_runtime_array(*base_word);

                        let struct_ty = self.spirv_builder.type_struct(vec![runtime_ty]);

                        self.spirv_builder.decorate(
                            struct_ty,
                            spirv::Decoration::BufferBlock,
                            vec![],
                        );

                        self.spirv_builder.member_decorate(
                            struct_ty,
                            0,
                            spirv::Decoration::Offset,
                            vec![rspirv::dr::Operand::LiteralBit32(0)],
                        );

                        self.types.insert(buffer.ty.clone(), struct_ty);

                        let ptr_ty =
                            self.spirv_builder
                                .type_pointer(None, StorageClass::Uniform, struct_ty);

                        vacant_entry.insert(ptr_ty);

                        ptr_ty
                    }
                };

                let word =
                    self.spirv_builder
                        .variable(ptr_ty_word, None, StorageClass::Uniform, None);

                {
                    let name = match kernel.name_map.get(uuid) {
                        Some(name) => name,
                        None => &format!("unnamed param #{}", idx),
                    };
                    self.spirv_builder.name(word, name);
                }

                // self.spirv_builder.decorate(
                //     word,
                //     spirv::Decoration::DescriptorSet,
                //     vec![rspirv::dr::Operand::LiteralBit32(idx as u32)]
                // );

                self.spirv_builder.decorate(
                    word,
                    rspirv::spirv::Decoration::DescriptorSet,
                    vec![rspirv::dr::Operand::LiteralBit32(0)], // Descriptor set 0 for uniform buffers
                );
                let binding_idx = self.buffer_order.iter().position(|id| id == *uuid).ok_or(
                    MimirJITCompilationError::Generic(format!(
                        "failed to finding index for buffer: {}",
                        kernel.name_map.get(uuid).unwrap_or(&"".to_string())
                    )),
                )?;
                self.spirv_builder.decorate(
                    word,
                    rspirv::spirv::Decoration::Binding,
                    vec![Operand::LiteralBit32(binding_idx as u32)],
                );

                self.vars.insert(
                    **uuid,
                    MimirVariable {
                        ty: mimir_ptr_ty.clone(),
                        word,
                    },
                );

                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.ptr_types.entry(MimirPtrType {
                        base: MimirTy::Primitive(element_type.clone()),
                        storage_class: StorageClass::Uniform,
                    })
                {
                    let base_word = match self.types.get(&element_type.to_mimir_ty()).ok_or_else(||
                        MimirJITCompilationError::TypeNotFound(MimirTy::Primitive(
                            element_type.clone(),
                        )),
                    ) {
                        Ok(w) => Ok(w),
                        Err(e) => {
                            println!("params.rs2");
                            Err(e)
                        }
                    }?;

                    let ptr =
                        self.spirv_builder
                            .type_pointer(None, StorageClass::Uniform, *base_word);

                    e.insert(ptr);
                }
            }
        }

        Ok(())
    }

    fn push_const_to_spirv(
        &mut self,
        kernel: &MimirKernelIR,
    ) -> Result<(), MimirJITCompilationError> {
        let types = kernel
            .param_order
            .iter()
            .map(|uuid| (uuid, kernel.var_map.get(uuid).unwrap().clone()))
            .filter(|(_, ty_var)| !ty_var.ty.is_global_array())
            .collect::<Vec<_>>();

        let mut tys = vec![];
        for (uuid, pc_type) in types.iter() {
            self.pc_order.push(**uuid);

            let mimir_ptr_type = MimirPtrType {
                base: pc_type.ty.clone(),
                storage_class: StorageClass::PushConstant,
            };

            let base_word = match self.types.get(&mimir_ptr_type.base).ok_or_else(
                || MimirJITCompilationError::TypeNotFound(mimir_ptr_type.base.clone()),
            ) {
                Ok(w) => Ok(w),
                Err(e) => {
                    println!("params.rs3");
                    Err(e)
                }
            }?;

            let _ = match self.ptr_types.entry(mimir_ptr_type.clone()) {
                Entry::Vacant(vacant_entry) => {
                    let ptr_ty = self.spirv_builder.type_pointer(
                        None,
                        mimir_ptr_type.storage_class,
                        *base_word,
                    );
                    vacant_entry.insert(ptr_ty);

                    ptr_ty
                }
                Entry::Occupied(occupied_entry) => *occupied_entry.get(),
            };

            // we lowk need to have the pointer type that is retrievable
            self.vars.insert(
                **uuid,
                MimirVariable {
                    ty: mimir_ptr_type,
                    word: (u32::MAX),
                },
            );

            tys.push(*base_word);
        }

        let pc_struct_ty_word = self.spirv_builder.type_struct(tys.clone());

        let ptr_pc_struct_ty =
            self.spirv_builder
                .type_pointer(None, StorageClass::PushConstant, pc_struct_ty_word);

        self.spirv_builder
            .decorate(pc_struct_ty_word, spirv::Decoration::Block, vec![]);

        let mut current_offset = 0;

        for (i, (_, pc_type)) in types.iter().enumerate() {
            // really this is just 4 bytes rn but bool size might be off and more types will be added later
            let size = match pc_type.ty {
                MimirTy::Primitive(MimirPrimitiveTy::Float32)
                | MimirTy::Primitive(MimirPrimitiveTy::Int32)
                | MimirTy::Primitive(MimirPrimitiveTy::Uint32) => 4,
                MimirTy::Primitive(MimirPrimitiveTy::Bool) => 4, // TODO: make sure this is right
                _ => 4, // default to 4 bytes for other types
            };

            self.spirv_builder.member_decorate(
                pc_struct_ty_word,
                i as u32,
                spirv::Decoration::Offset,
                vec![rspirv::dr::Operand::LiteralBit32(current_offset)],
            );

            // update the offset
            current_offset += size;
            // fix alignment issues
            if current_offset % 4 != 0 {
                current_offset = ((current_offset / 4) + 1) * 4;
            }
        }

        let pc_var =
            self.spirv_builder
                .variable(ptr_pc_struct_ty, None, StorageClass::PushConstant, None);

        self.pc_var_word = pc_var;

        Ok(())
    }
}
