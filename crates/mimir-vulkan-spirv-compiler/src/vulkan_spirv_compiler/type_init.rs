use std::collections::HashSet;

use crate::vulkan_spirv_compiler::{compiler::VulkanSpirVCompiler, ir::MimirPtrType};
use mimir_ir::{
    ir::{MimirConstExpr, MimirPrimitiveTy, MimirTy, MimirTyScope},
    kernel_ir::MimirKernelIR,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::{
    dr::Operand,
    spirv::{self, StorageClass},
};

impl VulkanSpirVCompiler {
    pub(crate) fn init_types(
        &mut self,
        kernel: &MimirKernelIR,
    ) -> Result<(), MimirJITCompilationError> {
        // get all the unique types in the kernel
        let mut type_set = kernel
            .var_map
            .values()
            .map(|t| MimirPtrType {
                base: t.ty.clone(),
                storage_class: match t.scope {
                    MimirTyScope::Param => {
                        if t.ty.is_global_array() {
                            StorageClass::Uniform
                        } else {
                            StorageClass::PushConstant
                        }
                    }
                    MimirTyScope::Local(_) => {
                        if t.ty.is_shared_mem() {
                            StorageClass::Workgroup
                        } else {
                            StorageClass::Function
                        }
                    }
                },
            })
            .collect::<HashSet<_>>();

        for base_ty in [
            MimirTy::Primitive(MimirPrimitiveTy::Float32),
            MimirTy::Primitive(MimirPrimitiveTy::Int32),
            MimirTy::Primitive(MimirPrimitiveTy::Bool),
            MimirTy::Primitive(MimirPrimitiveTy::Uint32),
        ] {
            type_set.insert(MimirPtrType {
                base: base_ty,
                storage_class: StorageClass::Function,
            });
        }

        // if kernel.const_generics > 0 {
        //     type_set.insert(MimirPtrType {
        //         base: MimirTy::Primitive(MimirPrimitiveTy::Uint32),
        //         storage_class: StorageClass::Function,
        //     });
        // }

        for ptr_ty in &type_set {
            if !self.ptr_types.contains_key(ptr_ty) {
                self.setup_ptr_ty(ptr_ty)?;
            }
        }

        Ok(())
    }

    pub(crate) fn setup_ptr_ty(
        &mut self,
        ptr_type: &MimirPtrType,
    ) -> Result<(), MimirJITCompilationError> {
        // just get and return it if it exists lowk
        if self.ptr_types.contains_key(ptr_type) {
            return Ok(());
        }

        if let MimirTy::GlobalArray { element_type } = &ptr_type.base {
            let base_ty = self.mimir_ty_to_word(&element_type.to_mimir_ty())?;

            let runtime_ty = self.spirv_builder.type_runtime_array(base_ty);

            let num_bytes = match &element_type {
                MimirPrimitiveTy::Int32 | MimirPrimitiveTy::Uint32 | MimirPrimitiveTy::Float32 => 4,
                MimirPrimitiveTy::Bool => 1,
            } as u32;

            self.spirv_builder.decorate(
                runtime_ty,
                spirv::Decoration::ArrayStride,
                vec![Operand::LiteralBit32(num_bytes)],
            );

            // create a struct type
            let struct_ty = self.spirv_builder.type_struct(vec![runtime_ty]);

            self.spirv_builder
                .decorate(struct_ty, rspirv::spirv::Decoration::BufferBlock, vec![]);

            // Also doesn't work for some reason
            self.spirv_builder.member_decorate(
                struct_ty,
                0,
                rspirv::spirv::Decoration::Offset,
                vec![rspirv::dr::Operand::LiteralBit32(0)],
            );

            self.types.insert(ptr_type.base.clone(), struct_ty);

            let ptr_ty = self
                .spirv_builder
                .type_pointer(None, StorageClass::Uniform, struct_ty);

            self.ptr_types.insert(ptr_type.clone(), ptr_ty);
        } else if let MimirTy::SharedMemArray { length } = &ptr_type.base {
            let element_type = MimirPrimitiveTy::Float32;
            let base_ty = self.mimir_ty_to_word(&element_type.to_mimir_ty())?;

            let length_lit = match length.as_ref() {
                MimirConstExpr::Literal(mimir_lit) => mimir_lit.clone(),
                _ => unreachable!(),
            };
            if length_lit.to_ty() != MimirPrimitiveTy::Uint32 {
                return Err(MimirJITCompilationError::Generic(
                    "The length of a shared memory array in Mimir must be an unsigned 32 bit integer".to_owned()
                ));
            }

            let lit_w = *self.get_literal(length_lit)?;
            let arr_ty = self.spirv_builder.type_array(base_ty, lit_w);

            self.types.insert(ptr_type.base.clone(), arr_ty);

            let ptr_ty = self
                .spirv_builder
                .type_pointer(None, StorageClass::Workgroup, arr_ty);

            self.ptr_types.insert(ptr_type.clone(), ptr_ty);
        } else {
            let ty_word = self.mimir_ty_to_word(&ptr_type.base)?;
            let ptr_word = self
                .spirv_builder
                .type_pointer(None, ptr_type.storage_class, ty_word);
            self.ptr_types.insert(ptr_type.clone(), ptr_word);
        }

        Ok(())
    }
}
