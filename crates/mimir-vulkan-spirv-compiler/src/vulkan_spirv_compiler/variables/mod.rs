use mimir_ir::{ir::{MimirPrimitiveTy, MimirTy}, kernel_ir::MimirKernelIR};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::{dr::Operand, spirv::StorageClass};

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler,
    ir::{MimirPtrType, MimirVariable},
};

pub(crate) mod shared_mem;
pub(crate) mod var_util;

impl VulkanSpirVCompiler {
    pub(crate) fn setup_vars(
        &mut self,
        kernel: &MimirKernelIR,
    ) -> Result<(), MimirJITCompilationError> {
        for (uuid, ty_var) in kernel.var_map.iter() {
            if !self.vars.contains_key(uuid) {
                let var_word = self.ty_var_to_word(ty_var, StorageClass::Function)?;

                self.spirv_builder.name(
                    var_word,
                    kernel.name_map.get(uuid).unwrap_or(&"".to_string()),
                );

                if ty_var.ty.is_global_array() {
                    let binding_idx = self.buffer_order.iter().position(|id| id == uuid).ok_or(
                        MimirJITCompilationError::Generic(format!(
                            "failed to finding index for buffer: {}",
                            kernel.name_map.get(uuid).unwrap_or(&"".to_string())
                        )),
                    )?;
                    self.spirv_builder.decorate(
                        var_word,
                        rspirv::spirv::Decoration::Binding,
                        vec![Operand::LiteralBit32(binding_idx as u32)],
                    );
                }

                self.vars.insert(
                    *uuid,
                    MimirVariable {
                        ty: MimirPtrType {
                            base: ty_var.ty.clone(),
                            storage_class: StorageClass::Function,
                        },
                        word: (var_word),
                    },
                );
            }
        }

        if !self.pc_order.is_empty() {
            let _ = self.get_ptr_ty(&MimirPtrType {
                base: MimirTy::Primitive(MimirPrimitiveTy::Uint32),
                storage_class: StorageClass::Function,
            });
        }

        Ok(())
    }
}
