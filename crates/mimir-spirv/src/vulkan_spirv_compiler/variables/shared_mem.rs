use mimir_ir::{ir::MimirTyVar, kernel_ir::MimirKernelIR};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::StorageClass;

use crate::{VulkanSpirVCompiler, vulkan_spirv_compiler::ir::{MimirPtrType, MimirVariable}};


impl VulkanSpirVCompiler {
    fn shared_mem_to_var(
        &mut self,
        ty_var: &MimirTyVar
    ) -> Result<u32, MimirJITCompilationError> {
        self.ty_var_to_word(ty_var, StorageClass::Workgroup)
    }

    pub(crate) fn setup_sh_mem_vars(
        &mut self,
        kernel: &MimirKernelIR
    ) -> Result<(), MimirJITCompilationError> {
        for (uuid, ty_var) in kernel.var_map.iter() {
            if !self.vars.contains_key(uuid) && ty_var.ty.is_shared_mem() {
                let var_word = self.shared_mem_to_var(ty_var)?;

                self.spirv_builder.name(
                    var_word,
                    kernel.name_map.get(uuid).unwrap_or(&"".to_string()),
                );

                self.vars.insert(
                    *uuid,
                    MimirVariable {
                        ty: MimirPtrType {
                            base: ty_var.ty.clone(),
                            storage_class: StorageClass::Workgroup,
                        },
                        word: (var_word),
                    },
                );
            } 
        }
        Ok(())
    }
}