use mimir_ir::ir::MimirTyVar;
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::{StorageClass, Word};

use crate::{VulkanSpirVCompiler, vulkan_spirv_compiler::ir::MimirPtrType};


impl VulkanSpirVCompiler {
    pub(crate) fn ty_var_to_word(
        &mut self,
        ty_var: &MimirTyVar,
        storage_class: StorageClass
    ) -> Result<Word, MimirJITCompilationError> {
        let mimir_ptr_ty = MimirPtrType {
            base: ty_var.ty.clone(),
            storage_class,
        };

        let ty_word = self.get_ptr_ty(&mimir_ptr_ty)?;

        Ok(self.spirv_builder.variable(
            ty_word,
            None,
            mimir_ptr_ty.storage_class,
            None
        ))
    }
}