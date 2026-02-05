use mimir_ir::ir::MimirLit;
use rspirv::spirv::StorageClass;

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir_to_spirv::expr_to_spirv::expr::ExprToWordResult,
    util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub fn var_to_word(&mut self, uuid: &u64) -> ExprToWordResult {
        let var = self.get_mimir_var(uuid)?;
        let ty = self.get_mimir_ty(&var.ty.base)?;

        if var.ty.storage_class == StorageClass::PushConstant {
            let idx = self.pc_order.iter().position(|x| *x == *uuid).unwrap();

            let push_const = self.pc_var_word;

            let ptr_ty = self.get_ptr_ty(&var.ty)?;

            // let ty = self.get_mimir_ty(&var.ty.base)?;

            let lit_idx = *match self.get_literal(MimirLit::Int32(idx as i32)) {
                Ok(w) => w,
                Err(e) => {
                    println!("Push const order: {:#?}, Idx: {idx}", self.pc_order);
                    return Err(e);
                }
            };
            let word = self
                .spirv_builder
                .access_chain(ptr_ty, None, push_const, vec![lit_idx])
                .map_err(map_err_closure)?;

            let load = self
                .spirv_builder
                .load(ty, None, word, None, vec![])
                .map_err(map_err_closure)?;

            Ok((load, var.ty.base.clone()))
        } else {
            let word = self
                .spirv_builder
                .load(ty, None, var.word, None, vec![])
                .map_err(map_err_closure)?;

            Ok((word, var.ty.base.clone()))
        }
    }
}
