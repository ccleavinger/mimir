use mimir_ir::ir::{MimirExpr, MimirLit, MimirPrimitiveTy, MimirStmt, MimirTy};
use mimir_runtime::generic::compiler::MimirJITCompilationError;

use crate::vulkan_spirv_compiler::{compiler::VulkanSpirVCompiler, util::map_err_closure};

impl VulkanSpirVCompiler {
    pub fn range_for_to_spirv(
        &mut self,
        var: &u64,
        start: &MimirExpr,
        end: &MimirExpr,
        step: Option<&MimirExpr>,
        body: &[MimirStmt],
    ) -> Result<(), MimirJITCompilationError> {
        let i32_ty = self.get_mimir_ty(&MimirTy::Primitive(MimirPrimitiveTy::Int32))?;
        let bool_ty = self.get_mimir_ty(&MimirPrimitiveTy::Bool.to_mimir_ty())?;

        let iter_var = self.get_mimir_var(var)?;
        let iter_var_word = iter_var.word;
        let iter_var_ty = iter_var.ty.base.clone();

        let (start_word, start_ty) = self.expr_to_word(start)?;
        let (end_word, end_ty) = self.expr_to_word(end)?;

        let header_block = self.spirv_builder.id();
        let body_block = self.spirv_builder.id();
        let continue_block = self.spirv_builder.id();
        let merge_block = self.spirv_builder.id();

        if start_ty != iter_var_ty || end_ty != iter_var_ty {
            return Err(MimirJITCompilationError::Generic(
                format!(
                    "Type mismatch in a ranged for loop.\nExpected `{:?}` got a `{:?}` for starting type in range and `{:?}` for ending type in range",
                    iter_var_ty,
                    start_ty,
                    end_ty
            )));
        }

        self.spirv_builder
            .store(iter_var_word, start_word, None, vec![])
            .map_err(map_err_closure)?;

        // update the variable, TODO: make sure this is actually needed
        if let Some(mut_var) = self.vars.get_mut(var) {
            mut_var.word = iter_var_word
        }

        self.spirv_builder
            .branch(header_block)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .begin_block(Some(header_block))
            .map_err(map_err_closure)?;

        let i_val = self
            .spirv_builder
            .load(i32_ty, None, iter_var_word, None, vec![])
            .map_err(map_err_closure)?;
        let cmp = self
            .spirv_builder
            .s_less_than(bool_ty, None, i_val, end_word)
            .map_err(map_err_closure)?;

        // Add loop merge instruction IMMEDIATELY before the branch - this is critical for SPIR-V validation
        self.spirv_builder
            .loop_merge(
                merge_block,
                continue_block,
                rspirv::spirv::LoopControl::NONE,
                vec![],
            )
            .map_err(map_err_closure)?;

        self.spirv_builder
            .branch_conditional(cmp, body_block, merge_block, vec![])
            .map_err(map_err_closure)?;

        // loop body
        self.spirv_builder
            .begin_block(Some(body_block))
            .map_err(map_err_closure)?;
        for ir in body {
            self.ir_to_spirv(ir)?;
        }

        // branch directly to continue block
        self.spirv_builder
            .branch(continue_block)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .begin_block(Some(continue_block))
            .map_err(map_err_closure)?;
        // load only once per operation
        let i_val = self
            .spirv_builder
            .load(i32_ty, None, iter_var_word, None, vec![])
            .map_err(map_err_closure)?;
        let step_word = if let Some(step_expr) = step {
            let (word, _) = self.expr_to_word(step_expr)?;
            word
        } else {
            *self.get_literal(MimirLit::Int32(1))?
        };

        let i_plus_step = self
            .spirv_builder
            .i_add(i32_ty, None, i_val, step_word)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .store(iter_var_word, i_plus_step, None, vec![])
            .map_err(map_err_closure)?;

        self.spirv_builder
            .branch(header_block)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .begin_block(Some(merge_block))
            .map_err(map_err_closure)?;

        Ok(())
    }
}
