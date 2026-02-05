use mimir_ir::ir::{MimirExpr, MimirLit, MimirPrimitiveTy, MimirStmt};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv::SelectionControl;

use crate::vulkan_spirv_compiler::{compiler::VulkanSpirVCompiler, util::map_err_closure};

impl VulkanSpirVCompiler {
    pub fn if_else_to_spirv(
        &mut self,
        cond: &MimirExpr,
        then: &[MimirStmt],
        else_branch: Option<&[MimirStmt]>,
    ) -> Result<(), MimirJITCompilationError> {
        // baby's first compiler optimization
        if let MimirExpr::Literal(MimirLit::Bool(var)) = *cond {
            if var {
                for ir in then {
                    self.ir_to_spirv(ir)?;
                }
            } else if let Some(e) = else_branch {
                for ir in e {
                    self.ir_to_spirv(ir)?;
                }
            }
            return Ok(());
        };

        let (cond_word, cond_ty) = self.expr_to_word(cond)?;

        if MimirPrimitiveTy::Bool.to_mimir_ty() != cond_ty {
            return Err(MimirJITCompilationError::Generic(format!(
                "Conditional in an if statement must be a boolean, not a `{:?}",
                cond_ty
            )));
        }

        if let Some(else_) = else_branch {
            let merge_block = self.spirv_builder.id();
            self.spirv_builder
                .selection_merge(merge_block, SelectionControl::NONE)
                .map_err(map_err_closure)?;

            let true_block = self.spirv_builder.id();
            let false_block = self.spirv_builder.id();

            self.spirv_builder
                .branch_conditional(cond_word, true_block, false_block, vec![])
                .map_err(map_err_closure)?;

            // true block (aka then block)
            self.spirv_builder
                .begin_block(Some(true_block))
                .map_err(map_err_closure)?;
            for ir in then {
                self.ir_to_spirv(ir)?;
            }
            self.spirv_builder
                .branch(merge_block)
                .map_err(map_err_closure)?;

            // false block (aka else block)
            self.spirv_builder
                .begin_block(Some(false_block))
                .map_err(map_err_closure)?;
            for ir in else_ {
                self.ir_to_spirv(ir)?;
            }
            self.spirv_builder
                .branch(merge_block)
                .map_err(map_err_closure)?;

            // Begin the merge block after both branches are complete
            self.spirv_builder
                .begin_block(Some(merge_block))
                .map_err(map_err_closure)?;
        } else {
            let merge_block = self.spirv_builder.id();
            let continue_block = self.spirv_builder.id();

            self.spirv_builder
                .selection_merge(merge_block, SelectionControl::NONE)
                .map_err(map_err_closure)?;

            self.spirv_builder
                .branch_conditional(cond_word, continue_block, merge_block, vec![])
                .map_err(map_err_closure)?;

            self.spirv_builder
                .begin_block(Some(continue_block))
                .map_err(map_err_closure)?;
            for ir in then {
                self.ir_to_spirv(ir)?;
            }
            self.spirv_builder
                .branch(merge_block)
                .map_err(map_err_closure)?;
            self.spirv_builder
                .begin_block(Some(merge_block))
                .map_err(map_err_closure)?;
        }

        Ok(())
    }
}
