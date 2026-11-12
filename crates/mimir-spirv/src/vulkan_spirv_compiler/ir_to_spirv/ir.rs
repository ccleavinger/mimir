use mimir_ir::ir::MimirStmt;
use mimir_runtime::generic::compiler::MimirJITCompilationError;

use crate::vulkan_spirv_compiler::{compiler::VulkanSpirVCompiler, util::map_err_closure};

impl VulkanSpirVCompiler {
    pub(crate) fn ir_to_spirv(&mut self, ir: &MimirStmt) -> Result<(), MimirJITCompilationError> {
        match ir {
            MimirStmt::Assign { lhs, rhs } => self.assign_to_spirv(lhs, rhs),
            MimirStmt::RangeFor {
                var,
                start,
                end,
                step,
                body,
            } => self.range_for_to_spirv(var, start, end, step.as_ref(), body),
            MimirStmt::If {
                condition,
                then_branch,
                else_branch,
            } => self.if_else_to_spirv(
                condition,
                then_branch,
                else_branch.as_ref().map(|x| x.as_slice()),
            ),
            MimirStmt::Return(mimir_expr_ir) => {
                if mimir_expr_ir.is_some() {
                    return Err(MimirJITCompilationError::Generic(
                        "Cannot return any values from a kernel!".to_owned(),
                    ));
                }

                self.spirv_builder.ret().map_err(map_err_closure)
            }
            MimirStmt::Syncthreads => self.syncthreads_to_spirv(),
        }
    }
}
