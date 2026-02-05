use mimir_ir::ir::MimirLit;
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::spirv;

use crate::vulkan_spirv_compiler::{compiler::VulkanSpirVCompiler, util::map_err_closure};

impl VulkanSpirVCompiler {
    pub fn syncthreads_to_spirv(&mut self) -> Result<(), MimirJITCompilationError> {
        let workgroup_scope =
            self.get_literal(MimirLit::Int32(spirv::Scope::Workgroup as u32 as i32))?;

        let device_scope = self.get_literal(MimirLit::Int32(spirv::Scope::Device as u32 as i32))?;

        let semantics = self.get_literal(MimirLit::Int32(
            (spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY)
                .bits() as i32,
        ))?;

        self.spirv_builder
            .control_barrier(*workgroup_scope, *device_scope, *semantics)
            .map_err(map_err_closure)?;

        Ok(())
    }
}
