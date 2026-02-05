use std::{collections::HashMap, iter::zip};

use mimir_ir::{
    ir::{MimirBuiltIn, MimirLit, MimirPrimitiveTy, MimirTy},
    kernel_ir::MimirKernelIR,
};
use mimir_runtime::generic::compiler::MimirJITCompilationError;
use rspirv::{
    dr::Operand,
    spirv::{self, StorageClass},
};

use crate::vulkan_spirv_compiler::{
    compiler::VulkanSpirVCompiler, ir::MimirPtrType, preprocess, util::map_err_closure,
};

impl VulkanSpirVCompiler {
    pub(crate) fn init_builder(
        &mut self,
        kernel: &MimirKernelIR,
    ) -> Result<(), MimirJITCompilationError> {
        // Set SPIR-V version first (requires SPIR-V version 1.2 for OpExecutionMode & LocalSizeId)
        self.spirv_builder.set_version(1, 2);

        // Add Shader capability - required for compute shaders
        self.spirv_builder.capability(spirv::Capability::Shader);

        // Set memory model after capabilities
        // I honestly don't know what this does but it works
        // TODO: research why/how this works
        self.spirv_builder
            .memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

        self.ext_inst = self
            .spirv_builder
            .ext_inst_import("GLSL.std.450".to_owned());

        let void = self.spirv_builder.type_void();

        let uint = self.mimir_ty_to_word(&MimirTy::Primitive(MimirPrimitiveTy::Uint32))?;
        let vec3_type = self.spirv_builder.type_vector(uint, 3);

        let voidf = self.spirv_builder.type_function(void, vec![]);

        let preprocess_info = preprocess::preprocess_kernel_ir(kernel);

        // get push constants set
        let (const_x, const_y, const_z) = {
            let const_x = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_x,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(0)],
            );

            let const_y = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_y,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(1)],
            );

            let const_z = self.spirv_builder.spec_constant_bit32(uint, 1);
            self.spirv_builder.decorate(
                const_z,
                spirv::Decoration::SpecId,
                vec![Operand::LiteralBit32(2)],
            );

            (const_x, const_y, const_z)
        };

        // convert preprocess struct from booleans to a vec of enums
        // (could be a hash set but due to constrained mutability a vec is more than sufficient)
        let ir_builtins = {
            let mut ir_builtins = vec![];
            // if preprocess_info.contains_block_dim {
            //     ir_builtins.push(MimirBuiltIn::BlockDim);
            // }
            if preprocess_info.contains_block_idx {
                ir_builtins.push(MimirBuiltIn::BlockIdx);
            }
            if preprocess_info.contains_gi_id {
                ir_builtins.push(MimirBuiltIn::GlobalInvocationId);
            }
            if preprocess_info.contains_thread_idx {
                ir_builtins.push(MimirBuiltIn::ThreadIdx);
            }
            ir_builtins
        };

        if preprocess_info.contains_block_dim {
            let workgroup_size = self
                .spirv_builder
                .spec_constant_composite(vec3_type, vec![const_x, const_y, const_z]);

            self.spirv_builder.decorate(
                workgroup_size,
                spirv::Decoration::BuiltIn,
                vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupSize)],
            );

            self.builtin_tys
                .insert(MimirBuiltIn::BlockDim, workgroup_size);
        }

        // logic to convert the builtins in `ir_builtins` to be words in the `self.builtin_tys` map
        let mut builtins = vec![];
        for builtin in ir_builtins.clone() {
            {
                let vec3_ptr_word = match self.vec3_ptr_word {
                    Some(word) => word,
                    None => {
                        let vec3_ptr_word =
                            self.spirv_builder
                                .type_pointer(None, StorageClass::Input, vec3_type);

                        self.vec3_ptr_word = Some(vec3_ptr_word);

                        vec3_ptr_word
                    }
                };

                // self.builtins = HashMap::from_iter(zip(ir_builtins.clone(), builtins.clone()));

                {
                    let k = MimirPtrType {
                        base: MimirPrimitiveTy::Uint32.to_mimir_ty(),
                        storage_class: StorageClass::Input,
                    };
                    if !self.ptr_types.contains_key(&k) {
                        let uint_word =
                            self.mimir_ty_to_word(&MimirPrimitiveTy::Uint32.to_mimir_ty())?;

                        let ptr_word =
                            self.spirv_builder
                                .type_pointer(None, StorageClass::Input, uint_word);

                        self.ptr_types.insert(k, ptr_word);
                    }
                }

                let word =
                    self.spirv_builder
                        .variable(vec3_ptr_word, None, StorageClass::Input, None);

                self.spirv_builder.decorate(
                    word,
                    spirv::Decoration::BuiltIn,
                    match builtin {
                        MimirBuiltIn::BlockIdx => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::WorkgroupId)]
                        }
                        MimirBuiltIn::BlockDim => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)]
                        }
                        MimirBuiltIn::ThreadIdx => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::LocalInvocationId)]
                        }
                        MimirBuiltIn::GlobalInvocationId => {
                            vec![Operand::BuiltIn(spirv::BuiltIn::GlobalInvocationId)]
                        }
                    },
                );

                builtins.push(word);
                self.builtin_tys.insert(builtin.clone(), word);
            }
        }
        self.builtins = HashMap::from_iter(zip(ir_builtins.clone(), builtins.clone()));

        self.spirv_builder.source(
            spirv::SourceLanguage::Unknown, // lowk can't do custom language strings for some reason :(
            1,
            None,
            None as Option<String>,
        );

        let mut literals_set = preprocess_info.literals.clone();

        literals_set.insert(MimirLit::Int32(0));
        literals_set.insert(MimirLit::Int32(1));
        literals_set.insert(MimirLit::Int32(2));

        for (i, _) in self.pc_order.iter().enumerate() {
            literals_set.insert(MimirLit::Int32(i as i32));
        }

        let literals = literals_set.into_iter().collect::<Vec<_>>();

        for lit in literals {
            let mimir_ty = lit.to_ty().to_mimir_ty();
            let word_ty = self.mimir_ty_to_word(&mimir_ty)?;

            let word = match lit {
                MimirLit::Bool(val) => {
                    if val {
                        self.spirv_builder.constant_true(word_ty)
                    } else {
                        self.spirv_builder.constant_false(word_ty)
                    }
                }
                _ => self
                    .spirv_builder
                    .constant_bit32(word_ty, lit.to_u32().unwrap()),
            };

            self.literals.insert(lit, word);
        }
        
        // handles both type declerations and pointer type declerations
        self.init_types(kernel)?;

        self.params_to_spirv(kernel)?;

        // we need to setup these not within a function
        self.setup_sh_mem_vars(kernel)?;

        let main = self
            .spirv_builder
            .begin_function(void, None, spirv::FunctionControl::empty(), voidf)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .begin_block(None)
            .map_err(map_err_closure)?;

        self.spirv_builder
            .entry_point(spirv::ExecutionModel::GLCompute, main, "main", &builtins);

        // Add ExecutionMode LocalSizeId using the specialization constants
        self.spirv_builder.execution_mode_id(
            main,
            spirv::ExecutionMode::LocalSizeId,
            vec![const_x, const_y, const_z],
        );

        Ok(())
    }
}
