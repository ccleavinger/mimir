use crate::spirv_compiler::compiler::SpirVCompiler;
use crate::spirv_compiler::ir::{MimirExprIR, MimirLit, MimirPtrType, MimirType};
use rspirv::spirv::StorageClass;

impl SpirVCompiler {
    pub fn for_loop(
        &mut self,
        iter_var_name: String,
        lhs: i32,
        rhs: i32,
        mimir_expr_irs: Vec<MimirExprIR>,
    ) -> Result<(), String> {
        let mut var = self
            .vars
            .get_mut(&iter_var_name)
            .ok_or(format!("Variable {} not found", iter_var_name))?;

        if var.word.is_some() {
            return Err(format!("Variable {} already exists", iter_var_name));
        }

        let int_ptr_ty = *self
            .ptr_types
            .get(&MimirPtrType {
                base: MimirType::Int32,
                storage_class: StorageClass::Function,
            })
            .unwrap();

        let int_ty = *self.types.get(&MimirType::Int32).unwrap();
        let bool_ty = *self.types.get(&MimirType::Bool).unwrap();

        let iter_var_word =
            self.spirv_builder
                .variable(int_ptr_ty, None, StorageClass::Function, None);

        self.spirv_builder
            .store(
                iter_var_word,
                self.literals.get(&MimirLit::Int32(lhs)).unwrap().clone(),
                None,
                vec![],
            )
            .map_err(|e| e.to_string())?;

        var.word = Some(iter_var_word);

        // the following is AI-generated from Claude.ai, TODO: research & properly implement
        // loop header
        let loop_header = self
            .spirv_builder
            .begin_block(None)
            .map_err(|e| e.to_string())?;
        self.spirv_builder
            .branch(loop_header)
            .map_err(|e| e.to_string())?;

        let i_val = self
            .spirv_builder
            .load(int_ty, None, iter_var_word, None, vec![])
            .map_err(|e| e.to_string())?;
        let cmp = self
            .spirv_builder
            .s_less_than(
                bool_ty,
                None,
                i_val,
                *self
                    .literals
                    .get(&MimirLit::Int32(rhs))
                    .ok_or("failed to retrieve literal of ")?,
            )
            .map_err(|e| e.to_string())?;

        let loop_body = self
            .spirv_builder
            .begin_block(None)
            .map_err(|e| e.to_string())?;
        let loop_merge = self
            .spirv_builder
            .begin_block(None)
            .map_err(|e| e.to_string())?;
        self.spirv_builder
            .branch_conditional(cmp, loop_body, loop_merge, vec![])
            .map_err(|e| e.to_string())?;

        for ir in mimir_expr_irs {
            self.build_ir(ir)?;
        }

        // increment counter and continue loop
        let loop_continue = self
            .spirv_builder
            .begin_block(None)
            .map_err(|e| e.to_string())?;
        self.spirv_builder
            .branch(loop_continue)
            .map_err(|e| e.to_string())?;

        // continue block - increment counter
        let i_val = self
            .spirv_builder
            .load(int_ty, None, iter_var_word, None, vec![])
            .map_err(|e| e.to_string())?;
        let i_plus_1 = self
            .spirv_builder
            .i_add(
                int_ty,
                None,
                i_val,
                *self
                    .literals
                    .get(&MimirLit::Int32(1))
                    .ok_or("failed to retrieve literal of 1 for loop increment")?,
            )
            .map_err(|e| e.to_string())?;
        self.spirv_builder
            .store(iter_var_word, i_plus_1, None, vec![])
            .map_err(|e| e.to_string())?;
        self.spirv_builder
            .branch(loop_header)
            .map_err(|e| e.to_string())?;

        Err("Unimplemented SPIR-V builder operation in `For` loop".to_string())
    }
}
