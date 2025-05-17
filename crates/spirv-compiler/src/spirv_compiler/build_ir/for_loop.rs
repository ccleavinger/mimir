use anyhow::{anyhow, Result};

use crate::spirv_compiler::compiler::SpirVCompiler;
use crate::spirv_compiler::ir::{MimirExprIR, MimirLit, MimirType};

impl SpirVCompiler {
    pub fn for_loop(
        &mut self,
        iter_var_name: String,
        lhs: &MimirExprIR,
        rhs: &MimirExprIR,
        mimir_expr_irs: Vec<MimirExprIR>,
        step: Option<Box<MimirExprIR>>,
    ) -> Result<()> {

        let int_ty = *self.types.get(&MimirType::Int32).unwrap();
        let bool_ty = *self.types.get(&MimirType::Bool).unwrap();

        // Get all the variable information we need upfront to avoid borrow conflicts
        let iter_var = self.vars.get(&iter_var_name).ok_or(anyhow!("Failed to find variable {} in the variable table", iter_var_name))?;
        let iter_var_word = iter_var.word.ok_or(anyhow!("The variable {} does not have a word assigned to it", iter_var_name))?;
        let iter_var_type = iter_var.ty.base.clone();
        
        // Process LHS and RHS with full ownership of self
        let lhs_stuff = self.ir_to_word(lhs)?;
        let rhs_stuff = self.ir_to_word(rhs)?;
        
        // Get the types of LHS
        let (lhs_ty, _) = self.parse_hand_side(lhs)?;
        
        // Create blocks for structured loop
        let header_block = self.spirv_builder.id();
        let body_block = self.spirv_builder.id();
        let continue_block = self.spirv_builder.id();
        let merge_block = self.spirv_builder.id();
        
        // Cast the initial value to the iterator variable type if needed
        let initial_value = if lhs_ty != iter_var_type {
            self.cast_word(lhs_stuff, &lhs_ty, &iter_var_type)?
        } else {
            lhs_stuff
        };
        
        // Initialize counter variable with proper type casting
        self.spirv_builder
            .store(
                iter_var_word,
                initial_value,
                None,
                vec![],
            )?;

        // Now update the variable after all other operations
        if let Some(var) = self.vars.get_mut(&iter_var_name) {
            var.word = Some(iter_var_word);
        }
        
        // Branch directly to header block which contains the condition
        self.spirv_builder.branch(header_block)?;
        
        // Loop header block with condition check integrated
        self.spirv_builder.begin_block(Some(header_block))?;
        
        // Condition check directly in header block - more efficient structure
        let i_val = self.spirv_builder.load(int_ty, None, iter_var_word, None, vec![])?;
        let cmp = self.spirv_builder.s_less_than(bool_ty, None, i_val, rhs_stuff)?;
        
        // Add loop merge instruction IMMEDIATELY before the branch - this is critical for SPIR-V validation
        self.spirv_builder.loop_merge(merge_block, continue_block, rspirv::spirv::LoopControl::NONE, vec![])?;
        
        // Branch based on condition - direct to body or merge
        self.spirv_builder.branch_conditional(cmp, body_block, merge_block, vec![])?;
        
        // Loop body
        self.spirv_builder.begin_block(Some(body_block))?;
        for expr in &mimir_expr_irs {
            self.build_ir(expr.clone())?;
        }
        // Branch directly to continue block
        self.spirv_builder.branch(continue_block)?;
        
        // Continue block (increment counter)
        self.spirv_builder.begin_block(Some(continue_block))?;
        // Load value only once per operation
        let i_val = self.spirv_builder.load(int_ty, None, iter_var_word, None, vec![])?;
        let step_word = if let Some(step_expr) = step {
            self.ir_to_word(&step_expr)?
        } else {
            *self.literals.get(&MimirLit::Int32(1)).ok_or(anyhow!("Failed to retrieve literal of 1"))?
        };

        // Increment counter by step
        let i_plus_step = self.spirv_builder.i_add(int_ty, None, i_val, step_word)?;
        self.spirv_builder.store(iter_var_word, i_plus_step, None, vec![])?;
        // Branch back to header/condition
        self.spirv_builder.branch(header_block)?;
        
        // Merge block (loop exit) - no changes needed
        self.spirv_builder.begin_block(Some(merge_block))?;
        
        Ok(())
    }
}
