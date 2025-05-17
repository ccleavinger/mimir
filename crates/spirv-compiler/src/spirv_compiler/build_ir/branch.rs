// if & if/else

use crate::spirv_compiler::compiler::SpirVCompiler;
use crate::spirv_compiler::ir::{MimirBinOp, MimirExprIR, MimirLit, MimirType, MimirUnOp, MimirVariable};
use anyhow::{anyhow, Result};
use rspirv::spirv::SelectionControl;

impl SpirVCompiler {
    pub fn build_branch(
        &mut self,
        cond: &MimirExprIR,
        then: &[MimirExprIR],
        opt_else: &Option<Vec<MimirExprIR>>
    ) -> Result<()> {
        // don't even generate branching SPIR-V for constants, no need to
        if let MimirExprIR::Literal(MimirLit::Bool(var)) = *cond {
            if var {
                for expr in then {
                    self.build_ir(expr.clone())?;
                }
            } else if let Some(e) = opt_else {
                for expr in e {
                    self.build_ir(expr.clone())?;
                }
            }
            return Ok(())
        };

        if let MimirExprIR::BinOp(lhs, op, rhs, _) = cond {
            self.branch_binop_case(then, opt_else, lhs, op, rhs)
        } else if let MimirExprIR::Unary(un_op, expr) = cond {
            if *un_op == MimirUnOp::Not {
                if let MimirExprIR::BinOp(ref lhs, ref bin_op, ref rhs, _) = **expr {
                    let negated_bin_op = SpirVCompiler::inverse_bin_op(bin_op)?;

                    return self.branch_binop_case(then, opt_else, lhs, &negated_bin_op, rhs);
                } else {
                    Err(anyhow!("Unexpected expression in Unary Not"))
                }
            } else {
                return Err(anyhow!("Encountered unexpected unary operator: {:?}", un_op));
            }
        } else {
            self.branch_var_case(cond, then, opt_else)
        }
    }

    fn branch_binop_case(
        &mut self,
        then: &[MimirExprIR],
        opt_else: &Option<Vec<MimirExprIR>>,
        lhs: &MimirExprIR,
        op: &MimirBinOp,
        rhs: &MimirExprIR
    ) -> Result<()> {
        let condition = self.build_binop((*lhs).clone(), op, (*rhs).clone())?;

        if MimirType::Bool != condition.0 {
            return Err(anyhow!("Encountered {:?} op instead of a compare op", op))
        }

        if opt_else.is_none() {
            let merge_block = self.spirv_builder.id();
            let continue_block = self.spirv_builder.id();

            self.spirv_builder.selection_merge(merge_block, SelectionControl::NONE)?;

            self.spirv_builder.branch_conditional(
                condition.1,
                continue_block,
                merge_block,
                vec![]
            ).map_err(|e| anyhow!("Failed to build conditional branch: {}", e))?;

            self.spirv_builder.begin_block(Some(continue_block))?;
            for expr in then {
                self.build_ir(expr.clone())?;
            }
            self.spirv_builder.branch(merge_block)?;
            self.spirv_builder.begin_block(Some(merge_block)).map_err(|e| anyhow!("Failed to begin block: {}", e))?;

            Ok(())
        } else {
            let merge_block = self.spirv_builder.id();
            self.spirv_builder.selection_merge(
                merge_block,
                SelectionControl::NONE
            )?;

            let true_block = self.spirv_builder.id();
            let false_block = self.spirv_builder.id();
            self.spirv_builder.branch_conditional(
                condition.1,
                true_block,
                false_block,
                vec![]
            )?;

            // true block (aka then block)
            self.spirv_builder.begin_block(Some(true_block))?;
            for expr in then {
                self.build_ir(expr.clone())?;
            }
            self.spirv_builder.branch(merge_block)?;

            // false block (aka else block)
            self.spirv_builder.begin_block(Some(false_block))?;
            for expr in opt_else.as_ref().unwrap() {
                self.build_ir(expr.clone())?;
            }            self.spirv_builder.branch(merge_block)?;

            // Begin the merge block after both branches are complete
            self.spirv_builder.begin_block(Some(merge_block))?;

            Ok(())
        }
    }


    pub fn branch_var_case(
        &mut self,
        cond: &MimirExprIR,
        then: &[MimirExprIR],
        opt_else: &Option<Vec<MimirExprIR>>,
    ) -> Result<()> {
        let ty = *self.types.get(&MimirType::Bool).ok_or(anyhow!("Failed to retrieve bool type"))?;

        // (&MimirVariable, bool) bool indicates whether the variable has a Not unOp. eg: !a
        let (var, is_negated) = self.var_tuple_insanity(cond)?;

        let var_word = var.word.ok_or(anyhow!("Variable {:?} is uninitialized", var))?; // Extract var.word here

        let word = self.spirv_builder.load(
            ty,
            None,
            var_word, // Use the extracted var_word here
            None,
            vec![]
        )?;

        let literal = *self.literals.get(&MimirLit::Bool(!is_negated)).ok_or(anyhow!("Failed to retrieve literal of true"))?;

        let condition = self.spirv_builder.logical_equal(
            ty,
            None,
            word,
            literal
        )?;        // Similar branching logic as above
        if opt_else.is_none() {
            let merge_block = self.spirv_builder.id();
            let continue_block = self.spirv_builder.id();

            self.spirv_builder.selection_merge(merge_block, SelectionControl::NONE)?;

            self.spirv_builder.branch_conditional(
                condition,
                continue_block,
                merge_block,
                vec![]
            )?;

            self.spirv_builder.begin_block(Some(continue_block))?;
            for expr in then {
                self.build_ir(expr.clone())?;
            }
            self.spirv_builder.branch(merge_block)?;
            self.spirv_builder.begin_block(Some(merge_block))?;
        } else {
            let merge_block = self.spirv_builder.id();
            self.spirv_builder.selection_merge(
                merge_block,
                SelectionControl::NONE
            )?;

            let true_block = self.spirv_builder.id();
            let false_block = self.spirv_builder.id();

            self.spirv_builder.branch_conditional(
                condition,
                true_block,
                false_block,
                vec![]
            )?;

            // true block (aka then block)
            self.spirv_builder.begin_block(Some(true_block))?;
            for expr in then {
                self.build_ir(expr.clone())?;
            }
            self.spirv_builder.branch(merge_block)?;

            // false block (aka else block)
            self.spirv_builder.begin_block(Some(false_block))?;
            for expr in opt_else.as_ref().unwrap() {
                self.build_ir(expr.clone())?;
            }
            self.spirv_builder.branch(merge_block)?;
            
            // Begin the merge block after both branches are complete
            self.spirv_builder.begin_block(Some(merge_block))?;
        }
        Ok(())
    }

    fn var_tuple_insanity(&mut self, cond: &MimirExprIR) -> Result<(&MimirVariable, bool)> {
        if let MimirExprIR::Var(name) = cond {
            let var = self.vars.get(name).ok_or(anyhow!("Variable {} not found", name))?;

            if var.word.is_none() {
                return Err(anyhow!("Variable {} is uninitialized", name));
            }

            if var.ty.base == MimirType::Bool {
                Ok((var, false))
            } else {
                Err(anyhow!("Variable {} is not a boolean", name))
            }
        } else {
            Err(anyhow!("Condition must be a variable"))
        }
    }

    #[inline]
    fn inverse_bin_op(bin_op: &MimirBinOp) -> Result<MimirBinOp> {
        match bin_op {
            MimirBinOp::Eq => Ok(MimirBinOp::Ne),
            MimirBinOp::Lt => Ok(MimirBinOp::Gte),
            MimirBinOp::Lte => Ok(MimirBinOp::Gt),
            MimirBinOp::Ne => Ok(MimirBinOp::Eq),
            MimirBinOp::Gt => Ok(MimirBinOp::Lte),
            MimirBinOp::Gte => Ok(MimirBinOp::Lt),
            _ => Err(anyhow!("Unsupported binary operator for inversion"))
        }
    }
}