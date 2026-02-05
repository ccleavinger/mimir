use std::collections::HashSet;

use mimir_ir::{
    ir::{
        self, MathIntrinsicExpr, MimirBinOpExpr, MimirBuiltInField, MimirCastExpr, MimirConstExpr, MimirExpr, MimirIndexExpr, MimirLit, MimirStmt, MimirTy, MimirUnOpExpr
    },
    kernel_ir::MimirKernelIR,
};
use rspirv::spirv;

#[derive(Debug)]
pub struct VkPreprocessInfo {
    pub literals: HashSet<MimirLit>,
    pub contains_block_idx: bool,
    pub contains_block_dim: bool,
    pub contains_thread_idx: bool,
    pub contains_gi_id: bool,
}

pub(crate) fn preprocess_kernel_ir(kernel: &MimirKernelIR) -> VkPreprocessInfo {
    let mut preprocess_info = VkPreprocessInfo {
        literals: HashSet::new(),
        contains_block_idx: false,
        contains_block_dim: false,
        contains_thread_idx: false,
        contains_gi_id: false,
    };
    for ir in &kernel.body {
        handle_ir_recursive(ir, &mut preprocess_info);
    }
    for var in kernel.var_map.values() {
        if let MimirTy::SharedMemArray { length  } = &var.ty {
            if let MimirConstExpr::Literal(lit) = length.as_ref() {
                preprocess_info.literals.insert(lit.clone());
            }
        }
    }

    let mut var_vec = kernel.param_order.iter()
        .map(|id| kernel.var_map.get(id).unwrap())
        .collect::<Vec<_>>();
    var_vec
        .retain(|var| var.ty.is_shared_mem());

    for (i, _) in kernel.param_order.iter().enumerate() {
        preprocess_info.literals.insert(MimirLit::Int32((i) as i32));
    }
    
    preprocess_info
}

fn handle_ir_recursive(ir: &MimirStmt, preprocess_info: &mut VkPreprocessInfo) {
    match ir {
        MimirStmt::Assign { lhs, rhs } => {
            handle_expr_ir_recursive(&lhs.to_norm_expr(), preprocess_info);
            handle_expr_ir_recursive(rhs, preprocess_info);
        }
        MimirStmt::RangeFor {
            start,
            end,
            step,
            body,
            ..
        } => {
            handle_expr_ir_recursive(start, preprocess_info);
            handle_expr_ir_recursive(end, preprocess_info);

            if let Some(expr_ir) = step {
                handle_expr_ir_recursive(expr_ir, preprocess_info);
            }

            for loop_ir in body {
                handle_ir_recursive(loop_ir, preprocess_info);
            }
        }
        MimirStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            handle_expr_ir_recursive(condition, preprocess_info);

            for then_ir in then_branch {
                handle_ir_recursive(then_ir, preprocess_info);
            }

            if let Some(else_branch_) = else_branch {
                for branch_ir in else_branch_ {
                    handle_ir_recursive(branch_ir, preprocess_info);
                }
            }
        }
        MimirStmt::Syncthreads => {
            preprocess_info
                .literals
                .insert(MimirLit::Int32(spirv::Scope::Workgroup as u32 as i32));
            preprocess_info
                .literals
                .insert(MimirLit::Int32(spirv::Scope::Device as u32 as i32));
            preprocess_info.literals.insert(MimirLit::Int32(
                (spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY)
                    .bits() as i32,
            ));
        }
        MimirStmt::Return(_mimir_expr_ir) => {}
    }
}

fn handle_expr_ir_recursive(expr_ir: &MimirExpr, preprocess_info: &mut VkPreprocessInfo) {
    match expr_ir {
        MimirExpr::BuiltinFieldAccess { built_in, field } => {
            match built_in {
                ir::MimirBuiltIn::BlockIdx => preprocess_info.contains_block_idx = true,
                ir::MimirBuiltIn::BlockDim => preprocess_info.contains_block_dim = true,
                ir::MimirBuiltIn::ThreadIdx => preprocess_info.contains_thread_idx = true,
                ir::MimirBuiltIn::GlobalInvocationId => preprocess_info.contains_gi_id = true,
            }
            let lit = MimirLit::Int32(match field {
                MimirBuiltInField::X => 0,
                MimirBuiltInField::Y => 1,
                MimirBuiltInField::Z => 2,
            });
            preprocess_info.literals.insert(lit);
        }
        MimirExpr::BinOp(MimirBinOpExpr { lhs, rhs, .. }) => {
            handle_expr_ir_recursive(lhs, preprocess_info);
            handle_expr_ir_recursive(rhs, preprocess_info);
        }
        MimirExpr::Index(MimirIndexExpr { index, .. }) => {
            handle_expr_ir_recursive(index, preprocess_info);
        }
        MimirExpr::Unary(MimirUnOpExpr { expr, .. }) => {
            handle_expr_ir_recursive(expr, preprocess_info);
        }
        MimirExpr::Literal(lit) => {
            preprocess_info.literals.insert(lit.clone());
        }
        MimirExpr::MathIntrinsic(MathIntrinsicExpr { args, .. }) => {
            for arg in args {
                handle_expr_ir_recursive(arg, preprocess_info);
            }
        }
        MimirExpr::Var(_) => {}
        MimirExpr::Cast(MimirCastExpr { from, .. }) => {
            handle_expr_ir_recursive(from, preprocess_info);
        }
        MimirExpr::ConstExpr(mimir_const_expr) => match mimir_const_expr {
            ir::MimirConstExpr::BinOp(mimir_bin_op_expr) => {
                handle_expr_ir_recursive(&mimir_bin_op_expr.lhs.to_norm_expr(), preprocess_info);
                handle_expr_ir_recursive(&mimir_bin_op_expr.rhs.to_norm_expr(), preprocess_info);
            }
            ir::MimirConstExpr::Unary(mimir_un_op_expr) => {
                handle_expr_ir_recursive(&mimir_un_op_expr.expr.to_norm_expr(), preprocess_info);
            }
            ir::MimirConstExpr::Literal(mimir_lit) => {
                preprocess_info.literals.insert(mimir_lit.clone());
            }
            ir::MimirConstExpr::MathIntrinsic(math_intrinsic_expr) => {
                for c_expr in &math_intrinsic_expr.args {
                    handle_expr_ir_recursive(&c_expr.to_norm_expr(), preprocess_info);
                }
            }
            ir::MimirConstExpr::Cast(mimir_cast_expr) => {
                handle_expr_ir_recursive(&mimir_cast_expr.from.to_norm_expr(), preprocess_info);
            }
            ir::MimirConstExpr::ConstGeneric { .. } => {}
        },
    }
}
