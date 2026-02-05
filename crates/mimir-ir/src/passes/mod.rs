use std::collections::{BTreeMap, HashSet};

use crate::{
    ir::{MimirStmt, MimirTy, MimirTyVar},
    kernel_ir::MimirKernelIR,
    passes::{
        optimize::{inline::InlineExprPass, to_const::pass::ConstExprPass},
        safety::kernel_spec_stmt_pass,
    },
    util::error::{MimirTypePassError, MultiPassError},
};

pub(crate) mod optimize;
pub(crate) mod safety;

pub(crate) trait KernelIRPrePass {
    fn stmt_pass(stmt: &MimirStmt) -> Result<Vec<MimirStmt>, MultiPassError>;
}

pub struct KernelIRPrePassSettings {
    pub const_expr_pass: bool,
}

// pass that is intended to be used after compilation to guarantee spec compliance and even provide basic optimizations
pub fn kernel_ir_pre_pass(
    ir: &MimirKernelIR,
    settings: KernelIRPrePassSettings,
) -> Result<MimirKernelIR, MultiPassError> {
    let mut ret_ir = ir.clone();

    // spec check enforcement
    if !ir.safety_sig {
        let mut init_var_list = HashSet::new();
        for stmt in &ir.body {
            kernel_spec_stmt_pass(ir, stmt, &mut init_var_list)
                .map_err(MultiPassError::SpecEnforce)?
        }
    }
    ret_ir.safety_sig = true;

    if settings.const_expr_pass {
        run_kernel_ir_pre_pass::<ConstExprPass>(&mut ret_ir.body)?;
    }

    Ok(ret_ir.clone())
}

fn run_kernel_ir_pre_pass<P>(body: &mut Vec<MimirStmt>) -> Result<(), MultiPassError>
where
    P: KernelIRPrePass,
{
    let mut new_body = vec![];
    for stmt in body {
        new_body.append(&mut P::stmt_pass(stmt)?);
    }
    Ok(())
}

pub(crate) trait KernelIRJitPass {
    fn stmt_pass(
        stmt: &MimirStmt,
        const_generics: &[u32],
    ) -> Result<Vec<MimirStmt>, MultiPassError>;

    // optional pass
    fn ty_pass(_ty: &MimirTy, _const_generics: &[u32]) -> Result<MimirTy, MultiPassError> {
        Err(MultiPassError::Type(MimirTypePassError::Unimplemented))
    }

    fn ty_pass_support() -> bool {
        false
    }
}

pub struct KernelIRPassSettings {
    pub const_generics: Vec<u32>,
    pub inline_pass: bool,
}

pub fn kernel_ir_pass(
    ir: &MimirKernelIR,
    settings: KernelIRPassSettings,
) -> Result<MimirKernelIR, MultiPassError> {
    let mut ret_ir = ir.clone();

    // spec check enforcement
    if !ir.safety_sig {
        let mut init_var_list = HashSet::new();
        for stmt in &ir.body {
            kernel_spec_stmt_pass(ir, stmt, &mut init_var_list)
                .map_err(MultiPassError::SpecEnforce)?;
        }
    }

    if settings.inline_pass {
        ret_ir.body = run_kernel_ir_pass::<InlineExprPass>(&ret_ir.body, &settings.const_generics)?;

        ret_ir.var_map =
            run_type_ir_pass::<InlineExprPass>(&ret_ir.var_map, &settings.const_generics)?;
    }

    Ok(ret_ir)
}

fn run_kernel_ir_pass<P>(
    body: &[MimirStmt],
    const_generics: &[u32],
) -> Result<Vec<MimirStmt>, MultiPassError>
where
    P: KernelIRJitPass,
{
    let mut new_body = vec![];
    for stmt in body {
        new_body.append(&mut P::stmt_pass(stmt, const_generics)?);
    }
    Ok(new_body)
}

fn run_type_ir_pass<P>(
    var_map: &BTreeMap<u64, MimirTyVar>,
    const_generics: &[u32],
) -> Result<BTreeMap<u64, MimirTyVar>, MultiPassError>
where
    P: KernelIRJitPass,
{
    assert!(P::ty_pass_support());
    let mut ret_map = BTreeMap::new();
    for (k, v) in var_map {
        let mut var = v.clone();
        var.ty = P::ty_pass(&var.ty, const_generics)?;

        ret_map.insert(*k, var);
    }
    Ok(ret_map)
}
