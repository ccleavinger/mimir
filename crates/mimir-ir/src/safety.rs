// // type checking, conformance to standards, etc.
// // will be run during IO

// use std::collections::HashMap;

// use crate::{error::ASTError, ir::{IRFunctionality, MimirExprIR, MimirIR, MimirTy}, util::get_output_ty};

// pub fn validate_ir(
//     ir: Box<dyn IRFunctionality>
// ) -> Result<(), ASTError> {

//     let map = ir.var_map();

//     let is_kernel_ir = ir.as_any().downcast_ref::<crate::kernel_ir::MimirKernelIR>().is_some();

//     for mimir_ir in ir.get_body() {
//         validate_stmt_ir(&map, &mimir_ir, is_kernel_ir)?;
//     }

//     Ok(())
// }

// fn validate_stmt_ir(
//     map: &HashMap<u32, (String, MimirTy)>,
//     mimir_ir: &MimirIR,
//     is_kernel_ir: bool
// ) -> Result<(), ASTError> {
//     match mimir_ir {
//         crate::ir::MimirIR::Assign { lhs, rhs } => {
//             let lhs_ty = get_ty_from_expr(map, lhs)?;
//             let rhs_ty = get_ty_from_expr(map, rhs)?;

//             if lhs_ty != rhs_ty {
//                 return Err(ASTError::Validation(
//                     format!(
//                         "Trying to set a {:?} to be a {:?} is not valid.",
//                         lhs_ty,
//                         rhs_ty
//                     )
//                 ))
//             }
//         },
//         crate::ir::MimirIR::RangeFor { var, start, end, step, body } => {
//             let (_, var_ty) = map.get(var).ok_or(
//                 ASTError::Validation(
//                     format!(
//                         "Could not retrieve variable at index {}, ir is invalid",
//                         var
//                     )
//                 )
//             )?;

//             // type checks for var, start, step, & end
//             {
//                 if *var_ty != MimirTy::Int32 {
//                     return Err(ASTError::Validation(
//                         "Iterable variable must be an Int32, ir is invalid".to_string()
//                     ));
//                 }

//                 let start_ty = get_ty_from_expr(map, start)?;

//                 if start_ty != MimirTy::Int32 {
//                     return Err(ASTError::Validation(
//                         format!(
//                             "Iterable must be initialized to be an Int32, not a {:?}, ir is invalid",
//                             start_ty
//                         )
//                     ));
//                 }

//                 let end_ty = get_ty_from_expr(map, end)?;

//                 if end_ty != MimirTy::Int32 {
//                     return Err(ASTError::Validation(
//                         format!(
//                             "End of iterable must be an Int32, not a {:?}, ir is invalid",
//                             end_ty
//                         )
//                     ));
//                 }

//                 if let Some(expr) = step {
//                     let ty = get_ty_from_expr(map, expr)?;

//                     if ty != MimirTy::Int32 {
//                         return Err(ASTError::Validation(
//                             format!(
//                                 "Step value of a for loop must be an Int32, not a {:?}, ir is invalid",
//                                 ty
//                             )
//                         ))
//                     }
//                 }
//             }

//             // check body
//             for loop_ir in body {
//                 validate_stmt_ir(map, loop_ir, is_kernel_ir)?;
//             }
//         },
//         crate::ir::MimirIR::If { condition, then_branch, else_branch } => {
//             let cond_ty = get_ty_from_expr(map, condition)?;

//             if !cond_ty.is_bool() {
//                 return Err(ASTError::Validation(
//                     "Conditions in if statements must create a boolean, ir is invalid".to_string()
//                 ));
//             }

//             for branch_ir in then_branch {
//                 validate_stmt_ir(map, branch_ir, is_kernel_ir)?;
//             }

//             if let Some(else_) = else_branch {
//                 for branch_ir in else_ {
//                     validate_stmt_ir(map, branch_ir, is_kernel_ir)?;
//                 }
//             }
//         },
//         crate::ir::MimirIR::Return(mimir_expr_ir) => {
//             if is_kernel_ir && mimir_expr_ir.is_some() {
//                 return Err(ASTError::Validation(
//                     "A kernel cannot return a value".to_string()
//                 ))
//             }
//         },
//         crate::ir::MimirIR::Syncthreads => {
//             // I lowk don't know what to do with this to ensure conformance. I think it's fine.
//         },
//     }

//     Ok(())
// }

// fn get_ty_from_expr(
//     map: &HashMap<u32, (String, MimirTy)>,
//     expr: &MimirExprIR
// ) -> Result<MimirTy, ASTError> {
//     match expr {
//         MimirExprIR::BinOp { lhs, op, rhs, .. } => {
//             // init lhs and rhs types to bool, they'll be updated as we go
//             let mut lhs_ty = MimirTy::Bool;
//             let mut rhs_ty = MimirTy::Bool;

//             // if both sides of the bin op are bin ops then we need to perform PEMDAS
//             if let MimirExprIR::BinOp { is_parenthesized: lhs_is_parenthesized, .. } = **lhs {
//                 if let MimirExprIR::BinOp { is_parenthesized: rhs_is_parenthesized, .. } = **rhs {
//                     if !lhs_is_parenthesized && rhs_is_parenthesized {
//                         // if the left side is not parenthesized and the right side is, we need to check the right side first
//                         rhs_ty = get_ty_from_expr(map, rhs)?;
//                         lhs_ty = get_ty_from_expr(map, lhs)?;
//                     } else {
//                         // otherwise we check the left side first
//                         lhs_ty = get_ty_from_expr(map, lhs)?;
//                         rhs_ty = get_ty_from_expr(map, rhs)?;
//                     }
//                 }
//             } else {
//                 // if either of them are not both bin ops, we just get their types
//                 lhs_ty = get_ty_from_expr(map, lhs)?;
//                 rhs_ty = get_ty_from_expr(map, rhs)?;
//             }

//             if lhs_ty != rhs_ty {
//                 Err(ASTError::Validation(
//                     format!(
//                         "Binary operation between {:?} and {:?} is not valid.",
//                         lhs_ty, rhs_ty
//                     )
//                 ))
//             } else if op.is_comparison() {
//                 // comparison ops always return a bool
//                 Ok(MimirTy::Bool)
//             } else {
//                 // otherwise we return the type of the lhs/rhs
//                 Ok(lhs_ty)
//             }
//         },
//         MimirExprIR::Index { var, index } => {
//             let (_, var_ty) = map.get(var).ok_or(
//                 ASTError::Validation(
//                     format!(
//                         "Could not retrieve variable at index {}, ir is invalid",
//                         var
//                     )
//                 )
//             )?;

//             let index_ty = get_ty_from_expr(map, index)?;

//             // Vulkan is cool with any signed integer type (i32, i64) for indexing, and casting from a u32 to a i32 shouldn't be problematic
//             // TODO: make sure this is consistent with other backends (i.e. CUDA, HIP, etc.)
//             if index_ty.is_numeric() && index_ty.is_floating_point() {
//                 return Err(ASTError::Validation(
//                     format!(
//                         "Indexing into an array must be an Int32, not a {:?}, ir is invalid",
//                         index_ty
//                     )
//                 ));
//             }

//             if let MimirTy::GlobalArray { element_type } = var_ty {
//                 Ok(*element_type.clone())
//             } else {
//                 Err(ASTError::Validation(
//                     format!(
//                         "Variable at index {} is not a global array, ir is invalid",
//                         var
//                     )
//                 ))
//             }
//         },
//         // MimirExprIR::Field { var, field } => {
//         //     // this'd require support for custom structs, which is not yet implemented
//         //     //

//         //     Err(ASTError::Validation(
//         //         format!(
//         //             "Field access is not yet implemented, ir is invalid. Tried to access field '{}' of variable at index {}",
//         //             field, var
//         //         )
//         //     ))
//         // },
//         MimirExprIR::Unary { un_op, expr } => {
//             let expr_ty = get_ty_from_expr(map, expr)?;

//             match un_op {
//                 crate::ir::MimirUnOp::Neg => {
//                     if !expr_ty.is_numeric() {
//                         return Err(ASTError::Validation(
//                             format!(
//                                 "Negation operator cannot be applied to a {:?}, ir is invalid",
//                                 expr_ty
//                             )
//                         ));
//                     }
//                     Ok(expr_ty)
//                 },
//                 crate::ir::MimirUnOp::Not => {
//                     if !expr_ty.is_bool() {
//                         return Err(ASTError::Validation(
//                             format!(
//                                 "Not operator can only be applied to a boolean, not a {:?}, ir is invalid",
//                                 expr_ty
//                             )
//                         ));
//                     }
//                     Ok(MimirTy::Bool)
//                 },
//             }
//         },
//         MimirExprIR::Literal(mimir_lit) => {
//             Ok(match mimir_lit {
//                 crate::ir::MimirLit::Int32(_) => MimirTy::Int32,
//                 crate::ir::MimirLit::Int64(_) => MimirTy::Int64,
//                 crate::ir::MimirLit::Float32(_) => MimirTy::Float32,
//                 crate::ir::MimirLit::Float64(_) => MimirTy::Float64,
//                 crate::ir::MimirLit::Bool(_) => MimirTy::Bool,
//                 crate::ir::MimirLit::Uint32(_) => MimirTy::Uint32,
//             })
//         },
//         MimirExprIR::Var(idx) => {
//             Ok(
//                 map.get(idx)
//                     .ok_or(ASTError::Validation(
//                         format!("Failed to retrieve variable at index {}, ir is invalid", idx)
//                     ))?.1.clone()
//             )
//         },
//         MimirExprIR::MathIntrinsic { func, args } => {
//             let tys = args
//                 .iter().map(|expr_ir| get_ty_from_expr(map, expr_ir)).collect::<Result<Vec<_>, _>>()?;

//             get_output_ty(func, &tys)
//         },
//         MimirExprIR::Cast { from, to } => {
//             let from_ty = get_ty_from_expr(map, from)?;

//             validate_casting(&from_ty, to)
//         },
//     }
// }

// fn validate_casting(
//     from: &MimirTy,
//     to: &MimirTy
// ) -> Result<MimirTy, ASTError> {
//     if from.is_global_array() {
//         Err(ASTError::Validation(format!("Cannot cast a GlobalArray to a {:?}", to)))
//     } else if to.is_global_array() {
//         Err(ASTError::Validation(format!("Cannot cast a {:?} to a GlobalArray", from)))
//     } else if from.is_bool() && to.is_numeric() {
//         Err(ASTError::Validation("Casting from a boolean to a numerical value is not permitted".to_string()))
//     } else if from.is_numeric() && to.is_bool() {
//         Err(ASTError::Validation("Casting from a numeric to a boolean is not permitted".to_string()))
//     } else {
//         Ok(to.clone())
//     }
// }
