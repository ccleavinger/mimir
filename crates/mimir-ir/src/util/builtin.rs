use crate::{
    ir::{MimirBuiltIn, MimirBuiltInField},
    util::error::ASTError,
};

#[inline]
pub fn str_to_builtin(string: &str) -> Result<MimirBuiltIn, ASTError> {
    match string {
        "block_idx" | "blockIdx" => Ok(MimirBuiltIn::BlockIdx),
        "block_dim" | "blockDim" => Ok(MimirBuiltIn::BlockDim),
        "thread_idx" | "threadIdx" => Ok(MimirBuiltIn::ThreadIdx),
        "global_invocation_id" | "globalInvocationId" => Ok(MimirBuiltIn::GlobalInvocationId),
        _ => Err(ASTError::Validation(format!(
            "`{string}` is an uknown built in"
        ))),
    }
}

#[inline]
pub fn str_to_builtin_field(string: &str) -> Result<MimirBuiltInField, ASTError> {
    match string {
        "x" => Ok(MimirBuiltInField::X),
        "y" => Ok(MimirBuiltInField::Y),
        "z" => Ok(MimirBuiltInField::Z),
        _ => Err(ASTError::Validation(format!(
            "`{string}` is an invalid field for builtin variables"
        ))),
    }
}
