use crate::{MIMIR_OMNI_RUNTIME, MimirGPUArray, error::CreateMimirArrayError};

pub fn from_iter<T: Copy + Send + Sync + Sized + 'static, I>(
    iter: I,
) -> Result<MimirGPUArray<T>, CreateMimirArrayError>
where
    I: IntoIterator<Item = T>,
{
    MIMIR_OMNI_RUNTIME.create_mimir_array(iter)
}
