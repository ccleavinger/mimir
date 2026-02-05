use crate::MIMIR_OMNI_RUNTIME;
use mimir_runtime::generic::launch::{MimirLaunch, MimirLaunchError, Param};

pub fn launch_kernel_name(
    name: String, // name of the kernel
    grid_dim: &[u32; 3],
    block_dim: &[u32; 3],
    params: &[Param],
    cgs: &[u32],
) -> Result<(), MimirLaunchError> {
    let ir = &MIMIR_OMNI_RUNTIME.ir_data;
    MIMIR_OMNI_RUNTIME.launch_kernel(ir, name, block_dim, grid_dim, params, cgs)
}
