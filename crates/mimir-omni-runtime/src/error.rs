use mimir_runtime::generic::{array::MimirArrayError, launch::LaunchDeviceError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CreateMimirArrayError {
    #[error("Mimir array error: {0}")]
    ArrayError(#[from] MimirArrayError),

    #[error("Launch device error: {0}")]
    LaunchDevError(#[from] LaunchDeviceError),
}
