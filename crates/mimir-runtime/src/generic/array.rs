use crate::{generic::device::MimirDevice, platforms::MimirPlatform};
use std::{any::Any, sync::Arc};
use thiserror::Error;

pub trait MimirArray<T: Copy + Sized + 'static> {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn size_bytes(&self) -> usize {
        std::mem::size_of::<T>() * self.len()
    }

    // Instead of having the MimirArray create itself the associated platform factory will instead create each array
    // fn from_iter_device<I>(mimir_device: Arc<dyn MimirDevice>, iter: I) -> Self
    // where
    //     I: IntoIterator<Item = T>,
    //     Self: Sized;

    // host -> device
    // fn from_iter<I>(iter: I) -> Self
    // where
    //     I: IntoIterator<Item = T>,
    //     Self: Sized;

    // lowk don't use this unless you're awesome sauce
    #[doc(hidden)]
    fn get_inner_as_any(&self) -> &dyn Any;

    // device -> host
    fn to_iter(&self) -> Box<dyn Iterator<Item = T>>;

    fn get_device(&self) -> Arc<dyn MimirDevice>;

    fn platform(&self) -> MimirPlatform {
        self.get_device().platform()
    }
}

pub trait MimirArrayFactory {
    fn create_from_iter_device<T: Copy + Send + Sync + Sized + 'static, I>(
        &self,
        mimir_device: Arc<&(dyn MimirDevice + Send + Sync)>,
        iter: I,
    ) -> Result<Arc<dyn MimirArray<T> + Send + Sync>, MimirArrayError>
    where
        I: IntoIterator<Item = T>;

    fn platform(&self) -> MimirPlatform;
}

#[derive(Error, Debug)]
pub enum MimirArrayError {
    #[error("Mimir GPU array error: {0}")]
    Generic(String),

    #[error(
        "Unsized Mimir GPU array error: A Mimir GPU array cannot be an empty buffer at initialization"
    )]
    UnsizedError,

    #[error("`{0:?}` is an invalid platform for initializing a Mimir array")]
    InvalidPlatformError(MimirPlatform),
}
