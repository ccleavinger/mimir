use crate::{Buffer, Device, ExecutionError, ExecutionResult, GpuBuffer, GpuScalar, LaunchError};
use std::marker::PhantomData;

/// A typed, shared GPU allocation. Cloning aliases the same allocation; it does
/// not copy its contents. Readback uses the buffer's owning device, independent
/// of the process-wide default context. Boolean elements use four-byte IR slots.
///
/// ```compile_fail
/// use mimir_runtime::GpuArray;
/// let array = GpuArray::<f32>::from_slice(&[1u32, 2]);
/// ```
#[derive(Clone)]
pub struct GpuArray<T: GpuScalar> {
    buffer: Buffer,
    element: PhantomData<T>,
}

impl<T: GpuScalar> GpuArray<T> {
    /// Allocate on the lazy global runtime's default device.
    pub fn from_slice(values: &[T]) -> Result<Self, LaunchError> {
        Ok(Self::from_slice_on(&crate::default_device()?, values)?)
    }

    /// Fallible allocation from host values on the default device.
    #[allow(
        clippy::should_implement_trait,
        reason = "GPU allocation is fallible; FromIterator cannot report device errors"
    )]
    pub fn from_iter(values: impl IntoIterator<Item = T>) -> Result<Self, LaunchError> {
        Ok(Self::from_iter_on(&crate::default_device()?, values)?)
    }

    /// Allocate on an explicitly selected device.
    pub fn from_slice_on(device: &Device, values: &[T]) -> ExecutionResult<Self> {
        Self::try_from(device.buffer_from_slice(values)?)
    }

    pub fn from_iter_on(
        device: &Device,
        values: impl IntoIterator<Item = T>,
    ) -> ExecutionResult<Self> {
        let words = values.into_iter().map(T::to_word).collect::<Vec<_>>();
        Self::try_from(device.buffer(T::TYPE, &words)?)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn device(&self) -> Device {
        self.buffer.device()
    }

    pub fn as_buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn into_buffer(self) -> Buffer {
        self.buffer
    }

    /// Synchronously copy the contents to a host vector.
    #[allow(unreachable_code)]
    pub fn to_vec(&self) -> ExecutionResult<Vec<T>> {
        self.device().read_typed(&self.buffer)
    }

    /// Synchronous readback followed by iteration over the host snapshot.
    pub fn to_iter(&self) -> ExecutionResult<std::vec::IntoIter<T>> {
        Ok(self.to_vec()?.into_iter())
    }
}
impl<T: GpuScalar> TryFrom<Buffer> for GpuArray<T> {
    type Error = ExecutionError;

    fn try_from(buffer: Buffer) -> ExecutionResult<Self> {
        if buffer.element_type() != T::TYPE {
            return Err(ExecutionError(
                "array element type does not match buffer".into(),
            ));
        }
        Ok(Self {
            buffer,
            element: PhantomData,
        })
    }
}

impl<T: GpuScalar> From<GpuArray<T>> for Buffer {
    fn from(array: GpuArray<T>) -> Self {
        array.into_buffer()
    }
}

impl<T: GpuScalar> AsRef<Buffer> for GpuArray<T> {
    fn as_ref(&self) -> &Buffer {
        self.as_buffer()
    }
}

impl<T: GpuScalar> GpuBuffer for GpuArray<T> {
    fn as_buffer(&self) -> &Buffer {
        &self.buffer
    }
}
