#![allow(unused_variables, unused_imports)]
use crate::KernelHandle;
use mimir_ir::flat::{Literal, ScalarType};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
#[error("Mimir execution: {0}")]
pub struct ExecutionError(pub String);
pub type ExecutionResult<T> = Result<T, ExecutionError>;

#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
use crate::vulkan::VulkanDevice;
// Compile-time backend variants keep selection in one runtime without loading
// backend plugins or shipping disabled implementations.
#[derive(Clone)]
pub(crate) enum DeviceBackend {
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    Vulkan(Arc<VulkanDevice>),
}
#[derive(Clone)]
pub(crate) enum BufferBackend {
    #[cfg(all(
        feature = "vulkan",
        any(target_os = "linux", target_os = "android", target_os = "windows")
    ))]
    Vulkan(Arc<crate::vulkan::VulkanBuffer>),
}
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: DeviceBackend,
}
#[derive(Clone)]
pub struct Buffer {
    pub(crate) element: ScalarType,
    pub(crate) length: usize,
    pub(crate) inner: BufferBackend,
}
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
impl Buffer {
    pub(crate) fn vulkan(&self) -> &Arc<crate::vulkan::VulkanBuffer> {
        match &self.inner {
            BufferBackend::Vulkan(buffer) => buffer,
        }
    }
}
pub enum Argument<'a> {
    Scalar(Literal),
    Buffer(&'a Buffer),
}
impl Buffer {
    /// Return a handle to the device retaining this allocation.
    pub fn device(&self) -> Device {
        match self.inner {
            #[cfg(all(
                feature = "vulkan",
                any(target_os = "linux", target_os = "android", target_os = "windows")
            ))]
            BufferBackend::Vulkan(ref buffer) => Device {
                inner: DeviceBackend::Vulkan(buffer.owner()),
            },
        }
    }
    pub fn len(&self) -> usize {
        let _ = &self.inner;
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn element_type(&self) -> ScalarType {
        self.element
    }
}
impl Device {
    pub fn discover() -> ExecutionResult<Vec<Self>> {
        #[allow(unused_mut)]
        let mut devices = Vec::new();
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            devices.extend(crate::vulkan::discover()?);
        }
        Ok(devices)
    }
    pub fn name(&self) -> &str {
        let _ = &self.inner;
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            return match &self.inner {
                DeviceBackend::Vulkan(device) => &device.name,
            };
        }
        #[allow(unreachable_code)]
        "unavailable"
    }
    pub fn buffer(&self, element: ScalarType, values: &[u32]) -> ExecutionResult<Buffer> {
        if values.is_empty() {
            return Err(ExecutionError("buffers must have positive length".into()));
        }
        if element == ScalarType::Bool && values.iter().any(|v| *v > 1) {
            return Err(ExecutionError(
                "boolean memory slots must contain 0 or 1".into(),
            ));
        }
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            return match &self.inner {
                DeviceBackend::Vulkan(device) => device.buffer(element, values),
            };
        }
        #[allow(unreachable_code)]
        Err(ExecutionError("no backend enabled".into()))
    }
    pub fn read(&self, buffer: &Buffer) -> ExecutionResult<Vec<u32>> {
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            return match &self.inner {
                DeviceBackend::Vulkan(device) => device.read(buffer),
            };
        }
        #[allow(unreachable_code)]
        Err(ExecutionError("no backend enabled".into()))
    }
    /// Submit a synchronous compute launch.
    ///
    /// # Safety
    /// The caller must ensure all accesses are in bounds, barriers are uniform,
    /// and the kernel satisfies the IR's execution preconditions (including
    /// arithmetic validity and absence of data races). Structural validation
    /// does not establish these properties.
    pub unsafe fn launch(
        &self,
        kernel: &KernelHandle,
        tuning: &[u32],
        grid: [u32; 3],
        workgroup: [u32; 3],
        args: &[Argument<'_>],
    ) -> ExecutionResult<()> {
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            return match &self.inner {
                DeviceBackend::Vulkan(device) => {
                    device.launch(kernel, tuning, grid, workgroup, args)
                }
            };
        }
        #[allow(unreachable_code)]
        Err(ExecutionError("no backend enabled".into()))
    }
}
impl crate::Runtime {
    pub fn devices(&self) -> ExecutionResult<Vec<Device>> {
        self.devices
            .get_or_init(|| Device::discover().map_err(|e| e.to_string()))
            .clone()
            .map_err(ExecutionError)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for bool {}
}
/// Values with the IR's four-byte storage representation.
pub trait GpuScalar: sealed::Sealed + Copy {
    const TYPE: ScalarType;
    fn to_word(self) -> u32;
    fn from_word(word: u32) -> Self;
}
impl GpuScalar for u32 {
    const TYPE: ScalarType = ScalarType::U32;
    fn to_word(self) -> u32 {
        self
    }
    fn from_word(v: u32) -> Self {
        v
    }
}
impl GpuScalar for i32 {
    const TYPE: ScalarType = ScalarType::I32;
    fn to_word(self) -> u32 {
        self as u32
    }
    fn from_word(v: u32) -> Self {
        v as i32
    }
}
impl GpuScalar for f32 {
    const TYPE: ScalarType = ScalarType::F32;
    fn to_word(self) -> u32 {
        self.to_bits()
    }
    fn from_word(v: u32) -> Self {
        Self::from_bits(v)
    }
}
impl GpuScalar for bool {
    const TYPE: ScalarType = ScalarType::Bool;
    fn to_word(self) -> u32 {
        u32::from(self)
    }
    fn from_word(v: u32) -> Self {
        v != 0
    }
}
impl Device {
    pub fn buffer_from_slice<T: GpuScalar>(&self, values: &[T]) -> ExecutionResult<Buffer> {
        self.buffer(
            T::TYPE,
            &values.iter().map(|v| v.to_word()).collect::<Vec<_>>(),
        )
    }
    pub fn read_typed<T: GpuScalar>(&self, buffer: &Buffer) -> ExecutionResult<Vec<T>> {
        if buffer.element != T::TYPE {
            return Err(ExecutionError(
                "read element type does not match buffer".into(),
            ));
        }
        Ok(self.read(buffer)?.into_iter().map(T::from_word).collect())
    }
}

/// Borrow a host value as a checked kernel argument without moving buffers.
pub trait AsArgument {
    fn as_argument(&self) -> Argument<'_>;
}
/// A borrowable GPU allocation, shared by raw buffers, typed arrays and custom
/// owners. This trait is object-safe; it exposes the underlying allocation,
/// without introducing mutable access to typed wrappers' storage metadata.
pub trait GpuBuffer {
    fn as_buffer(&self) -> &Buffer;
    fn len(&self) -> usize {
        self.as_buffer().len()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn element_type(&self) -> ScalarType {
        self.as_buffer().element_type()
    }
    fn device(&self) -> Device {
        self.as_buffer().device()
    }
}
impl GpuBuffer for Buffer {
    fn as_buffer(&self) -> &Buffer {
        self
    }
}
impl<B: GpuBuffer + ?Sized> GpuBuffer for &B {
    fn as_buffer(&self) -> &Buffer {
        (**self).as_buffer()
    }
}
impl<B: GpuBuffer + ?Sized> GpuBuffer for &mut B {
    fn as_buffer(&self) -> &Buffer {
        (**self).as_buffer()
    }
}
impl<B: GpuBuffer + ?Sized> GpuBuffer for Box<B> {
    fn as_buffer(&self) -> &Buffer {
        (**self).as_buffer()
    }
}
impl<B: GpuBuffer + ?Sized> GpuBuffer for Arc<B> {
    fn as_buffer(&self) -> &Buffer {
        (**self).as_buffer()
    }
}
impl<B: GpuBuffer + ?Sized> AsArgument for B {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Buffer(self.as_buffer())
    }
}
impl AsArgument for Literal {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Scalar(*self)
    }
}
impl AsArgument for Argument<'_> {
    fn as_argument(&self) -> Argument<'_> {
        match self {
            Argument::Scalar(v) => Argument::Scalar(*v),
            Argument::Buffer(b) => Argument::Buffer(b),
        }
    }
}
impl AsArgument for u32 {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Scalar(Literal::U32(*self))
    }
}
impl AsArgument for i32 {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Scalar(Literal::I32(*self))
    }
}
impl AsArgument for f32 {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Scalar(Literal::F32(self.to_bits()))
    }
}
impl AsArgument for bool {
    fn as_argument(&self) -> Argument<'_> {
        Argument::Scalar(Literal::Bool(*self))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LaunchError {
    #[error(transparent)]
    Registration(#[from] crate::RegistrationError),
    #[error(transparent)]
    Execution(#[from] ExecutionError),
}
impl crate::Runtime {
    /// Resolve a platform-independent kernel hash and execute it on a device.
    ///
    /// # Safety
    /// The same bounds, race, arithmetic and barrier obligations as
    /// `Device::launch` apply.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch(
        &self,
        device: &Device,
        kernel: crate::KernelHash,
        tuning: &[u32],
        grid: [u32; 3],
        workgroup: [u32; 3],
        args: &[Argument<'_>],
    ) -> Result<(), LaunchError> {
        let kernel = self.kernel(kernel)?;
        unsafe {
            device.launch(&kernel, tuning, grid, workgroup, args)?;
        }
        Ok(())
    }
}

impl crate::Runtime {
    /// The first discovered device, retained by this runtime. No mutable global
    /// device selection is involved; use explicit handles to choose a device.
    pub fn default_device(&self) -> ExecutionResult<Device> {
        self.devices
            .get_or_init(|| Device::discover().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| ExecutionError(e.clone()))?
            .first()
            .cloned()
            .ok_or_else(|| ExecutionError("no compute device is available".into()))
    }
}
pub fn default_device() -> Result<Device, LaunchError> {
    Ok(crate::default_runtime()?.default_device()?)
}
/// Launch using the process-wide runtime and its default device.
///
/// # Safety
/// The execution obligations documented on `Device::launch` apply.
pub unsafe fn launch_default(
    kernel: crate::KernelHash,
    tuning: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
    args: &[Argument<'_>],
) -> Result<(), LaunchError> {
    let runtime = crate::default_runtime()?;
    let device = runtime.default_device()?;
    unsafe { runtime.launch(&device, kernel, tuning, grid, workgroup, args) }
}
/// Launch on an explicit device using the process-wide runtime.
///
/// # Safety
/// The execution obligations documented on `Device::launch` apply.
pub unsafe fn launch_on_default(
    device: &Device,
    kernel: crate::KernelHash,
    tuning: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
    args: &[Argument<'_>],
) -> Result<(), LaunchError> {
    unsafe { crate::default_runtime()?.launch(device, kernel, tuning, grid, workgroup, args) }
}
