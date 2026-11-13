use crate::KernelHandle;
use mimir_ir::flat::{Literal, ScalarType};
use mimir_runtime_core::{
    BufferBackend, DeviceBackend, Launch, validate_buffer_owner, validate_buffer_values,
};
pub use mimir_runtime_core::{
    DeviceLuid, ExecutionError, ExecutionResult, GpuScalar, HardwareInfo, PciAddress,
};
use std::sync::Arc;

/// A shared backend device context.
#[derive(Clone)]
pub struct Device {
    inner: Arc<dyn DeviceBackend>,
}
/// A shared backend allocation, retaining its owning context.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<dyn BufferBackend>,
}
pub enum Argument<'a> {
    Scalar(Literal),
    Buffer(&'a Buffer),
}
impl Buffer {
    pub fn device(&self) -> Device {
        Device::from_backend(self.inner.device())
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn element_type(&self) -> ScalarType {
        self.inner.element_type()
    }
}
impl Device {
    /// Integrate an independently supplied backend context without rediscovery
    /// or automatic duplicate selection.
    pub fn from_backend(backend: Arc<dyn DeviceBackend>) -> Self {
        Self { inner: backend }
    }
    /// Discover enabled backends, retaining the preferred implementation of
    /// confidently matched duplicate hardware.
    pub fn discover() -> ExecutionResult<Vec<Self>> {
        let backends: &[&dyn mimir_runtime_core::Backend] = &[
            #[cfg(all(
                feature = "vulkan",
                any(target_os = "linux", target_os = "android", target_os = "windows")
            ))]
            &mimir_vulkan_runtime::VulkanBackend,
            #[cfg(all(
                feature = "cuda",
                any(target_os = "linux", target_os = "windows"),
                target_pointer_width = "64"
            ))]
            &mimir_cuda_runtime::CudaBackend,
        ];
        Self::discover_with_backends(backends)
    }
    /// Discover and resolve duplicate hardware from an explicit set of providers.
    /// Each physical device is represented by its highest-priority backend.
    /// Ambiguous identities are retained rather than hiding distinct devices.
    pub fn discover_with_backends(
        backends: &[&dyn mimir_runtime_core::Backend],
    ) -> ExecutionResult<Vec<Self>> {
        let mut devices = Vec::new();
        for backend in backends {
            devices.extend(backend.discover()?.into_iter().map(Self::from_backend));
        }
        Ok(crate::discovery::select(devices))
    }
    pub fn priority(&self) -> usize {
        self.inner.priority()
    }
    pub fn hardware_info(&self) -> mimir_runtime_core::HardwareInfo {
        self.inner.hardware_info()
    }
    pub fn name(&self) -> &str {
        self.inner.name()
    }
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }
    pub fn buffer(&self, element: ScalarType, values: &[u32]) -> ExecutionResult<Buffer> {
        validate_buffer_values(element, values)?;
        let inner = self.inner.clone().buffer(element, values)?;
        validate_buffer_owner(self.inner.as_ref(), inner.as_ref())?;
        if inner.element_type() != element || inner.len() != values.len() {
            return Err(ExecutionError(
                "backend returned incorrect buffer metadata".into(),
            ));
        }
        Ok(Buffer { inner })
    }
    pub fn read(&self, buffer: &Buffer) -> ExecutionResult<Vec<u32>> {
        validate_buffer_owner(self.inner.as_ref(), buffer.inner.as_ref())?;
        let values = self.inner.read(buffer.inner.as_ref())?;
        if values.len() != buffer.len() {
            return Err(ExecutionError(
                "backend returned incorrect readback length".into(),
            ));
        }
        validate_buffer_values(buffer.element_type(), &values)?;
        Ok(values)
    }
    /// Submit a synchronous compute launch.
    ///
    /// # Safety
    /// The caller must ensure bounds, uniform barriers, valid arithmetic and
    /// absence of data races, including all IR execution preconditions.
    pub unsafe fn launch(
        &self,
        kernel: &KernelHandle,
        tuning: &[u32],
        grid: [u32; 3],
        workgroup: [u32; 3],
        args: &[Argument<'_>],
    ) -> ExecutionResult<()> {
        let arguments: Vec<_> = args
            .iter()
            .map(|a| match a {
                Argument::Scalar(v) => mimir_runtime_core::Argument::Scalar(*v),
                Argument::Buffer(b) => mimir_runtime_core::Argument::Buffer(b.inner.as_ref()),
            })
            .collect();
        let launch = Launch::new(
            self.inner.as_ref(),
            kernel,
            tuning,
            grid,
            workgroup,
            &arguments,
        )?;
        unsafe { self.inner.launch(&launch) }
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

impl Device {
    pub fn buffer_from_slice<T: GpuScalar>(&self, values: &[T]) -> ExecutionResult<Buffer> {
        self.buffer(
            T::TYPE,
            &values.iter().map(|v| v.to_word()).collect::<Vec<_>>(),
        )
    }
    pub fn read_typed<T: GpuScalar>(&self, buffer: &Buffer) -> ExecutionResult<Vec<T>> {
        if buffer.element_type() != T::TYPE {
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
