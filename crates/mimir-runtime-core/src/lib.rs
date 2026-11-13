//! Backend-neutral execution contracts. This crate has no device discovery
//! policy, linked-module registry, global runtime, or platform dependencies.
use mimir_ir::flat::{
    Literal, ScalarType, SpecializedKernel, Type, ValidatedModule, ValidationError,
};
use std::{any::Any, sync::Arc};

/// Platform-independent identity of canonical kernel IR, including its callees.
pub use mimir_ir::flat::SemanticHash as KernelHash;

#[derive(Debug, Clone, thiserror::Error)]
#[error("Mimir execution: {0}")]
pub struct ExecutionError(pub String);
pub type ExecutionResult<T> = Result<T, ExecutionError>;

/// Retains validated IR independently of the registry that supplied it.
#[derive(Clone)]
pub struct KernelHandle {
    module: Arc<ValidatedModule>,
    index: usize,
    hash: KernelHash,
}
impl KernelHandle {
    /// Select a kernel and compute its identity; an invalid index is rejected.
    pub fn new(module: Arc<ValidatedModule>, index: usize) -> Result<Self, ValidationError> {
        let hash = module.kernel_hash(index)?;
        Ok(Self {
            module,
            index,
            hash,
        })
    }
    pub fn ir(&self) -> &ValidatedModule {
        &self.module
    }
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn semantic_hash(&self) -> KernelHash {
        self.hash
    }
}

/// Identity of an execution context, rather than of a physical GPU. Clones
/// identify the same context; separately created contexts must use new tokens.
#[derive(Clone, Debug)]
pub struct DeviceId(Arc<()>);

impl DeviceId {
    pub fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for DeviceId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for DeviceId {}

/// A complete PCI domain/bus/device/function address. All fields participate in
/// matching; neither an enumeration ordinal nor a device model identifies a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}
impl PciAddress {
    pub fn is_valid(self) -> bool {
        self.bus <= 255 && self.device <= 31 && self.function <= 7
    }
}

/// Windows adapter identity plus the node within a linked adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLuid {
    pub bytes: [u8; 8],
    pub node_mask: u32,
}

/// Optional physical-device hints for discovery across APIs. UUIDs must identify
/// the device/partition, never the driver or a pipeline-cache format. Memory is
/// total device memory, not free memory or the current allocation budget.
/// These hints never replace `DeviceId` for allocation ownership or execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HardwareInfo {
    pub uuid: Option<[u8; 16]>,
    pub luid: Option<DeviceLuid>,
    pub pci: Option<PciAddress>,
    pub vendor_id: Option<u32>,
    pub total_memory_bytes: Option<u64>,
}

/// A compiled-in provider of device contexts. Discovery policy and ordering
/// belong to the runtime that aggregates providers.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn discover(&self) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>>;
}

/// An owned context with synchronous execution and four-byte scalar slots.
/// Implementations synchronize host access to their queues and allocations.
pub trait DeviceBackend: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> DeviceId;
    fn name(&self) -> &str;
    fn backend_name(&self) -> &'static str;
    /// Preference over alternative backends for the same hardware. Larger wins;
    /// equal priorities retain the first discovered implementation.
    fn priority(&self) -> usize {
        0
    }
    /// Discovery hints. Unavailable identifiers remain `None`; context tokens
    /// and backend enumeration ordinals are not physical identities.
    fn hardware_info(&self) -> HardwareInfo {
        HardwareInfo::default()
    }

    /// Validate values with `validate_buffer_values` before allocating. The
    /// returned buffer must retain this context and preserve type/length.
    fn buffer(
        self: Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>>;
    /// Reject buffers from another context or implementation. Readback must be
    /// synchronized with submissions and return exactly `buffer.len()` slots.
    fn read(&self, buffer: &dyn BufferBackend) -> ExecutionResult<Vec<u32>>;
    /// Execute a structurally checked launch synchronously. Implementations
    /// must call `launch.validate_device(self)` and enforce device-specific
    /// dimensions, storage and argument limits before submitting GPU work.
    ///
    /// # Safety
    /// The caller must ensure bounds, uniform barriers, arithmetic validity and
    /// absence of data races, including all IR execution preconditions.
    unsafe fn launch(&self, launch: &Launch<'_>) -> ExecutionResult<()>;
}

/// Shared ownership of an allocation. Metadata is immutable and the allocation
/// retains its context until destruction. Backends use checked `Any` downcasts
/// before accessing native handles; a matching context token alone is not enough.
pub trait BufferBackend: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn device(&self) -> Arc<dyn DeviceBackend>;
    fn element_type(&self) -> ScalarType;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy)]
pub enum Argument<'a> {
    Scalar(Literal),
    Buffer(&'a dyn BufferBackend),
}

pub fn validate_buffer_values(element: ScalarType, values: &[u32]) -> ExecutionResult<()> {
    if values.is_empty() {
        return Err(ExecutionError("buffers must have positive length".into()));
    }
    if element == ScalarType::Bool && values.iter().any(|v| *v > 1) {
        return Err(ExecutionError(
            "boolean memory slots must contain 0 or 1".into(),
        ));
    }
    Ok(())
}

/// Check both implementation type and execution-context identity before use.
pub fn validate_buffer_owner<D: DeviceBackend + ?Sized>(
    device: &D,
    buffer: &dyn BufferBackend,
) -> ExecutionResult<()> {
    let owner = buffer.device();
    if device.as_any().type_id() != owner.as_any().type_id() || device.id() != owner.id() {
        return Err(ExecutionError("buffer belongs to another device".into()));
    }
    Ok(())
}

/// Immutable, structurally checked launch request. Creation validates dimensions,
/// argument types/ownership, tuning arithmetic and specialized allocations. It
/// does not prove memory bounds, barrier uniformity or freedom from races.
pub struct Launch<'a> {
    device: DeviceId,
    device_type: std::any::TypeId,
    kernel: &'a KernelHandle,
    specialized: SpecializedKernel,
    grid: [u32; 3],
    workgroup: [u32; 3],
    arguments: &'a [Argument<'a>],
}

impl<'a> Launch<'a> {
    pub fn new<D: DeviceBackend + ?Sized>(
        device: &D,
        kernel: &'a KernelHandle,
        tuning: &[u32],
        grid: [u32; 3],
        workgroup: [u32; 3],
        arguments: &'a [Argument<'a>],
    ) -> ExecutionResult<Self> {
        if grid.contains(&0) || workgroup.contains(&0) {
            return Err(ExecutionError("launch dimensions must be positive".into()));
        }
        workgroup
            .iter()
            .try_fold(1u64, |a, &b| a.checked_mul(b as u64))
            .ok_or_else(|| ExecutionError("workgroup size overflow".into()))?;
        let parameters = &kernel.ir().module().kernels[kernel.index()]
            .routine
            .header
            .parameters;
        if parameters.len() != arguments.len() {
            return Err(ExecutionError("kernel argument count mismatch".into()));
        }
        for (parameter, argument) in parameters.iter().zip(arguments) {
            match (parameter.ty, argument) {
                (Type::Scalar(t), Argument::Scalar(v)) if t == v.ty() => {}
                (Type::Buffer { element, .. }, Argument::Buffer(b))
                    if element == b.element_type() =>
                {
                    validate_buffer_owner(device, *b)?
                }
                _ => return Err(ExecutionError("kernel argument type mismatch".into())),
            }
        }
        let specialized = kernel
            .ir()
            .specialize_kernel(kernel.index(), tuning)
            .map_err(|e| ExecutionError(e.to_string()))?;
        Ok(Self {
            device: device.id(),
            device_type: device.as_any().type_id(),
            kernel,
            specialized,
            grid,
            workgroup,
            arguments,
        })
    }

    pub fn validate_device<D: DeviceBackend + ?Sized>(&self, device: &D) -> ExecutionResult<()> {
        if self.device != device.id() || self.device_type != device.as_any().type_id() {
            return Err(ExecutionError("launch belongs to another device".into()));
        }
        Ok(())
    }

    pub fn kernel(&self) -> &KernelHandle {
        self.kernel
    }

    pub fn specialized(&self) -> &SpecializedKernel {
        &self.specialized
    }

    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    pub fn arguments(&self) -> &[Argument<'a>] {
        self.arguments
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

#[cfg(test)]
mod tests {
    use super::*;
    use mimir_ir::flat::*;
    #[test]
    fn context_identity_is_retained_by_clones_and_unique_per_context() {
        let first = DeviceId::new();
        assert_eq!(first, first.clone());
        assert_ne!(first, DeviceId::new());
    }
    #[test]
    fn standalone_kernel_handles_check_indices_and_retain_validated_ir() {
        let module = Arc::new(
            Module {
                kernels: vec![Kernel {
                    debug: DebugInfo::default(),
                    routine: Routine {
                        header: Header::default(),
                        body: Body {
                            entry: BlockId(0),
                            blocks: vec![Block {
                                id: BlockId(0),
                                arguments: vec![],
                                instructions: vec![],
                                merge: None,
                                terminator: Terminator::Return(None),
                            }],
                        },
                    },
                }],
                functions: vec![],
            }
            .validate()
            .unwrap(),
        );
        assert!(KernelHandle::new(module.clone(), 1).is_err());
        let handle = KernelHandle::new(module.clone(), 0).unwrap();
        assert_eq!(handle.semantic_hash(), module.kernel_hash(0).unwrap());
        drop(module);
        assert_eq!(handle.ir().module().kernels.len(), 1);
    }
}
