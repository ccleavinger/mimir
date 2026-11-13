//! Exercise the facade and core contracts without a GPU or platform libraries.
use mimir_ir::flat::*;
use mimir_runtime::backend::{self, Backend, BufferBackend, DeviceBackend, DeviceId, Launch};
use mimir_runtime::{Argument, Device, ExecutionError, ExecutionResult, GpuArray, Runtime};
use std::{
    any::Any,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Default)]
struct Stats {
    allocations: usize,
    reads: usize,
    launches: usize,
    grid: [u32; 3],
    workgroup: [u32; 3],
}
struct MockDevice {
    id: DeviceId,
    stats: Mutex<Stats>,
    bad_metadata: bool,
    bad_read: bool,
}
impl MockDevice {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            id: DeviceId::new(),
            stats: Mutex::new(Stats::default()),
            bad_metadata: false,
            bad_read: false,
        })
    }
}
struct MockBuffer {
    device: Arc<MockDevice>,
    element: ScalarType,
    words: Mutex<Vec<u32>>,
}
impl BufferBackend for MockBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn device(&self) -> Arc<dyn DeviceBackend> {
        self.device.clone()
    }
    fn element_type(&self) -> ScalarType {
        self.element
    }
    fn len(&self) -> usize {
        self.words.lock().unwrap().len()
    }
}
impl DeviceBackend for MockDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn id(&self) -> DeviceId {
        self.id.clone()
    }
    fn name(&self) -> &str {
        "mock context"
    }
    fn backend_name(&self) -> &'static str {
        "mock"
    }
    fn buffer(
        self: Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        backend::validate_buffer_values(element, values)?;
        self.stats.lock().unwrap().allocations += 1;
        let element = if self.bad_metadata {
            ScalarType::Bool
        } else {
            element
        };
        Ok(Arc::new(MockBuffer {
            device: self,
            element,
            words: Mutex::new(values.to_vec()),
        }))
    }
    fn read(&self, buffer: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        backend::validate_buffer_owner(self, buffer)?;
        self.stats.lock().unwrap().reads += 1;
        let native = buffer.as_any().downcast_ref::<MockBuffer>().unwrap();
        let mut words = native.words.lock().unwrap().clone();
        if self.bad_read {
            words.push(9);
        }
        Ok(words)
    }
    unsafe fn launch(&self, launch: &Launch<'_>) -> ExecutionResult<()> {
        launch.validate_device(self)?;
        let mut stats = self.stats.lock().unwrap();
        stats.launches += 1;
        stats.grid = launch.grid();
        stats.workgroup = launch.workgroup();
        let [
            backend::Argument::Scalar(Literal::U32(value)),
            backend::Argument::Buffer(buffer),
        ] = launch.arguments()
        else {
            panic!("mock kernel signature")
        };
        let native = buffer.as_any().downcast_ref::<MockBuffer>().unwrap();
        native.words.lock().unwrap()[0] =
            value + launch.specialized().kernel.constants[&ConstantId(0)];
        Ok(())
    }
}
struct MockBackend(Arc<MockDevice>);
impl Backend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }
    fn discover(&self) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
        Ok(vec![self.0.clone()])
    }
}

fn kernel() -> Module {
    Module {
        kernels: vec![Kernel {
            debug: DebugInfo::default(),
            routine: Routine {
                header: Header {
                    parameters: vec![
                        Parameter {
                            id: ParameterId(0),
                            ty: Type::Scalar(ScalarType::U32),
                        },
                        Parameter {
                            id: ParameterId(1),
                            ty: Type::Buffer {
                                element: ScalarType::U32,
                                space: AddressSpace::Global,
                                access: Access::Write,
                            },
                        },
                    ],
                    constants: vec![ConstantId(0)],
                    ..Header::default()
                },
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
}
fn registered(runtime: &Runtime) -> mimir_runtime::KernelHandle {
    let module = kernel();
    let hash = module.clone().validate().unwrap().kernel_hash(0).unwrap();
    runtime.register(hash, module).unwrap()
}

#[test]
fn independent_backend_dispatches_typed_arrays_and_registered_launches() {
    let context = MockDevice::new();
    let provider = MockBackend(context.clone());
    let device = Device::from_backend(provider.discover().unwrap().remove(0));
    assert_eq!(device.name(), "mock context");
    assert_eq!(device.backend_name(), "mock");
    let runtime = Runtime::default();
    let handle = registered(&runtime);
    let array = GpuArray::<u32>::from_slice_on(&device, &[0, 2]).unwrap();
    // Mock execution accesses only the first slot of this nonempty array.
    unsafe {
        runtime
            .launch(
                &device,
                handle.semantic_hash(),
                &[3],
                [2, 1, 1],
                [4, 1, 1],
                &[
                    Argument::Scalar(Literal::U32(7)),
                    Argument::Buffer(array.as_buffer()),
                ],
            )
            .unwrap();
    }
    assert_eq!(array.to_vec().unwrap(), vec![10, 2]);
    let stats = context.stats.lock().unwrap();
    assert_eq!((stats.allocations, stats.reads, stats.launches), (1, 1, 1));
    assert_eq!((stats.grid, stats.workgroup), ([2, 1, 1], [4, 1, 1]));
}

#[test]
fn buffers_and_arrays_retain_their_context_without_a_runtime() {
    let context = MockDevice::new();
    let weak = Arc::downgrade(&context);
    let device = Device::from_backend(context.clone());
    let array = GpuArray::<f32>::from_slice_on(&device, &[1.25, -0.0]).unwrap();
    let alias = array.clone();
    drop(array);
    drop(device);
    drop(context);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        alias
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![1.25f32.to_bits(), (-0.0f32).to_bits()]
    );
    drop(alias);
    assert!(weak.upgrade().is_none());
}

#[test]
fn invalid_requests_are_rejected_before_backend_dispatch() {
    let context = MockDevice::new();
    let device = Device::from_backend(context.clone());
    assert!(device.buffer(ScalarType::U32, &[]).is_err());
    assert!(device.buffer(ScalarType::Bool, &[2]).is_err());
    assert_eq!(context.stats.lock().unwrap().allocations, 0);
    let buffer = device.buffer_from_slice(&[0u32]).unwrap();
    assert!(device.read_typed::<f32>(&buffer).is_err());
    let runtime = Runtime::default();
    let handle = registered(&runtime);
    let args = [Argument::Scalar(Literal::U32(1)), Argument::Buffer(&buffer)];
    // All calls below fail structural validation before mock execution.
    unsafe {
        assert!(
            device
                .launch(&handle, &[1], [0, 1, 1], [1; 3], &args)
                .is_err()
        );
        assert!(
            device
                .launch(&handle, &[1], [1; 3], [0, 1, 1], &args)
                .is_err()
        );
        assert!(
            device
                .launch(&handle, &[1], [1; 3], [u32::MAX; 3], &args)
                .is_err()
        );
        assert!(device.launch(&handle, &[], [1; 3], [1; 3], &args).is_err());
        assert!(device.launch(&handle, &[1], [1; 3], [1; 3], &[]).is_err());
        assert!(
            device
                .launch(
                    &handle,
                    &[1],
                    [1; 3],
                    [1; 3],
                    &[
                        Argument::Scalar(Literal::Bool(true)),
                        Argument::Buffer(&buffer)
                    ]
                )
                .is_err()
        );
    }
    assert_eq!(context.stats.lock().unwrap().launches, 0);
}

#[test]
fn foreign_context_buffers_are_rejected_for_reads_and_launches() {
    let first = MockDevice::new();
    let second = MockDevice::new();
    let a = Device::from_backend(first.clone());
    let b = Device::from_backend(second.clone());
    let buffer = a.buffer_from_slice(&[0u32]).unwrap();
    assert!(b.read(&buffer).is_err());
    let handle = registered(&Runtime::default());
    unsafe {
        assert!(
            b.launch(
                &handle,
                &[1],
                [1; 3],
                [1; 3],
                &[Argument::Scalar(Literal::U32(1)), Argument::Buffer(&buffer)]
            )
            .is_err()
        );
    }
    assert_eq!(
        (
            second.stats.lock().unwrap().reads,
            first.stats.lock().unwrap().reads
        ),
        (0, 0)
    );
    assert_eq!(second.stats.lock().unwrap().launches, 0);
    let native = first.clone().buffer(ScalarType::U32, &[0]).unwrap();
    assert!(second.read(native.as_ref()).is_err());
    let arguments = [
        backend::Argument::Scalar(Literal::U32(1)),
        backend::Argument::Buffer(native.as_ref()),
    ];
    let launch = Launch::new(first.as_ref(), &handle, &[1], [1; 3], [1; 3], &arguments).unwrap();
    unsafe {
        assert!(second.launch(&launch).is_err());
    }
}

struct OtherDevice {
    id: DeviceId,
    calls: AtomicUsize,
}
impl DeviceBackend for OtherDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn id(&self) -> DeviceId {
        self.id.clone()
    }
    fn name(&self) -> &str {
        "other backend"
    }
    fn backend_name(&self) -> &'static str {
        "other"
    }
    fn buffer(
        self: Arc<Self>,
        _: ScalarType,
        _: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        Err(ExecutionError("unused".into()))
    }
    fn read(&self, _: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![])
    }
    unsafe fn launch(&self, _: &Launch<'_>) -> ExecutionResult<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[test]
fn foreign_implementation_is_rejected_even_with_the_same_context_token() {
    let first = MockDevice::new();
    let second = Arc::new(OtherDevice {
        id: first.id(),
        calls: AtomicUsize::new(0),
    });
    let a = Device::from_backend(first);
    let b = Device::from_backend(second.clone());
    let buffer = a.buffer_from_slice(&[0u32]).unwrap();
    assert!(b.read(&buffer).is_err());
    let handle = registered(&Runtime::default());
    unsafe {
        assert!(
            b.launch(
                &handle,
                &[1],
                [1; 3],
                [1; 3],
                &[Argument::Scalar(Literal::U32(1)), Argument::Buffer(&buffer)]
            )
            .is_err()
        );
    }
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn backend_metadata_and_readback_contracts_are_checked() {
    let bad = Arc::new(MockDevice {
        bad_metadata: true,
        ..Arc::try_unwrap(MockDevice::new()).ok().unwrap()
    });
    let device = Device::from_backend(bad);
    assert!(device.buffer_from_slice(&[1u32]).is_err());
    let bad = Arc::new(MockDevice {
        bad_read: true,
        ..Arc::try_unwrap(MockDevice::new()).ok().unwrap()
    });
    let device = Device::from_backend(bad);
    let buffer = device.buffer_from_slice(&[1u32]).unwrap();
    assert!(device.read(&buffer).is_err());
}

#[test]
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn backend_free_runtime_has_no_discovered_devices() {
    assert!(Device::discover().unwrap().is_empty());
    let runtime = Runtime::default();
    assert!(runtime.devices().unwrap().is_empty());
    assert!(runtime.default_device().is_err());
}
