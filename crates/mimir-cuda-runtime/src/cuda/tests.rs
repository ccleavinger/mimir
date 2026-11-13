//! Inject the real FFI boundary, not a separate backend implementation. Tests
//! exercise context stacks, raw argument pointers, cleanup and error propagation.
use super::*;
use mimir_ir::flat::*;
use mimir_runtime_core::KernelHandle;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    ffi::{CStr, c_char, c_int, c_uint},
    sync::{LazyLock, MutexGuard},
};

static SERIAL: Mutex<()> = Mutex::new(());
static MOCK: LazyLock<Mutex<Mock>> = LazyLock::new(|| Mutex::new(Mock::default()));
thread_local! { static STACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) }; }
fn current() -> usize {
    STACK.with(|s| s.borrow().last().copied().unwrap_or(0))
}
fn mock() -> MutexGuard<'static, Mock> {
    MOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
struct Memory {
    context: usize,
    bytes: Vec<u8>,
}
struct ModuleRecord {
    context: usize,
    parameters: Vec<usize>,
    ptx: String,
}
#[derive(Clone)]
struct Submission {
    arguments: Vec<u64>,
    grid: [u32; 3],
    block: [u32; 3],
    context: usize,
}
struct Mock {
    next: usize,
    count: i32,
    major: i32,
    minor: i32,
    limits: HashMap<i32, i32>,
    function_threads: i32,
    function_shared: i32,
    fail: Option<(&'static str, i32)>,
    contexts: HashSet<usize>,
    streams: HashMap<usize, usize>,
    memory: HashMap<u64, Memory>,
    modules: HashMap<usize, ModuleRecord>,
    functions: HashMap<usize, usize>,
    loads: usize,
    load_attempts: usize,
    unloads: usize,
    allocations: usize,
    frees: usize,
    syncs: usize,
    context_syncs: usize,
    created: usize,
    destroyed: usize,
    submissions: Vec<Submission>,
    trace: Vec<&'static str>,
    destroyed_live_resources: bool,
}
impl Default for Mock {
    fn default() -> Self {
        Self {
            next: 1000,
            count: 2,
            major: 8,
            minor: 0,
            limits: [
                (1, 1024),
                (2, 1024),
                (3, 1024),
                (4, 64),
                (5, i32::MAX),
                (6, 65535),
                (7, 65535),
                (8, 49152),
            ]
            .into(),
            function_threads: 1024,
            function_shared: 0,
            fail: None,
            contexts: HashSet::new(),
            streams: HashMap::new(),
            memory: HashMap::new(),
            modules: HashMap::new(),
            functions: HashMap::new(),
            loads: 0,
            load_attempts: 0,
            unloads: 0,
            allocations: 0,
            frees: 0,
            syncs: 0,
            context_syncs: 0,
            created: 0,
            destroyed: 0,
            submissions: vec![],
            trace: vec![],
            destroyed_live_resources: false,
        }
    }
}
impl Mock {
    fn handle(&mut self) -> usize {
        self.next += 1;
        self.next
    }
    fn failure(&mut self, name: &str) -> i32 {
        match self.fail {
            Some((operation, status)) if name == operation => {
                self.fail = None;
                status
            }
            _ => 0,
        }
    }
    fn valid_context(&self) -> bool {
        self.contexts.contains(&current())
    }
}
struct Harness {
    _serial: MutexGuard<'static, ()>,
    driver: Arc<Driver>,
}
impl Harness {
    fn new() -> Self {
        let serial = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *mock() = Mock::default();
        STACK.with(|s| s.borrow_mut().clear());
        Self {
            _serial: serial,
            driver: Arc::new(Driver {
                _library: None,
                device_uuid: Some(device_uuid),
                device_pci: Some(device_pci),
                device_luid: Some(device_luid),
                device_memory: Some(device_memory),
                init,
                driver_version,
                device_count,
                device_get,
                device_name,
                device_attribute,
                context_create,
                context_destroy,
                context_push,
                context_pop,
                context_sync,
                stream_create,
                stream_destroy,
                stream_sync,
                mem_alloc,
                mem_free,
                copy_to_device,
                copy_to_host,
                module_load,
                module_unload,
                module_function,
                function_attribute,
                launch,
                error_name,
                error_string,
            }),
        }
    }
    fn devices(&self) -> Vec<Arc<dyn DeviceBackend>> {
        discover_with(self.driver.clone()).unwrap()
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        STACK.with(|s| s.borrow_mut().clear());
    }
}

extern "system" fn init(_: c_uint) -> Status {
    mock().failure("init")
}
extern "system" fn driver_version(value: *mut c_int) -> Status {
    unsafe {
        *value = 12090;
    }
    0
}
extern "system" fn device_count(value: *mut c_int) -> Status {
    unsafe {
        *value = mock().count;
    }
    0
}
extern "system" fn device_get(value: *mut c_int, ordinal: c_int) -> Status {
    unsafe {
        *value = ordinal;
    }
    0
}
extern "system" fn device_name(value: *mut c_char, length: c_int, _: c_int) -> Status {
    let name = b"Mock NVIDIA device\0";
    if length < name.len() as i32 {
        return 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(name.as_ptr(), value.cast(), name.len());
    }
    0
}
extern "system" fn device_attribute(value: *mut c_int, attribute: c_int, _: c_int) -> Status {
    let m = mock();
    let result = match attribute {
        COMPUTE_CAPABILITY_MAJOR => m.major,
        COMPUTE_CAPABILITY_MINOR => m.minor,
        _ => *m.limits.get(&attribute).unwrap_or(&0),
    };
    unsafe {
        *value = result;
    }
    0
}
extern "system" fn context_create(value: *mut Handle, _: c_uint, _: c_int) -> Status {
    let mut m = mock();
    let failure = m.failure("context_create");
    if failure != 0 {
        return failure;
    }
    let id = m.handle();
    m.contexts.insert(id);
    m.created += 1;
    STACK.with(|s| s.borrow_mut().push(id));
    unsafe {
        *value = id as Handle;
    }
    0
}
extern "system" fn context_destroy(context: Handle) -> Status {
    let mut m = mock();
    let context = context as usize;
    if m.streams.values().any(|c| *c == context)
        || m.modules.values().any(|r| r.context == context)
        || m.memory.values().any(|r| r.context == context)
    {
        m.destroyed_live_resources = true;
    }
    m.contexts.remove(&context);
    m.destroyed += 1;
    m.trace.push("destroy_context");
    STACK.with(|s| s.borrow_mut().retain(|c| *c != context));
    0
}
extern "system" fn context_push(context: Handle) -> Status {
    let mut m = mock();
    let failure = m.failure("context_push");
    if failure != 0 {
        return failure;
    }
    if !m.contexts.contains(&(context as usize)) {
        return 201;
    }
    STACK.with(|s| s.borrow_mut().push(context as usize));
    0
}
extern "system" fn context_pop(value: *mut Handle) -> Status {
    let context = STACK.with(|s| s.borrow_mut().pop());
    match context {
        Some(c) => {
            unsafe {
                *value = c as Handle;
            }
            0
        }
        None => 201,
    }
}
extern "system" fn context_sync() -> Status {
    let mut m = mock();
    m.context_syncs += 1;
    m.failure("context_sync")
}
extern "system" fn stream_create(value: *mut Handle, flags: c_uint) -> Status {
    let mut m = mock();
    let failure = m.failure("stream_create");
    if failure != 0 {
        return failure;
    }
    if !m.valid_context() || flags != STREAM_NON_BLOCKING {
        return 201;
    }
    let id = m.handle();
    m.streams.insert(id, current());
    unsafe {
        *value = id as Handle;
    }
    0
}
extern "system" fn stream_destroy(stream: Handle) -> Status {
    let mut m = mock();
    if m.streams.get(&(stream as usize)) != Some(&current()) {
        return 201;
    }
    m.streams.remove(&(stream as usize));
    m.trace.push("destroy_stream");
    0
}
extern "system" fn stream_sync(stream: Handle) -> Status {
    let mut m = mock();
    m.syncs += 1;
    if m.streams.get(&(stream as usize)) != Some(&current()) {
        return 201;
    }
    m.failure("stream_sync")
}
extern "system" fn mem_alloc(value: *mut DevicePtr, bytes: usize) -> Status {
    let mut m = mock();
    let failure = m.failure("mem_alloc");
    if failure != 0 {
        return failure;
    }
    if !m.valid_context() {
        return 201;
    }
    let pointer = 0x1_0000_0000 + (m.handle() as u64) * 4096;
    m.memory.insert(
        pointer,
        Memory {
            context: current(),
            bytes: vec![0; bytes],
        },
    );
    m.allocations += 1;
    unsafe {
        *value = pointer;
    }
    0
}
extern "system" fn mem_free(pointer: DevicePtr) -> Status {
    let mut m = mock();
    if m.memory.get(&pointer).map(|b| b.context) != Some(current()) {
        return 201;
    }
    m.memory.remove(&pointer);
    m.frees += 1;
    m.trace.push("free");
    0
}
extern "system" fn copy_to_device(
    pointer: DevicePtr,
    source: *const c_void,
    bytes: usize,
) -> Status {
    let mut m = mock();
    let failure = m.failure("copy_to_device");
    if failure != 0 {
        return failure;
    }
    let Some(memory) = m.memory.get_mut(&pointer) else {
        return 1;
    };
    if memory.context != current() || bytes > memory.bytes.len() || source.is_null() {
        return 201;
    }
    memory.bytes[..bytes]
        .copy_from_slice(unsafe { std::slice::from_raw_parts(source.cast(), bytes) });
    0
}
extern "system" fn copy_to_host(
    destination: *mut c_void,
    pointer: DevicePtr,
    bytes: usize,
) -> Status {
    let mut m = mock();
    let failure = m.failure("copy_to_host");
    if failure != 0 {
        return failure;
    }
    let Some(memory) = m.memory.get(&pointer) else {
        return 1;
    };
    if memory.context != current() || bytes > memory.bytes.len() || destination.is_null() {
        return 201;
    }
    unsafe {
        ptr::copy_nonoverlapping(memory.bytes.as_ptr(), destination.cast(), bytes);
    }
    0
}
extern "system" fn module_load(
    value: *mut Handle,
    image: *const c_void,
    count: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> Status {
    let mut m = mock();
    m.load_attempts += 1;
    if !m.valid_context() {
        return 201;
    }
    let failure = m.failure("module_load");
    if failure != 0 {
        for i in 0..count as usize {
            if unsafe { *options.add(i) } == JIT_ERROR_LOG_BUFFER {
                let message = b"mock PTX JIT diagnostic\0";
                unsafe {
                    ptr::copy_nonoverlapping(
                        message.as_ptr(),
                        (*values.add(i)).cast(),
                        message.len(),
                    );
                }
            }
        }
        return failure;
    }
    let ptx = unsafe { CStr::from_ptr(image.cast()) }
        .to_string_lossy()
        .into_owned();
    let Some((_, rest)) = ptx.split_once(".visible .entry mimir_kernel(") else {
        return 218;
    };
    let Some((parameters, _)) = rest.split_once(')') else {
        return 218;
    };
    let parameters = if parameters.is_empty() {
        vec![]
    } else {
        parameters
            .split(", ")
            .map(|p| if p.contains(".u64") { 8 } else { 4 })
            .collect()
    };
    let id = m.handle();
    m.modules.insert(
        id,
        ModuleRecord {
            context: current(),
            parameters,
            ptx,
        },
    );
    m.loads += 1;
    unsafe {
        *value = id as Handle;
    }
    0
}
extern "system" fn module_unload(module: Handle) -> Status {
    let mut m = mock();
    if m.modules.get(&(module as usize)).map(|r| r.context) != Some(current()) {
        return 201;
    }
    m.modules.remove(&(module as usize));
    m.functions.retain(|_, id| *id != module as usize);
    m.unloads += 1;
    m.trace.push("unload");
    0
}
extern "system" fn module_function(
    value: *mut Handle,
    module: Handle,
    name: *const c_char,
) -> Status {
    let mut m = mock();
    let failure = m.failure("module_function");
    if failure != 0 {
        return failure;
    }
    if m.modules.get(&(module as usize)).map(|r| r.context) != Some(current()) {
        return 201;
    }
    if unsafe { CStr::from_ptr(name) }.to_bytes() != b"mimir_kernel" {
        return 500;
    }
    let id = m.handle();
    m.functions.insert(id, module as usize);
    unsafe {
        *value = id as Handle;
    }
    0
}
extern "system" fn function_attribute(value: *mut c_int, attribute: c_int, _: Handle) -> Status {
    let mut m = mock();
    let failure = m.failure("function_attribute");
    if failure != 0 {
        return failure;
    }
    unsafe {
        *value = match attribute {
            FUNCTION_MAX_THREADS => m.function_threads,
            FUNCTION_SHARED_BYTES => m.function_shared,
            _ => 0,
        };
    }
    0
}
#[allow(clippy::too_many_arguments)]
extern "system" fn launch(
    function: Handle,
    gx: c_uint,
    gy: c_uint,
    gz: c_uint,
    bx: c_uint,
    by: c_uint,
    bz: c_uint,
    shared: c_uint,
    stream: Handle,
    parameters: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Status {
    let mut m = mock();
    let failure = m.failure("launch");
    if failure != 0 {
        return failure;
    }
    let Some(module) = m.functions.get(&(function as usize)) else {
        return 500;
    };
    let module = &m.modules[module];
    if module.context != current()
        || m.streams.get(&(stream as usize)) != Some(&current())
        || shared != 0
        || !extra.is_null()
    {
        return 201;
    }
    if !module.parameters.is_empty() && parameters.is_null() {
        return 1;
    }
    let arguments = module
        .parameters
        .iter()
        .enumerate()
        .map(|(i, size)| unsafe {
            let pointer = *parameters.add(i);
            if *size == 8 {
                *pointer.cast::<u64>()
            } else {
                u64::from(*pointer.cast::<u32>())
            }
        })
        .collect();
    m.submissions.push(Submission {
        arguments,
        grid: [gx, gy, gz],
        block: [bx, by, bz],
        context: current(),
    });
    0
}
extern "system" fn error_name(_: Status, value: *mut *const c_char) -> Status {
    unsafe {
        *value = c"CUDA_ERROR_MOCK".as_ptr();
    }
    0
}
extern "system" fn error_string(_: Status, value: *mut *const c_char) -> Status {
    unsafe {
        *value = c"mock driver failure".as_ptr();
    }
    0
}

fn kernel(types: &[Type], tuning: bool) -> KernelHandle {
    let module = Module {
        kernels: vec![Kernel {
            debug: DebugInfo::default(),
            routine: Routine {
                header: Header {
                    parameters: types
                        .iter()
                        .enumerate()
                        .map(|(i, ty)| Parameter {
                            id: ParameterId(i as u32),
                            ty: *ty,
                        })
                        .collect(),
                    constants: if tuning { vec![ConstantId(0)] } else { vec![] },
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
    .validate()
    .unwrap();
    KernelHandle::new(Arc::new(module), 0).unwrap()
}
fn buffer_type(element: ScalarType) -> Type {
    Type::Buffer {
        element,
        space: AddressSpace::Global,
        access: Access::ReadWrite,
    }
}
fn submit(
    device: &dyn DeviceBackend,
    kernel: &KernelHandle,
    tuning: &[u32],
    block: [u32; 3],
    arguments: &[Argument<'_>],
) -> ExecutionResult<()> {
    let launch = Launch::new(device, kernel, tuning, [2, 1, 1], block, arguments)?;
    // All mock kernels have no memory accesses and no execution preconditions.
    unsafe { device.launch(&launch) }
}

#[test]
fn discovery_handles_absence_unsupported_gpus_and_initialization_errors() {
    let h = Harness::new();
    mock().fail = Some(("init", NO_DEVICE));
    assert!(discover_with(h.driver.clone()).unwrap().is_empty());
    mock().count = 0;
    assert!(h.devices().is_empty());
    mock().count = 2;
    mock().major = 3;
    assert!(h.devices().is_empty());
    assert_eq!(mock().created, 0);
    mock().fail = Some(("init", 803));
    let error = discover_with(h.driver.clone()).err().unwrap();
    assert!(error.0.contains("cuInit") && error.0.contains("803"));
}

#[test]
fn context_stacks_typed_transfers_and_resource_lifetimes() {
    let h = Harness::new();
    STACK.with(|s| s.borrow_mut().push(0xfeed));
    let mut devices = h.devices();
    assert_eq!(current(), 0xfeed);
    let device = devices.remove(0);
    drop(devices);
    let weak = Arc::downgrade(&device);
    for (element, words) in [
        (ScalarType::U32, vec![1, u32::MAX]),
        (ScalarType::I32, vec![(-7i32) as u32, i32::MIN as u32]),
        (
            ScalarType::F32,
            vec![(-0.0f32).to_bits(), f32::NAN.to_bits()],
        ),
        (ScalarType::Bool, vec![0, 1]),
    ] {
        let buffer = device.clone().buffer(element, &words).unwrap();
        assert_eq!(device.read(buffer.as_ref()).unwrap(), words);
        assert_eq!(current(), 0xfeed);
    }
    let buffer = device.clone().buffer(ScalarType::U32, &[42]).unwrap();
    let retained = buffer.clone();
    drop(buffer);
    drop(device);
    assert!(weak.upgrade().is_some());
    assert_eq!(retained.device().read(retained.as_ref()).unwrap(), vec![42]);
    drop(retained);
    let m = mock();
    assert_eq!((m.created, m.destroyed), (2, 2));
    assert_eq!(m.allocations, m.frees);
    assert!(m.memory.is_empty() && m.contexts.is_empty() && m.streams.is_empty());
    assert!(!m.destroyed_live_resources);
    assert_eq!(current(), 0xfeed);
}

#[test]
fn raw_kernel_arguments_cache_keys_and_launch_dimensions() {
    let h = Harness::new();
    let devices = h.devices();
    let device = devices[0].as_ref();
    let buffer = devices[0].clone().buffer(ScalarType::U32, &[0]).unwrap();
    let kernel = kernel(
        &[
            Type::Scalar(ScalarType::Bool),
            buffer_type(ScalarType::U32),
            Type::Scalar(ScalarType::I32),
            Type::Scalar(ScalarType::F32),
            Type::Scalar(ScalarType::U32),
        ],
        true,
    );
    let arguments = [
        Argument::Scalar(Literal::Bool(true)),
        Argument::Buffer(buffer.as_ref()),
        Argument::Scalar(Literal::I32(-9)),
        Argument::Scalar(Literal::f32(-0.0)),
        Argument::Scalar(Literal::U32(u32::MAX)),
    ];
    submit(device, &kernel, &[3], [4, 1, 1], &arguments).unwrap();
    submit(device, &kernel, &[3], [8, 1, 1], &arguments).unwrap();
    assert_eq!(mock().loads, 1);
    submit(device, &kernel, &[4], [4, 1, 1], &arguments).unwrap();
    let m = mock();
    assert_eq!(m.loads, 2);
    let first = &m.submissions[0];
    assert_eq!(first.arguments[0], 1);
    assert!(first.arguments[1] > u64::from(u32::MAX));
    assert_eq!(
        first.arguments[2..],
        [
            u64::from((-9i32) as u32),
            u64::from((-0.0f32).to_bits()),
            u64::from(u32::MAX)
        ]
    );
    assert_eq!((first.grid, first.block), ([2, 1, 1], [4, 1, 1]));
    assert_eq!(
        first.context,
        device
            .as_any()
            .downcast_ref::<CudaDevice>()
            .unwrap()
            .context
            .raw as usize
    );
    assert!(m.modules.values().all(|r| r.ptx.contains(".target sm_50")));
    assert_eq!(m.syncs, 3);
}

#[test]
fn concurrent_launches_compile_one_module_and_restore_each_thread() {
    let h = Harness::new();
    let devices = h.devices();
    let device = &devices[0];
    let kernel = kernel(&[], false);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let kernel = &kernel;
            scope.spawn(move || {
                submit(device.as_ref(), kernel, &[], [1; 3], &[]).unwrap();
                assert_eq!(current(), 0);
            });
        }
    });
    let m = mock();
    assert_eq!((m.loads, m.submissions.len()), (1, 8));
}

#[test]
fn foreign_contexts_and_invalid_requests_never_reach_launch() {
    let h = Harness::new();
    let devices = h.devices();
    let buffer = devices[0].clone().buffer(ScalarType::U32, &[0]).unwrap();
    assert!(devices[1].read(buffer.as_ref()).is_err());
    let kernel = kernel(&[buffer_type(ScalarType::U32)], false);
    assert!(
        submit(
            devices[1].as_ref(),
            &kernel,
            &[],
            [1; 3],
            &[Argument::Buffer(buffer.as_ref())]
        )
        .is_err()
    );
    let arguments = [Argument::Buffer(buffer.as_ref())];
    let launch = Launch::new(
        devices[0].as_ref(),
        &kernel,
        &[],
        [1; 3],
        [1; 3],
        &arguments,
    )
    .unwrap();
    unsafe {
        assert!(devices[1].launch(&launch).is_err());
    }
    assert!(devices[0].clone().buffer(ScalarType::Bool, &[2]).is_err());
    assert!(devices[0].clone().buffer(ScalarType::U32, &[]).is_err());
    assert_eq!(mock().loads, 0);
    assert!(mock().submissions.is_empty());
}

#[test]
fn errors_release_allocations_modules_and_restore_contexts() {
    let h = Harness::new();
    let devices = h.devices();
    let device = &devices[0];
    STACK.with(|s| s.borrow_mut().push(0xfeed));
    mock().fail = Some(("copy_to_device", 1));
    assert!(device.clone().buffer(ScalarType::U32, &[1]).is_err());
    {
        let m = mock();
        assert_eq!((m.allocations, m.frees), (1, 1));
    }
    assert_eq!(current(), 0xfeed);
    let kernel = kernel(&[], false);
    mock().fail = Some(("module_load", 218));
    let error = submit(device.as_ref(), &kernel, &[], [1; 3], &[]).unwrap_err();
    assert!(error.0.contains("mock PTX JIT diagnostic"));
    assert_eq!(current(), 0xfeed);
    mock().fail = Some(("module_function", 500));
    assert!(submit(device.as_ref(), &kernel, &[], [1; 3], &[]).is_err());
    assert_eq!(mock().unloads, 1);
    assert!(mock().modules.is_empty());
    mock().fail = Some(("function_attribute", 1));
    assert!(submit(device.as_ref(), &kernel, &[], [1; 3], &[]).is_err());
    assert_eq!(mock().unloads, 2);
    submit(device.as_ref(), &kernel, &[], [1; 3], &[]).unwrap();
    assert_eq!(mock().load_attempts, 4);
    mock().fail = Some(("stream_sync", 700));
    assert!(
        submit(device.as_ref(), &kernel, &[], [1; 3], &[])
            .unwrap_err()
            .0
            .contains("cuStreamSynchronize")
    );
    assert_eq!(mock().context_syncs, 1);
    assert_eq!(current(), 0xfeed);
    drop(devices);
    let m = mock();
    assert!(m.modules.is_empty());
    assert_eq!(m.loads, m.unloads);
    assert!(!m.destroyed_live_resources);
    let unload = m.trace.iter().rposition(|e| *e == "unload").unwrap();
    let destroy_stream = m.trace[..unload]
        .iter()
        .rposition(|e| *e == "destroy_stream");
    assert!(destroy_stream.is_some());
}

#[test]
fn device_function_and_parameter_limits_are_checked() {
    let h = Harness::new();
    let devices = h.devices();
    let device = devices[0].as_ref();
    let empty = kernel(&[], false);
    assert!(submit(device, &empty, &[], [1024, 2, 1], &[]).is_err());
    let launch = Launch::new(device, &empty, &[], [1, 65536, 1], [1; 3], &[]).unwrap();
    unsafe {
        assert!(device.launch(&launch).is_err());
    }
    assert_eq!(mock().loads, 0);
    mock().function_threads = 32;
    assert!(submit(device, &empty, &[], [64, 1, 1], &[]).is_err());
    assert!(mock().submissions.is_empty());
    submit(device, &empty, &[], [32, 1, 1], &[]).unwrap();
    let too_many = kernel(&vec![Type::Scalar(ScalarType::U32); 1025], false);
    assert!(
        submit(
            device,
            &too_many,
            &[],
            [1; 3],
            &vec![Argument::Scalar(Literal::U32(0)); 1025]
        )
        .is_err()
    );
    assert_eq!(mock().loads, 1);
}

#[test]
fn partial_discovery_failure_destroys_created_contexts() {
    let h = Harness::new();
    mock().fail = Some(("stream_create", 1));
    assert!(discover_with(h.driver.clone()).is_err());
    let m = mock();
    assert_eq!((m.created, m.destroyed), (1, 1));
    assert!(!m.destroyed_live_resources);
    assert_eq!(current(), 0);
}

#[test]
fn transfer_and_submission_errors_are_not_silenced() {
    let h = Harness::new();
    let devices = h.devices();
    let device = &devices[0];
    STACK.with(|s| s.borrow_mut().push(0xfeed));
    mock().fail = Some(("mem_alloc", 2));
    assert!(
        device
            .clone()
            .buffer(ScalarType::U32, &[1])
            .err()
            .unwrap()
            .0
            .contains("cuMemAlloc")
    );
    mock().fail = Some(("context_sync", 700));
    assert!(
        device
            .clone()
            .buffer(ScalarType::U32, &[1])
            .err()
            .unwrap()
            .0
            .contains("after upload")
    );
    {
        let m = mock();
        assert_eq!((m.allocations, m.frees), (1, 1));
    }
    let buffer = device.clone().buffer(ScalarType::U32, &[1]).unwrap();
    mock().fail = Some(("copy_to_host", 1));
    assert!(
        device
            .read(buffer.as_ref())
            .unwrap_err()
            .0
            .contains("cuMemcpyDtoH")
    );
    let kernel = kernel(&[], false);
    mock().fail = Some(("launch", 701));
    assert!(
        submit(device.as_ref(), &kernel, &[], [1; 3], &[])
            .unwrap_err()
            .0
            .contains("cuLaunchKernel")
    );
    assert_eq!(mock().syncs, 1);
    assert_eq!(current(), 0xfeed);
    assert!(mock().submissions.is_empty());
}

#[test]
fn shared_memory_limits_are_checked_before_and_after_jit() {
    let h = Harness::new();
    let devices = h.devices();
    let device = devices[0].as_ref();
    let base = kernel(&[], false);
    let mut module = base.ir().module().clone();
    module.kernels[0]
        .routine
        .header
        .variables
        .push(Variable::Storage {
            id: StorageId(0),
            element: ScalarType::U32,
            space: AddressSpace::Workgroup,
            length: ConstantOperand::Literal(12289),
        });
    let oversized = KernelHandle::new(Arc::new(module.validate().unwrap()), 0).unwrap();
    assert!(submit(device, &oversized, &[], [1; 3], &[]).is_err());
    assert_eq!(mock().loads, 0);
    mock().function_shared = 50000;
    assert!(
        submit(device, &base, &[], [1; 3], &[])
            .unwrap_err()
            .0
            .contains("compiled kernel resource limits")
    );
    assert_eq!(mock().loads, 1);
    assert!(mock().submissions.is_empty());
}

#[test]
fn compilation_caches_are_owned_by_each_context() {
    let h = Harness::new();
    let devices = h.devices();
    let kernel = kernel(&[], false);
    for device in &devices {
        submit(device.as_ref(), &kernel, &[], [1; 3], &[]).unwrap();
    }
    let m = mock();
    assert_eq!(m.loads, 2);
    assert_ne!(m.submissions[0].context, m.submissions[1].context);
}

struct AliasedBuffer {
    native: Arc<dyn BufferBackend>,
    owner: Arc<dyn DeviceBackend>,
    length: usize,
}
impl BufferBackend for AliasedBuffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self.native.as_any()
    }
    fn device(&self) -> Arc<dyn DeviceBackend> {
        self.owner.clone()
    }
    fn element_type(&self) -> ScalarType {
        self.native.element_type()
    }
    fn len(&self) -> usize {
        self.length
    }
}
#[test]
fn native_ownership_and_metadata_are_checked_after_downcasting() {
    let h = Harness::new();
    let devices = h.devices();
    let native = devices[1].clone().buffer(ScalarType::U32, &[1]).unwrap();
    let foreign = AliasedBuffer {
        native: native.clone(),
        owner: devices[0].clone(),
        length: 1,
    };
    assert!(devices[0].read(&foreign).is_err());
    let inflated = AliasedBuffer {
        native,
        owner: devices[1].clone(),
        length: 1024,
    };
    assert!(devices[1].read(&inflated).is_err());
}

extern "system" fn device_uuid(value: *mut [u8; 16], device: c_int) -> Status {
    if mock().failure("identity_uuid") != 0 {
        return 1;
    }
    unsafe {
        *value = [device as u8 + 1; 16];
    }
    0
}
extern "system" fn device_pci(value: *mut c_char, size: c_int, device: c_int) -> Status {
    if mock().failure("identity_pci") != 0 {
        return 1;
    }
    let text = format!("0000:{:02x}:00.0\0", device + 1);
    if text.len() > size as usize {
        return 1;
    }
    unsafe {
        ptr::copy_nonoverlapping(text.as_ptr(), value.cast(), text.len());
    }
    0
}
extern "system" fn device_luid(_: *mut c_char, _: *mut c_uint, _: c_int) -> Status {
    801
}
extern "system" fn device_memory(value: *mut usize, _: c_int) -> Status {
    if mock().failure("identity_memory") != 0 {
        return 1;
    }
    unsafe {
        *value = 16 * 1024 * 1024 * 1024;
    }
    0
}
#[test]
fn identity_queries_report_stable_hardware_without_changing_context_identity() {
    let h = Harness::new();
    let devices = h.devices();
    let first = devices[0].hardware_info();
    assert_eq!(first.uuid, Some([1; 16]));
    assert_eq!(
        first.pci,
        Some(PciAddress {
            domain: 0,
            bus: 1,
            device: 0,
            function: 0
        })
    );
    assert_eq!(first.vendor_id, Some(0x10de));
    assert_eq!(first.total_memory_bytes, Some(16 * 1024 * 1024 * 1024));
    assert!(first.luid.is_none());
    assert_ne!(first.uuid, devices[1].hardware_info().uuid);
    assert_eq!(devices[0].priority(), 200);
    let rediscovered = h.devices();
    assert_eq!(rediscovered[0].hardware_info(), first);
    assert_ne!(rediscovered[0].id(), devices[0].id());
}
#[test]
fn unavailable_identity_queries_do_not_prevent_discovery() {
    let mut h = Harness::new();
    {
        let driver = Arc::get_mut(&mut h.driver).unwrap();
        driver.device_uuid = None;
        driver.device_pci = None;
        driver.device_luid = None;
        driver.device_memory = None;
    }
    let devices = h.devices();
    assert_eq!(
        devices[0].hardware_info(),
        HardwareInfo {
            vendor_id: Some(0x10de),
            ..HardwareInfo::default()
        }
    );
}

#[test]
fn failed_optional_identity_queries_are_ignored_individually() {
    let h = Harness::new();
    mock().fail = Some(("identity_uuid", 1));
    let info = hardware_info(&h.driver, 0);
    assert!(info.uuid.is_none());
    assert!(info.pci.is_some());
    mock().fail = Some(("identity_pci", 1));
    let info = hardware_info(&h.driver, 0);
    assert!(info.pci.is_none());
    assert!(info.uuid.is_some());
    mock().fail = Some(("identity_memory", 1));
    let info = hardware_info(&h.driver, 0);
    assert!(info.total_memory_bytes.is_none());
    assert!(info.uuid.is_some());
}
#[test]
fn pci_parser_preserves_full_addresses_and_rejects_invalid_ones() {
    assert_eq!(
        pci_address("0001:ab:1f.7"),
        Some(PciAddress {
            domain: 1,
            bus: 171,
            device: 31,
            function: 7
        })
    );
    for value in [
        "ab:00.0",
        "0000:100:00.0",
        "0000:01:20.0",
        "0000:01:00.8",
        "0000:01:00.0:0",
        "invalid",
    ] {
        assert!(pci_address(value).is_none(), "{value}");
    }
}

extern "system" fn valid_luid(value: *mut c_char, node: *mut c_uint, device: c_int) -> Status {
    unsafe {
        ptr::copy_nonoverlapping([3u8; 8].as_ptr(), value.cast(), 8);
        *node = 1 << device;
    }
    0
}
#[test]
fn windows_adapter_identity_includes_its_node_mask() {
    let mut h = Harness::new();
    Arc::get_mut(&mut h.driver).unwrap().device_luid = Some(valid_luid);
    assert_eq!(
        hardware_info(&h.driver, 0).luid,
        Some(DeviceLuid {
            bytes: [3; 8],
            node_mask: 1
        })
    );
    assert_eq!(
        hardware_info(&h.driver, 1).luid,
        Some(DeviceLuid {
            bytes: [3; 8],
            node_mask: 2
        })
    );
}
