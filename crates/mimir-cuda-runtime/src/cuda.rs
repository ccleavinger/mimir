use crate::driver::*;
use mimir_ir::flat::{Literal, ScalarType, VariantKey};
use mimir_runtime_core::{
    Argument, BufferBackend, DeviceBackend, DeviceId, DeviceLuid, ExecutionError, ExecutionResult,
    HardwareInfo, Launch, PciAddress, validate_buffer_owner, validate_buffer_values,
};
use std::{
    collections::HashMap,
    ffi::{CString, c_void},
    marker::PhantomData,
    ptr,
    rc::Rc,
    sync::{Arc, Mutex},
};

fn err(message: impl Into<String>) -> ExecutionError {
    ExecutionError(format!("CUDA {}", message.into()))
}

struct Context {
    driver: Arc<Driver>,
    raw: Handle,
}
// CUDA contexts may be made current on different threads. Each operation pushes
// this context on that thread's stack; device submissions are serialized below.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    fn create(driver: Arc<Driver>, device: i32) -> ExecutionResult<Arc<Self>> {
        let mut raw = ptr::null_mut();
        driver.check(
            unsafe { (driver.context_create)(&mut raw, 0, device) },
            "cuCtxCreate_v2",
        )?;
        let context = Arc::new(Self { driver, raw });
        // cuCtxCreate pushes the newly-created context; restore the caller's
        // previous context before exposing it to other threads.
        let mut popped = ptr::null_mut();
        context.driver.check(
            unsafe { (context.driver.context_pop)(&mut popped) },
            "cuCtxPopCurrent_v2 after creation",
        )?;
        if popped != raw {
            return Err(err("context creation did not restore the expected stack"));
        }
        Ok(context)
    }

    fn with_current<T>(
        &self,
        operation: impl FnOnce() -> ExecutionResult<T>,
    ) -> ExecutionResult<T> {
        self.driver.check(
            unsafe { (self.driver.context_push)(self.raw) },
            "cuCtxPushCurrent_v2",
        )?;
        let mut guard = CurrentContext {
            context: self,
            active: true,
            _thread: PhantomData,
        };
        let result = operation();
        let restored = guard.pop();
        match result {
            Ok(value) => {
                restored?;
                Ok(value)
            }
            Err(error) => Err(error),
        }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.driver.context_destroy)(self.raw);
        }
    }
}
struct CurrentContext<'a> {
    context: &'a Context,
    active: bool,
    _thread: PhantomData<Rc<()>>,
}
impl CurrentContext<'_> {
    fn pop(&mut self) -> ExecutionResult<()> {
        self.active = false;
        let mut popped = ptr::null_mut();
        self.context.driver.check(
            unsafe { (self.context.driver.context_pop)(&mut popped) },
            "cuCtxPopCurrent_v2",
        )?;
        if popped != self.context.raw {
            return Err(err("unexpected context on the calling thread's stack"));
        }
        Ok(())
    }
}
impl Drop for CurrentContext<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.pop();
        }
    }
}

struct Stream {
    context: Arc<Context>,
    raw: Handle,
}
// Access is serialized by the device mutex; destruction occurs with its context current.
unsafe impl Send for Stream {}
impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.context.with_current(|| {
            unsafe {
                (self.context.driver.stream_sync)(self.raw);
                (self.context.driver.stream_destroy)(self.raw);
            }
            Ok(())
        });
    }
}
struct CompiledKernel {
    context: Arc<Context>,
    module: Handle,
    function: Handle,
    max_threads: u32,
    shared_bytes: u32,
}
// Module/function handles are accessed only while the device mutex is held.
unsafe impl Send for CompiledKernel {}
impl Drop for CompiledKernel {
    fn drop(&mut self) {
        let _ = self.context.with_current(|| {
            unsafe {
                (self.context.driver.module_unload)(self.module);
            }
            Ok(())
        });
    }
}
struct State {
    // Drop the stream (synchronize/destroy) before unloading its kernel modules.
    stream: Stream,
    kernels: HashMap<VariantKey, CompiledKernel>,
}
struct Limits {
    threads: u32,
    block: [u32; 3],
    grid: [u32; 3],
    shared: u32,
}

/// An isolated CUDA context with a private stream and in-memory module cache.
/// Buffers retain this context; no process-wide primary context is changed.
pub struct CudaDevice {
    id: DeviceId,
    hardware: HardwareInfo,
    name: String,
    target: String,
    limits: Limits,
    context: Arc<Context>,
    state: Mutex<State>,
}
struct Allocation {
    context: Arc<Context>,
    pointer: DevicePtr,
    bytes: usize,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // No device lock here: allocation cleanup also runs on upload errors
        // while the caller already holds that lock. Retained borrows prevent
        // freeing an allocation still used by a synchronous submission.
        let _ = self.context.with_current(|| {
            unsafe {
                (self.context.driver.mem_free)(self.pointer);
            }
            Ok(())
        });
    }
}
/// A CUDA device allocation using Mimir's four-byte element representation.
pub struct CudaBuffer {
    // Free the allocation before releasing the owning device's final reference.
    allocation: Allocation,
    device: Arc<CudaDevice>,
    element: ScalarType,
    length: usize,
}

pub(crate) fn discover() -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
    let Some(driver) = Driver::load()? else {
        return Ok(Vec::new());
    };
    discover_with(driver)
}

fn discover_with(driver: Arc<Driver>) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
    let status = unsafe { (driver.init)(0) };
    if status == NO_DEVICE {
        return Ok(Vec::new());
    }
    driver.check(status, "cuInit")?;
    let mut count = 0;
    let mut version = 0;
    let status = unsafe { (driver.device_count)(&mut count) };
    if status == NO_DEVICE {
        return Ok(Vec::new());
    }
    driver.check(status, "cuDeviceGetCount")?;
    driver.check(
        unsafe { (driver.driver_version)(&mut version) },
        "cuDriverGetVersion",
    )?;
    let mut devices: Vec<Arc<dyn DeviceBackend>> = Vec::new();
    for ordinal in 0..count {
        let mut device = 0;
        driver.check(
            unsafe { (driver.device_get)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let major = driver.attribute(device, COMPUTE_CAPABILITY_MAJOR)?;
        let minor = driver.attribute(device, COMPUTE_CAPABILITY_MINOR)?;
        if major < 5 {
            continue;
        } // PTX emitter baseline is sm_50.
        let mut name = [0u8; 256];
        driver.check(
            unsafe { (driver.device_name)(name.as_mut_ptr().cast(), name.len() as i32, device) },
            "cuDeviceGetName",
        )?;
        let end = name.iter().position(|c| *c == 0).unwrap_or(name.len());
        let name = String::from_utf8_lossy(&name[..end]).into_owned();
        let limits = Limits {
            threads: driver.attribute(device, MAX_THREADS_PER_BLOCK)?,
            block: [
                driver.attribute(device, MAX_BLOCK_DIM[0])?,
                driver.attribute(device, MAX_BLOCK_DIM[1])?,
                driver.attribute(device, MAX_BLOCK_DIM[2])?,
            ],
            grid: [
                driver.attribute(device, MAX_GRID_DIM[0])?,
                driver.attribute(device, MAX_GRID_DIM[1])?,
                driver.attribute(device, MAX_GRID_DIM[2])?,
            ],
            shared: driver.attribute(device, MAX_SHARED_MEMORY_PER_BLOCK)?,
        };
        let hardware = hardware_info(&driver, device);
        let context = Context::create(driver.clone(), device)?;
        let stream = context.with_current(|| {
            let mut raw = ptr::null_mut();
            driver.check(
                unsafe { (driver.stream_create)(&mut raw, STREAM_NON_BLOCKING) },
                "cuStreamCreate",
            )?;
            Ok(Stream {
                context: context.clone(),
                raw,
            })
        })?;
        devices.push(Arc::new(CudaDevice {
            id: DeviceId::new(),
            hardware,
            name,
            target: format!("cuda:driver{version}:device{ordinal}:sm{major}{minor}"),
            limits,
            context,
            state: Mutex::new(State {
                stream,
                kernels: HashMap::new(),
            }),
        }));
    }
    Ok(devices)
}

fn pci_address(text: &str) -> Option<PciAddress> {
    let mut parts = text.split(':');
    let domain = u32::from_str_radix(parts.next()?, 16).ok()?;
    let bus = u32::from_str_radix(parts.next()?, 16).ok()?;
    let (device, function) = parts.next()?.split_once('.')?;
    if parts.next().is_some() {
        return None;
    }
    let address = PciAddress {
        domain,
        bus,
        device: u32::from_str_radix(device, 16).ok()?,
        function: u32::from_str_radix(function, 16).ok()?,
    };
    address.is_valid().then_some(address)
}
fn hardware_info(driver: &Driver, device: i32) -> HardwareInfo {
    let uuid = driver.device_uuid.and_then(|query| {
        let mut bytes = [0u8; 16];
        (unsafe { query(&mut bytes, device) } == 0 && bytes != [0; 16]).then_some(bytes)
    });
    let pci = driver.device_pci.and_then(|query| {
        let mut text = [0u8; 32];
        if unsafe { query(text.as_mut_ptr().cast(), text.len() as i32, device) } != 0 {
            return None;
        }
        let end = text.iter().position(|b| *b == 0).unwrap_or(text.len());
        pci_address(std::str::from_utf8(&text[..end]).ok()?)
    });
    let luid = driver.device_luid.and_then(|query| {
        let mut bytes = [0u8; 8];
        let mut node_mask = 0;
        (unsafe { query(bytes.as_mut_ptr().cast(), &mut node_mask, device) } == 0
            && bytes != [0; 8]
            && node_mask.count_ones() == 1)
            .then_some(DeviceLuid { bytes, node_mask })
    });
    let total_memory_bytes = driver.device_memory.and_then(|query| {
        let mut bytes = 0usize;
        (unsafe { query(&mut bytes, device) } == 0 && bytes > 0).then_some(bytes as u64)
    });

    HardwareInfo {
        uuid,
        pci,
        luid,
        vendor_id: Some(0x10de),
        total_memory_bytes,
    }
}

impl CudaDevice {
    fn native_buffer<'a>(&self, buffer: &'a dyn BufferBackend) -> ExecutionResult<&'a CudaBuffer> {
        validate_buffer_owner(self, buffer)?;
        let native = buffer
            .as_any()
            .downcast_ref::<CudaBuffer>()
            .ok_or_else(|| err("buffer is not a CUDA allocation"))?;
        if !ptr::eq(self, native.device.as_ref()) {
            return Err(err("buffer belongs to another context"));
        }
        if native.element != buffer.element_type() || native.length != buffer.len() {
            return Err(err("buffer metadata does not match CUDA allocation"));
        }
        Ok(native)
    }

    fn compile(&self, launch: &Launch<'_>) -> ExecutionResult<CompiledKernel> {
        let ptx = mimir_codegen_ptx::compile(launch.kernel().ir(), launch.specialized())
            .map_err(|e| err(e.to_string()))?;
        let ptx = CString::new(ptx).map_err(|e| err(e.to_string()))?;
        let driver = &self.context.driver;
        let mut module = ptr::null_mut();
        let mut log = [0u8; 8192];
        let mut options = [JIT_ERROR_LOG_BUFFER, JIT_ERROR_LOG_BUFFER_SIZE_BYTES];
        // CUDA encodes numeric JIT option values directly in these pointer slots.
        let mut values = [log.as_mut_ptr().cast::<c_void>(), log.len() as *mut c_void];
        let status = unsafe {
            (driver.module_load)(
                &mut module,
                ptx.as_ptr().cast(),
                options.len() as u32,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if let Err(error) = driver.check(status, "cuModuleLoadDataEx") {
            let end = log.iter().position(|b| *b == 0).unwrap_or(log.len());
            return Err(ExecutionError(format!(
                "{}; PTX JIT log: {}",
                error.0,
                String::from_utf8_lossy(&log[..end])
            )));
        }
        let mut compiled = CompiledKernel {
            context: self.context.clone(),
            module,
            function: ptr::null_mut(),
            max_threads: 0,
            shared_bytes: 0,
        };
        let entry = CString::new(mimir_codegen_ptx::ENTRY_NAME).unwrap();
        driver.check(
            unsafe { (driver.module_function)(&mut compiled.function, module, entry.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        let mut threads = 0;
        let mut shared = 0;
        driver.check(
            unsafe {
                (driver.function_attribute)(&mut threads, FUNCTION_MAX_THREADS, compiled.function)
            },
            "cuFuncGetAttribute(max threads)",
        )?;
        driver.check(
            unsafe {
                (driver.function_attribute)(&mut shared, FUNCTION_SHARED_BYTES, compiled.function)
            },
            "cuFuncGetAttribute(shared bytes)",
        )?;
        compiled.max_threads =
            u32::try_from(threads).map_err(|_| err("negative function thread limit"))?;
        compiled.shared_bytes =
            u32::try_from(shared).map_err(|_| err("negative function shared-memory size"))?;
        Ok(compiled)
    }
}

impl DeviceBackend for CudaDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn id(&self) -> DeviceId {
        self.id.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn backend_name(&self) -> &'static str {
        "cuda"
    }

    fn priority(&self) -> usize {
        200
    }

    fn hardware_info(&self) -> HardwareInfo {
        self.hardware.clone()
    }

    fn buffer(
        self: Arc<Self>,
        element: ScalarType,
        values: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        validate_buffer_values(element, values)?;
        let bytes = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| err("buffer size overflow"))?;
        let allocation = {
            let _state = self
                .state
                .lock()
                .map_err(|_| err("device mutex poisoned"))?;
            self.context.with_current(|| {
                let driver = &self.context.driver;
                let mut pointer = 0;
                driver.check(
                    unsafe { (driver.mem_alloc)(&mut pointer, bytes) },
                    "cuMemAlloc_v2",
                )?;
                let allocation = Allocation {
                    context: self.context.clone(),
                    pointer,
                    bytes,
                };
                driver.check(
                    unsafe { (driver.copy_to_device)(pointer, values.as_ptr().cast(), bytes) },
                    "cuMemcpyHtoD_v2",
                )?;
                // A pageable host upload may return after staging but before
                // DMA finishes. Our nonblocking launch stream does not wait on
                // the legacy copy stream, so explicitly complete the transfer.
                driver.check(
                    unsafe { (driver.context_sync)() },
                    "cuCtxSynchronize after upload",
                )?;
                Ok(allocation)
            })?
        };
        Ok(Arc::new(CudaBuffer {
            allocation,
            device: self,
            element,
            length: values.len(),
        }))
    }

    fn read(&self, buffer: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        let buffer = self.native_buffer(buffer)?;
        let _state = self
            .state
            .lock()
            .map_err(|_| err("device mutex poisoned"))?;
        self.context.with_current(|| {
            let mut values = vec![0u32; buffer.length];
            self.context.driver.check(
                unsafe {
                    (self.context.driver.copy_to_host)(
                        values.as_mut_ptr().cast(),
                        buffer.allocation.pointer,
                        buffer.allocation.bytes,
                    )
                },
                "cuMemcpyDtoH_v2",
            )?;
            Ok(values)
        })
    }

    unsafe fn launch(&self, launch: &Launch<'_>) -> ExecutionResult<()> {
        launch.validate_device(self)?;
        let grid = launch.grid();
        let block = launch.workgroup();
        let threads = block.iter().map(|n| *n as u64).product::<u64>(); // checked by Launch
        if threads > u64::from(self.limits.threads)
            || (0..3).any(|i| block[i] > self.limits.block[i] || grid[i] > self.limits.grid[i])
        {
            return Err(err("launch exceeds device dimensions"));
        }
        launch
            .specialized()
            .kernel
            .check_allocation_limits(u64::MAX, u64::from(self.limits.shared))
            .map_err(|e| err(e.to_string()))?;
        let mut arguments = Vec::with_capacity(launch.arguments().len());
        let mut parameter_bytes = 0usize;
        for argument in launch.arguments() {
            let value = match argument {
                Argument::Scalar(literal) => PackedArgument::Scalar(match literal {
                    Literal::Bool(v) => u32::from(*v),
                    Literal::I32(v) => *v as u32,
                    Literal::U32(v) | Literal::F32(v) => *v,
                }),
                Argument::Buffer(buffer) => {
                    PackedArgument::Buffer(self.native_buffer(*buffer)?.allocation.pointer)
                }
            };
            let size = match value {
                PackedArgument::Scalar(_) => 4,
                PackedArgument::Buffer(_) => 8,
            };
            parameter_bytes = parameter_bytes
                .checked_add(size - 1)
                .map(|n| n & !(size - 1))
                .and_then(|n| n.checked_add(size))
                .ok_or_else(|| err("kernel parameter size overflow"))?;
            arguments.push(value);
        }
        // Conservative ABI limit valid on every supported (sm_50+) architecture.
        if parameter_bytes > 4096 {
            return Err(err(
                "kernel parameters exceed the supported 4096-byte limit",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| err("device mutex poisoned"))?;
        self.context.with_current(|| {
            let key = launch.specialized().variant_key(
                self.target.as_bytes(),
                concat!("mimir.ptx.v1:ptx6.0:sm50:abi1:", env!("CARGO_PKG_VERSION")).as_bytes(),
            );
            if let std::collections::hash_map::Entry::Vacant(entry) = state.kernels.entry(key) {
                entry.insert(self.compile(launch)?);
            }
            let kernel = &state.kernels[&key];
            if threads > u64::from(kernel.max_threads) || kernel.shared_bytes > self.limits.shared {
                return Err(err("launch exceeds compiled kernel resource limits"));
            }
            let mut parameters: Vec<*mut c_void> = arguments
                .iter_mut()
                .map(|argument| match argument {
                    PackedArgument::Scalar(value) => (value as *mut u32).cast(),
                    PackedArgument::Buffer(value) => (value as *mut u64).cast(),
                })
                .collect();
            let parameters = if parameters.is_empty() {
                ptr::null_mut()
            } else {
                parameters.as_mut_ptr()
            };
            let driver = &self.context.driver;
            let launched = driver.check(
                unsafe {
                    (driver.launch)(
                        kernel.function,
                        grid[0],
                        grid[1],
                        grid[2],
                        block[0],
                        block[1],
                        block[2],
                        0,
                        state.stream.raw,
                        parameters,
                        ptr::null_mut(),
                    )
                },
                "cuLaunchKernel",
            );
            let synced = driver.check(
                unsafe { (driver.stream_sync)(state.stream.raw) },
                "cuStreamSynchronize",
            );
            if synced.is_err() {
                unsafe {
                    (driver.context_sync)();
                }
            }
            launched.and(synced)
        })
    }
}

enum PackedArgument {
    Scalar(u32),
    Buffer(u64),
}

impl BufferBackend for CudaBuffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Arc<dyn DeviceBackend> {
        self.device.clone()
    }

    fn element_type(&self) -> ScalarType {
        self.element
    }

    fn len(&self) -> usize {
        self.length
    }
}

#[cfg(test)]
mod tests;
