//! The small, explicitly versioned CUDA Driver ABI used by this backend.
//! No headers, SDK discovery, bindgen, static CUDA linkage or build script.
use libloading::Library;
use mimir_runtime_core::{ExecutionError, ExecutionResult};
use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    sync::Arc,
};

pub(crate) type Handle = *mut c_void;
pub(crate) type DevicePtr = u64;
pub(crate) type Status = c_int;
pub(crate) const NO_DEVICE: Status = 100;
pub(crate) const JIT_ERROR_LOG_BUFFER: c_int = 5;
pub(crate) const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_int = 6;
pub(crate) const MAX_THREADS_PER_BLOCK: c_int = 1;
pub(crate) const MAX_BLOCK_DIM: [c_int; 3] = [2, 3, 4];
pub(crate) const MAX_GRID_DIM: [c_int; 3] = [5, 6, 7];
pub(crate) const MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub(crate) const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub(crate) const COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub(crate) const FUNCTION_MAX_THREADS: c_int = 0;
pub(crate) const FUNCTION_SHARED_BYTES: c_int = 1;
pub(crate) const STREAM_NON_BLOCKING: c_uint = 1;

pub(crate) type DeviceUuidFn = unsafe extern "system" fn(*mut [u8; 16], c_int) -> Status;
pub(crate) type DevicePciFn = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> Status;
pub(crate) type DeviceLuidFn = unsafe extern "system" fn(*mut c_char, *mut c_uint, c_int) -> Status;
pub(crate) type DeviceMemoryFn = unsafe extern "system" fn(*mut usize, c_int) -> Status;

// `system` is CUDAAPI (stdcall on 32-bit Windows, C on supported 64-bit targets).
macro_rules! driver {
    ($($name:ident: $symbol:literal ($($arg:ty),*) ),* $(,)?) => {
        pub(crate) struct Driver {
            // The library must outlive every copied function pointer.
            pub(crate) _library: Option<Library>,
            // Identity queries are best-effort and must not prevent execution
            // on a driver that does not expose one of these optional symbols.
            pub(crate) device_uuid: Option<DeviceUuidFn>,
            pub(crate) device_pci: Option<DevicePciFn>,
            pub(crate) device_luid: Option<DeviceLuidFn>,
            pub(crate) device_memory: Option<DeviceMemoryFn>,
            $(pub(crate) $name: unsafe extern "system" fn($($arg),*) -> Status,)*
        }
        impl Driver {
            pub(crate) fn load() -> ExecutionResult<Option<Arc<Self>>> {
                #[cfg(target_os = "linux")]
                let filename = "libcuda.so.1";
                #[cfg(target_os = "windows")]
                let filename = "nvcuda.dll";
                // No driver installation is a normal discovery outcome. Do not
                // fall back to toolkit stub libraries (libcuda.so).
                let library = match unsafe { Library::new(filename) } {
                    Ok(library) => library,
                    Err(_) => return Ok(None),
                };
                // Use the v2 ABI explicitly where CUDA header aliases otherwise
                // select a different signature depending on toolkit version.
                $(let $name = unsafe {
                    *library.get::<unsafe extern "system" fn($($arg),*) -> Status>($symbol)
                        .map_err(|e| ExecutionError(format!("CUDA driver symbol {}: {e}", String::from_utf8_lossy($symbol))))?
                };)*
                let device_uuid = unsafe {
                    library.get::<DeviceUuidFn>(b"cuDeviceGetUuid_v2\0")
                        .or_else(|_| library.get::<DeviceUuidFn>(b"cuDeviceGetUuid\0")).ok().map(|s| *s)
                };
                let device_pci = unsafe { library.get::<DevicePciFn>(b"cuDeviceGetPCIBusId\0").ok().map(|s| *s) };
                let device_luid = unsafe { library.get::<DeviceLuidFn>(b"cuDeviceGetLuid\0").ok().map(|s| *s) };
                let device_memory = unsafe { library.get::<DeviceMemoryFn>(b"cuDeviceTotalMem_v2\0").ok().map(|s| *s) };
                Ok(Some(Arc::new(Self { _library: Some(library), device_uuid, device_pci, device_luid, device_memory, $($name,)* })))
            }
        }
    };
}

driver! {
    init: b"cuInit\0"(c_uint),
    driver_version: b"cuDriverGetVersion\0"(*mut c_int),
    device_count: b"cuDeviceGetCount\0"(*mut c_int),
    device_get: b"cuDeviceGet\0"(*mut c_int, c_int),
    device_name: b"cuDeviceGetName\0"(*mut c_char, c_int, c_int),
    device_attribute: b"cuDeviceGetAttribute\0"(*mut c_int, c_int, c_int),
    context_create: b"cuCtxCreate_v2\0"(*mut Handle, c_uint, c_int),
    context_destroy: b"cuCtxDestroy_v2\0"(Handle),
    context_push: b"cuCtxPushCurrent_v2\0"(Handle),
    context_pop: b"cuCtxPopCurrent_v2\0"(*mut Handle),
    context_sync: b"cuCtxSynchronize\0"(),
    stream_create: b"cuStreamCreate\0"(*mut Handle, c_uint),
    stream_destroy: b"cuStreamDestroy_v2\0"(Handle),
    stream_sync: b"cuStreamSynchronize\0"(Handle),
    mem_alloc: b"cuMemAlloc_v2\0"(*mut DevicePtr, usize),
    mem_free: b"cuMemFree_v2\0"(DevicePtr),
    copy_to_device: b"cuMemcpyHtoD_v2\0"(DevicePtr, *const c_void, usize),
    copy_to_host: b"cuMemcpyDtoH_v2\0"(*mut c_void, DevicePtr, usize),
    module_load: b"cuModuleLoadDataEx\0"(*mut Handle, *const c_void, c_uint, *mut c_int, *mut *mut c_void),
    module_unload: b"cuModuleUnload\0"(Handle),
    module_function: b"cuModuleGetFunction\0"(*mut Handle, Handle, *const c_char),
    function_attribute: b"cuFuncGetAttribute\0"(*mut c_int, c_int, Handle),
    launch: b"cuLaunchKernel\0"(Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle, *mut *mut c_void, *mut *mut c_void),
    error_name: b"cuGetErrorName\0"(Status, *mut *const c_char),
    error_string: b"cuGetErrorString\0"(Status, *mut *const c_char),
}

impl Driver {
    pub(crate) fn check(&self, status: Status, operation: &str) -> ExecutionResult<()> {
        if status == 0 {
            return Ok(());
        }
        let mut name = std::ptr::null();
        let mut description = std::ptr::null();
        // CUDA supplies static NUL-terminated diagnostic strings.
        let name = unsafe {
            if (self.error_name)(status, &mut name) == 0 && !name.is_null() {
                CStr::from_ptr(name).to_string_lossy().into_owned()
            } else {
                "CUDA_ERROR_UNKNOWN".into()
            }
        };
        let description = unsafe {
            if (self.error_string)(status, &mut description) == 0 && !description.is_null() {
                CStr::from_ptr(description).to_string_lossy().into_owned()
            } else {
                "no driver description".into()
            }
        };
        Err(ExecutionError(format!(
            "CUDA {operation}: {name} ({status}): {description}"
        )))
    }
    pub(crate) fn attribute(&self, device: c_int, attribute: c_int) -> ExecutionResult<u32> {
        let mut value = 0;
        self.check(
            unsafe { (self.device_attribute)(&mut value, attribute, device) },
            "cuDeviceGetAttribute",
        )?;
        u32::try_from(value)
            .map_err(|_| ExecutionError("CUDA driver returned a negative device limit".into()))
    }
}
