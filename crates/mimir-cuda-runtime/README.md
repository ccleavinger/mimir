# mimir-cuda-runtime

Standalone CUDA backend for Mimir's core execution contracts. CUDA is loaded at
runtime through the Driver API; building requires no NVIDIA GPU, driver, CUDA
Toolkit, nvcc, LLVM or libdevice. Runtime dependencies are `mimir-runtime-core`,
`mimir-ir`, `mimir-codegen-ptx` and the small `libloading` crate.

```rust
use mimir_runtime_core::{Backend, GpuScalar};
use mimir_cuda_runtime::CudaBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for device in CudaBackend.discover()? {
        let buffer = device.clone().buffer(<u32 as GpuScalar>::TYPE, &[1, 2, 3])?;
        assert_eq!(device.read(buffer.as_ref())?, vec![1, 2, 3]);
    }
    Ok(())
}
```

The unified `mimir` facade enables this backend with the optional `cuda` feature.
Use `default-features = false, features = ["cuda"]` for CUDA without Vulkan.
Discovery supports 64-bit Linux and Windows with compute capability 5.0 or newer.
Missing drivers, no NVIDIA devices, and unsupported targets return no devices;
other driver initialization failures return errors. Windows loads `nvcuda.dll`;
Linux loads `libcuda.so.1`, never toolkit stub libraries.

CUDA reports a discovery priority of `200` (Vulkan reports `100`). Best-effort
UUID, full PCI address, Windows adapter/node ID and total-memory queries let the
unified runtime select one implementation of duplicate hardware. UUID v2 is
preferred where available to distinguish device partitions. Missing identity
symbols or failed metadata queries do not prevent execution. Standalone discovery
returns all CUDA contexts; duplicate resolution belongs to `mimir-runtime`.

Each discovered device owns an isolated context and nonblocking stream. Contexts
are pushed/popped on the calling thread, preserving the caller's current context.
Transfers and launches are synchronous and serialized per context; allocations
retain their context. All four scalar types use four-byte memory slots. Kernel
arguments follow PTX header order, with four-byte scalars and eight-byte pointers.
The conservative kernel argument limit is 4096 bytes.

The first launch JIT-compiles PTX through `cuModuleLoadDataEx` and caches its module
and entry point. Keys include kernel/callee identity, tuning values, device/driver
identity and compiler/ABI format. Block sizes do not affect emitted PTX, so launches
with different block sizes share compiled modules. Device and compiled-function
limits, shared storage and allocation ownership are checked before submission.
JIT failures include the driver's error log. Math precision follows the PTX
codegen crate's documented approximation policy.

Launches retain core's unsafe obligations: valid bounds/arithmetic, uniform
barriers and absence of races. This backend does not expose asynchronous streams,
primary-context sharing, external allocation imports or persistent disk caches.

`cargo test -p mimir-cuda-runtime` exercises an injected Driver API table without
hardware. Ignored hardware tests can be run on a compatible NVIDIA installation:

```sh
cargo test -p mimir-cuda-runtime -- --ignored
```
