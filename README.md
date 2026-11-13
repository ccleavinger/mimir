# Mimir

Mimir compiles Rust GPU kernels through a reusable AST into a language-neutral,
validated SSA IR. One runtime owns registration, devices, buffers, compilation,
and execution. The `mimir` crate is the public entry point.

```rust
use mimir::{launch, mimir_global, GpuArray};

#[mimir_global]
fn add(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = GpuArray::from_slice(&[1u32, 2, 3, 4])?;
    // Four invocations access four elements; this kernel has no races or barriers.
    unsafe {
        launch!(add<<<[1, 1, 1], [4, 1, 1]>>>(&values, 3u32))?;
    }
    assert_eq!(values.to_vec()?, [4, 5, 6, 7]);
    Ok(())
}
```

Grid dimensions count workgroups; workgroup dimensions count invocations per
workgroup. Launches are synchronous. Buffer and kernel handles retain their
owners; a buffer cannot be used on another device. All scalar memory slots,
including booleans, occupy four bytes. Typed buffer helpers encode that layout.

## Typed GPU storage

`GpuArray<T>` lives in `mimir-runtime` and is re-exported by `mimir`. It supports
`u32`, `i32`, `f32` and `bool`, including the IR's four-byte boolean slots.
Use `GpuArray::from_slice` or `GpuArray::from_iter` with the shared default device,
or `from_slice_on(&device, ...)` / `from_iter_on` with an explicit device.
`to_vec()` and `to_iter()` synchronously read back typed host values using the
allocation's owning device. Clones share the allocation; they do not copy it.
The underlying buffer is private, and `GpuArray::<T>::try_from(buffer)` checks
its element type. `as_buffer()` borrows the raw buffer; `into_buffer()` unwraps
it. Empty allocations are rejected by the existing buffer API.

`GpuBuffer` is an object-safe trait for borrowable GPU allocations. Implement
`as_buffer(&self) -> &Buffer` on a custom owner to use it directly with `launch!`.
The trait supplies `len`, `is_empty`, `element_type` and `device` metadata. Raw
`Buffer`, `GpuArray<T>`, references, `Box` and `Arc` wrappers implement it.
`&dyn GpuBuffer` also works as a kernel argument. It represents a whole backing
allocation; suballocation offsets and GPU slices are not currently supported.
Typed arrays enforce host element types, while runtime validation still checks
those types against each kernel's parameter interface.

## Shared default runtime

`mimir::MIMIR_RUNTIME` is a process-wide `LazyLock<Result<Runtime,
RegistrationError>>`. `mimir::default_runtime()` returns a shared reference to it.
Initialization and initialization errors are cached, and device discovery remains
lazy. `mimir::default_device()` clones the default device handle from that same
runtime. No additional GPU context is created on subsequent calls.

The kernel registry uses an internal `RwLock`. Uploads use `&self`, so
`mimir::default_runtime()?.register(hash, module)?` works from concurrent callers.
Validation and hashing happen before taking the write lock; handles are cloned
under a read lock, which is released before compilation or execution. There is
no global lock held across GPU work. This also gives future C wrappers a shared
initialization, registration and launch entry point without a new runtime owner.

`Runtime::new()` remains available for explicitly isolated contexts, while
`Runtime::default()` creates an empty registry for applications that only upload
IR. The default device is the first discovered device; it is not a mutable
global current-device selection.

## Embedded and uploaded IR

`#[mimir_global]` parses Rust into `mimir-ast`, validates it, lowers it to
`mimir-ir::flat`, validates the module, and emits versioned bytes into a `linkme`
distributed slice. The default runtime collects those descriptors once on first use. There are no
macro-generated sidecar files, runtime kernel-file reads, or compile-order
shared files. Each kernel emits a same-named `KernelHash` constant: the canonical
SHA-256 identity of its validated IR and reachable device functions, independent
of Rust paths, debug names, module indices and target backend. Equal kernels
share one registry entry even when embedded by different crates or uploaded
later. Backend-specific tuning and compilation fingerprints remain separate
variant-cache keys.

To group kernels and device functions, use a self-contained GPU module:

```rust
#[mimir::mimir_module]
mod math {
    #[device]
    fn twice(x: u32) -> u32 { return x * 2; }
    #[kernel]
    pub fn transform(values: &mut [u32]) {
        let i: u32 = global_id.x;
        values[i] = twice(values[i]);
    }
}
```

Device functions use module-local names, explicit scalar return types and
explicit `return` statements. Whole-module validation rejects unresolved calls,
argument mismatches and recursion. The Vulkan compiler specializes and inlines
the reachable function graph. Generic device functions and cross-module device
calls are currently rejected; kernel u32 const generics are supported.

For runtime uploads, compute the hash from validated IR and supply it during
registration. The same `KernelHash` type is used by Rust macros and uploads:

```rust,ignore
let validated = module.clone().validate()?;
let hash: mimir::KernelHash = validated.kernel_hash(0)?;
let handle = runtime.register(hash, module)?;
// Or: runtime.register_bytes(hash, &bytes)?;
let same_kernel = runtime.kernel(hash)?;
```

Registration validates and checks the supplied hash against the actual IR,
including on duplicate uploads. A wrong hash or invalid module leaves the
registry unchanged. Lookup uses a `HashMap<KernelHash, KernelHandle>`; names and
module positions are not public identifiers. Registering a hash from a module
with several kernels selects that kernel and retains its module and callees.
`runtime.kernel_count()` reports unique registered kernel identities.

`Module::to_bytes` / `Module::from_bytes` encode the transport format: the eight
bytes `MIMIRIR\0`, a little-endian u32 schema version, and exactly one postcard
module. Decoding rejects unknown versions and trailing bytes. Linked descriptors
carry the ordered kernel hashes for their embedded module; initialization checks
all of them before registering that descriptor. Existing handles remain valid
after the runtime is dropped.

`mimir::embed_ir!(&[hash_a, hash_b], STATIC_BYTES)` embeds an independently
produced artifact. Hashes must match the embedded module's kernel order. An empty
module uses `&[]`. IR can be produced without the Rust frontend or AST crate at
runtime.

## CUDA-style launches

```rust,ignore
unsafe {
    mimir::launch!(runtime, device;
        kernels::transform<N, { N + 1 }><<<grid, workgroup>>>(buffer, 2u32)
    )?;
}
```

The default form is `launch!(kernel<<<grid, workgroup>>>(arguments...))`.
Use `launch!(device; kernel<<<...>>>(...))` to select a device with the shared
runtime, or `launch!(runtime, device; kernel<<<...>>>(...))` for an isolated
runtime. Tuning arguments are optional.
Launches return `Result<(), LaunchError>` and require the same unsafe execution
obligations as `Device::launch`. Scalars (`u32`, `i32`, `f32`, `bool`), buffers,
references, `Literal` and existing `Argument` values are accepted. Unsuffixed
integer arguments default to Rust's `i32`; use `2u32` for an unsigned parameter.
Expressions are evaluated once and buffers are borrowed. The macro uses Logos
and chumsky for the launch envelope, then syn for Rust expression fragments.
Original token trees preserve spans and hygiene through the string-based lexer,
including calls generated by `macro_rules!`.

## Platforms and compilation

The default `vulkan` feature enables Vulkan on Linux, Android and Windows.
Vulkan 1.1 compute devices are supported; devices are discovered lazily and
retained per runtime. The optional `cuda` feature enables CUDA Driver API execution
on 64-bit Linux and Windows for compute capability 5.0 or newer. CUDA is loaded at
runtime: building requires no NVIDIA GPU, driver, CUDA Toolkit or nvcc. Missing
CUDA drivers and machines without NVIDIA devices yield no CUDA devices. Other
target systems can use registration with no backend and receive no devices.

```toml
# Registration and uploads without a GPU backend:
mimir = { path = "crates/mimir", default-features = false }
# Explicit Vulkan deployment:
# mimir = { path = "crates/mimir", default-features = false, features = ["vulkan"] }
# CUDA without Vulkan:
# mimir = { path = "crates/mimir", default-features = false, features = ["cuda"] }
```

Cargo features are additive: every dependency in a deployment must avoid enabling
backends it does not want. Backend dependencies are optional and target gated.
Future backends belong behind features of the same runtime, with the facade
remaining the application interface.

Compilation uses the flat IR's specialization and semantic variant keys. Each
device caches compiled kernels in memory, including tuning values, target and
driver identity and compiler format version in its fingerprints. Vulkan also
includes workgroup dimensions; CUDA reads them at launch and reuses modules
across block sizes.

Unified discovery reports each confidently identified physical GPU through its
highest-priority backend: CUDA reports `200`, Vulkan `100`, and larger `usize`
priorities win. Ties retain the first discovered implementation. Replacement
preserves physical discovery order rather than sorting unrelated GPUs by priority.
Discovery attempts every enabled backend even if another backend fails to initialize.
Usable devices are returned despite failures elsewhere. If no backend finds a device,
any backend failures are reported together; an empty list is returned when discovery
completes without errors but finds no devices.
UUIDs, Windows adapter/node IDs and full PCI domain/bus/device/function addresses
supply strong matches; conflicting identifiers prevent merging. A normalized model
name and total memory within 5% provide a fallback only when the correspondence is
unambiguous. Identical model names or memory totals alone never hide another GPU;
ambiguous matches and distinct device partitions are retained.

Backends expose these optional hints through `HardwareInfo` and `priority()` in
`mimir-runtime-core`; failed identity queries do not disable an otherwise usable
backend. `Device::discover_with_backends` applies the same selection to custom
providers. Selection applies to unified discovery (`Runtime::devices` and default
devices included); standalone platform discovery and explicit `Device::from_backend`
handles remain independent. Hardware identity never permits cross-context buffer
use: allocation ownership still uses the separate `DeviceId` context token.
A per-device lock coordinates compilation and queue submission. Launch limits,
argument types, buffer ownership, shared allocations and capabilities are
checked before dispatch. Persistent disk caches are not used.

Vulkan buffers currently use host-visible coherent memory, and execution uses a
synchronous queue. This provides a working portable baseline; dedicated device
memory, transfer queues and asynchronous submissions remain future performance
work. Floating-point operations follow Vulkan/SPIR-V device precision rules.
The backend reports unsupported pointer/block-argument patterns rather than
silently changing their semantics. Calls in loop headers are rejected; move the
call into a preceding block or the loop body.

CUDA uses isolated contexts, device memory, synchronous transfers and a private
stream. Generated PTX is JIT-compiled by the installed driver, with JIT diagnostics
included in compilation errors. Contexts are restored on the calling thread, and
allocations retain their context. CUDA supports all current IR operations; math
precision follows the [PTX codegen policy](crates/mimir-codegen-ptx/README.md).

IR validation proves type and control-flow structure. It does not prove bounds,
race freedom, arithmetic preconditions or barrier uniformity. `Device::launch`
is therefore unsafe and documents those caller obligations. SHA-256 identities
are content identifiers, not signatures. Cryptographic trust policy is deferred.

## Compiler pipeline

The Rust frontend has two separate stages. Chumsky parses original function-body
token trees into a source-spanned syntax tree with explicit operator precedence.
A resolver then assigns lexical IDs, resolves device calls and applies literal
typing. Parsing does not mutate symbol tables or emit IR. `syn` handles outer
Rust item signatures, attributes and literal decoding; macro bodies go directly
to the GPU grammar. The existing `parse_kernel(&syn::ItemFn)` and `compile_module`
interfaces remain available alongside token-preserving `SourceFunction` entry
points.

Validated AST lowering is divided into storage/declaration builders, expression
lowering and structured statement lowering. Statement sequences use explicit
continuations for branches, loop bodies, merges and continue targets, including
early returns. The backend specializes/inlines device routines, lays out the
structured CFG with an iterative work stack and indexed block lookup, then emits
SPIR-V through typed instruction dispatch. AST and IR are kept as structured
data rather than encoded as parser token streams. Recursive expression visitors
and the separate acyclic function-inlining pass remain ordinary compiler passes.

## Workspace and verification

- `mimir`: public facade, macros and typed host interfaces.
- `mimir-runtime-core`: backend contracts, kernel handles and shared launch validation.
- `mimir-vulkan-runtime`: standalone Vulkan devices, allocations and pipeline caching.
- `mimir-cuda-runtime`: dynamically loaded CUDA devices, transfers and PTX module caching.
- `mimir-runtime`: unified registration, typed handles and feature-selected execution.
- `mimir-ir`: flat IR, validation, specialization, identity and transport bytes.
- `mimir-ast`: reusable lexical AST, validation, passes and lowering.
- `mimir-rs-compiler` / `mimir-macros`: Rust parsing and compile-time embedding.
- `mimir-codegen-spirv`: validated flat IR to Vulkan SPIR-V, independent of runtime.
- `mimir-codegen-ptx`: self-contained PTX text generation, independent of runtime.

Platform runtimes depend on `mimir-runtime-core`, never on the unified runtime.
The core has no platform or linked-registration dependencies; disabling the unified
runtime's default features excludes Vulkan and SPIR-V dependencies. Independent
backends can be supplied through `Device::from_backend`; contracts are also
re-exported as `mimir_runtime::backend`.

The old omni-runtime, legacy runtime traits and AST-shaped IR are retired.
Both matrix multiplication and MNIST examples use the public typed arrays and
the library-owned lazy runtime; there is no example-only buffer wrapper.

```sh
cargo test --workspace
cargo check -p mimir --no-default-features
cargo check -p mimir --no-default-features --features cuda
cargo test -p mimir --test kernel -- --ignored
cargo test -p mimir --test launch -- --ignored
cargo test -p mimir --test default_runtime -- --ignored
cargo test -p mimir --test array -- --ignored
cargo run -p matmul
# CUDA only (including Jetson):
cargo run -p matmul --release --no-default-features --features cuda
```

The matmul example defaults to Vulkan. Its `cuda` and `vulkan` features select
backends; use `--no-default-features --features cuda` to avoid Vulkan initialization.

The ignored test requires a Vulkan compute device. Shader tests invoke
`spirv-val --target-env vulkan1.1` when installed. Matrix multiplication checks
both naive and shared-memory implementations against a CPU reference, and
prints warm pipeline timings. Initial local debug measurements for 64³ were
1.04 ms naive, 0.97 ms tiled and 2.02 ms CPU; these are smoke measurements, not a
performance guarantee. The previous implementation did not build at the start
of migration, so a trustworthy before/after benchmark is unavailable.

The `measure` example records current hash verification/registration and
compilation costs: `cargo run -p mimir --example measure --release`. Run with
`--no-default-features` for a registration-only binary. Measurements are specific
to the machine and include synchronous submission and synchronization overhead.
