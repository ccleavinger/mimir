# mimir-codegen-ptx

Generate self-contained PTX text directly from validated and specialized Mimir
flat IR. The crate depends only on `mimir-ir` and `thiserror`; generating code
requires no CUDA toolkit, driver, GPU, LLVM, or libdevice.

```rust
use mimir_codegen_ptx::{compile, CompileError};
use mimir_ir::flat::ValidatedModule;

fn generate(module: &ValidatedModule) -> Result<String, Box<dyn std::error::Error>> {
    let specialized = module.specialize_kernel(0, &[])?;
    Ok(compile(module, &specialized)?)
}
```

The compiler covers every current flat IR instruction and value operation:
scalar and buffer copies; integer, float and boolean operations; numeric casts;
all built-ins and math intrinsics; global, local and workgroup memory; barriers;
branches, loop-carried values and parallel edge assignments; and specialized,
transitive device calls with scalar or void returns. Calls use native PTX
functions and are supported in loop headers. Unreachable structural blocks emit
`trap`. Specializations are checked against the module before emission.

## Output and ABI

- Output is deterministic PTX 6.0 targeting `sm_50`, with 64-bit addresses and
  the fixed visible entry name `mimir_kernel` (`ENTRY_NAME`). Each compilation
  produces one kernel and its reachable specialized device functions.
- Kernel arguments follow `Header::parameters` order. Buffers are 64-bit generic
  device pointers, without slice lengths. Scalars occupy 32 bits; bool parameters
  accept zero/nonzero, and bool returns/stores use 0/1.
- All buffer elements, including bool, occupy four bytes. Local storage is private
  to a thread; shared storage is owned by the kernel's block. Storage is uninitialized.
  Storage pointers are converted to/from generic addresses when passed to functions.
- Workgroup size comes from `%ntid`, so code is reusable across launch block sizes.
  Global IDs are `block_id * block_size + thread_id` on each axis.
- `bar.sync 0` supplies block synchronization and shared-memory visibility.
  Callers must ensure barriers are encountered uniformly, as with the IR itself.
- The returned `String` has no trailing NUL. Use `CString` before passing it to
  `cuModuleLoadDataEx`. Loading, launches, device limits and buffer ownership belong
  to the runtime; this crate does not implement a CUDA runtime.

## Floating-point behavior

Basic arithmetic uses explicit round-to-nearest instructions, division uses
`div.rn`, and FMA uses `fma.rn`. Integer casts preserve bits; integer/float casts
use PTX conversion instructions (float-to-integer truncates toward zero).
Overflow, invalid casts, and integer division by zero follow PTX behavior.
Floating remainder uses binary long division rather than a rounded quotient,
including for subnormal divisors and very large quotients. Float inequality
includes unordered (NaN) comparisons. Min/max use native PTX NaN behavior.

Transcendentals are **approximations**, not correctly rounded libm replacements.
Sin/cos, base-2 exponential and logarithm use PTX SFU instructions; other functions
use generated arithmetic, range reduction, and series. Trigonometric accuracy
outside ±100π follows PTX's best-effort behavior; tangent also magnifies error near
its poles. Compositions such as pow and hyperbolic functions accumulate SFU and
rounding error. Domain errors produce NaNs; sign-sensitive operations handle
signed zeros and infinities. No external math symbols or linking step is needed.
The [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
describes instruction-specific precision and exceptional-value behavior.

## Validation

`cargo test -p mimir-codegen-ptx` runs numerical checks using a CPU interpreter of
the generated scalar instructions, CFG copy-cycle regressions, specialization
checks, and coverage kernels for all IR operations. CPU numerical checks validate
the lowering algorithms; they do not model hardware SFU approximation errors.

To also assemble the coverage kernels without a GPU, put `ptxas` on `PATH` or set:

```sh
PTXAS=/path/to/ptxas cargo test -p mimir-codegen-ptx
```

Assembly checks are skipped if the tool is absent. Actual GPU execution still
requires a compatible NVIDIA driver and GPU.
