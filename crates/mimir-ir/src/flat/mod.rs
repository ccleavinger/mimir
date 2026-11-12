//! Language-neutral SSA IR with a standard-defined type vocabulary.
//!
//! # Layout and values
//!
//! [`Kernel`] and [`DeviceFunction`] contain a [`Routine`]: a [`Header`] followed
//! by a [`Body`] of structured basic blocks. Header parameter order defines the
//! invocation interface; constant order defines the independent u32 tuning
//! interface. Variables declare every SSA value and allocation. Each SSA value
//! is defined exactly once by an instruction or a block argument. All ID spaces
//! are routine-local except module-local function IDs. There are no type IDs.
//!
//! Operands reference parameters, constants, SSA values, storage, or immediate
//! scalar literals. Pure operations define values; stores, barriers and calls
//! are explicit instructions. A storage reference preserves its element type,
//! address space and permissions. Calls may reduce access permissions, but
//! cannot promote them. Device functions cannot capture kernel resources or
//! constants, and only kernels may own workgroup allocations.
//!
//! Workgroup dimensions remain launch configuration. The `WorkgroupSize`
//! builtin only reads that launch state; no workgroup size is stored in the IR.
//!
//! # Control flow
//!
//! [`Edge`] arguments bind destination block arguments simultaneously. Conditional
//! headers carry selection or loop [`Merge`] metadata. Loop backedges pass
//! through the declared continue target. Unreachable executable blocks, irreducible cycles,
//! branches entering constructs outside their headers, and selections that
//! reconverge before their merge are rejected. Constructs may return early or
//! exit to an enclosing merge/continue target. When a construct has no path to
//! its merge or continue target, that block is an empty `Unreachable` structural
//! placeholder. Entry blocks have no arguments
//! or predecessors. Device call graphs must be acyclic.
//!
//! # Specialization and backend handoff
//!
//! Build or deserialize a [`Module`], then call [`Module::validate`]. The returned
//! [`ValidatedModule`] provides immutable access to the original IR. Its
//! [`ValidatedModule::specialize_kernel`] method evaluates the constant DAG and
//! resolves storage lengths and all reachable device-function instances. It
//! returns metadata indexed by original IDs alongside the unchanged validated
//! IR, so backends can consume both without reconstructing instruction identity.
//!
//! Derived constants use checked u32 arithmetic; overflow, underflow and zero
//! divisors are errors. Allocations must have positive length. Every scalar uses
//! a four-byte memory slot; stored booleans use 0 or 1. Backends check their byte
//! limits through [`SpecializedRoutine::check_allocation_limits`] and apply any
//! additional target constraints. Local limits are per routine/invocation;
//! workgroup limits are per kernel/workgroup.
//!
//! Body integer add/subtract/multiply wrap to 32 bits. Integer division truncates
//! toward zero; remainder follows the dividend's sign. Integer zero divisors and
//! signed division overflow are invalid executions. F32 values use IEEE-754
//! binary32 bit patterns; floating arithmetic/intrinsic precision follows the
//! backend's documented guarantees. Numeric casts use value conversion, not bit
//! reinterpretation. Float-to-integer conversion truncates toward zero and is
//! invalid for non-finite or out-of-range values. Boolean conversions are not
//! implicit. Validation proves typing/structure, not bounds, races, arithmetic
//! preconditions or barrier uniformity. Storage starts uninitialized.
//!
//! # Semantic identity
//!
//! [`ValidatedModule::kernel_hash`] and [`ValidatedModule::function_hash`] use
//! SHA-256 over versioned canonical semantic content. Debug names, numeric IDs,
//! declaration-table order and physical block order do not affect identity.
//! Equal derived constant definitions are coalesced. Operand order, instruction
//! order and ordered interfaces remain significant; arbitrary algebraic or CFG
//! equivalence is not inferred. Caller identities include callee identities.
//!
//! [`SpecializedKernel::variant_key`] adds tuning values and explicit backend
//! target/options fingerprints. Runtime compilation and caching are separate
//! consumers of these keys. This module does not manage a runtime cache.
//!
//! Serde supports editable IR artifacts; deserialized modules require validation.
//! Canonical bytes use a versioned postcard schema for identity, not a finalized
//! GPU instruction encoding. Semantic/schema changes require a new identity
//! version. Legacy AST-shaped IR modules remain until frontend/backend migration.
mod identity;
mod specialize;
mod types;
mod validate;
pub use identity::{SemanticHash, VariantKey};
pub use specialize::{
    Allocation, FunctionInstance, SpecializationError, SpecializedKernel, SpecializedRoutine,
};
pub use types::*;
pub use validate::{ValidatedModule, ValidationError};
