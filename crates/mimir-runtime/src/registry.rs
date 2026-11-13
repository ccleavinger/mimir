use mimir_ir::flat::{Module, ValidatedModule, ValidationError};
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};
use thiserror::Error;

/// Platform-independent identity of canonical kernel IR, including its callees.
pub use mimir_ir::flat::SemanticHash as KernelHash;

/// A module embedded in the executable, with compile-time kernel identities.
pub struct LinkedModule {
    pub hashes: &'static [KernelHash],
    pub bytes: &'static [u8],
}
#[linkme::distributed_slice]
pub static LINKED_MODULES: [LinkedModule];

#[derive(Debug, Clone, Error)]
pub enum RegistrationError {
    #[error("kernel is not registered: {0:?}")]
    MissingKernel(KernelHash),
    #[error(transparent)]
    Artifact(#[from] mimir_ir::flat::ArtifactError),
    #[error(transparent)]
    Invalid(#[from] ValidationError),
    #[error("supplied kernel hash does not match any kernel in the validated IR: {0:?}")]
    HashMismatch(KernelHash),
    #[error("linked kernel hashes do not match the embedded IR")]
    LinkedHashesMismatch,
}

/// Retains validated module storage; module indices are backend-local metadata,
/// never registry keys or public kernel identifiers.
#[derive(Clone)]
pub struct KernelHandle {
    module: Arc<ValidatedModule>,
    index: usize,
    hash: KernelHash,
}
impl KernelHandle {
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
#[derive(Default)]
pub struct Runtime {
    kernels: RwLock<HashMap<KernelHash, KernelHandle>>,
    pub(crate) devices: std::sync::OnceLock<Result<Vec<crate::Device>, String>>,
}
impl Runtime {
    pub fn new() -> Result<Self, RegistrationError> {
        let mut runtime = Self::default();
        for descriptor in LINKED_MODULES {
            let validated = Arc::new(Module::from_bytes(descriptor.bytes)?.validate()?);
            let hashes = (0..validated.module().kernels.len())
                .map(|i| validated.kernel_hash(i))
                .collect::<Result<Vec<_>, _>>()?;
            if hashes != descriptor.hashes {
                return Err(RegistrationError::LinkedHashesMismatch);
            }
            for (index, hash) in hashes.into_iter().enumerate() {
                runtime
                    .kernels
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .entry(hash)
                    .or_insert_with(|| KernelHandle {
                        module: validated.clone(),
                        index,
                        hash,
                    });
            }
        }
        Ok(runtime)
    }
    pub fn kernel(&self, hash: KernelHash) -> Result<KernelHandle, RegistrationError> {
        self.kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&hash)
            .cloned()
            .ok_or(RegistrationError::MissingKernel(hash))
    }
    pub fn kernel_count(&self) -> usize {
        self.kernels
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
    pub fn register_bytes(
        &self,
        hash: KernelHash,
        bytes: &[u8],
    ) -> Result<KernelHandle, RegistrationError> {
        self.register(hash, Module::from_bytes(bytes)?)
    }
    /// Validate and verify the supplied identity even when already registered.
    /// A duplicate retains the existing allocation and compiled cache identity.
    pub fn register(
        &self,
        hash: KernelHash,
        module: Module,
    ) -> Result<KernelHandle, RegistrationError> {
        let validated = module.validate()?;
        for index in 0..validated.module().kernels.len() {
            if validated.kernel_hash(index)? == hash {
                return Ok(self
                    .kernels
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .entry(hash)
                    .or_insert_with(|| KernelHandle {
                        module: Arc::new(validated),
                        index,
                        hash,
                    })
                    .clone());
            }
        }
        Err(RegistrationError::HashMismatch(hash))
    }
}

/// Process-wide default runtime. Initialization, including failures, is cached.
/// Registry and device synchronization live inside Runtime; callers never hold
/// a global lock across validation, compilation or GPU execution.
pub static MIMIR_RUNTIME: LazyLock<Result<Runtime, RegistrationError>> =
    LazyLock::new(Runtime::new);

pub fn default_runtime() -> Result<&'static Runtime, RegistrationError> {
    MIMIR_RUNTIME.as_ref().map_err(Clone::clone)
}
