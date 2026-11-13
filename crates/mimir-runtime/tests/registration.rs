use mimir_ir::flat::*;
use mimir_runtime::*;

fn module(name: &str) -> Module {
    Module {
        kernels: vec![Kernel {
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
            debug: DebugInfo {
                name: name.into(),
                ..Default::default()
            },
        }],
        functions: vec![],
    }
}

fn hash(module: &Module) -> KernelHash {
    module.clone().validate().unwrap().kernel_hash(0).unwrap()
}
#[test]
fn uploads_validate_and_handles_retain_ir() {
    let runtime = Runtime::default();
    let module = module("uploaded");
    let hash = hash(&module);
    let handle = runtime
        .register_bytes(hash, &module.to_bytes().unwrap())
        .unwrap();
    assert_eq!(runtime.kernel(hash).unwrap().semantic_hash(), hash);
    drop(runtime);
    assert_eq!(handle.ir().module().kernels.len(), 1);
}
#[test]
fn registration_deduplicates_by_semantics_and_checks_supplied_hash() {
    let runtime = Runtime::default();
    let original = module("one");
    let key = hash(&original);
    runtime.register(key, original).unwrap();
    let duplicate = runtime
        .register(key, module("different debug name"))
        .unwrap();
    assert!(std::ptr::eq(
        duplicate.ir(),
        runtime.kernel(key).unwrap().ir()
    ));
    assert_eq!(runtime.kernel_count(), 1);
    assert!(matches!(
        runtime.register(key, Module::default()),
        Err(RegistrationError::HashMismatch(_))
    ));
    let wrong = KernelHash([99; 32]);
    assert!(matches!(
        runtime.register(wrong, module("valid")),
        Err(RegistrationError::HashMismatch(_))
    ));
    let mut invalid = module("invalid");
    invalid.kernels[0].routine.body.blocks.clear();
    assert!(matches!(
        runtime.register(key, invalid),
        Err(RegistrationError::Invalid(_))
    ));
    assert_eq!(runtime.kernel_count(), 1);
    assert!(runtime.kernel(wrong).is_err());
}
#[test]
fn hash_lookup_is_independent_of_module_order_and_unused_kernels() {
    let original = module("one");
    let key = hash(&original);
    let mut combined = original.clone();
    let mut other = original.kernels[0].clone();
    other.routine.header.parameters.push(Parameter {
        id: ParameterId(0),
        ty: Type::Scalar(ScalarType::U32),
    });
    combined.kernels.insert(0, other);
    let runtime = Runtime::default();
    let handle = runtime.register(key, combined).unwrap();
    assert_eq!(handle.index(), 1);
    assert_eq!(handle.semantic_hash(), key);
    runtime.register(key, original).unwrap();
    assert_eq!(runtime.kernel_count(), 1);
}

#[test]
fn artifact_rejects_wrong_versions_truncation_and_trailing_data() {
    let bytes = module("test").to_bytes().unwrap();
    assert_eq!(Module::from_bytes(&bytes).unwrap(), module("test"));
    for end in 0..bytes.len() {
        assert!(Module::from_bytes(&bytes[..end]).is_err());
    }
    let mut changed = bytes.clone();
    changed[8..12].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        Module::from_bytes(&changed),
        Err(ArtifactError::Version(99))
    ));
    let mut trailing = bytes;
    trailing.push(0);
    assert!(matches!(
        Module::from_bytes(&trailing),
        Err(ArtifactError::TrailingBytes)
    ));
}

// Empty module artifact. A real static exercises linker collection, not a mocked list.
#[linkme::distributed_slice(LINKED_MODULES)]
static LINKED: LinkedModule = LinkedModule {
    hashes: &[],
    bytes: b"MIMIRIR\0\x01\0\0\0\0\0",
};

#[test]
fn discovers_linked_artifacts() {
    let runtime = Runtime::new().unwrap();
    assert_eq!(runtime.kernel_count(), 0);
}
