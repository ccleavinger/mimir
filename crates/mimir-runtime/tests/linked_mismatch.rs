use mimir_runtime::*;
#[linkme::distributed_slice(LINKED_MODULES)]
static INVALID: LinkedModule = LinkedModule {
    hashes: &[KernelHash([7; 32])],
    bytes: b"MIMIRIR\0\x01\0\0\0\0\0",
};
#[test]
fn linked_hashes_are_verified_against_ir() {
    assert!(matches!(
        Runtime::new(),
        Err(RegistrationError::LinkedHashesMismatch)
    ));
}
#[test]
fn default_initialization_caches_errors_without_panicking() {
    for _ in 0..2 {
        assert!(matches!(
            default_runtime(),
            Err(RegistrationError::LinkedHashesMismatch)
        ));
    }
    assert!(matches!(
        &*MIMIR_RUNTIME,
        Err(RegistrationError::LinkedHashesMismatch)
    ));
}
