#[mimir::mimir_global]
fn same_name(values: &mut [u32]) {
    values[global_id.x] = 2u32;
}
#[test]
fn dependency_exports_survive_linking_and_names_do_not_collide() {
    let runtime = mimir::Runtime::new().unwrap();
    let dependency = runtime.kernel(mimir_linked_fixture::same_name).unwrap();
    let local = runtime.kernel(same_name).unwrap();
    assert_ne!(dependency.semantic_hash(), local.semantic_hash());
}

#[mimir::mimir_global]
fn identical_dependency_ir(values: &mut [u32]) {
    values[global_id.x] = 1u32;
}
#[test]
fn identical_ir_from_different_crates_is_one_entry() {
    let runtime = mimir::Runtime::new().unwrap();
    assert_eq!(identical_dependency_ir, mimir_linked_fixture::same_name);
    assert_eq!(runtime.kernel_count(), 2);
}
