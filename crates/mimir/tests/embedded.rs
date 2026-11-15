mimir::embed_ir!(&[], b"MIMIRIR\0\x01\0\0\0\0\0");
mimir::embed_ir!(&[], b"MIMIRIR\0\x01\0\0\0\0\0");

#[test]
fn facade_collects_embedded_modules() {
    assert_eq!(mimir::Runtime::new().unwrap().kernel_count(), 0);
}
