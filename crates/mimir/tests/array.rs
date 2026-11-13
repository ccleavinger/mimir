use mimir::{Buffer, Device, GpuArray, GpuBuffer, GpuScalar, mimir_global};
#[mimir_global]
fn increment(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}
struct CustomArray(GpuArray<u32>);
impl GpuBuffer for CustomArray {
    fn as_buffer(&self) -> &Buffer {
        self.0.as_buffer()
    }
}
#[allow(dead_code)]
unsafe fn accepts_custom_and_erased_buffers(
    device: &Device,
    array: &CustomArray,
    erased: &dyn GpuBuffer,
) {
    unsafe {
        let _ = mimir::launch!(device;increment<<<[1,1,1],[4,1,1]>>>(array,&1u32));
        let _ = mimir::launch!(device;increment<<<[1,1,1],[4,1,1]>>>(erased,1u32));
    }
}
#[test]
fn typed_arrays_are_shareable() {
    fn shareable<T: Send + Sync>() {}
    shareable::<GpuArray<u32>>();
    shareable::<GpuArray<f32>>();
    shareable::<GpuArray<bool>>();
}
#[test]
#[ignore = "requires a Vulkan compute device"]
fn typed_arrays_use_their_owning_device_and_launch_through_the_trait() {
    // Independent from the global default context: readback must retain this owner.
    let runtime = mimir::Runtime::new().unwrap();
    let device = runtime.default_device().unwrap();
    fn roundtrip<T: GpuScalar + PartialEq + std::fmt::Debug>(device: &Device, values: &[T]) {
        let array = GpuArray::from_slice_on(device, values).unwrap();
        assert_eq!(array.to_vec().unwrap(), values);
        assert_eq!(array.len(), values.len());
        assert!(!array.is_empty());
        assert_eq!(array.element_type(), T::TYPE);
        assert_eq!(array.to_iter().unwrap().collect::<Vec<_>>(), values);
    }
    roundtrip(&device, &[-2i32, 0, 9]);
    roundtrip(&device, &[1.5f32, -2.0, 3.25]);
    roundtrip(&device, &[true, false, true]);
    let array = GpuArray::from_iter_on(&device, 1u32..=4).unwrap();
    let clone = array.clone();
    let wrapper = CustomArray(array);
    let erased: &dyn GpuBuffer = &wrapper;
    unsafe {
        mimir::launch!(&device;increment<<<[1,1,1],[4,1,1]>>>(erased,2u32)).unwrap();
        mimir::launch!(&device;increment<<<[1,1,1],[4,1,1]>>>(&wrapper,&3u32)).unwrap();
    }
    assert_eq!(clone.to_vec().unwrap(), [6, 7, 8, 9]);
    let raw = clone.into_buffer();
    assert!(GpuArray::<f32>::try_from(raw.clone()).is_err());
    assert_eq!(
        GpuArray::<u32>::try_from(raw).unwrap().to_vec().unwrap(),
        [6, 7, 8, 9]
    );
    assert!(GpuArray::<u32>::from_slice_on(&device, &[]).is_err());
    let default_array = GpuArray::from_iter([1u32, 2, 3, 4]).unwrap();
    unsafe {
        mimir::launch!(increment<<<[1,1,1],[4,1,1]>>>(&default_array,1u32)).unwrap();
    }
    assert_eq!(default_array.to_vec().unwrap(), [2, 3, 4, 5]);
}
