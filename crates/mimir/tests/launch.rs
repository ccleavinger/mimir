use mimir::{KernelHash, Runtime, launch, mimir_global};

#[mimir_global]
fn add(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}

#[mimir_global]
fn same_ir_different_name(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}

#[mimir_global]
fn tuned<const N: u32>(values: &mut [u32]) {
    let i: u32 = global_id.x;
    values[i] = values[i] + N;
}

#[test]
fn generated_bindings_are_semantic_hashes_and_deduplicate() {
    let _: KernelHash = add;
    assert_eq!(add, same_ir_different_name);
    let runtime = Runtime::new().unwrap();
    assert_eq!(runtime.kernel_count(), 2);
    assert_eq!(runtime.kernel(add).unwrap().semantic_hash(), add);
    let module = runtime.kernel(add).unwrap().ir().module().clone();
    let uploaded = Runtime::default();
    assert_eq!(uploaded.register(add, module).unwrap().semantic_hash(), add);
}

// Compile the macro on backend-free builds and exercise missing-hash errors
// without submitting any GPU work.
#[allow(dead_code)]
unsafe fn compile_shapes(runtime: &Runtime, device: &mimir::Device, buffer: &mimir::Buffer) {
    let _ = unsafe { launch!(runtime,device;add<<<[1,1,1],[4,1,1]>>>(buffer,3u32)) };
    let _ = unsafe { launch!(runtime,device;tuned<4u32><<<[1,1,1],[4,1,1]>>>(&buffer);) };
}

macro_rules! hygienic_launch {
    ($r:expr,$d:expr,$k:expr,$b:expr)=>{{
        let runtime=$r;let device=$d;let kernel=$k;let buffer=$b;
        mimir::launch!(runtime,device;kernel<<<[1,1,1],[4,1,1]>>>(buffer,1u32))
    }};
}

#[test]
#[ignore = "requires a Vulkan compute device"]
fn cuda_launch_executes_once_and_preserves_hygiene() {
    let runtime = Runtime::new().unwrap();
    let devices = runtime.devices().unwrap();
    let device = &devices[0];
    let buffer = device.buffer_from_slice(&[1u32, 2, 3, 4]).unwrap();
    let mut evaluations = 0;
    unsafe {
        launch!(&runtime,device;add<<<[1,1,1],[4,1,1]>>>(&buffer,{evaluations+=1;3u32})).unwrap();
        launch!(&runtime,device;tuned<{2u32+2}><<<[1,1,1],[4,1,1]>>>(buffer.clone());).unwrap();
        hygienic_launch!(&runtime, device, add, &buffer).unwrap();
    }
    assert_eq!(evaluations, 1);
    assert_eq!(device.read_typed::<u32>(&buffer).unwrap(), [9, 10, 11, 12]);
}
