use mimir::{default_runtime, mimir_global};

#[mimir_global]
fn increment(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}

#[test]
fn default_runtime_is_shared_and_registration_is_concurrent() {
    let runtime = default_runtime().unwrap();
    let mut module = runtime.kernel(increment).unwrap().ir().module().clone();
    module.kernels[0]
        .routine
        .header
        .parameters
        .push(mimir::ir::Parameter {
            id: mimir::ir::ParameterId(9),
            ty: mimir::ir::Type::Scalar(mimir::ir::ScalarType::U32),
        });
    let hash = module.clone().validate().unwrap().kernel_hash(0).unwrap();
    let handles = std::thread::scope(|scope| {
        let threads = (0..16)
            .map(|_| {
                let module = &module;
                scope.spawn(move || {
                    let shared = default_runtime().unwrap();
                    assert!(std::ptr::eq(runtime, shared));
                    let handle = shared.register(hash, module.clone()).unwrap();
                    assert_eq!(shared.kernel(hash).unwrap().semantic_hash(), hash);
                    handle
                })
            })
            .collect::<Vec<_>>();
        threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Vec<_>>()
    });
    for handle in &handles {
        assert!(std::ptr::eq(handle.ir(), handles[0].ir()));
    }
    assert!(std::ptr::eq(
        runtime,
        mimir::MIMIR_RUNTIME.as_ref().unwrap()
    ));
}

#[test]
#[ignore = "requires a Vulkan compute device"]
fn default_and_explicit_device_launches_share_the_context() {
    let device = mimir::default_device().unwrap();
    let buffer = device.buffer_from_slice(&[1u32, 2, 3, 4]).unwrap();

    unsafe {
        mimir::launch!(increment<<<[1,1,1],[4,1,1]>>>(&buffer, 2)).unwrap();
        mimir::launch!(&device; increment<<<[1,1,1],[4,1,1]>>>(&buffer, 3)).unwrap();
    }
    assert_eq!(device.read_typed::<u32>(&buffer).unwrap(), [6, 7, 8, 9]);
}
