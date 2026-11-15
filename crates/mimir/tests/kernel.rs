use mimir::{Runtime, mimir_global};
#[mimir_global]
pub fn add(values: &mut [u32], amount: u32) {
    let i: u32 = global_id.x;
    values[i] = values[i] + amount;
}

#[mimir_global]
fn loop_kernel(values: &mut [u32]) {
    let x: u32 = global_id.x;
    let mut total: u32 = 0;
    for i in 0u32..4u32 {
        if i < 2 {
            total = total + i;
        } else {
            total = total + 2;
        }
    }
    values[x] = total;
}

#[test]
fn rust_kernel_registers_flat_ir() {
    let runtime = Runtime::new().unwrap();
    let kernel = runtime.kernel(add).unwrap();

    assert_eq!(
        kernel.ir().module().kernels[0]
            .routine
            .header
            .parameters
            .len(),
        2
    );
    runtime.kernel(loop_kernel).unwrap();
}

#[test]
#[ignore = "requires a Vulkan compute device"]
fn vulkan_executes_embedded_and_uploaded_kernels() {
    unsafe {
        use mimir::{Argument, ir::ScalarType};
        let runtime = Runtime::new().unwrap();
        let devices = runtime.devices().unwrap();
        let device = devices.first().expect("Vulkan compute device");
        let buffer = device.buffer(ScalarType::U32, &[1, 2, 3, 4]).unwrap();
        let handle = runtime.kernel(add).unwrap();

        for _ in 0..2 {
            device
                .launch(
                    &handle,
                    &[],
                    [1, 1, 1],
                    [4, 1, 1],
                    &[
                        Argument::Buffer(&buffer),
                        Argument::Scalar(mimir::ir::Literal::U32(3)),
                    ],
                )
                .unwrap();
        }

        assert_eq!(device.read(&buffer).unwrap(), [7, 8, 9, 10]);
        let bytes = handle.ir().module().to_bytes().unwrap();
        let uploaded = runtime.register_bytes(add, &bytes).unwrap();
        device
            .launch(
                &uploaded,
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[
                    Argument::Buffer(&buffer),
                    Argument::Scalar(mimir::ir::Literal::U32(1)),
                ],
            )
            .unwrap();
        assert_eq!(device.read(&buffer).unwrap(), [8, 9, 10, 11]);
        device
            .launch(
                &runtime.kernel(loop_kernel).unwrap(),
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&buffer)],
            )
            .unwrap();
        assert_eq!(device.read(&buffer).unwrap(), [5; 4]);
        device
            .launch(
                &runtime.kernel(helpers::transform).unwrap(),
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&buffer)],
            )
            .unwrap();
        assert_eq!(device.read(&buffer).unwrap(), [8; 4]);
        device
            .launch(
                &runtime.kernel(helpers::loop_calls).unwrap(),
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&buffer)],
            )
            .unwrap();
        assert_eq!(device.read(&buffer).unwrap(), [13; 4]);
        let reverse = device.buffer_from_slice(&[1u32, 2, 3, 4]).unwrap();
        device
            .launch(
                &runtime.kernel(reverse_group).unwrap(),
                &[4],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&reverse)],
            )
            .unwrap();
        assert_eq!(device.read_typed::<u32>(&reverse).unwrap(), [4, 3, 2, 1]);
        let booleans = device
            .buffer_from_slice(&[true, false, true, false])
            .unwrap();
        device
            .launch(
                &runtime.kernel(bool_not).unwrap(),
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&booleans)],
            )
            .unwrap();
        assert_eq!(
            device.read_typed::<bool>(&booleans).unwrap(),
            [false, true, false, true]
        );
        let floats = device.buffer_from_slice(&[0.0f32, 1.0, 2.0, 3.0]).unwrap();
        device
            .launch(
                &runtime.kernel(float_math).unwrap(),
                &[],
                [1, 1, 1],
                [4, 1, 1],
                &[Argument::Buffer(&floats)],
            )
            .unwrap();

        for (actual, input) in device
            .read_typed::<f32>(&floats)
            .unwrap()
            .iter()
            .zip([0.0f32, 1.0, 2.0, 3.0])
        {
            assert!((*actual - (input.sin() + 1.0)).abs() < 0.0001);
        }

        assert!(device.read_typed::<u32>(&floats).is_err());
        assert!(
            device
                .launch(&handle, &[], [0, 1, 1], [4, 1, 1], &[])
                .is_err()
        );
        assert!(
            device
                .launch(
                    &handle,
                    &[],
                    [1, 1, 1],
                    [4, 1, 1],
                    &[
                        Argument::Buffer(&floats),
                        Argument::Scalar(mimir::ir::Literal::U32(1))
                    ]
                )
                .is_err()
        );
    }
}

#[test]
fn generated_spirv_passes_vulkan_validation() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let runtime = Runtime::new().unwrap();
    for (definition, tuning) in [
        (add, vec![]),
        (loop_kernel, vec![]),
        (helpers::transform, vec![]),
        (helpers::loop_calls, vec![]),
        (reverse_group, vec![4]),
        (bool_not, vec![]),
        (float_math, vec![]),
    ] {
        let handle = runtime.kernel(definition).unwrap();
        let module = handle.ir();
        let specialized = module.specialize_kernel(handle.index(), &tuning).unwrap();
        let words = mimir_codegen_spirv::compile(module, &specialized, [4, 1, 1]).unwrap();
        let mut child = match Command::new("spirv-val")
            .args(["--target-env", "vulkan1.1", "-"])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
            Err(e) => panic!("{e}"),
        };
        let mut stdin = child.stdin.take().unwrap();
        for word in words {
            stdin.write_all(&word.to_le_bytes()).unwrap();
        }
        drop(stdin);
        let result = child.wait_with_output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[mimir::mimir_module]
mod helpers {
    #[device]
    fn twice(x: u32) -> u32 {
        return x * 2;
    }

    #[device]
    fn choose(x: u32) -> u32 {
        if x < 2 {
            return twice(x);
        } else {
            return x + 3;
        }
    }

    #[kernel]
    pub fn loop_calls(values: &mut [u32]) {
        let mut sum: u32 = 0;
        for j in 0u32..4u32 {
            sum = sum + choose(j);
        }
        values[global_id.x] = sum;
    }
    #[kernel]
    pub fn transform(values: &mut [u32]) {
        let i: u32 = global_id.x;
        values[i] = choose(values[i]);
    }
}

#[mimir_global]
fn reverse_group<const N: u32>(values: &mut [u32]) {
    #[shared]
    let mut scratch = [0u32; N];
    let i: u32 = thread_idx.x;
    scratch[i] = values[i];
    syncthreads();
    values[i] = scratch[N - 1u32 - i];
}

#[mimir_global]
fn bool_not(values: &mut [bool]) {
    let i: u32 = global_id.x;
    values[i] = !values[i];
}

#[mimir_global]
fn float_math(values: &mut [f32]) {
    let i: u32 = global_id.x;
    values[i] = sin(values[i]) + 1.0;
}
