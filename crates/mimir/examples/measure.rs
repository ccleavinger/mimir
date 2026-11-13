use mimir::{Argument, Runtime, mimir_global};
use std::time::Instant;
#[mimir_global]
fn increment(values: &mut [u32]) {
    let i: u32 = global_id.x;
    values[i] = values[i] + 1u32;
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let runtime = Runtime::new()?;
    println!("linked registration: {:?}", start.elapsed());
    let handle = runtime.kernel(increment)?;
    let bytes = handle.ir().module().to_bytes()?;
    println!("artifact: {} bytes", bytes.len());
    let start = Instant::now();
    for _ in 0..100 {
        let runtime = Runtime::default();
        runtime.register_bytes(increment, &bytes)?;
    }
    println!("upload registration mean: {:?}", start.elapsed() / 100);
    let devices = runtime.devices()?;
    let Some(device) = devices.first() else {
        println!("no device/backend");
        return Ok(());
    };
    println!("device: {}", device.name());
    let buffer = device.buffer_from_slice(&[0u32; 64])?;
    let handle = runtime.kernel(increment)?;
    let start = Instant::now();
    unsafe {
        device.launch(
            &handle,
            &[],
            [1, 1, 1],
            [64, 1, 1],
            &[Argument::Buffer(&buffer)],
        )?;
    }
    println!("first compile and launch: {:?}", start.elapsed());
    let start = Instant::now();
    for _ in 0..100 {
        unsafe {
            device.launch(
                &handle,
                &[],
                [1, 1, 1],
                [64, 1, 1],
                &[Argument::Buffer(&buffer)],
            )?;
        }
    }
    println!("cached launch mean: {:?}", start.elapsed() / 100);
    assert_eq!(device.read_typed::<u32>(&buffer)?, [101; 64]);
    Ok(())
}
