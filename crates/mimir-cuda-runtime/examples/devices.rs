use mimir_cuda_runtime::CudaBackend;
use mimir_runtime_core::Backend;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = CudaBackend.discover()?;
    if devices.is_empty() {
        println!("No compatible CUDA devices found.");
    }
    for (index, device) in devices.iter().enumerate() {
        println!("{index}: {} ({})", device.name(), device.backend_name());
    }
    Ok(())
}
