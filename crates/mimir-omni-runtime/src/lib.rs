// bundle all the seperate Mimir runtimes into a dynamic runtime

use std::{
    cell::Cell,
    collections::HashMap,
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use crate::error::CreateMimirArrayError;
use mimir_ir::ir::MimirIRData;
use mimir_runtime::generic::runtime::MimirJITRuntime;
use mimir_runtime::{
    generic::{
        array::{MimirArray, MimirArrayFactory},
        device::{MimirDevice, MimirDeviceManager},
        launch::{LaunchDeviceError, MimirLaunch, MimirLaunchError},
    },
    platforms::MimirPlatform,
};
use mimir_vulkan::{array::MIMIR_VK_ARR_FACTORY, runtime::MIMIR_VK_RUNTIME};
use parking_lot::Mutex;

pub use mimir_runtime::generic::launch::{MimirBuffer, MimirVar, Param};

pub static MIMIR_OMNI_RUNTIME: LazyLock<MimirOmniRuntime> = LazyLock::new(MimirOmniRuntime::init);
const FILENAME: &str = "mimir.bin";

pub type MimirGPUArray<T> = Arc<dyn MimirArray<T> + Send + Sync>;

pub mod error;
pub mod mimir_gpu_arr;
pub mod mimir_kernel;

pub struct MimirOmniRuntime {
    pub devices: Vec<Box<dyn MimirDevice + Send + Sync>>,
    pub ir_data: MimirIRData,
    pub current_device_idx: Mutex<Cell<usize>>,
}

impl MimirOmniRuntime {
    pub fn init() -> Self {
        let runtime = match MIMIR_VK_RUNTIME.as_ref() {
            Some(r) => r,
            None => {
                return Self {
                    devices: vec![],
                    ir_data: MimirIRData {
                        irs: HashMap::new(),
                        source_hashes: HashMap::new(),
                    },
                    current_device_idx: Mutex::new(Cell::new(0)),
                };
            }
        };

        let vulkan_devices = runtime.get_platform_devices();

        let devices = vulkan_devices.get_devices();

        let ir_data = Self::get_ir_data().unwrap();

        Self {
            devices,
            ir_data,
            current_device_idx: Mutex::new(Cell::new(0)),
        }
    }

    pub(crate) fn get_curr_dev_idx(&self) -> usize {
        self.current_device_idx.lock().get()
    }

    fn get_ir_data() -> Result<MimirIRData, Box<dyn std::error::Error>> {
        // Try runtime location first (same dir as executable)
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path
            .parent()
            .ok_or("Failed to get executable directory")?;
        let exe_name = exe_path
            .file_stem()
            .ok_or("Failed to get executable name")?
            .to_str()
            .ok_or("Invalid UTF-8 in executable name")?;

        let file_name = format!("{}.{}", exe_name, FILENAME);

        let mut __file__: Box<dyn BufRead> = match File::open(exe_dir.join(&file_name)) {
            Ok(file) => Box::new(BufReader::new(file)),
            Err(_) => {
                // Fallback: try the workspace/project root where the mimir-macros saved it
                // CARGO_MANIFEST_DIR at compile time is the BINARY's manifest dir
                let compile_time_dir = env::var("CARGO_MANIFEST_DIR")?;
                let path_buf = PathBuf::from(compile_time_dir).join(&file_name);
                let file = File::open(path_buf)?;
                Box::new(BufReader::new(file))
            }
        };

        let ir_data = MimirIRData::load(&mut __file__)?;
        Ok(ir_data)
    }

    pub(crate) fn get_device_internal(
        &self,
    ) -> Result<&(dyn MimirDevice + Send + Sync), LaunchDeviceError> {
        self.get_device(self.get_curr_dev_idx())
            .ok_or(LaunchDeviceError::OutOfRange(
                self.get_curr_dev_idx(),
                self.num_devices(),
            ))
    }

    pub(crate) fn create_mimir_array<T: Copy + Send + Sync + Sized + 'static, I>(
        &self,
        iter: I,
    ) -> Result<MimirGPUArray<T>, CreateMimirArrayError>
    where
        I: IntoIterator<Item = T>,
    {
        let device = Arc::new(self.get_device_internal()?);
        Ok(match device.platform() {
            MimirPlatform::Vulkan => MIMIR_VK_ARR_FACTORY.create_from_iter_device(device, iter)?,
        })
    }
}

impl MimirDeviceManager for MimirOmniRuntime {
    fn get_device(&self, idx: usize) -> Option<&(dyn MimirDevice + Send + Sync)> {
        self.devices
            .get(idx)
            .map(|d| d.as_ref() as &(dyn MimirDevice + Send + Sync))
    }

    fn num_devices(&self) -> usize {
        self.devices.len()
    }
}

impl MimirLaunch for MimirOmniRuntime {
    fn launch_kernel(
        &self,
        ir: &MimirIRData,
        name: String, // name of the kernel
        block_dim: &[u32; 3],
        grid_dim: &[u32; 3],
        params: &[Param],
        cgs: &[u32],
    ) -> Result<(), MimirLaunchError> {
        let device = Box::new(self.get_device_internal()?);

        match device.platform() {
            MimirPlatform::Vulkan => {
                let vk_r = match MIMIR_VK_RUNTIME.as_ref() {
                    Some(r) => r,
                    None => {
                        return Err(MimirLaunchError::UnsupportedPlatform(MimirPlatform::Vulkan));
                    }
                };
                vk_r.execute_kernel(device, ir, &name, block_dim, grid_dim, params, cgs)?
            }
        }

        Ok(())
    }

    fn set_device(
        &self,
        idx: usize,
    ) -> Result<(), mimir_runtime::generic::launch::LaunchDeviceError> {
        if idx < self.num_devices() {
            *self.current_device_idx.lock().get_mut() = idx;
            Ok(())
        } else {
            Err(LaunchDeviceError::OutOfRange(idx, self.num_devices()))
        }
    }
}
