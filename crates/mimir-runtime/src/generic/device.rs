
pub trait MimirDevice {
    fn device_name(&self) -> String;

    fn vram_size(&self) -> usize;

    fn get_device(&self) -> Option<&dyn std::any::Any>;
}

pub trait MimirDeviceManager {
    fn get_device(&self, idx: u32) -> Option<&dyn MimirDevice>;

    fn num_devices(&self) -> usize;

    fn get_device_name(&self, idx: u32) -> String {
        if let Some(device) = self.get_device(idx) {
            device.device_name()
        } else {
            String::new()
        }
    }

    fn get_device_vram_size(&self, idx: u32) -> usize {
        if let Some(device) = self.get_device(idx) {
            device.vram_size()
        } else {
            0
        }
    }
}