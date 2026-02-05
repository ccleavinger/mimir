use std::hash::{DefaultHasher, Hash, Hasher};

use crate::platforms::MimirPlatform;

pub trait MimirDevice {
    fn device_name(&self) -> String;

    // in bytes
    fn vram_size(&self) -> usize;

    fn as_platform_device(&self) -> Box<dyn std::any::Any>;

    fn platform(&self) -> MimirPlatform;

    fn get_id(&self) -> DeviceID;
}

#[derive(Clone, Copy)]
pub enum DeviceID {
    PCI(PCIBusID),
    NAME(DeviceNameID),
}

#[derive(Clone, Copy)]
pub struct PCIBusID {
    pub pci_domain: u32,
    pub pci_bus: u32,
    pub pci_device: u32,
}

#[derive(Clone, Copy)]
pub struct DeviceNameID(u64);

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

impl DeviceNameID {
    pub fn new(device_name: String) -> Self {
        Self(calculate_hash(&device_name))
    }

    pub fn get_id(&self) -> u64 {
        self.0
    }
}

pub trait MimirDeviceManager {
    fn get_device(&self, idx: usize) -> Option<&(dyn MimirDevice + Send + Sync)>;

    fn num_devices(&self) -> usize;

    // fn submit_devices(
    //     &mut self,
    //     devices: &[Box<dyn MimirDevice>]
    // );

    fn get_device_name(&self, idx: usize) -> Option<String> {
        self.get_device(idx).map(|device| device.device_name())
    }

    fn get_device_vram_size(&self, idx: usize) -> usize {
        if let Some(device) = self.get_device(idx) {
            device.vram_size()
        } else {
            0
        }
    }
}

pub trait MimirPlatformDevices {
    fn get_devices(&self) -> Vec<Box<dyn MimirDevice + Send + Sync>>;

    fn get_platform(&self) -> MimirPlatform;
}
