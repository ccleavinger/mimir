// supported platforms
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MimirPlatform {
    Vulkan,
    // CUDA, // WIP
}

pub trait PlatformExclusive {
    fn platform(&self) -> MimirPlatform;
}
