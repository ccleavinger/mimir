use mimir_ir::flat::ScalarType;
use mimir_runtime::backend::{
    Backend, BufferBackend, DeviceBackend, DeviceId, DeviceLuid, HardwareInfo, Launch, PciAddress,
};
use mimir_runtime::{Device, ExecutionError, ExecutionResult};
use std::{any::Any, sync::Arc};

struct Gpu {
    id: DeviceId,
    backend: &'static str,
    name: &'static str,
    priority: usize,
    hardware: HardwareInfo,
}
impl DeviceBackend for Gpu {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn id(&self) -> DeviceId {
        self.id.clone()
    }
    fn name(&self) -> &str {
        self.name
    }
    fn backend_name(&self) -> &'static str {
        self.backend
    }
    fn priority(&self) -> usize {
        self.priority
    }
    fn hardware_info(&self) -> HardwareInfo {
        self.hardware.clone()
    }
    fn buffer(
        self: Arc<Self>,
        _: ScalarType,
        _: &[u32],
    ) -> ExecutionResult<Arc<dyn BufferBackend>> {
        Err(ExecutionError("discovery-only test device".into()))
    }
    fn read(&self, _: &dyn BufferBackend) -> ExecutionResult<Vec<u32>> {
        Err(ExecutionError("discovery-only test device".into()))
    }
    unsafe fn launch(&self, _: &Launch<'_>) -> ExecutionResult<()> {
        Err(ExecutionError("discovery-only test device".into()))
    }
}
struct Provider(Vec<Arc<dyn DeviceBackend>>);
impl Backend for Provider {
    fn name(&self) -> &'static str {
        "test"
    }
    fn discover(&self) -> ExecutionResult<Vec<Arc<dyn DeviceBackend>>> {
        Ok(self.0.clone())
    }
}
fn gpu(backend: &'static str, priority: usize, hardware: HardwareInfo) -> Arc<dyn DeviceBackend> {
    named(backend, priority, "NVIDIA GeForce RTX 4090", hardware)
}
fn named(
    backend: &'static str,
    priority: usize,
    name: &'static str,
    hardware: HardwareInfo,
) -> Arc<dyn DeviceBackend> {
    Arc::new(Gpu {
        id: DeviceId::new(),
        backend,
        priority,
        name,
        hardware,
    })
}
fn discover(devices: Vec<Arc<dyn DeviceBackend>>) -> Vec<Device> {
    Device::discover_with_backends(&[&Provider(devices)]).unwrap()
}
fn uuid(id: u8) -> HardwareInfo {
    HardwareInfo {
        uuid: Some([id; 16]),
        ..HardwareInfo::default()
    }
}
fn pci(domain: u32, bus: u32, function: u32) -> HardwareInfo {
    HardwareInfo {
        pci: Some(PciAddress {
            domain,
            bus,
            device: 0,
            function,
        }),
        ..HardwareInfo::default()
    }
}
fn soft(memory_gib: u64) -> HardwareInfo {
    HardwareInfo {
        vendor_id: Some(0x10de),
        total_memory_bytes: Some(memory_gib * 1024 * 1024 * 1024),
        ..HardwareInfo::default()
    }
}

#[test]
fn priority_replaces_duplicates_and_preserves_order_of_unrelated_hardware() {
    let low = gpu("vulkan", 100, uuid(1));
    let high = gpu("cuda", 200, uuid(1));
    let other = gpu("vulkan", 100, uuid(2));
    let result = Device::discover_with_backends(&[
        &Provider(vec![low, other.clone()]),
        &Provider(vec![high.clone()]),
    ])
    .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].backend_name(), "cuda");
    assert_eq!(result[0].priority(), 200);
    assert_eq!(result[0].hardware_info(), high.hardware_info());
    assert_eq!(result[1].hardware_info(), other.hardware_info());
    // Priority is supplied by the implementation, not hard-coded to a backend name.
    let result = discover(vec![
        gpu("vulkan", usize::MAX, uuid(1)),
        gpu("cuda", 200, uuid(1)),
    ]);
    assert_eq!(result[0].backend_name(), "vulkan");
}
#[test]
fn equal_priorities_keep_the_first_discovered_implementation() {
    let result = discover(vec![gpu("vulkan", 7, uuid(1)), gpu("cuda", 7, uuid(1))]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].backend_name(), "vulkan");
}
#[test]
fn pci_matching_uses_domain_bus_device_and_function() {
    assert_eq!(
        discover(vec![
            gpu("vulkan", 1, pci(0, 1, 0)),
            gpu("cuda", 2, pci(0, 1, 0))
        ])
        .len(),
        1
    );
    for different in [
        pci(1, 1, 0),
        pci(0, 2, 0),
        pci(0, 1, 1),
        HardwareInfo {
            pci: Some(PciAddress {
                domain: 0,
                bus: 1,
                device: 1,
                function: 0,
            }),
            ..HardwareInfo::default()
        },
    ] {
        assert_eq!(
            discover(vec![
                gpu("vulkan", 1, pci(0, 1, 0)),
                gpu("cuda", 2, different)
            ])
            .len(),
            2
        );
    }
}
#[test]
fn windows_adapter_identity_includes_the_linked_adapter_node() {
    let a = HardwareInfo {
        luid: Some(DeviceLuid {
            bytes: [1; 8],
            node_mask: 1,
        }),
        ..HardwareInfo::default()
    };
    let b = HardwareInfo {
        luid: Some(DeviceLuid {
            bytes: [1; 8],
            node_mask: 2,
        }),
        ..HardwareInfo::default()
    };
    assert_eq!(
        discover(vec![gpu("vulkan", 1, a.clone()), gpu("cuda", 2, a.clone())]).len(),
        1
    );
    assert_eq!(
        discover(vec![gpu("vulkan", 1, a), gpu("cuda", 2, b)]).len(),
        2
    );
}
#[test]
fn strong_identity_does_not_require_matching_names_or_memory_totals() {
    let mut a = uuid(1);
    a.total_memory_bytes = Some(24 << 30);
    let mut b = uuid(1);
    b.total_memory_bytes = Some(23 << 30);
    let result = discover(vec![
        named("vulkan", 1, "Graphics Device", a),
        named("cuda", 2, "GeForce RTX 4090", b),
    ]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].backend_name(), "cuda");
}
#[test]
fn conflicting_identifiers_veto_pci_name_and_memory_matches() {
    let mut a = soft(24);
    a.uuid = Some([1; 16]);
    a.pci = pci(0, 1, 0).pci;
    let mut b = a.clone();
    b.uuid = Some([2; 16]);
    assert_eq!(
        discover(vec![gpu("vulkan", 1, a.clone()), gpu("cuda", 2, b)]).len(),
        2
    );
    let mut b = a.clone();
    b.pci = pci(0, 2, 0).pci;
    assert_eq!(
        discover(vec![gpu("vulkan", 1, a.clone()), gpu("cuda", 2, b)]).len(),
        2
    );
    let mut b = a.clone();
    b.vendor_id = Some(0x1002);
    assert_eq!(
        discover(vec![gpu("vulkan", 1, a), gpu("cuda", 2, b)]).len(),
        2
    );
}
#[test]
fn fallback_matches_vendor_name_variations_and_reserved_memory() {
    let mut b = soft(24);
    b.total_memory_bytes = Some((24 << 30) - (256 << 20));
    let result = discover(vec![
        gpu("vulkan", 100, soft(24)),
        named("cuda", 200, "GeForce RTX4090 (NVIDIA)", b),
    ]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].backend_name(), "cuda");
    assert_eq!(
        discover(vec![gpu("vulkan", 1, soft(24)), gpu("cuda", 2, soft(16))]).len(),
        2
    );
    assert_eq!(
        discover(vec![
            gpu("vulkan", 1, soft(24)),
            named("cuda", 2, "GeForce RTX 4080", soft(24))
        ])
        .len(),
        2
    );
}
#[test]
fn identical_gpu_models_require_an_unambiguous_correspondence() {
    let result = discover(vec![
        gpu("vulkan", 1, soft(24)),
        gpu("vulkan", 1, soft(24)),
        gpu("cuda", 2, soft(24)),
        gpu("cuda", 2, soft(24)),
    ]);
    assert_eq!(result.len(), 4);
    let result = discover(vec![
        gpu("vulkan", 1, soft(24)),
        gpu("vulkan", 1, soft(24)),
        gpu("cuda", 2, soft(24)),
    ]);
    assert_eq!(result.len(), 3);
    let result = discover(vec![
        gpu("vulkan", 1, uuid(1)),
        gpu("vulkan", 1, uuid(2)),
        gpu("cuda", 2, uuid(2)),
        gpu("cuda", 2, uuid(1)),
    ]);
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|d| d.backend_name() == "cuda"));
    assert_eq!(result[0].hardware_info().uuid, Some([1; 16]));
    assert_eq!(result[1].hardware_info().uuid, Some([2; 16]));
}
#[test]
fn missing_metadata_names_or_memory_alone_do_not_establish_identity() {
    let empty = HardwareInfo::default();
    assert_eq!(
        discover(vec![gpu("vulkan", 1, empty.clone()), gpu("cuda", 2, empty)]).len(),
        2
    );
    assert_eq!(
        discover(vec![
            named("vulkan", 1, "Graphics Device", soft(24)),
            named("cuda", 2, "Graphics Device", soft(24))
        ])
        .len(),
        2
    );
    assert_eq!(
        discover(vec![gpu("vulkan", 1, soft(24)), gpu("vulkan", 2, soft(24))]).len(),
        2
    );
    assert!(Device::discover_with_backends(&[]).unwrap().is_empty());
}
#[test]
fn conflicting_identity_bridges_do_not_merge_distinct_partitions() {
    let mut a = pci(0, 1, 0);
    a.uuid = Some([1; 16]);
    let bridge = pci(0, 1, 0);
    let mut b = pci(0, 1, 0);
    b.uuid = Some([2; 16]);
    assert_eq!(
        discover(vec![gpu("a", 1, a), gpu("b", 2, bridge), gpu("c", 3, b)]).len(),
        3
    );
}
#[test]
fn strong_matches_survive_ambiguous_fallback_matches() {
    let mut a = soft(24);
    a.uuid = Some([1; 16]);
    let mut b = soft(24);
    b.uuid = Some([2; 16]);
    let result = discover(vec![
        gpu("vulkan", 1, a.clone()),
        gpu("vulkan", 1, b),
        gpu("cuda", 2, a),
        gpu("other", 3, soft(24)),
    ]);
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].backend_name(), "cuda");
}
#[test]
fn zero_identifiers_are_unavailable_not_shared_hardware_ids() {
    let invalid = HardwareInfo {
        uuid: Some([0; 16]),
        luid: Some(DeviceLuid {
            bytes: [0; 8],
            node_mask: 0,
        }),
        pci: Some(PciAddress {
            domain: 0,
            bus: 999,
            device: 0,
            function: 0,
        }),
        vendor_id: Some(0),
        total_memory_bytes: Some(0),
    };
    assert_eq!(
        discover(vec![
            gpu("vulkan", 1, invalid.clone()),
            gpu("cuda", 2, invalid)
        ])
        .len(),
        2
    );
}

#[test]
fn strong_duplicate_icds_are_represented_only_by_the_highest_priority_backend() {
    let result = discover(vec![
        gpu("vulkan", 1, uuid(1)),
        gpu("vulkan", 1, uuid(1)),
        gpu("cuda", 2, uuid(1)),
    ]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].backend_name(), "cuda");
}
#[test]
fn fallback_memory_tolerance_does_not_chain_across_distinct_totals() {
    let result = discover(vec![
        gpu("a", 1, soft(24)),
        gpu("b", 2, soft(23)),
        gpu("c", 3, soft(22)),
    ]);
    assert_eq!(result.len(), 3);
}
#[test]
fn losing_contexts_are_released_after_selection() {
    let low = gpu("vulkan", 1, uuid(1));
    let high = gpu("cuda", 2, uuid(1));
    let low_weak = Arc::downgrade(&low);
    let high_weak = Arc::downgrade(&high);
    let selected = discover(vec![low, high]);
    assert_eq!(selected.len(), 1);
    assert!(low_weak.upgrade().is_none());
    assert!(high_weak.upgrade().is_some());
}
