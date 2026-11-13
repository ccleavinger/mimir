//! Resolve identity before ranking. Priority must never influence whether two
//! devices are considered the same, or turn an ambiguous match into a guess.
use crate::Device;
use mimir_runtime_core::HardwareInfo;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Evidence {
    Conflict,
    Unknown,
    Strong,
    Heuristic,
}
struct Candidate {
    device: Device,
    info: HardwareInfo,
    model: String,
}
struct Group {
    members: Vec<usize>,
    ambiguous: bool,
}

fn normalized_model(name: &str) -> String {
    let mut depth = 0usize;
    let name: String = name
        .chars()
        .filter_map(|c| match c {
            '(' => {
                depth += 1;
                None
            }
            ')' => {
                depth = depth.saturating_sub(1);
                None
            }
            _ if depth > 0 => None,
            _ => Some(if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }),
        })
        .collect();
    name.split_whitespace()
        .filter(|word| {
            !matches!(
                *word,
                "nvidia"
                    | "geforce"
                    | "amd"
                    | "radeon"
                    | "intel"
                    | "graphics"
                    | "corporation"
                    | "gpu"
                    | "device"
                    | "advanced"
                    | "micro"
                    | "devices"
                    | "tm"
            )
        })
        .collect()
}
fn clean(mut info: HardwareInfo) -> HardwareInfo {
    info.uuid = info.uuid.filter(|id| *id != [0; 16]);
    info.luid = info
        .luid
        .filter(|id| id.bytes != [0; 8] && id.node_mask.count_ones() == 1);
    info.pci = info.pci.filter(|id| id.is_valid());
    info.vendor_id = info.vendor_id.filter(|id| *id != 0);
    info.total_memory_bytes = info.total_memory_bytes.filter(|size| *size > 0);
    info
}
fn evidence(a: &Candidate, b: &Candidate) -> Evidence {
    let (a_info, b_info) = (&a.info, &b.info);
    let mut strong = false;
    macro_rules! compare {
        ($field:ident, $is_strong:expr) => {
            if let (Some(a), Some(b)) = (a_info.$field, b_info.$field) {
                if a != b {
                    return Evidence::Conflict;
                }
                strong |= $is_strong;
            }
        };
    }
    compare!(uuid, true);
    compare!(luid, true);
    compare!(pci, true);
    compare!(vendor_id, false);
    if strong {
        return Evidence::Strong;
    }
    // A model name (including its numeric model code) plus approximately equal
    // total memory is only a fallback, and is resolved for uniqueness below.
    if a.model == b.model
        && a.model.bytes().any(|c| c.is_ascii_digit())
        && let (Some(a), Some(b)) = (a_info.total_memory_bytes, b_info.total_memory_bytes)
    {
        // APIs may exclude small driver reservations from their totals.
        let tolerance = a.max(b) / 20;
        if a.abs_diff(b) <= tolerance {
            return Evidence::Heuristic;
        }
    }
    Evidence::Unknown
}

fn components(count: usize, connected: impl Fn(usize, usize) -> bool) -> Vec<Vec<usize>> {
    let mut seen = vec![false; count];
    let mut output = Vec::new();
    for start in 0..count {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut pending = vec![start];
        let mut component = Vec::new();
        while let Some(i) = pending.pop() {
            component.push(i);
            for (j, visited) in seen.iter_mut().enumerate() {
                if !*visited && connected(i, j) {
                    *visited = true;
                    pending.push(j);
                }
            }
        }
        output.push(component);
    }
    output
}
fn compatible(members: &[usize], candidates: &[Candidate], matches: &[Vec<Evidence>]) -> bool {
    members.iter().enumerate().all(|(i, a)| {
        members[i + 1..].iter().all(|b| {
            candidates[*a].device.backend_name() != candidates[*b].device.backend_name()
                && matches[*a][*b] != Evidence::Conflict
        })
    })
}

fn strong_compatible(
    members: &[usize],
    candidates: &[Candidate],
    matches: &[Vec<Evidence>],
) -> bool {
    let consistent = members.iter().enumerate().all(|(i, a)| {
        members[i + 1..]
            .iter()
            .all(|b| matches[*a][*b] != Evidence::Conflict)
    });
    let first = &candidates[members[0]].info;
    let same_uuid = first.uuid.is_some()
        && members
            .iter()
            .all(|i| candidates[*i].info.uuid == first.uuid);
    let same_luid = first.luid.is_some()
        && members
            .iter()
            .all(|i| candidates[*i].info.luid == first.luid);
    // Multiple APIs/ICDs can expose the same UUID. A PCI-only one-to-many
    // correspondence, however, might represent distinct hardware partitions.
    consistent && (compatible(members, candidates, matches) || same_uuid || same_luid)
}

pub(crate) fn select(devices: Vec<Device>) -> Vec<Device> {
    let candidates: Vec<_> = devices
        .into_iter()
        .map(|device| Candidate {
            info: clean(device.hardware_info()),
            model: normalized_model(device.name()),
            device,
        })
        .collect();
    let count = candidates.len();
    let matches: Vec<Vec<_>> = (0..count)
        .map(|a| {
            (0..count)
                .map(|b| evidence(&candidates[a], &candidates[b]))
                .collect()
        })
        .collect();
    let strong = components(count, |a, b| {
        candidates[a].device.backend_name() != candidates[b].device.backend_name()
            && matches[a][b] == Evidence::Strong
    });
    let mut groups = Vec::new();
    for members in strong {
        if strong_compatible(&members, &candidates, &matches) {
            groups.push(Group {
                members,
                ambiguous: false,
            });
        } else {
            // An identifier bridge with conflicting UUIDs, or several contexts
            // from one backend, cannot establish an unambiguous correspondence.
            groups.extend(members.into_iter().map(|i| Group {
                members: vec![i],
                ambiguous: true,
            }));
        }
    }
    let fallback_connected = |a: usize, b: usize| {
        if groups[a].ambiguous || groups[b].ambiguous {
            return false;
        }
        let members: Vec<_> = groups[a]
            .members
            .iter()
            .chain(&groups[b].members)
            .copied()
            .collect();
        compatible(&members, &candidates, &matches)
            && groups[a].members.iter().any(|a| {
                groups[b]
                    .members
                    .iter()
                    .any(|b| matches[*a][*b] == Evidence::Heuristic)
            })
    };
    let weak = components(groups.len(), fallback_connected);
    let mut resolved = Vec::new();
    for component in weak {
        let members: Vec<_> = component
            .iter()
            .flat_map(|i| groups[*i].members.iter().copied())
            .collect();
        let pairwise = component.iter().enumerate().all(|(i, a)| {
            component[i + 1..]
                .iter()
                .all(|b| fallback_connected(*a, *b))
        });
        if compatible(&members, &candidates, &matches) && pairwise {
            resolved.push(members);
        } else {
            resolved.extend(component.into_iter().map(|i| groups[i].members.clone()));
        }
    }
    // Preserve physical discovery order. A replacement occupies the first slot
    // for that GPU; higher priority does not reorder unrelated hardware.
    resolved.sort_by_key(|members| *members.iter().min().unwrap());
    resolved
        .into_iter()
        .map(|members| {
            let winner = members
                .iter()
                .copied()
                .min_by_key(|i| (std::cmp::Reverse(candidates[*i].device.priority()), *i))
                .unwrap();
            candidates[winner].device.clone()
        })
        .collect()
}
