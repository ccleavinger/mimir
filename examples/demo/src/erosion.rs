//! Closed-boundary grid erosion: conservative water/sediment transport, then erosion/deposition.
//! All simulation buffers remain on the selected device between requests.
use super::Error;
use mimir::{launch, mimir_global, GpuArray};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Instant;
const PREVIEW: usize = 64;
static NEXT: AtomicU64 = AtomicU64::new(1);
static SIMULATIONS: Mutex<BTreeMap<u64, Arc<Mutex<Simulation>>>> = Mutex::new(BTreeMap::new());

#[mimir_global]
fn flow(state: &[f32], flux: &mut [f32], size: i32, rain: f32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < size * size {
        let x = i % size;
        let y = i / size;
        let surface = state[i * 3] + state[i * 3 + 1] + rain;
        let mut total = 0.0f32;
        for d in 0..4 {
            let mut j = i;
            let mut valid = true;
            if d == 0 {
                j = i - 1;
                valid = x > 0;
            }
            if d == 1 {
                j = i + 1;
                valid = x < size - 1;
            }
            if d == 2 {
                j = i - size;
                valid = y > 0;
            }
            if d == 3 {
                j = i + size;
                valid = y < size - 1;
            }
            let mut f = 0.0f32;
            if valid {
                f = max(
                    0.0,
                    (surface - state[j * 3] - state[j * 3 + 1] - rain) * 0.22,
                );
            }
            flux[i * 4 + d] = f;
            total += f;
        }
        let scale = min(1.0, (state[i * 3 + 1] + rain) / (total + 0.00000001));
        for d in 0..4 {
            flux[i * 4 + d] *= scale;
        }
    }
}
#[mimir_global]
fn transport(state: &[f32], flux: &[f32], next: &mut [f32], size: i32, rain: f32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < size * size {
        let x = i % size;
        let y = i / size;
        let height = state[i * 3];
        let water = state[i * 3 + 1] + rain;
        let sediment = state[i * 3 + 2];
        let concentration = sediment / (water + 0.00000001);
        let mut incoming = 0.0f32;
        let mut outgoing = 0.0f32;
        let mut carried = sediment;
        let mut slope = 0.0f32;
        for d in 0..4 {
            let mut j = i;
            let mut opposite = 0;
            let mut valid = true;
            if d == 0 {
                j = i - 1;
                opposite = 1;
                valid = x > 0;
            }
            if d == 1 {
                j = i + 1;
                opposite = 0;
                valid = x < size - 1;
            }
            if d == 2 {
                j = i - size;
                opposite = 3;
                valid = y > 0;
            }
            if d == 3 {
                j = i + size;
                opposite = 2;
                valid = y < size - 1;
            }
            let out = flux[i * 4 + d];
            outgoing += out;
            carried -= out * concentration;
            if valid {
                let f = flux[j * 4 + opposite];
                incoming += f;
                carried += f * state[j * 3 + 2] / (state[j * 3 + 1] + rain + 0.00000001);
                slope = max(slope, max(height - state[j * 3], state[j * 3] - height));
            }
        }
        carried = max(0.0, carried);
        let capacity = max(
            0.0,
            (incoming + outgoing) * slope * (size as f32 / 128.0) * 60.0,
        );
        let mut bed = height;
        if carried < capacity {
            let eroded = min(height, min(0.002, (capacity - carried) * 0.12));
            bed -= eroded;
            carried += eroded;
        } else {
            let deposited = (carried - capacity) * 0.15;
            bed += deposited;
            carried -= deposited;
        }
        next[i * 3] = bed;
        next[i * 3 + 1] = max(0.0, (water + incoming - outgoing) * 0.97);
        next[i * 3 + 2] = carried;
    }
}
struct Simulation {
    size: usize,
    state: GpuArray<f32>,
    next: GpuArray<f32>,
    flux: GpuArray<f32>,
    steps: usize,
    seed: u32,
}
impl Simulation {
    fn new(size: usize, state: &[f32], seed: u32) -> Result<Self, Error> {
        Ok(Self {
            size,
            state: GpuArray::from_slice(state)?,
            next: GpuArray::from_slice(&vec![0.0; state.len()])?,
            flux: GpuArray::from_slice(&vec![0.0f32; size * size * 4])?,
            steps: 0,
            seed,
        })
    }
    fn advance(&mut self, steps: usize, rain: f32) -> Result<f64, Error> {
        let size = self.size as i32;
        let blocks = ((self.size * self.size) as u32).div_ceil(128);
        let start = Instant::now();
        for _ in 0..steps {
            // Each invocation owns a cell, writes disjoint slots, and reads only
            // the previous pass. Flux and next-state buffers never alias their inputs.
            unsafe {
                launch!(flow<<<[blocks,1,1],[128,1,1]>>>(&self.state,&self.flux,size,rain))?;
                launch!(transport<<<[blocks,1,1],[128,1,1]>>>(&self.state,&self.flux,&self.next,size,rain))?;
            }
            std::mem::swap(&mut self.state, &mut self.next);
            self.steps += 1;
        }
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }
}
// Scalar, single-thread reference with the same f32 operation order and boundaries.
fn cpu_advance(
    state: &mut Vec<f32>,
    next: &mut Vec<f32>,
    flux: &mut [f32],
    size: usize,
    steps: usize,
    rain: f32,
) {
    for _ in 0..steps {
        for i in 0..size * size {
            let x = i % size;
            let y = i / size;
            let surface = state[i * 3] + state[i * 3 + 1] + rain;
            let mut total = 0.0f32;
            for d in 0..4 {
                let j = neighbor(i, x, y, d, size);
                let f = j
                    .map(|j| ((surface - state[j * 3] - state[j * 3 + 1] - rain) * 0.22).max(0.0))
                    .unwrap_or(0.0);
                flux[i * 4 + d] = f;
                total += f;
            }
            let scale = ((state[i * 3 + 1] + rain) / (total + 0.00000001)).min(1.0);
            for d in 0..4 {
                flux[i * 4 + d] *= scale;
            }
        }
        for i in 0..size * size {
            let x = i % size;
            let y = i / size;
            let height = state[i * 3];
            let water = state[i * 3 + 1] + rain;
            let sediment = state[i * 3 + 2];
            let concentration = sediment / (water + 0.00000001);
            let mut incoming = 0.0f32;
            let mut outgoing = 0.0f32;
            let mut carried = sediment;
            let mut slope = 0.0f32;
            for d in 0..4 {
                let out = flux[i * 4 + d];
                outgoing += out;
                carried -= out * concentration;
                if let Some(j) = neighbor(i, x, y, d, size) {
                    let f = flux[j * 4 + (d ^ 1)];
                    incoming += f;
                    carried += f * state[j * 3 + 2] / (state[j * 3 + 1] + rain + 0.00000001);
                    slope = slope.max((height - state[j * 3]).abs());
                }
            }
            carried = carried.max(0.0);
            let capacity = ((incoming + outgoing) * slope * (size as f32 / 128.0) * 60.0).max(0.0);
            let mut bed = height;
            if carried < capacity {
                let eroded = height.min(0.002f32.min((capacity - carried) * 0.12));
                bed -= eroded;
                carried += eroded;
            } else {
                let deposited = (carried - capacity) * 0.15;
                bed += deposited;
                carried -= deposited;
            }
            next[i * 3] = bed;
            next[i * 3 + 1] = ((water + incoming - outgoing) * 0.97).max(0.0);
            next[i * 3 + 2] = carried;
        }
        std::mem::swap(state, next);
    }
}
fn neighbor(i: usize, x: usize, y: usize, d: usize, size: usize) -> Option<usize> {
    match d {
        0 if x > 0 => Some(i - 1),
        1 if x + 1 < size => Some(i + 1),
        2 if y > 0 => Some(i - size),
        3 if y + 1 < size => Some(i + size),
        _ => None,
    }
}
fn noise(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263) ^ seed;
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) & 65535) as f32 / 65535.0
}
fn terrain(size: usize, seed: u32) -> Vec<f32> {
    let mut state = vec![0.0; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let mut h = 0.0;
            let mut weight = 0.4;
            for frequency in [3.0, 6.0, 12.0, 24.0, 48.0] {
                let px = x as f32 / size as f32 * frequency;
                let py = y as f32 / size as f32 * frequency;
                let ix = px.floor() as i32;
                let iy = py.floor() as i32;
                let fx = px - ix as f32;
                let fy = py - iy as f32;
                let u = fx * fx * (3.0 - 2.0 * fx);
                let v = fy * fy * (3.0 - 2.0 * fy);
                let a = noise(ix, iy, seed) * (1.0 - u) + noise(ix + 1, iy, seed) * u;
                let b = noise(ix, iy + 1, seed) * (1.0 - u) + noise(ix + 1, iy + 1, seed) * u;
                h += (a * (1.0 - v) + b * v) * weight;
                weight *= 0.5;
            }
            state[(y * size + x) * 3] = 0.12 + h;
        }
    }
    state
}
fn bounded(args: &Value, name: &str, default: u64, min: u64, max: u64) -> Result<usize, Error> {
    let n = if args[name].is_null() {
        default
    } else {
        args[name]
            .as_u64()
            .ok_or_else(|| format!("{name} must be an integer"))?
    };
    if n < min || n > max {
        return Err(format!("{name} must be {min}–{max}").into());
    }
    Ok(n as usize)
}
fn rainfall(args: &Value) -> Result<f32, Error> {
    let v = if args["rain"].is_null() {
        0.002
    } else {
        args["rain"].as_f64().ok_or("rain must be numeric")?
    };
    if !v.is_finite() || !(0.0..=0.01).contains(&v) {
        return Err("rain must be between 0 and 0.01".into());
    }
    Ok(v as f32)
}
fn snapshot(id: u64, sim: &Simulation, gpu_ms: f64) -> Result<Value, Error> {
    let start = Instant::now();
    let state = sim.state.to_vec()?;
    let readback_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut height = String::with_capacity(PREVIEW * PREVIEW * 2);
    let mut water = String::with_capacity(PREVIEW * PREVIEW * 2);
    use std::fmt::Write;
    let stride = sim.size / PREVIEW;
    for y in 0..PREVIEW {
        for x in 0..PREVIEW {
            let mut h = 0.0;
            let mut w = 0.0;
            for dy in 0..stride {
                for dx in 0..stride {
                    let i = ((y * stride + dy) * sim.size + x * stride + dx) * 3;
                    h += state[i];
                    w += state[i + 1];
                }
            }
            let n = (stride * stride) as f32;
            write!(height, "{:02x}", (h / n * 255.0).clamp(0.0, 255.0) as u8)?;
            write!(water, "{:02x}", (w / n * 2000.0).clamp(0.0, 255.0) as u8)?;
        }
    }
    Ok(
        json!({"simulation":id,"size":sim.size,"seed":sim.seed,"steps":sim.steps,"preview_size":PREVIEW,"height":height,"water":water,"gpu_ms":gpu_ms,"readback_ms":readback_ms}),
    )
}
fn comparison(sim: &Simulation, steps: usize, rain: f32) -> Result<Value, Error> {
    let source = sim.state.to_vec()?;
    let mut gpu = Simulation::new(sim.size, &source, sim.seed)?;
    // Scratch warm-up leaves the actual comparison input untouched.
    gpu.advance(1, rain)?;
    let mut gpu = Simulation::new(sim.size, &source, sim.seed)?;
    let mut cpu = source.clone();
    let mut next = vec![0.0; source.len()];
    let mut flux = vec![0.0; sim.size * sim.size * 4];
    let gpu_ms = gpu.advance(steps, rain)?;
    let start = Instant::now();
    let actual = gpu.state.to_vec()?;
    let readback_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    cpu_advance(&mut cpu, &mut next, &mut flux, sim.size, steps, rain);
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut max = 0.0f32;
    let mut terrain_max = 0.0f32;
    let mut squared = 0.0f64;
    for (i, (a, b)) in actual.iter().zip(&cpu).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err("simulation produced non-finite values".into());
        }
        let e = (a - b).abs();
        max = max.max(e);
        if i % 3 == 0 {
            terrain_max = terrain_max.max(e);
        }
        squared += (e as f64).powi(2);
    }
    let tolerance = 0.002f32;
    Ok(
        json!({"comparison":true,"size":sim.size,"steps":steps,"gpu_ms":gpu_ms,"cpu_ms":cpu_ms,"readback_ms":readback_ms,"speedup":cpu_ms/gpu_ms.max(0.000001),"max_error":max,"terrain_max_error":terrain_max,"rmse":(squared/cpu.len() as f64).sqrt(),"tolerance":tolerance,"passed":max<=tolerance}),
    )
}
pub fn run(path: &str, args: &Value) -> Result<Value, Error> {
    let rain = rainfall(args)?;
    if path == "/api/erosion-create" {
        let size = bounded(args, "size", 512, 128, 512)?;
        if ![128, 256, 512].contains(&size) {
            return Err("size must be 128, 256 or 512".into());
        }
        let seed = bounded(args, "seed", 42, 0, u32::MAX as u64)? as u32;
        let initial = terrain(size, seed);
        let mut warm = Simulation::new(size, &initial, seed)?;
        warm.advance(1, rain)?;
        let sim = Simulation::new(size, &initial, seed)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let result = snapshot(id, &sim, 0.0)?;
        let mut map = SIMULATIONS
            .lock()
            .map_err(|_| "simulation store unavailable")?;
        map.insert(id, Arc::new(Mutex::new(sim)));
        while map.len() > 4 {
            map.pop_first();
        }
        return Ok(result);
    }
    let id = args["simulation"].as_u64().ok_or("missing simulation id")?;
    let sim = SIMULATIONS
        .lock()
        .map_err(|_| "simulation store unavailable")?
        .get(&id)
        .cloned()
        .ok_or("Terrain expired or backend restarted. Create a new terrain.")?;
    let mut sim = sim
        .try_lock()
        .map_err(|_| "This terrain is busy. Wait for the current operation.")?;
    let steps = bounded(args, "steps", 32, 1, 128)?;
    if path == "/api/erosion-compare" {
        return comparison(&sim, steps, rain);
    }
    if path != "/api/erosion-step" {
        return Err("unknown erosion operation".into());
    }
    let ms = sim.advance(steps, rain)?;
    snapshot(id, &sim, ms)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_is_finite_and_conserves_terrain_plus_sediment() {
        let n = 32;
        let mut state = terrain(n, 42);
        let original = state.clone();
        let before: f64 = state.chunks_exact(3).map(|p| (p[0] + p[2]) as f64).sum();
        let mut next = vec![0.0; state.len()];
        let mut flux = vec![0.0; n * n * 4];
        cpu_advance(&mut state, &mut next, &mut flux, n, 128, 0.002);
        assert!(state.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(state
            .iter()
            .zip(original)
            .any(|(a, b)| (a - b).abs() > 0.001));
        let after: f64 = state.chunks_exact(3).map(|p| (p[0] + p[2]) as f64).sum();
        assert!((before - after).abs() < 0.001);
    }
    #[test]
    fn dry_terrain_without_rain_is_unchanged() {
        let mut state = terrain(32, 7);
        let before = state.clone();
        let mut next = vec![0.0; state.len()];
        let mut flux = vec![0.0; 32 * 32 * 4];
        cpu_advance(&mut state, &mut next, &mut flux, 32, 16, 0.0);
        assert_eq!(state, before);
    }
    #[test]
    #[ignore = "requires GPU"]
    fn gpu_matches_reference_and_comparison_does_not_advance_session() -> Result<(), Error> {
        let initial = terrain(128, 17);
        let sim = Simulation::new(128, &initial, 17)?;
        let result = comparison(&sim, 64, 0.002)?;
        assert_eq!(result["passed"], true);
        assert_eq!(sim.steps, 0);
        assert_eq!(sim.state.to_vec()?, initial);
        Ok(())
    }
}
