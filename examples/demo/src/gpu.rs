use super::Error;
use mimir::{launch, mimir_global, GpuArray};
use serde_json::{json, Value};
use std::time::Instant;

#[mimir_global]
fn mandelbrot(out: &mut [f32], width: i32, height: i32, limit: i32, zoom: f32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < width * height {
        let cx = ((i % width) as f32 / width as f32 - 0.5) * 3.2 / zoom - 0.7436439;
        let cy = ((i / width) as f32 / height as f32 - 0.5) * 2.4 / zoom + 0.1318259;
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        let mut n = 0;
        for step in 0..limit {
            if x * x + y * y < 4.0 {
                let next = x * x - y * y + cx;
                y = 2.0 * x * y + cy;
                x = next;
                n += 1;
            }
        }
        let mut r = 12.0f32;
        let mut g = 20.0f32;
        let mut b = 30.0f32;
        if n < limit {
            let t = n as f32 * 0.09;
            r = 128.0 + 127.0 * sin(t);
            g = 128.0 + 127.0 * sin(t + 2.0);
            b = 128.0 + 127.0 * sin(t + 4.0);
        }
        out[i * 4] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = 255.0;
    }
}

#[mimir_global]
fn blur(input: &[f32], out: &mut [f32], width: i32, height: i32, radius: i32, horizontal: bool) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < width * height * 4 {
        let channel = i % 4;
        let x = (i / 4) % width;
        let y = i / 4 / width;
        let mut sum = 0.0f32;
        for offset in -radius..radius + 1 {
            let mut sx = x;
            let mut sy = y;
            if horizontal {
                sx = min(max(x + offset, 0), width - 1);
            } else {
                sy = min(max(y + offset, 0), height - 1);
            }
            sum += input[(sy * width + sx) * 4 + channel];
        }
        out[i] = sum / (radius * 2 + 1) as f32;
    }
}

pub fn run(path: &str, args: &Value) -> Result<Value, Error> {
    if path.starts_with("/api/teach-") {
        return super::teach::run(path, args);
    }
    if path == "/api/mnist" {
        return super::mnist::predict(args);
    }
    if path == "/api/status" {
        let devices = mimir::default_runtime()?.devices()?;
        return Ok(
            json!({"devices": devices.iter().map(|d| d.name()).collect::<Vec<_>>(),
            "backend": if cfg!(feature="cuda") { "CUDA" } else if cfg!(feature="vulkan") { "Vulkan" } else { "none" },
            "demos": ["fractal", "blur", "mnist", "teach"]}),
        );
    }
    let start = Instant::now();
    let (width, height, pixels, detail) = if path == "/api/fractal" {
        let zoom = args["zoom"].as_f64().unwrap_or(1.0);
        if !zoom.is_finite() || !(1.0..=10000.0).contains(&zoom) {
            return Err("zoom must be 1–10000".into());
        }
        let width = 960i32;
        let height = 720i32;
        let out = GpuArray::from_slice(&vec![0.0f32; (width * height * 4) as usize])?;
        // Each invocation writes one pixel; the guard covers the rounded-up grid.
        unsafe {
            launch!(mandelbrot<<<[(width * height) as u32 / 128,1,1],[128,1,1]>>>(&out,width,height,512,zoom as f32))?;
        }
        (
            width,
            height,
            out.to_vec()?,
            "691,200 parallel pixels · up to 512 iterations per pixel".to_owned(),
        )
    } else if path == "/api/blur" {
        let width = args["width"].as_i64().ok_or("missing width")?;
        let height = args["height"].as_i64().ok_or("missing height")?;
        let radius = args["radius"].as_i64().ok_or("missing radius")?;
        if !(1..=768).contains(&width)
            || !(1..=768).contains(&height)
            || !(1..=32).contains(&radius)
        {
            return Err("image must be at most 768×768; radius must be 1–32".into());
        }
        let values = args["pixels"].as_array().ok_or("missing RGBA pixels")?;
        if values.len() != (width * height * 4) as usize {
            return Err("invalid pixel count".into());
        }
        let pixels = values
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|v| *v <= 255)
                    .map(|v| v as f32)
                    .ok_or("invalid pixel value")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = GpuArray::from_slice(&pixels)?;
        let temp = GpuArray::from_slice(&vec![0.0f32; pixels.len()])?;
        let out = GpuArray::from_slice(&vec![0.0f32; pixels.len()])?;
        let blocks = (pixels.len() as u32).div_ceil(128);
        // Separate buffers avoid races; clamped sample coordinates are in bounds.
        unsafe {
            launch!(blur<<<[blocks,1,1],[128,1,1]>>>(&input,&temp,width as i32,height as i32,radius as i32,true))?;
            launch!(blur<<<[blocks,1,1],[128,1,1]>>>(&temp,&out,width as i32,height as i32,radius as i32,false))?;
        }
        (
            width as i32,
            height as i32,
            out.to_vec()?,
            format!("Two-pass box blur · radius {radius}px"),
        )
    } else {
        return Err("unknown operation".into());
    };
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    Ok(
        json!({"image": super::bitmap(width as u32,height as u32,&pixels), "width": width,"height":height,"elapsed_ms":elapsed,"detail":detail}),
    )
}
