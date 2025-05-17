use mimir::{launch, mimir_global};
use mimir_vulkan::*;
use rand::Rng;
use std::time::Instant;

const M: usize = 256;
const N: usize = 256;
const K: usize = 256;

#[mimir_global]
fn matmul_naive_kernel(A: &[f32], B: &[f32], C: &mut [f32], m: i32, n: i32, k: i32) {
    let row = block_idx.y * block_dim.y + thread_idx.y;
    let col = block_idx.x * block_dim.x + thread_idx.x;

    if row < m && col < k {
        let mut sum = 0.0;
        for i in 0..n {
            sum += A[(row * n + i) as usize] * B[(i * k + col) as usize];
        }
        C[(row * k + col) as usize] = sum;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    // Initialize matrices
    let a_host: Vec<f32> = (0..M * N).map(|_| rng.random::<f32>()).collect();
    let b_host: Vec<f32> = (0..N * K).map(|_| rng.random::<f32>()).collect();

    // Allocate GPU memory
    let a_gpu = GPUArray::from_iter(a_host);
    let b_gpu = GPUArray::from_iter(b_host);
    let mut c_gpu = GPUArray::from_iter(vec![0.0f32; M * K]);

    // Configure launch parameters
    let block_dim = [16, 16, 1];
    let grid_dim = [
        (K as u32).div_ceil(block_dim[0]),
        (M as u32).div_ceil(block_dim[1]),
        1,
    ];

    // Warm-up run
    launch! {
        matmul_naive_kernel<<<grid_dim, block_dim>>>(
            &a_gpu,
            &b_gpu,
            &mut c_gpu,
            M as i32,
            N as i32,
            K as i32
        );
    }?;

    // Timed run
    let start_time = Instant::now();
    launch! {
        matmul_naive_kernel<<<grid_dim, block_dim>>>(
            &a_gpu,
            &b_gpu,
            &mut c_gpu,
            M as i32,
            N as i32,
            K as i32
        );
    }?;
    let duration = start_time.elapsed();

    println!(
        "GPU Matmul ({}x{}x{}) took: {:.2?}",
        M, N, K, duration
    );
    
    // Optional: Add CPU matmul for verification and comparison
    // ...

    Ok(())
}
