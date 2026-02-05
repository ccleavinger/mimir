use mimir_macros::{launch, mimir_global};
use mimir_omni_runtime::*;
use rand::Rng;
use std::time::Instant;

const M: i32 = 256*4;
const N: i32 = 256*4;
const K: i32 = 256*4;

#[mimir_global]
fn matmul_naive_kernel(A: &[f32], B: &[f32], C: &mut [f32], m: i32, n: i32, k: i32) {
    let row = (block_idx.y * block_dim.y + thread_idx.y) as i32;
    let col = (block_idx.x * block_dim.x + thread_idx.x) as i32;

    if row < m && col < k {
        let mut sum = 0.0;
        for i in 0..n {
            sum += A[(row * n + i) as i32] * B[(i * k + col) as i32];
        }
        C[(row * k + col) as i32] = sum;
    }
}

#[mimir_global]
fn sgemm_shared_mem_block<const BLOCKSIZE: u32>(
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    A: &[f32],
    B: &[f32],
    beta: f32,
    C: &mut [f32],
) {
    let c_row = block_idx.x;  // Which C tile row this block computes
    let c_col = block_idx.y;  // Which C tile col this block computes

    #[shared]
    let mut As = [0f32; (BLOCKSIZE * BLOCKSIZE)];
    #[shared]
    let mut Bs = [0f32; (BLOCKSIZE * BLOCKSIZE)];

    // Calculate 1D thread index and convert to row/col within block
    let thread_id = (thread_idx.y * block_dim.x + thread_idx.x);
    let thread_row = thread_id / BLOCKSIZE;
    let thread_col = thread_id % BLOCKSIZE;

    // Starting positions in global memory
    let mut a_offset = c_row * BLOCKSIZE * (k as u32);
    let mut b_offset = c_col * BLOCKSIZE;
    let c_offset = c_row * BLOCKSIZE * (n as u32) + c_col * BLOCKSIZE;

    // Each thread accumulates its result here
    let mut tmp: f32 = 0.0;

    // Loop over tiles in K dimension
    for bk_idx in (0..k).step_by(BLOCKSIZE as i32) {
        // Load tile from A into shared memory (As)
        if thread_row < BLOCKSIZE && thread_col < BLOCKSIZE {
            let a_global_row = c_row * BLOCKSIZE + thread_row;
            let a_global_col = (bk_idx as u32) + thread_col;
            
            if a_global_row < (m as u32) && a_global_col < (k as u32) {
                let a_idx = a_global_row * (k as u32) + a_global_col;
                As[(thread_row * BLOCKSIZE + thread_col) as i32] = A[a_idx as i32];
            } else {
                As[(thread_row * BLOCKSIZE + thread_col) as i32] = 0.0;
            }
        }

        // Load tile from B into shared memory (Bs)
        if thread_row < BLOCKSIZE && thread_col < BLOCKSIZE {
            let b_global_row = (bk_idx as u32) + thread_row;
            let b_global_col = c_col * BLOCKSIZE + thread_col;
            
            if b_global_row < (k as u32) && b_global_col < (n as u32) {
                let b_idx = b_global_row * (n as u32) + b_global_col;
                Bs[(thread_row * BLOCKSIZE + thread_col) as i32] = B[b_idx as i32];
            } else {
                Bs[(thread_row * BLOCKSIZE + thread_col) as i32] = 0.0;
            }
        }

        __syncthreads();

        // Compute dot product for this tile
        if thread_row < BLOCKSIZE && thread_col < BLOCKSIZE {
            for dot_idx in 0..(BLOCKSIZE as i32) {
                tmp += As[(thread_row * BLOCKSIZE + (dot_idx as u32)) as i32] *
                       Bs[((dot_idx as u32) * BLOCKSIZE + thread_col) as i32];
            }
        }

        __syncthreads();
    }

    // Write result to C
    if thread_row < BLOCKSIZE && thread_col < BLOCKSIZE {
        let c_global_row = c_row * BLOCKSIZE + thread_row;
        let c_global_col = c_col * BLOCKSIZE + thread_col;
        
        if c_global_row < (m as u32) && c_global_col < (n as u32) {
            let c_idx = c_global_row * (n as u32) + c_global_col;
            C[c_idx as i32] = alpha * tmp + beta * C[c_idx as i32];
        }
    }
}

fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: i32, n: i32, k: i32) {
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0;
            for p in 0..n {
                sum += a[(i * n + p) as usize] * b[(p * k + j) as usize];
            }
            c[(i * k + j) as usize] = sum;
        }
    }
}

fn run_naive(
    a_gpu: &MimirGPUArray<f32>,
    b_gpu: &MimirGPUArray<f32>,
    mut c_gpu: MimirGPUArray<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure launch parameters
    let block_dim = [16, 16, 1];
    let grid_dim = [
        (K as u32).div_ceil(block_dim[0]),
        (M as u32).div_ceil(block_dim[1]),
        1,
    ];

    // Naive matmul warm-up run
    launch! {
        matmul_naive_kernel<<<grid_dim, block_dim>>>(
            &a_gpu,
            &b_gpu,
            &mut c_gpu,
            M,
            N,
            K,
        );
    }?;

    // Timed run
    let start_time = Instant::now();
    launch! {
        matmul_naive_kernel<<<grid_dim, block_dim>>>(
            &a_gpu,
            &b_gpu,
            &mut c_gpu,
            M,
            N,
            K
        );
    }?;
    let duration = start_time.elapsed();

    println!(
        "Naive GPU Matmul ({}x{}x{}) took: {:.2?}",
        M, N, K, duration
    );
    Ok(())
}

fn run_sgemm_shared_block(
    a_gpu: &MimirGPUArray<f32>,
    b_gpu: &MimirGPUArray<f32>,
    mut c_gpu: MimirGPUArray<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    const BETA: f32 = 0.0f32;
    const ALPHA: f32 = 1.0f32;
    const BLOCKSIZE: u32 = 16;

    // For tiled version, need BLOCKSIZE^2 threads per block
    let block_dim = [BLOCKSIZE, BLOCKSIZE, 1];
    let grid_dim = [
        (M as u32).div_ceil(BLOCKSIZE),
        (K as u32).div_ceil(BLOCKSIZE),
        1,
    ];

    println!("Launching tiled kernel with grid: {:?}, block: {:?}", grid_dim, block_dim);

    // sgemm shared block warm-up run
    launch! {
        sgemm_shared_mem_block<BLOCKSIZE><<<grid_dim, block_dim>>>(
            M,
            K,
            N,
            ALPHA,
            &a_gpu,
            &b_gpu,
            BETA,
            &mut c_gpu,
        );
    }?;

    // Timed run
    let start_time = Instant::now();
    launch! {
        sgemm_shared_mem_block<BLOCKSIZE><<<grid_dim, block_dim>>>(
            M,
            K,
            N,
            ALPHA,
            &a_gpu,
            &b_gpu,
            BETA,
            &mut c_gpu,
        );
    }?;
    let duration = start_time.elapsed();

    println!(
        "Sgemm shared mem block GPU Matmul ({}x{}x{}) took: {:.2?}",
        M, N, K, duration
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();

    // Initialize matrices with correct dimensions
    let a_host: Vec<f32> = (0..(M * N) as usize).map(|_| rng.random::<f32>()).collect();
    let b_host: Vec<f32> = (0..(N * K) as usize).map(|_| rng.random::<f32>()).collect();

    {
        // Allocate GPU memory
        let a_gpu = mimir_gpu_arr::from_iter(a_host.clone())?;
        let b_gpu = mimir_gpu_arr::from_iter(b_host.clone())?;
        let c_gpu = mimir_gpu_arr::from_iter(vec![0.0f32; (M * K) as usize])?;

        run_naive(&a_gpu, &b_gpu, c_gpu.clone())?;

        let c_gpu2 = mimir_gpu_arr::from_iter(vec![0.0f32; (M * K) as usize])?;
        run_sgemm_shared_block(&a_gpu, &b_gpu, c_gpu2.clone())?;

        // Verify results for naive kernel
        let c_host_gpu: Vec<f32> = c_gpu.to_iter().collect();
        let mut c_host_cpu = vec![0.0f32; (M * K) as usize];

        let start_time = Instant::now();
        cpu_matmul(&a_host, &b_host, &mut c_host_cpu, M, N, K);
        let duration = start_time.elapsed();

        println!(
            "CPU Matmul ({}x{}x{}) took: {:.2?}",
            M, N, K, duration
        );

        let mut naive_mismatch = false;
        for i in 0..(M * K) as usize {
            if (c_host_gpu[i] - c_host_cpu[i]).abs() > 1e-3 {
                println!(
                    "Naive kernel mismatch at index {}: GPU result = {}, CPU result = {}",
                    i, c_host_gpu[i], c_host_cpu[i]
                );
                naive_mismatch = true;
                if i < 10 { break; } // Show first few mismatches
            }
        }
        if !naive_mismatch {
            println!("Naive kernel verification passed!");
        }

        // Verify results for tiled kernel
        let c_host_gpu2: Vec<f32> = c_gpu2.to_iter().collect();
        let mut tiled_mismatch = false;
        for i in 0..(M * K) as usize {
            if (c_host_gpu2[i] - c_host_cpu[i]).abs() > 1e-3 {
                println!(
                    "Tiled kernel mismatch at index {}: GPU result = {}, CPU result = {}",
                    i, c_host_gpu2[i], c_host_cpu[i]
                );
                tiled_mismatch = true;
                if i < 10 { break; } // Show first few mismatches
            }
        }
        if !tiled_mismatch {
            println!("Tiled kernel verification passed!");
        }
    }

    Ok(())
}