# Mimir

High-performance Kernel programming language extension for Rust

## Overview
---
Mimir is the Norse god of wisdom who happens to be a floating head in a well. GPU programming is a lot like Mimir I've found. Nearly boundless knowledge and potential but as difficult to interact with as a disembodied head that is barely afloat.

Mimir the library aims to address this by enabling the simplicity of CUDA kernel programming with the widespread adoption of the Vulkan standards.

This project is still in early development, expect breaking changes and/or unexpected behavior. 

## Example
---
Below is an example taken directly from the naive Matmul example
```rust
// Matrix multiplication kernel (A @ B = C) with global memory coalescing
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
```
Running the kernel:

```rust
    const M: usize = 256;
    const N: usize = 256;
    const K: usize = 256;

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
```

If you wish to use Mimir in a project include `mimir-gpu` and your backend of choice (only `mimir-vulkan` as of now). Import whatever macros you need from mimir-gpu and import all from your backend of choice (i.e. `use mimir_vulkan::*;`)


## Development
---
So far this has been my passion project and solo developed by me. I'm always adding new features and working on new concepts. I'd love to continue this project and I'd appreciate any and all help I can get. If anybody is interested/willing to contribute please email me (caleb.cleavinger@gmail.com).

## Roadmap
---

- [x] Binary intermediary format; smaller file sizes, more efficient JIT compilation, and langauge agnostic
- [x] Add shared memory features
- [ ] Stricter typing
    - specification checks do exist but are not good enough yet 
- [ ] Kernel Templating
    - [x] Constant generics
    - [ ] Type Generics  
- [ ] CUDA backend
- [ ] Tenstorrent/RISC-V accelerator backend???
