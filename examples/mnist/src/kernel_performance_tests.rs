//! Tests for MNIST kernel performance

use std::time::Instant;
use mimir::{launch, mimir_global};
use mimir_vulkan::*;

// Import kernels from mnist_nn
use crate::mnist_nn::{INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};

#[test]
fn test_kernel_performance() {
    let batch_size = 32;
    let m = batch_size as i32;
    let n = INPUT_SIZE as i32;
    let k = HIDDEN_SIZE as i32;
    let output_k = OUTPUT_SIZE as i32;
    let size = m * k;

    // Allocate dummy data as GPUArray
    let a = mimir_vulkan::GPUArray::from_iter(vec![1.0f32; (m * n) as usize].into_iter());
    let b = mimir_vulkan::GPUArray::from_iter(vec![1.0f32; (n * k) as usize].into_iter());
    let mut c = mimir_vulkan::GPUArray::from_iter(vec![0.0f32; (m * k) as usize].into_iter());
    let bias = mimir_vulkan::GPUArray::from_iter(vec![0.5f32; k as usize].into_iter());
    let mut grad = mimir_vulkan::GPUArray::from_iter(vec![0.0f32; (m * k) as usize].into_iter());
    let labels = mimir_vulkan::GPUArray::from_iter(vec![0i32; m as usize].into_iter());

    // Kernel launch configs
    let block_size_1d = [256, 1, 1];
    let grid_size_1d = [((m * k) as u32 + 255) / 256, 1, 1];
    let block_size_2d = [32, 32, 1];
    let grid_size_2d = [((k as u32 + 31) / 32), ((m as u32 + 31) / 32), 1];
    let grid_size_softmax = [batch_size as u32, 1, 1];
    let block_size_softmax = [1, 1, 1];

    // Test each kernel with launch!
    let start = Instant::now();
    launch! {
        matmul_a_b_kernel<<<grid_size_2d, block_size_2d>>>(
            &a, &b, &c, m, n, k
        );
    }.unwrap();
    println!("matmul_a_b_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        matmul_a_bt_kernel<<<grid_size_2d, block_size_2d>>>(
            &a, &b, &c, m, n, k
        );
    }.unwrap();
    println!("matmul_a_bt_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        matmul_at_b_kernel<<<grid_size_2d, block_size_2d>>>(
            &a, &b, &c, m, n, k
        );
    }.unwrap();
    println!("matmul_at_b_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        relu_kernel<<<grid_size_1d, block_size_1d>>>(
            &c, m * k
        );
    }.unwrap();
    println!("relu_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        bias_add_kernel<<<grid_size_1d, block_size_1d>>>(
            &c, &bias, m, k
        );
    }.unwrap();
    println!("bias_add_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        softmax_kernel<<<grid_size_softmax, block_size_softmax>>>(
            &c, m, k
        );
    }.unwrap();
    println!("softmax_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        zero_grad_kernel<<<grid_size_1d, block_size_1d>>>(
            &grad, m * k
        );
    }.unwrap();
    println!("zero_grad_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        compute_output_gradients_kernel<<<grid_size_1d, block_size_1d>>>(
            &c, &a, &labels, m
        );
    }.unwrap();
    println!("compute_output_gradients_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        drelu_kernel<<<grid_size_1d, block_size_1d>>>(
            &c, &grad, m * k
        );
    }.unwrap();
    println!("drelu_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        multiply_gradients_kernel<<<grid_size_1d, block_size_1d>>>(
            &grad, &c, m * k
        );
    }.unwrap();
    println!("multiply_gradients_kernel: {:?}", start.elapsed());

    let start = Instant::now();
    launch! {
        update_weights_kernel<<<grid_size_1d, block_size_1d>>>(
            &c, &grad, m * k
        );
    }.unwrap();
    println!("update_weights_kernel: {:?}", start.elapsed());
}
