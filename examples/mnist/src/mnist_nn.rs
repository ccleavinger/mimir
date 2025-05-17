use mimir::{launch, mimir_global};
use mimir_vulkan::*;
use mnist::MnistBuilder;
use rand::distr::{Distribution, Uniform};
use std::time::Instant;
use indicatif::ProgressBar;

// Constants matching the CUDA example
pub const INPUT_SIZE: usize = 784; // 28x28 pixels
pub const HIDDEN_SIZE: usize = 4096;
pub const OUTPUT_SIZE: usize = 10;
pub const BATCH_SIZE: usize = 32;
pub const EPOCHS: usize = 20;
pub const LEARNING_RATE: f32 = 0.05;

// Neural Network Structure
pub struct NeuralNetwork {
    weights1: GPUArray<f32>,
    weights2: GPUArray<f32>,
    bias1: GPUArray<f32>,
    bias2: GPUArray<f32>,
    grad_weights1: GPUArray<f32>,
    grad_weights2: GPUArray<f32>,
    grad_bias1: GPUArray<f32>,
    grad_bias2: GPUArray<f32>,
}

// Matrix multiplication kernel (A @ B = C) with global memory coalescing
#[mimir_global]
fn matmul_a_b_kernel(
    A: &[f32],
    B: &[f32],
    C: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) {
    // BLOCKSIZE should match the launch configuration
    let BLOCKSIZE = 32;
    let cRow = block_idx.x * BLOCKSIZE + (thread_idx.x / BLOCKSIZE);
    let cCol = block_idx.y * BLOCKSIZE + (thread_idx.x % BLOCKSIZE);

    if cRow < m && cCol < k {
        let mut tmp = 0.0;
        for i in 0..n {
            tmp += A[(cRow * n + i) ] * B[(i * k + cCol) ];
        }
        // alpha = 1.0, beta = 0.0 for this use case
        C[(cRow * k + cCol)] = tmp;
    }
}

// Matrix multiplication kernel (A @ B.T = C)
#[mimir_global]
fn matmul_a_bt_kernel(
    A: &[f32],
    B: &[f32],
    C: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) {
    let row = block_idx.y * block_dim.y + thread_idx.y;
    let col = block_idx.x * block_dim.x + thread_idx.x;

    if row < m && col < k {
        let mut sum = 0.0;
        for i in 0..n {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

// Matrix multiplication kernel (A.T @ B = C)
#[mimir_global]
fn matmul_at_b_kernel(
    A: &[f32],
    B: &[f32],
    C: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) {
    let row = block_idx.y * block_dim.y + thread_idx.y;
    let col = block_idx.x * block_dim.x + thread_idx.x;

    if row < n && col < k {
        let mut sum = 0.0;
        for i in 0..m {
            sum += A[i  * n  + row ] * B[i  * k  + col ];
        }
        C[row  * k  + col ] = sum;
    }
}

// ReLU activation kernel
#[mimir_global]
fn relu_kernel(x: &mut [f32], size: i32) {
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    if idx < size {
        x[idx] = max(x[idx], 0.0);
    }
}

// Bias addition kernel
#[mimir_global]
fn bias_add_kernel(x: &mut [f32], bias: &[f32], batch_size: i32, size: i32) {
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    let b = idx / size;
    let i = idx % size;

    if b < batch_size && i < size {
        x[b * size + i] += bias[i];
    }
}

// Softmax kernel
#[mimir_global]
fn softmax_kernel(x: &mut [f32], batch_size: i32, size: i32) {
    let b = block_idx.x;
    if b < batch_size {
        // Find max for numerical stability
        let mut max_val = f32::MIN;
        for i in 0..size {
            max_val = max(max_val, x[b * size + i]);
        }
        
        // Compute exp and sum
        let mut sum = 0.0;
        for i in 0..size {
            let exp_val = exp(x[b * size + i] - max_val);
            x[b * size + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for i in 0..size {
            x[b * size + i] /= sum;
        }
    }
}

// Zero gradient kernel
#[mimir_global]
fn zero_grad_kernel(grad: &mut [f32], size: i32) {
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    if idx < size {
        grad[idx] = 0.0;
    }
}

// Compute output gradients kernel
#[mimir_global]
fn compute_output_gradients_kernel(
    grad_output: &mut [f32],
    output: &[f32],
    labels: &[i32],
    batch_size: i32,
) {
    let OUTPUT_SIZE = 10;

    let b = block_idx.x * block_dim.x + thread_idx.x;
    if b < batch_size {
        let output_size = OUTPUT_SIZE;
        
        // Copy output probabilities to gradients
        for i in 0..output_size {
            grad_output[b * output_size + i] = output[b * output_size + i];
        }
        
        // Subtract 1.0 from the correct class (y_pred - one_hot(y_true))
        let label = labels[b];
        if label < output_size {
            grad_output[b * output_size + label] -= 1.0;
        }
        
        // Divide by batch size (average over batch)
        for i in 0..output_size {
            grad_output[b * output_size + i] /= batch_size;
        }
    }
}

// ReLU derivative kernel
#[mimir_global]
fn drelu_kernel(x: &[f32], d_relu_out: &mut [f32], size: i32) {
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    if idx < size {
        d_relu_out[idx] = if x[idx] > 0.0 { 1.0 } else { 0.0 };
    }
}

// Element-wise multiplication of gradients
#[mimir_global]
fn multiply_gradients_kernel(grad1: &mut [f32], grad2: &[f32], size: i32) {
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    if idx < size {
        grad1[idx] *= grad2[idx];
    }
}

// Update weights kernel
#[mimir_global]
fn update_weights_kernel(weights: &mut [f32], grad_weights: &[f32], size: i32) {
    let LEARNING_RATE = 0.05;
    let idx = block_idx.x * block_dim.x + thread_idx.x;
    if idx < size {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

impl NeuralNetwork {
    pub fn new() -> anyhow::Result<Self> {
        // Initialize random weights with Kaiming initialization
        let mut rng = rand::rng();
        let scale_w1 = (2.0 / INPUT_SIZE as f32).sqrt();
        let scale_w2 = (2.0 / HIDDEN_SIZE as f32).sqrt();
        
        let w1_dist = Uniform::new(-scale_w1, scale_w1).unwrap();
        let w2_dist = Uniform::new(-scale_w2, scale_w2).unwrap();
        
        let mut weights1 = vec![0.0; HIDDEN_SIZE * INPUT_SIZE];
        let mut weights2 = vec![0.0; OUTPUT_SIZE * HIDDEN_SIZE];
        let mut bias1 = vec![0.0; HIDDEN_SIZE];
        let mut bias2 = vec![0.0; OUTPUT_SIZE];
        
        for w in weights1.iter_mut() {
            *w = w1_dist.sample(&mut rng);
        }
        
        for w in weights2.iter_mut() {
            *w = w2_dist.sample(&mut rng);
        }
        
        // Initialize biases to zeros
        for b in bias1.iter_mut() {
            *b = 0.0;
        }
        
        for b in bias2.iter_mut() {
            *b = 0.0;
        }
        
        // Create GPU arrays
        let weights1_gpu = GPUArray::from_iter(weights1.into_iter());
        let weights2_gpu = GPUArray::from_iter(weights2.into_iter());
        let bias1_gpu = GPUArray::from_iter(bias1.into_iter());
        let bias2_gpu = GPUArray::from_iter(bias2.into_iter());
        
        // Create gradient arrays (initialized to zeros during training)
        let grad_weights1_gpu = GPUArray::from_iter(vec![0.0; HIDDEN_SIZE * INPUT_SIZE].into_iter());
        let grad_weights2_gpu = GPUArray::from_iter(vec![0.0; OUTPUT_SIZE * HIDDEN_SIZE].into_iter());
        let grad_bias1_gpu = GPUArray::from_iter(vec![0.0; HIDDEN_SIZE].into_iter());
        let grad_bias2_gpu = GPUArray::from_iter(vec![0.0; OUTPUT_SIZE].into_iter());
        
        Ok(NeuralNetwork {
            weights1: weights1_gpu,
            weights2: weights2_gpu,
            bias1: bias1_gpu,
            bias2: bias2_gpu,
            grad_weights1: grad_weights1_gpu,
            grad_weights2: grad_weights2_gpu,
            grad_bias1: grad_bias1_gpu,
            grad_bias2: grad_bias2_gpu,
        })
    }
    
    pub fn forward(
        &self,
        input: &GPUArray<f32>,
        hidden: &mut GPUArray<f32>,
        output: &mut GPUArray<f32>,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        // Define block and grid sizes
        let block_size = [32, 32, 1] as [u32; 3];
        let grid_size_hidden = [
            (HIDDEN_SIZE as u32 + block_size[0] - 1) / block_size[0],
            (batch_size as u32 + block_size[1] - 1) / block_size[1],
            1,
        ] as [u32; 3];


        //println!("Input to Hidden (X @ W1)");
        launch! {
            matmul_a_b_kernel<<<grid_size_hidden, block_size>>>(
                input,
                &self.weights1,
                hidden,
                batch_size as i32,
                INPUT_SIZE as i32,
                HIDDEN_SIZE as i32
            );
        }?;
        //println!("Finished Input to Hidden (X @ W1)");
        
        //println!("Add bias1");
        let bias_block_size = 256;
        let bias_grid_size = ((batch_size * HIDDEN_SIZE) as u32 + bias_block_size - 1) / bias_block_size;

        {
            let block_size = [bias_block_size, 1, 1] as [u32; 3];
            let grid_size = [bias_grid_size, 1, 1] as [u32; 3];

            launch! {
                bias_add_kernel<<<grid_size, block_size>>>(
                    hidden,
                    &self.bias1,
                    batch_size as i32,
                    HIDDEN_SIZE as i32
                );
            }?;
        }
        //println!("Finished add bias1");

        //println!("Apply ReLU");
        let relu_block_size = 256;
        let relu_grid_size = ((batch_size * HIDDEN_SIZE) as u32 + relu_block_size - 1) / relu_block_size;

        {
            let block_size = [relu_block_size, 1, 1] as [u32; 3];
            let grid_size = [relu_grid_size, 1, 1] as [u32; 3];

            launch! {
                relu_kernel<<<grid_size, block_size>>>(
                    hidden,
                    (batch_size * HIDDEN_SIZE) as i32
                );
            }?;
        }
        //println!("Finished Apply ReLU");
        
        //println!("Hidden to Output (Hidden @ W2)");
        let grid_size_output = [
            (OUTPUT_SIZE as u32 + block_size[0] - 1) / block_size[0],
            (batch_size as u32 + block_size[1] - 1) / block_size[1],
            1,
        ] as [u32; 3];

        launch! {
            matmul_a_b_kernel<<<grid_size_output, block_size>>>(
                hidden,
                &self.weights2,
                output,
                batch_size as i32,
                HIDDEN_SIZE as i32,
                OUTPUT_SIZE as i32
            );
        }?;
        //println!("Finished Hidden to Output (W2)");
        
        //println!("Add bias2");
        let bias_grid_size = ((batch_size * OUTPUT_SIZE) as u32 + bias_block_size - 1) / bias_block_size;

        {
            let grid_size = [bias_grid_size, 1, 1] as [u32; 3];
            let block_size = [bias_block_size, 1, 1] as [u32; 3];

            launch! {
                bias_add_kernel<<<grid_size, block_size>>>(
                    output,
                    &self.bias2,
                    batch_size as i32,
                    OUTPUT_SIZE as i32
                );
            }?;
        }
        //println!("Finished Add bias2");

        //println!("Softmax Results");
        {
            let grid_size = [batch_size as u32, 1, 1];
            let block_size = [1, 1, 1];

            // Apply softmax
            launch! {
                softmax_kernel<<<grid_size, block_size>>>(
                    output,
                    batch_size as i32,
                    OUTPUT_SIZE as i32
                );
            }?;
        }
        //println!("Finished Softmax");
        
        Ok(())
    }
    
    pub fn backward(
        &mut self,
        input: &GPUArray<f32>,
        hidden: &GPUArray<f32>,
        output: &GPUArray<f32>,
        labels: &GPUArray<i32>,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        // Zero out gradients
        let zero_block_size = 256;
        
        let w1_size = HIDDEN_SIZE * INPUT_SIZE;
        let w1_grid_size = (w1_size as u32 + zero_block_size - 1) / zero_block_size;
        
        launch! {
            zero_grad_kernel<<<[w1_grid_size, 1, 1], [zero_block_size, 1, 1]>>>(
                &self.grad_weights1,
                w1_size as i32
            );
        }?;
        
        let w2_size = OUTPUT_SIZE * HIDDEN_SIZE;
        let w2_grid_size = (w2_size as u32 + zero_block_size - 1) / zero_block_size;
        
        launch! {
            zero_grad_kernel<<<[w2_grid_size, 1, 1], [zero_block_size, 1, 1]>>>(
                &self.grad_weights2,
                w2_size as i32
            );
        }?;
        
        let b1_grid_size = (HIDDEN_SIZE as u32 + zero_block_size - 1) / zero_block_size;
        
        launch! {
            zero_grad_kernel<<<[b1_grid_size, 1, 1], [zero_block_size, 1, 1]>>>(
                &self.grad_bias1,
                HIDDEN_SIZE as i32
            );
        }?;
        
        let b2_grid_size = (OUTPUT_SIZE as u32 + zero_block_size - 1) / zero_block_size;
        
        launch! {
            zero_grad_kernel<<<[b2_grid_size, 1, 1], [zero_block_size, 1, 1]>>>(
                &self.grad_bias2,
                OUTPUT_SIZE as i32
            );
        }?;
        
        // Compute output gradients
        let mut grad_output = GPUArray::from_iter(vec![0.0; batch_size * OUTPUT_SIZE].into_iter());
        let output_grad_block_size = 256;
        let output_grad_grid_size = ((batch_size) as u32 + output_grad_block_size - 1) / output_grad_block_size;
        
        launch! {
            compute_output_gradients_kernel<<<[output_grad_grid_size, 1, 1], [output_grad_block_size, 1, 1]>>>(
                &grad_output,
                output,
                labels,
                batch_size as i32
            );
        }?;
        
        // Update gradients for weights2 (W2.grad = hidden.T @ grad_output)
        let block_size = [32, 32, 1] as [u32; 3];
        let grid_size_w2 = [
            (HIDDEN_SIZE as u32 + block_size[0] - 1) / block_size[0],
            (OUTPUT_SIZE as u32 + block_size[1] - 1) / block_size[1],
            1,
        ] as [u32; 3];
        
        launch! {
            matmul_at_b_kernel<<<grid_size_w2, block_size>>>(
                hidden,
                &grad_output,
                &self.grad_weights2,
                batch_size as i32,
                HIDDEN_SIZE as i32,
                OUTPUT_SIZE as i32
            );
        }?;
        
        // Compute dX2 (gradient of loss w.r.t. input of second layer)
        let mut d_dX2 = GPUArray::from_iter(vec![0.0; batch_size * HIDDEN_SIZE].into_iter());
        let grid_size_dX2 = [
            (HIDDEN_SIZE as u32 + block_size[0] - 1) / block_size[0],
            (batch_size as u32 + block_size[1] - 1) / block_size[1],
            1,
        ] as [u32; 3];
        
        launch! {
            matmul_a_bt_kernel<<<grid_size_dX2, block_size>>>(
                &grad_output,
                &self.weights2,
                &mut d_dX2,
                batch_size as i32,
                OUTPUT_SIZE as i32,
                HIDDEN_SIZE as i32
            );
        }?;
        
        // Compute ReLU derivative
        let mut grad_hidden = GPUArray::from_iter(vec![0.0; batch_size * HIDDEN_SIZE].into_iter());
        let hidden_grad_block_size = 256;
        let hidden_grad_grid_size = ((batch_size * HIDDEN_SIZE) as u32 + hidden_grad_block_size - 1) / hidden_grad_block_size;
        
        launch! {
            drelu_kernel<<<[hidden_grad_grid_size, 1, 1], [hidden_grad_block_size, 1, 1]>>>(
                hidden,
                &mut grad_hidden,
                (batch_size * HIDDEN_SIZE) as i32
            );
        }?;
        
        // Element-wise multiplication for backprop through ReLU
        launch! {
            multiply_gradients_kernel<<<[hidden_grad_grid_size, 1, 1], [hidden_grad_block_size, 1, 1]>>>(
                &mut d_dX2,
                &grad_hidden,
                (batch_size * HIDDEN_SIZE) as i32
            );
        }?;
        
        // Update gradients for weights1 (W1.grad = input.T @ d_dX2)
        let grid_size_w1 = [
            (INPUT_SIZE as u32 + block_size[0] - 1) / block_size[0],
            (HIDDEN_SIZE as u32 + block_size[1] - 1) / block_size[1],
            1,
        ] as [u32; 3];
        
        launch! {
            matmul_at_b_kernel<<<grid_size_w1, block_size>>>(
                input,
                &d_dX2,
                &self.grad_weights1,
                batch_size as i32,
                INPUT_SIZE as i32,
                HIDDEN_SIZE as i32
            );
        }?;
        
        // Manually accumulate bias gradients
        // For bias2
        let bias_output_vec = grad_output.to_iter().collect::<Vec<_>>();
        let mut bias2_grad_vec = vec![0.0; OUTPUT_SIZE];
        
        for b in 0..batch_size {
            for i in 0..OUTPUT_SIZE {
                bias2_grad_vec[i] += bias_output_vec[b * OUTPUT_SIZE + i];
            }
        }
        
        // For bias1
        let d_dX2_vec = d_dX2.to_iter().collect::<Vec<_>>();
        let mut bias1_grad_vec = vec![0.0; HIDDEN_SIZE];
        
        for b in 0..batch_size {
            for i in 0..HIDDEN_SIZE {
                bias1_grad_vec[i] += d_dX2_vec[b * HIDDEN_SIZE + i];
            }
        }
        
        // Copy back to GPU
        self.grad_bias1 = GPUArray::from_iter(bias1_grad_vec.into_iter());
        self.grad_bias2 = GPUArray::from_iter(bias2_grad_vec.into_iter());
        
        Ok(())
    }
    
    pub fn update_weights(&mut self) -> anyhow::Result<()> {
        let block_size = 256;
        
        // Update weights1
        let w1_size = HIDDEN_SIZE * INPUT_SIZE;
        let w1_grid_size = (w1_size as u32 + block_size - 1) / block_size;
        
        launch! {
            update_weights_kernel<<<[w1_grid_size, 1, 1], [block_size, 1, 1]>>>(
                &mut self.weights1,
                &self.grad_weights1,
                w1_size as i32
            );
        }?;
        
        // Update weights2
        let w2_size = OUTPUT_SIZE * HIDDEN_SIZE;
        let w2_grid_size = (w2_size as u32 + block_size - 1) / block_size;
        
        launch! {
            update_weights_kernel<<<[w2_grid_size, 1, 1], [block_size, 1, 1]>>>(
                &mut self.weights2,
                &self.grad_weights2,
                w2_size as i32
            );
        }?;
        
        // Update bias1
        let b1_grid_size = (HIDDEN_SIZE as u32 + block_size - 1) / block_size;
        
        launch! {
            update_weights_kernel<<<[b1_grid_size, 1, 1], [block_size, 1, 1]>>>(
                &mut self.bias1,
                &self.grad_bias1,
                HIDDEN_SIZE as i32
            );
        }?;
        
        // Update bias2
        let b2_grid_size = (OUTPUT_SIZE as u32 + block_size - 1) / block_size;
        
        launch! {
            update_weights_kernel<<<[b2_grid_size, 1, 1], [block_size, 1, 1]>>>(
                &mut self.bias2,
                &self.grad_bias2,
                OUTPUT_SIZE as i32
            );
        }?;
        
        Ok(())
    }
}

// Function to calculate cross-entropy loss for batches
fn cross_entropy_loss(output: &[f32], labels: &[i32], batch_size: usize) -> f32 {
    let mut total_loss = 0.0;
    
    for b in 0..batch_size {
        let label = labels[b] as usize;
        if label < OUTPUT_SIZE {
            // Clip to avoid log(0)
            let prob = output[b * OUTPUT_SIZE + label].max(1e-15);
            total_loss -= prob.ln();
        }
    }
    
    total_loss / batch_size as f32
}

// Evaluate model accuracy
fn evaluate_accuracy(
    nn: &NeuralNetwork,
    x_test: &[f32],
    y_test: &[u8],
    test_size: usize,
) -> anyhow::Result<f32> {
    let mut correct = 0;
    let mut total = 0;
    
    // Process in batches to avoid GPU memory issues
    for batch_start in (0..test_size).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(test_size);
        let current_batch_size = batch_end - batch_start;
        
        // Prepare batch data
        let mut batch_x = vec![0.0; current_batch_size * INPUT_SIZE];
        let mut batch_y = vec![0; current_batch_size];
        
        for i in 0..current_batch_size {
            for j in 0..INPUT_SIZE {
                batch_x[i * INPUT_SIZE + j] = x_test[(batch_start + i) * INPUT_SIZE + j];
            }
            batch_y[i] = y_test[batch_start + i];
        }
        
        // Copy to GPU
        let d_input = GPUArray::from_iter(batch_x.into_iter());
        let mut d_hidden = GPUArray::from_iter(vec![0.0; current_batch_size * HIDDEN_SIZE].into_iter());
        let mut d_output = GPUArray::from_iter(vec![0.0; current_batch_size * OUTPUT_SIZE].into_iter());
        
        // Forward pass
        nn.forward(&d_input, &mut d_hidden, &mut d_output, current_batch_size)?;
        
        // Copy output back to CPU for evaluation
        let output = d_output.to_iter().collect::<Vec<_>>();
        
        // Count correct predictions
        for i in 0..current_batch_size {
            let mut max_idx = 0;
            let mut max_val = output[i * OUTPUT_SIZE];
            
            for j in 1..OUTPUT_SIZE {
                if output[i * OUTPUT_SIZE + j] > max_val {
                    max_val = output[i * OUTPUT_SIZE + j];
                    max_idx = j;
                }
            }
            
            if max_idx as i32 == batch_y[i] as i32 {
                correct += 1;
            }
            total += 1;
        }
    }
    
    Ok(100.0 * correct as f32 / total as f32)
}

// Train the neural network
pub fn train_mnist() -> anyhow::Result<()> {
    println!("Loading MNIST dataset...");
    
    // Load MNIST dataset
    let mnist = MnistBuilder::new()
        .base_path("./data")
        .training_set_length(60000)
        .test_set_length(10000)
        .finalize()
        .normalize();
    
    // Preprocess data
    let train_size = mnist.trn_lbl.len();
    let test_size = mnist.tst_lbl.len();
    
    println!("Training set size: {}", train_size);
    println!("Test set size: {}", test_size);
    
    // Create neural network
    let mut nn = NeuralNetwork::new()?;
    
    println!("Starting training...");
    let start_time = Instant::now();
    
    // Training loop
    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0;
        let mut batch_count = 0;


        let bar = ProgressBar::new((train_size/BATCH_SIZE) as u64);
        // Process in batches
        for batch_start in (0..train_size).step_by(BATCH_SIZE) {
            bar.inc(1);
            let batch_end = (batch_start + BATCH_SIZE).min(train_size);
            let current_batch_size = batch_end - batch_start;
            
            // Prepare batch data
            let mut batch_x = vec![0.0; current_batch_size * INPUT_SIZE];
            let mut batch_y = vec![0; current_batch_size];

            //println!("Started data normalization");
            for i in 0..current_batch_size {
                for j in 0..INPUT_SIZE {
                    batch_x[i * INPUT_SIZE + j] = mnist.trn_img[(batch_start + i) * INPUT_SIZE + j] as f32 / 255.0;
                }
                batch_y[i] = mnist.trn_lbl[batch_start + i] as i32;
            }
            //println!("end data normalization");
            
            // Copy to GPU
            let d_input = GPUArray::from_iter(batch_x.iter().cloned());
            let d_labels = GPUArray::from_iter(batch_y.iter().cloned());
            let mut d_hidden = GPUArray::from_iter(vec![0.0; current_batch_size * HIDDEN_SIZE].into_iter());
            let mut d_output = GPUArray::from_iter(vec![0.0; current_batch_size * OUTPUT_SIZE].into_iter());
            
            // Forward pass
            //println!("Started forward pass");
            nn.forward(&d_input, &mut d_hidden, &mut d_output, current_batch_size)?;
            //println!("Finished forward pass");


            // Calculate loss
            let output = d_output.to_iter().collect::<Vec<_>>();
            let loss = cross_entropy_loss(&output, &batch_y, current_batch_size);
            total_loss += loss;
            batch_count += 1;
            
            // Backward pass
            nn.backward(&d_input, &d_hidden, &d_output, &d_labels, current_batch_size)?;
            
            // Update weights
            nn.update_weights()?;
        }
        
        // Calculate average loss for this epoch
        let avg_loss = total_loss / batch_count as f32;
        
        // Evaluate accuracy on test set
        let accuracy = evaluate_accuracy(&nn, &mnist.tst_img, &mnist.tst_lbl, test_size)?;
        
        println!("Epoch {}/{}: loss = {:.4}, accuracy = {:.2}%", 
                 epoch + 1, EPOCHS, avg_loss, accuracy);
    }
    
    let elapsed = start_time.elapsed();
    println!("Training completed in {:.2?}", elapsed);
    
    // Final evaluation
    let final_accuracy = evaluate_accuracy(&nn, &mnist.tst_img, &mnist.tst_lbl, test_size)?;
    println!("Final test accuracy: {:.2}%", final_accuracy);
    
    Ok(())
}
