use crate::model::{probabilities, Error, Model, INPUT, OUTPUT};
use mimir::{launch, mimir_global, GpuArray};
use rand::{seq::SliceRandom, RngExt, SeedableRng};
use serde_json::{json, Value};
use std::time::Instant;
const BATCH: usize = 64;

#[mimir_global]
fn output_gradient(logits: &[f32], labels: &[i32], grad: &mut [f32], batch: i32) {
    let row = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if row < batch {
        let mut largest = logits[row * 10];
        for j in 1..10 {
            largest = max(largest, logits[row * 10 + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..10 {
            sum += exp(logits[row * 10 + j] - largest);
        }
        for j in 0..10 {
            let mut v = exp(logits[row * 10 + j] - largest) / sum;
            if j == labels[row] {
                v -= 1.0;
            }
            grad[row * 10 + j] = v / batch as f32;
        }
    }
}

#[mimir_global]
fn backward(
    grad: &[f32],
    weights: &[f32],
    activation: &[f32],
    previous: &mut [f32],
    inputs: i32,
    outputs: i32,
    count: i32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let row = i / inputs;
        let col = i % inputs;
        let mut sum = 0.0f32;
        for j in 0..outputs {
            sum += grad[row * outputs + j] * weights[col * outputs + j];
        }
        // ReLU's derivative depends on the forward activation, never the gradient sign.
        if activation[i] <= 0.0 {
            sum = 0.0;
        }
        previous[i] = sum;
    }
}

#[mimir_global]
fn weight_gradient(
    input: &[f32],
    grad: &[f32],
    dw: &mut [f32],
    inputs: i32,
    outputs: i32,
    batch: i32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < inputs * outputs {
        let row = i / outputs;
        let col = i % outputs;
        let mut sum = 0.0f32;
        for b in 0..batch {
            sum += input[b * inputs + row] * grad[b * outputs + col];
        }
        dw[i] = sum;
    }
}

#[mimir_global]
fn bias_gradient(grad: &[f32], db: &mut [f32], outputs: i32, batch: i32) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < outputs {
        let mut sum = 0.0f32;
        for b in 0..batch {
            sum += grad[b * outputs + i];
        }
        db[i] = sum;
    }
}

#[mimir_global]
fn adam(
    weights: &mut [f32],
    grad: &[f32],
    mean: &mut [f32],
    variance: &mut [f32],
    count: i32,
    lr: f32,
    correction1: f32,
    correction2: f32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let g = grad[i];
        let m = 0.9 * mean[i] + 0.1 * g;
        let v = 0.999 * variance[i] + 0.001 * g * g;
        mean[i] = m;
        variance[i] = v;
        weights[i] -= lr * (m / correction1) / (sqrt(v / correction2) + 0.00000001);
    }
}

struct Optimizer {
    dw: GpuArray<f32>,
    db: GpuArray<f32>,
    mw: GpuArray<f32>,
    vw: GpuArray<f32>,
    mb: GpuArray<f32>,
    vb: GpuArray<f32>,
}

fn zeros(n: usize) -> Result<GpuArray<f32>, Error> {
    Ok(GpuArray::from_slice(&vec![0.0f32; n])?)
}

fn train_batch(
    model: &Model,
    optim: &[Optimizer],
    x: Vec<f32>,
    y: &[i32],
    step: i32,
    lr: f32,
) -> Result<(), Error> {
    let batch = y.len();
    let a = model.forward(GpuArray::from_slice(&x)?, batch)?;
    let labels = GpuArray::from_slice(y)?;
    let mut grad = zeros(batch * OUTPUT)?;
    // All launches write disjoint output slots; separate buffers preserve forward activations.
    unsafe {
        launch!(output_gradient<<<[(batch as u32).div_ceil(64),1,1],[64,1,1]>>>(a.last().unwrap(),&labels,&grad,batch as i32))?;
    }
    for index in (0..model.layers.len()).rev() {
        let l = &model.layers[index];
        let o = &optim[index];
        let previous = if index > 0 {
            let count = batch * l.inputs;
            let previous = zeros(count)?;
            // Propagate using the current weights before Adam changes them.
            unsafe {
                launch!(backward<<<[(count as u32).div_ceil(128),1,1],[128,1,1]>>>(&grad,&l.weights,&a[index],&previous,l.inputs as i32,l.outputs as i32,count as i32))?;
            }
            Some(previous)
        } else {
            None
        };
        let count = l.inputs * l.outputs;
        unsafe {
            launch!(weight_gradient<<<[(count as u32).div_ceil(128),1,1],[128,1,1]>>>(&a[index],&grad,&o.dw,l.inputs as i32,l.outputs as i32,batch as i32))?;
            launch!(bias_gradient<<<[(l.outputs as u32).div_ceil(128),1,1],[128,1,1]>>>(&grad,&o.db,l.outputs as i32,batch as i32))?;
            launch!(adam<<<[(count as u32).div_ceil(128),1,1],[128,1,1]>>>(&l.weights,&o.dw,&o.mw,&o.vw,count as i32,lr,1.0-0.9f32.powi(step),1.0-0.999f32.powi(step)))?;
            launch!(adam<<<[(l.outputs as u32).div_ceil(128),1,1],[128,1,1]>>>(&l.bias,&o.db,&o.mb,&o.vb,l.outputs as i32,lr,1.0-0.9f32.powi(step),1.0-0.999f32.powi(step)))?;
        }
        if let Some(previous) = previous {
            grad = previous;
        }
    }
    Ok(())
}

fn dataset() -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), Error> {
    let base = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/data"));
    fn read(
        base: &std::path::Path,
        name: &str,
        images: bool,
        count: usize,
    ) -> Result<Vec<u8>, Error> {
        let data = std::fs::read(base.join(name))?;
        let header = if images { 16 } else { 8 };
        let word = |i| u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        if data.len() < header
            || word(0) != if images { 2051 } else { 2049 }
            || word(4) != count
            || (images && (word(8) != 28 || word(12) != 28))
            || data.len() != header + count * if images { INPUT } else { 1 }
        {
            return Err(format!("invalid IDX file: {name}").into());
        }
        let out = data[header..].to_vec();
        if !images && out.iter().any(|v| *v >= 10) {
            return Err("invalid label".into());
        }
        Ok(out)
    }
    Ok((
        read(base, "train-images-idx3-ubyte", true, 60000)?,
        read(base, "train-labels-idx1-ubyte", false, 60000)?,
        read(base, "t10k-images-idx3-ubyte", true, 10000)?,
        read(base, "t10k-labels-idx1-ubyte", false, 10000)?,
    ))
}

pub fn evaluate(model: &Model, images: &[u8], labels: &[u8]) -> Result<f64, Error> {
    let mut correct = 0;
    for start in (0..labels.len()).step_by(256) {
        let end = (start + 256).min(labels.len());
        let x = images[start * INPUT..end * INPUT]
            .iter()
            .map(|p| *p as f32 / 255.0)
            .collect::<Vec<_>>();
        let a = model.forward(GpuArray::from_slice(&x)?, end - start)?;
        let logits = a.last().unwrap().to_vec()?;
        for (row, score) in logits.chunks_exact(OUTPUT).enumerate() {
            let p = probabilities(score)?;
            let prediction = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            correct += usize::from(prediction == labels[start + row] as usize);
        }
    }
    Ok(correct as f64 / labels.len() as f64)
}

pub fn predict_file(path: &str) -> Result<(), Error> {
    let model = Model::from_value(&serde_json::from_slice(&std::fs::read(path)?)?)?;
    let (_, _, images, labels) = dataset()?;
    println!(
        "Full MNIST test accuracy: {:.2}% (10,000 images)",
        evaluate(&model, &images, &labels)? * 100.0
    );
    Ok(())
}

fn snapshot(model: &Model) -> Result<Value, Error> {
    let layers=model.layers.iter().map(|l| Ok(json!({"inputs":l.inputs,"outputs":l.outputs,"weights":l.weights.to_vec()?,"bias":l.bias.to_vec()?}))).collect::<Result<Vec<_>,Error>>()?;
    Ok(json!({"format_version":2,"layers":layers}))
}

pub fn train_mnist() -> Result<(), Error> {
    let epochs = std::env::var("MNIST_EPOCHS")
        .unwrap_or_else(|_| "30".into())
        .parse::<usize>()?;
    if epochs == 0 {
        return Err("MNIST_EPOCHS must be positive".into());
    }
    let output = std::env::var("MNIST_OUTPUT").unwrap_or_else(|_| "mnist-model.json".into());
    let seed = std::env::var("MNIST_SEED")
        .unwrap_or_else(|_| "42".into())
        .parse::<u64>()?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let (images, labels, test_images, test_labels) = dataset()?;
    let mut split = (0..60000).collect::<Vec<usize>>();
    split.shuffle(&mut rng);
    let (validation, training) = split.split_at(5000);
    let val_images = validation
        .iter()
        .flat_map(|i| images[i * INPUT..(i + 1) * INPUT].iter().copied())
        .collect::<Vec<_>>();
    let val_labels = validation.iter().map(|i| labels[*i]).collect::<Vec<_>>();
    let mut training = training.to_vec();
    let dims = [INPUT, 512, 128, OUTPUT];
    let layers=dims.windows(2).map(|d| {
        let scale=(6.0/d[0] as f32).sqrt();
        json!({"inputs":d[0],"outputs":d[1],"weights":(0..d[0]*d[1]).map(|_|rng.random_range(-scale..scale)).collect::<Vec<f32>>(),"bias":vec![0.0f32;d[1]]})
    }).collect::<Vec<_>>();
    let model = Model::from_value(&json!({"format_version":2,"layers":layers}))?;
    let optim = model
        .layers
        .iter()
        .map(|l| {
            Ok(Optimizer {
                dw: zeros(l.inputs * l.outputs)?,
                db: zeros(l.outputs)?,
                mw: zeros(l.inputs * l.outputs)?,
                vw: zeros(l.inputs * l.outputs)?,
                mb: zeros(l.outputs)?,
                vb: zeros(l.outputs)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let start = Instant::now();
    let mut step = 0;
    let mut best = 0.0;
    let mut checkpoint: Option<Value> = None;
    println!(
        "Training 784 → 512 → 128 → 10; Adam; 55,000 training / 5,000 validation; seed {seed}"
    );
    for epoch in 0..epochs {
        training.shuffle(&mut rng);
        let lr = if epoch < 10 {
            0.001f32
        } else if epoch < 20 {
            0.0003
        } else {
            0.0001
        };
        for indices in training.chunks(BATCH) {
            let mut x = Vec::with_capacity(indices.len() * INPUT);
            let mut y = Vec::with_capacity(indices.len());
            for &i in indices {
                // Translation augmentation helps drawings that are slightly off center.
                let dx = rng.random_range(-2i32..=2);
                let dy = rng.random_range(-2i32..=2);
                for row in 0..28i32 {
                    for col in 0..28i32 {
                        let sx = col - dx;
                        let sy = row - dy;
                        x.push(if (0..28).contains(&sx) && (0..28).contains(&sy) {
                            images[i * INPUT + (sy * 28 + sx) as usize] as f32 / 255.0
                        } else {
                            0.0
                        });
                    }
                }
                y.push(labels[i] as i32);
            }
            step += 1;
            train_batch(&model, &optim, x, &y, step, lr)?;
        }
        let accuracy = evaluate(&model, &val_images, &val_labels)?;
        println!(
            "Epoch {}/{} · validation {:.2}% · elapsed {:.1}s",
            epoch + 1,
            epochs,
            accuracy * 100.0,
            start.elapsed().as_secs_f64()
        );
        if accuracy > best {
            best = accuracy;
            checkpoint = Some(snapshot(&model)?);
        }
    }
    let mut checkpoint = checkpoint.ok_or("training did not produce a valid checkpoint")?;
    let best_model = Model::from_value(&checkpoint)?;
    let test = evaluate(&best_model, &test_images, &test_labels)?;
    println!(
        "Best validation: {:.2}%; full test accuracy: {:.2}%",
        best * 100.0,
        test * 100.0
    );
    if test < 0.95 {
        return Err(format!(
            "test accuracy {:.2}% is below 95%; existing output was not replaced",
            test * 100.0
        )
        .into());
    }
    checkpoint["metrics"] = json!({"validation_accuracy":best,"test_accuracy":test,"test_samples":10000,"seed":seed,"epochs":epochs});
    let temp = format!("{output}.tmp");
    std::fs::write(&temp, serde_json::to_vec(&checkpoint)?)?;
    std::fs::rename(temp, &output)?;
    println!("Saved qualified model to {output}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a compute device"]
    fn relu_backward_preserves_negative_gradients_for_active_neurons() -> Result<(), Error> {
        let grad = GpuArray::from_slice(&[-2.0f32, -3.0])?;
        let weights = GpuArray::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let activation = GpuArray::from_slice(&[2.0f32, -3.0, 0.0])?;
        let previous = zeros(3)?;
        unsafe {
            launch!(backward<<<[1,1,1],[32,1,1]>>>(&grad,&weights,&activation,&previous,3,2,3))?;
        }
        assert_eq!(previous.to_vec()?, vec![-8.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    #[ignore = "requires a compute device"]
    fn output_gradient_averages_batch_and_subtracts_correct_label() -> Result<(), Error> {
        let logits = zeros(20)?;
        let labels = GpuArray::from_slice(&[3i32, 7])?;
        let grad = zeros(20)?;
        unsafe {
            launch!(output_gradient<<<[1,1,1],[32,1,1]>>>(&logits,&labels,&grad,2))?;
        }
        let actual = grad.to_vec()?;
        for (i, value) in actual.into_iter().enumerate() {
            let expected = if i == 3 || i == 17 { -0.45 } else { 0.05 };
            assert!((value - expected).abs() < 1e-6);
        }
        Ok(())
    }
}
