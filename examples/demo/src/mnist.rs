use super::Error;
use mimir::{launch, mimir_global, GpuArray};
use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::Instant;

const INPUT: usize = 784;
const HIDDEN: usize = 256;
const OUTPUT: usize = 10;
static MODEL: OnceLock<Model> = OnceLock::new();
struct Model {
    w1: GpuArray<f32>,
    w2: GpuArray<f32>,
    b1: GpuArray<f32>,
    b2: GpuArray<f32>,
}

// Saved dense weights are row-major [input, output], matching mnist-cnn.
#[mimir_global]
fn dense(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    inputs: i32,
    outputs: i32,
    relu: bool,
) {
    let col = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if col < outputs {
        let mut sum = 0.0f32;
        for row in 0..inputs {
            sum += input[row] * weights[row * outputs + col];
        }
        sum += bias[col];
        if relu {
            sum = max(sum, 0.0);
        }
        output[col] = sum;
    }
}

pub fn init() -> Result<(), Error> {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/mnist-model.json"
    ));
    let load = || -> Result<Model, Error> {
        let value: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if value["format_version"].as_u64() != Some(1)
            || value["input_size"].as_u64() != Some(INPUT as u64)
            || value["hidden_size"].as_u64() != Some(HIDDEN as u64)
            || value["output_size"].as_u64() != Some(OUTPUT as u64)
        {
            return Err("expected version 1 dense model with architecture 784 → 256 → 10".into());
        }
        fn parameters(value: &Value, name: &str, count: usize) -> Result<Vec<f32>, Error> {
            let array = value[name]
                .as_array()
                .ok_or_else(|| format!("missing {name}"))?;
            if array.len() != count {
                return Err(
                    format!("{name}: expected {count} parameters, got {}", array.len()).into(),
                );
            }
            array
                .iter()
                .map(|v| {
                    let n = v.as_f64().ok_or("parameter must be numeric")? as f32;
                    if !n.is_finite() {
                        return Err("parameter must be finite f32".into());
                    }
                    Ok(n)
                })
                .collect()
        }
        // Validate the entire file before touching the GPU.
        let w1 = parameters(&value, "weights1", INPUT * HIDDEN)?;
        let w2 = parameters(&value, "weights2", HIDDEN * OUTPUT)?;
        let b1 = parameters(&value, "bias1", HIDDEN)?;
        let b2 = parameters(&value, "bias2", OUTPUT)?;
        Ok(Model {
            w1: GpuArray::from_slice(&w1)?,
            w2: GpuArray::from_slice(&w2)?,
            b1: GpuArray::from_slice(&b1)?,
            b2: GpuArray::from_slice(&b2)?,
        })
    };
    let model =
        load().map_err(|e| format!("Cannot initialize MNIST model at {}: {e}", path.display()))?;
    MODEL
        .set(model)
        .map_err(|_| "MNIST model already initialized")?;
    println!("MNIST model loaded: {}", path.display());
    Ok(())
}

pub fn predict(args: &Value) -> Result<Value, Error> {
    let values = args["pixels"]
        .as_array()
        .ok_or("missing 28×28 grayscale pixels")?;
    if values.len() != INPUT {
        return Err("MNIST expects exactly 784 grayscale byte values".into());
    }
    let pixels = values
        .iter()
        .map(|v| {
            v.as_u64()
                .filter(|n| *n <= 255)
                .map(|n| n as f32)
                .ok_or("pixels must be bytes 0–255")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if pixels.iter().all(|p| *p == 0.0) {
        return Err("Draw a digit before predicting.".into());
    }
    let model = MODEL.get().ok_or("MNIST model not initialized")?;
    let start = Instant::now();
    let normalized = pixels.iter().map(|p| p / 255.0).collect::<Vec<_>>();
    let input = GpuArray::from_slice(&normalized)?;
    let hidden = GpuArray::from_slice(&[0.0f32; HIDDEN])?;
    let output = GpuArray::from_slice(&[0.0f32; OUTPUT])?;
    // Each invocation owns one output neuron; inputs and model weights are read-only.
    unsafe {
        launch!(dense<<<[2,1,1],[128,1,1]>>>(&input,&model.w1,&model.b1,&hidden,INPUT as i32,HIDDEN as i32,true))?;
        launch!(dense<<<[1,1,1],[32,1,1]>>>(&hidden,&model.w2,&model.b2,&output,HIDDEN as i32,OUTPUT as i32,false))?;
    }
    let logits = output.to_vec()?;
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("model produced non-finite scores".into());
    }
    // Stable softmax over ten read-back logits.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    let probabilities: Vec<f32> = exp.iter().map(|v| v / total).collect();
    let prediction = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let rgba = pixels
        .iter()
        .flat_map(|p| [*p, *p, *p, 255.0])
        .collect::<Vec<_>>();
    Ok(
        json!({"prediction":prediction,"probabilities":probabilities,"elapsed_ms":elapsed,"image":super::bitmap(28,28,&rgba)}),
    )
}
