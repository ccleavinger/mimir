//! A small, session-scoped webcam classifier trained entirely through Mimir kernels.
use super::Error;
use mimir::{launch, mimir_global, GpuArray};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
const INPUT: usize = 32 * 32 * 3;
const HIDDEN: usize = 64;
const EPOCHS: usize = 150;
static NEXT: AtomicU64 = AtomicU64::new(1);
static MODELS: Mutex<BTreeMap<u64, Arc<Classifier>>> = Mutex::new(BTreeMap::new());
// Avoid concurrent training runs contending for the same demo device.
static TRAINING: Mutex<()> = Mutex::new(());
struct Classifier {
    names: Vec<String>,
    w1: GpuArray<f32>,
    b1: GpuArray<f32>,
    w2: GpuArray<f32>,
    b2: GpuArray<f32>,
}
fn array(data: &[f32]) -> Result<GpuArray<f32>, Error> {
    Ok(GpuArray::from_slice(data)?)
}
fn zeros(n: usize) -> Result<GpuArray<f32>, Error> {
    array(&vec![0.0; n])
}

#[mimir_global]
fn dense(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    y: &mut [f32],
    inputs: i32,
    outputs: i32,
    count: i32,
    relu: bool,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let row = i / outputs;
        let col = i % outputs;
        let mut sum = b[col];
        for j in 0..inputs {
            sum += x[row * inputs + j] * w[j * outputs + col];
        }
        if relu {
            sum = max(sum, 0.0);
        }
        y[i] = sum;
    }
}
#[mimir_global]
fn softmax_gradient(logits: &[f32], labels: &[i32], grad: &mut [f32], batch: i32, classes: i32) {
    let row = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if row < batch {
        let mut largest = logits[row * classes];
        for c in 1..classes {
            largest = max(largest, logits[row * classes + c]);
        }
        let mut total = 0.0f32;
        for c in 0..classes {
            total += exp(logits[row * classes + c] - largest);
        }
        for c in 0..classes {
            let mut v = exp(logits[row * classes + c] - largest) / total;
            if c == labels[row] {
                v -= 1.0;
            }
            grad[row * classes + c] = v / batch as f32;
        }
    }
}
#[mimir_global]
fn backward(
    grad: &[f32],
    weights: &[f32],
    hidden: &[f32],
    previous: &mut [f32],
    classes: i32,
    count: i32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let row = i / 64;
        let col = i % 64;
        let mut v = 0.0f32;
        for c in 0..classes {
            v += grad[row * classes + c] * weights[col * classes + c];
        }
        if hidden[i] <= 0.0 {
            v = 0.0;
        }
        previous[i] = v;
    }
}
#[mimir_global]
fn gradients(
    x: &[f32],
    grad: &[f32],
    dw: &mut [f32],
    db: &mut [f32],
    inputs: i32,
    outputs: i32,
    batch: i32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < inputs * outputs {
        let row = i / outputs;
        let col = i % outputs;
        let mut v = 0.0f32;
        for b in 0..batch {
            v += x[b * inputs + row] * grad[b * outputs + col];
        }
        dw[i] = v;
    }
    if i < outputs {
        let mut v = 0.0f32;
        for b in 0..batch {
            v += grad[b * outputs + i];
        }
        db[i] = v;
    }
}
#[mimir_global]
fn update(
    w: &mut [f32],
    g: &[f32],
    mean: &mut [f32],
    variance: &mut [f32],
    count: i32,
    lr: f32,
    c1: f32,
    c2: f32,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let v = g[i];
        let m = 0.9 * mean[i] + 0.1 * v;
        let s = 0.999 * variance[i] + 0.001 * v * v;
        mean[i] = m;
        variance[i] = s;
        w[i] -= lr * (m / c1) / (sqrt(s / c2) + 0.00000001);
    }
}
struct Adam {
    grad: GpuArray<f32>,
    mean: GpuArray<f32>,
    variance: GpuArray<f32>,
}
impl Adam {
    fn new(n: usize) -> Result<Self, Error> {
        Ok(Self {
            grad: zeros(n)?,
            mean: zeros(n)?,
            variance: zeros(n)?,
        })
    }
    fn step(&self, w: &GpuArray<f32>, epoch: usize) -> Result<(), Error> {
        let n = w.len();
        let lr = if epoch < 75 { 0.001f32 } else { 0.0003 };
        // Each invocation writes one distinct parameter and its optimizer state.
        unsafe {
            launch!(update<<<[(n as u32).div_ceil(128),1,1],[128,1,1]>>>(w,&self.grad,&self.mean,&self.variance,n as i32,lr,1.0-0.9f32.powi(epoch as i32+1),1.0-0.999f32.powi(epoch as i32+1)))?;
        }
        Ok(())
    }
}
impl Classifier {
    fn forward(
        &self,
        x: &GpuArray<f32>,
        hidden: &GpuArray<f32>,
        out: &GpuArray<f32>,
        batch: usize,
    ) -> Result<(), Error> {
        let classes = self.names.len();
        // Fixed, validated shapes; each invocation owns one output neuron.
        unsafe {
            launch!(dense<<<[((batch*HIDDEN) as u32).div_ceil(128),1,1],[128,1,1]>>>(x,&self.w1,&self.b1,hidden,INPUT as i32,HIDDEN as i32,(batch*HIDDEN) as i32,true))?;
            launch!(dense<<<[((batch*classes) as u32).div_ceil(128),1,1],[128,1,1]>>>(hidden,&self.w2,&self.b2,out,HIDDEN as i32,classes as i32,(batch*classes) as i32,false))?;
        }
        Ok(())
    }
    fn scores(&self, pixels: &[f32]) -> Result<Vec<f32>, Error> {
        let x = array(pixels)?;
        let hidden = zeros(HIDDEN)?;
        let out = zeros(self.names.len())?;
        self.forward(&x, &hidden, &out, 1)?;
        probabilities(&out.to_vec()?)
    }
    fn accuracy(&self, x: &[f32], labels: &[usize]) -> Result<f64, Error> {
        let input = array(x)?;
        let hidden = zeros(labels.len() * HIDDEN)?;
        let out = zeros(labels.len() * self.names.len())?;
        self.forward(&input, &hidden, &out, labels.len())?;
        let logits = out.to_vec()?;
        let mut correct = 0;
        for (scores, label) in logits.chunks_exact(self.names.len()).zip(labels) {
            let p = probabilities(scores)?;
            if p.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
                == *label
            {
                correct += 1;
            }
        }
        Ok(correct as f64 / labels.len() as f64)
    }
}
fn probabilities(scores: &[f32]) -> Result<Vec<f32>, Error> {
    if scores.iter().any(|v| !v.is_finite()) {
        return Err("Training diverged; collect more varied examples and retrain.".into());
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p = scores.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum: f32 = p.iter().sum();
    Ok(p.into_iter().map(|v| v / sum).collect())
}
fn pixels(value: &Value) -> Result<Vec<f32>, Error> {
    let p = value.as_array().ok_or("expected RGB image pixels")?;
    if p.len() != INPUT {
        return Err("expected a 32×32 RGB image (3072 bytes)".into());
    }
    p.iter()
        .map(|v| {
            v.as_u64()
                .filter(|v| *v <= 255)
                .map(|v| v as f32 / 255.0 - 0.5)
                .ok_or_else(|| "pixels must be bytes 0–255".into())
        })
        .collect()
}
struct Dataset {
    names: Vec<String>,
    x: Vec<f32>,
    y: Vec<usize>,
    validation_x: Vec<f32>,
    validation_y: Vec<usize>,
    samples: usize,
}
fn dataset(args: &Value) -> Result<Dataset, Error> {
    let classes = args["classes"].as_array().ok_or("missing classes")?;
    if !(2..=4).contains(&classes.len()) {
        return Err("create two to four classes".into());
    }
    let mut data = Dataset {
        names: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        validation_x: Vec::new(),
        validation_y: Vec::new(),
        samples: 0,
    };
    for (label, class) in classes.iter().enumerate() {
        let name = class["name"].as_str().ok_or("missing class name")?.trim();
        if name.is_empty()
            || name.chars().count() > 40
            || data
                .names
                .iter()
                .any(|n| n.to_lowercase() == name.to_lowercase())
        {
            return Err("use unique class names of 1–40 characters".into());
        }
        let samples = class["samples"].as_array().ok_or("missing samples")?;
        if !(10..=60).contains(&samples.len()) {
            return Err("collect 10–60 samples for every class".into());
        }
        data.names.push(name.into());
        data.samples += samples.len();
        for (index, sample) in samples.iter().enumerate() {
            let p = pixels(sample)?;
            // Reserve every fifth capture for validation; never train on those frames.
            if index % 5 == 4 {
                data.validation_x.extend(p);
                data.validation_y.push(label);
            } else {
                data.x.extend(p);
                data.y.push(label);
            }
        }
    }
    Ok(data)
}
fn train(args: &Value) -> Result<Value, Error> {
    let _training = TRAINING
        .try_lock()
        .map_err(|_| "Another training run is active on this backend. Try again shortly.")?;
    let data = dataset(args)?;
    let classes = data.names.len();
    let batch = data.y.len();
    let start = Instant::now();
    let mut seed = 42u32;
    let mut random = |count: usize, inputs: usize| {
        let scale = (6.0 / inputs as f32).sqrt();
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 16777216.0 * 2.0 - 1.0) * scale
            })
            .collect::<Vec<_>>()
    };
    let model = Classifier {
        names: data.names,
        w1: array(&random(INPUT * HIDDEN, INPUT))?,
        b1: zeros(HIDDEN)?,
        w2: array(&random(HIDDEN * classes, HIDDEN))?,
        b2: zeros(classes)?,
    };
    let x = array(&data.x)?;
    let labels = GpuArray::from_slice(&data.y.iter().map(|v| *v as i32).collect::<Vec<_>>())?;
    let hidden = zeros(batch * HIDDEN)?;
    let output = zeros(batch * classes)?;
    let grad = zeros(batch * classes)?;
    let previous = zeros(batch * HIDDEN)?;
    let w1 = Adam::new(INPUT * HIDDEN)?;
    let b1 = Adam::new(HIDDEN)?;
    let w2 = Adam::new(HIDDEN * classes)?;
    let b2 = Adam::new(classes)?;
    for epoch in 0..EPOCHS {
        model.forward(&x, &hidden, &output, batch)?;
        // The forward buffers and gradient buffers are distinct. Parameter updates happen
        // after both layers' gradients are computed using the unchanged forward weights.
        unsafe {
            launch!(softmax_gradient<<<[(batch as u32).div_ceil(64),1,1],[64,1,1]>>>(&output,&labels,&grad,batch as i32,classes as i32))?;
            launch!(backward<<<[((batch*HIDDEN) as u32).div_ceil(128),1,1],[128,1,1]>>>(&grad,&model.w2,&hidden,&previous,classes as i32,(batch*HIDDEN) as i32))?;
            launch!(gradients<<<[((INPUT*HIDDEN) as u32).div_ceil(128),1,1],[128,1,1]>>>(&x,&previous,&w1.grad,&b1.grad,INPUT as i32,HIDDEN as i32,batch as i32))?;
            launch!(gradients<<<[((HIDDEN*classes) as u32).div_ceil(128),1,1],[128,1,1]>>>(&hidden,&grad,&w2.grad,&b2.grad,HIDDEN as i32,classes as i32,batch as i32))?;
        }
        w1.step(&model.w1, epoch)?;
        b1.step(&model.b1, epoch)?;
        w2.step(&model.w2, epoch)?;
        b2.step(&model.b2, epoch)?;
    }
    let training = model.accuracy(&data.x, &data.y)?;
    let validation = model.accuracy(&data.validation_x, &data.validation_y)?;
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let response = json!({"model":id,"names":model.names,"training_accuracy":training,"validation_accuracy":validation,"samples":data.samples,"epochs":EPOCHS,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0});
    let mut models = MODELS.lock().map_err(|_| "model store unavailable")?;
    models.insert(id, Arc::new(model));
    while models.len() > 8 {
        models.pop_first();
    }
    Ok(response)
}
pub fn run(path: &str, args: &Value) -> Result<Value, Error> {
    if path == "/api/teach-train" {
        return train(args);
    }
    let id = args["model"].as_u64().ok_or("missing model id")?;
    let model = MODELS
        .lock()
        .map_err(|_| "model store unavailable")?
        .get(&id)
        .cloned()
        .ok_or("This model has expired or the backend restarted. Train again.")?;
    let p = pixels(&args["pixels"])?;
    let start = Instant::now();
    let scores = model.scores(&p)?;
    let winner = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    Ok(
        json!({"names":model.names,"scores":scores,"winner":winner,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn samples(color: [u8; 3]) -> Vec<Value> {
        (0..10).map(|_| json!(color.repeat(1024))).collect()
    }
    #[test]
    fn validates_dataset_and_keeps_validation_frames_out_of_training() {
        let mut args = json!({"classes":[{"name":"Red","samples":samples([255,0,0])},{"name":"Blue","samples":samples([0,0,255])}]});
        let data = dataset(&args).unwrap();
        assert_eq!(data.y.len(), 16);
        assert_eq!(data.validation_y.len(), 4);
        assert_eq!(data.x.len(), 16 * INPUT);
        args["classes"][1]["name"] = json!(" red ");
        assert!(dataset(&args).is_err());
        args["classes"][1]["name"] = json!("Blue");
        args["classes"][1]["samples"][0] = json!([1, 2]);
        assert!(dataset(&args).is_err());
        assert!(pixels(&json!(vec![256; INPUT])).is_err());
    }
    #[test]
    #[ignore = "requires GPU"]
    fn learns_two_image_classes_and_predicts_unseen_colors() -> Result<(), Error> {
        let args = json!({"classes":[{"name":"Red","samples":samples([255,0,0])},{"name":"Blue","samples":samples([0,0,255])}]});
        let result = run("/api/teach-train", &args)?;
        assert_eq!(result["validation_accuracy"], 1.0);
        for (color, label) in [([220, 15, 10], 0), ([10, 15, 220], 1)] {
            let predicted = run(
                "/api/teach-predict",
                &json!({"model":result["model"],"pixels":color.repeat(1024)}),
            )?;
            assert_eq!(predicted["winner"], label);
        }
        Ok(())
    }
}
