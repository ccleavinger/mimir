use super::Error;
use mimir::GpuArray;
use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::Instant;
#[path = "../../mnist/src/model.rs"]
mod model;

use model::{Model, INPUT};
static MODEL: OnceLock<Model> = OnceLock::new();

pub fn init() -> Result<(), Error> {
    let path = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/mnist-model.json"
    ));
    let load = || -> Result<Model, Error> {
        Model::from_value(&serde_json::from_slice(&std::fs::read(path)?)?)
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
    let activations = model.forward(input, 1)?;
    let logits = activations.last().unwrap().to_vec()?;
    let probabilities = model::probabilities(&logits)?;
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
