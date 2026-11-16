//! Shared GPU inference used by the trainer and the demo. Version 1 remains readable.
use mimir::{launch, mimir_global, GpuArray};
use serde_json::{json, Value};
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub const INPUT: usize = 784;
pub const OUTPUT: usize = 10;
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub weights: GpuArray<f32>,
    pub bias: GpuArray<f32>,
}
pub struct Model {
    pub layers: Vec<Layer>,
}

#[mimir_global]
fn dense(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    inputs: i32,
    outputs: i32,
    count: i32,
    relu: bool,
) {
    let i = (block_idx.x * block_dim.x + thread_idx.x) as i32;
    if i < count {
        let col = i % outputs;
        let row = i / outputs;
        let mut sum = 0.0f32;
        for k in 0..inputs {
            sum += input[row * inputs + k] * weights[k * outputs + col];
        }
        sum += bias[col];
        if relu {
            sum = max(sum, 0.0);
        }
        output[i] = sum;
    }
}

impl Model {
    pub fn from_value(value: &Value) -> Result<Self, Error> {
        let raw = match value["format_version"].as_u64() {
            Some(1) => {
                if value["input_size"] != INPUT || value["output_size"] != OUTPUT {
                    return Err("invalid version 1 architecture".into());
                }
                vec![
                    json!({"inputs":INPUT,"outputs":value["hidden_size"],"weights":value["weights1"],"bias":value["bias1"]}),
                    json!({"inputs":value["hidden_size"],"outputs":OUTPUT,"weights":value["weights2"],"bias":value["bias2"]}),
                ]
            }
            Some(2) => value["layers"].as_array().ok_or("missing layers")?.clone(),
            _ => return Err("unsupported model version (expected 1 or 2)".into()),
        };
        if !(2..=5).contains(&raw.len()) {
            return Err("expected 2–5 dense layers".into());
        }
        fn parameters(v: &Value, count: usize) -> Result<Vec<f32>, Error> {
            let a = v.as_array().ok_or("missing parameters")?;
            if a.len() != count {
                return Err(format!("expected {count} parameters, found {}", a.len()).into());
            }
            a.iter()
                .map(|n| {
                    let f = n.as_f64().ok_or("non-numeric parameter")? as f32;
                    if !f.is_finite() {
                        return Err("non-finite parameter".into());
                    }
                    Ok(f)
                })
                .collect()
        }
        let mut expected = INPUT;
        let mut validated = Vec::new();
        for l in raw {
            let inputs = l["inputs"].as_u64().ok_or("missing inputs")? as usize;
            let outputs = l["outputs"].as_u64().ok_or("missing outputs")? as usize;
            if inputs != expected || !(1..=1024).contains(&outputs) {
                return Err("invalid or disconnected layer dimensions".into());
            }
            validated.push((
                inputs,
                outputs,
                parameters(&l["weights"], inputs * outputs)?,
                parameters(&l["bias"], outputs)?,
            ));
            expected = outputs;
        }
        if expected != OUTPUT {
            return Err("expected ten output classes".into());
        }
        // Check every parameter before allocating GPU storage.
        let layers = validated
            .into_iter()
            .map(|(inputs, outputs, w, b)| {
                Ok(Layer {
                    inputs,
                    outputs,
                    weights: GpuArray::from_slice(&w)?,
                    bias: GpuArray::from_slice(&b)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self { layers })
    }
    // Activations include the input; output logits are the final array.
    pub fn forward(&self, input: GpuArray<f32>, batch: usize) -> Result<Vec<GpuArray<f32>>, Error> {
        if batch == 0 || input.len() != batch * INPUT {
            return Err("invalid input shape".into());
        }
        let mut activations = vec![input];
        for (index, l) in self.layers.iter().enumerate() {
            let count = batch * l.outputs;
            let output = GpuArray::from_slice(&vec![0.0f32; count])?;
            // Each invocation owns one output; dimensions and buffers are checked above.
            unsafe {
                launch!(dense<<<[(count as u32).div_ceil(128),1,1],[128,1,1]>>>(activations.last().unwrap(),&l.weights,&l.bias,&output,l.inputs as i32,l.outputs as i32,count as i32,index+1<self.layers.len()))?;
            }
            activations.push(output);
        }
        Ok(activations)
    }
}

pub fn probabilities(logits: &[f32]) -> Result<Vec<f32>, Error> {
    if logits.len() != OUTPUT || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid model output".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum: f32 = exp.iter().sum();
    Ok(exp.into_iter().map(|v| v / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_architecture_and_parameter_counts_before_gpu_allocation() {
        assert!(Model::from_value(&json!({"format_version":99}))
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported"));
        let mut value = json!({"format_version":2,"layers":[
            {"inputs":784,"outputs":2,"weights":vec![0.0f32;1568],"bias":[0.0,0.0]},
            {"inputs":3,"outputs":10,"weights":vec![0.0f32;30],"bias":vec![0.0f32;10]}]});
        assert!(Model::from_value(&value)
            .err()
            .unwrap()
            .to_string()
            .contains("disconnected"));
        value["layers"][1]["inputs"] = json!(2);
        assert!(Model::from_value(&value)
            .err()
            .unwrap()
            .to_string()
            .contains("expected 20 parameters"));
        value["layers"][1]["weights"] = json!(vec![0.0f32; 20]);
        value["layers"][1]["weights"][0] = json!(1e100);
        assert!(Model::from_value(&value)
            .err()
            .unwrap()
            .to_string()
            .contains("non-finite"));
    }
}
