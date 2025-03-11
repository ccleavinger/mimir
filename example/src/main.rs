use mimir::mimir_global;

#[mimir_global]
fn assign_mult_by_2(input: &[f32], output: &mut [f32]) {
    let idx = block_idx * block_dim + thread_idx;
    output[idx] = input[idx] * 2.0;
}

fn main() {
    println!("If you see this message, the macro has been successfully executed!");
}
