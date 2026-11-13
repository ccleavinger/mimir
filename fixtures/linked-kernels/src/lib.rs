#[mimir::mimir_global]
pub fn same_name(values: &mut [u32]) {
    values[global_id.x] = 1u32;
}
