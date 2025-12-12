use anyhow::{anyhow, Context, Result};
use metal::{Device, MTLResourceOptions, MTLSize};
use std::mem;
use std::path::Path;
use std::slice;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatrixDims {
    m: u32,
    n: u32,
    k: u32,
}

const SHADER_PATH: &str = "shaders/sgemm.metal";
const MATRIX_SIZE: usize = 64;

fn main() -> Result<()> {
    let device = Device::system_default().ok_or_else(|| anyhow!("Metal device unavailable"))?;
    println!("Using Metal device: {}", device.name());

    let shader_source = std::fs::read_to_string(Path::new(SHADER_PATH))
        .with_context(|| format!("failed to read shader at {SHADER_PATH}"))?;
    let compile_options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(&shader_source, &compile_options)
        .map_err(|err| anyhow!("failed to compile Metal shader source: {err}"))?;
    let function = library
        .get_function("sgemm", None)
        .map_err(|err| anyhow!("failed to find `sgemm` kernel: {err}"))?;
    let pipeline_state = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|err| anyhow!("failed to create compute pipeline state: {err}"))?;

    let (m, n, k) = (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    let dims = MatrixDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };

    let a_host = generate_matrix(m, k, 0.01);
    let b_host = generate_matrix(k, n, 0.02);
    let cpu_ref = cpu_sgemm(m, n, k, &a_host, &b_host);

    let buffer_len_bytes = |len: usize| (len * mem::size_of::<f32>()) as u64;
    let buffer_a = device.new_buffer_with_data(
        a_host.as_ptr() as *const _,
        buffer_len_bytes(a_host.len()),
        MTLResourceOptions::CPUCacheModeDefaultCache,
    );
    let buffer_b = device.new_buffer_with_data(
        b_host.as_ptr() as *const _,
        buffer_len_bytes(b_host.len()),
        MTLResourceOptions::CPUCacheModeDefaultCache,
    );
    let buffer_c =
        device.new_buffer(buffer_len_bytes(m * n), MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::write_bytes(
            buffer_c.contents(),
            0,
            mem::size_of::<f32>() * m * n,
        );
    }
    let dims_buffer = device.new_buffer_with_data(
        &dims as *const _ as *const _,
        mem::size_of::<MatrixDims>() as u64,
        MTLResourceOptions::CPUCacheModeDefaultCache,
    );

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&buffer_a), 0);
    encoder.set_buffer(1, Some(&buffer_b), 0);
    encoder.set_buffer(2, Some(&buffer_c), 0);
    encoder.set_buffer(3, Some(&dims_buffer), 0);

    let threads_per_threadgroup = MTLSize {
        width: 8,
        height: 8,
        depth: 1,
    };
    let grid_size = MTLSize {
        width: n as u64,
        height: m as u64,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: ceil_div(grid_size.width, threads_per_threadgroup.width),
        height: ceil_div(grid_size.height, threads_per_threadgroup.height),
        depth: 1,
    };
    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let c_result = unsafe {
        slice::from_raw_parts(buffer_c.contents() as *const f32, m * n).to_vec()
    };

    let max_diff = c_result
        .iter()
        .zip(cpu_ref.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);

    println!("SGEMM complete for {m}x{n} (k = {k})");
    println!("Maximum difference vs CPU reference: {max_diff:e}");

    Ok(())
}

fn generate_matrix(rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| (idx as f32 + 1.0) * scale)
        .collect()
}

fn cpu_sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = acc;
        }
    }
    c
}

fn ceil_div(num: u64, denom: u64) -> u64 {
    if denom == 0 {
        return 0;
    }
    (num + denom - 1) / denom
}
