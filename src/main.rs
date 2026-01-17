use anyhow::{anyhow, Result};
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use rand::Rng;
use std::collections::HashMap;
use std::io::Write;
use std::mem;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct MatrixDims {
    m: u32,
    n: u32,
    k: u32,
}

// ============================================================================
// Kernel registry - add new kernels here
// ============================================================================

struct Kernel {
    name: &'static str,
    shader_path: &'static str,
    function_name: &'static str,
    threads_per_threadgroup: MTLSize,
}

const KERNELS: &[Kernel] = &[
    Kernel {
        name: "Naive",
        shader_path: "shaders/naive.metal",
        function_name: "naive",
        threads_per_threadgroup: MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        },
    },
    Kernel {
        name: "Contiguous Global",
        shader_path: "shaders/contiguous_global.metal",
        function_name: "sgemm_v1_contig_global",
        threads_per_threadgroup: MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        },
    },
    Kernel {
        name: "Threadgroup Tiling",
        shader_path: "shaders/threadgroup_tiling.metal",
        function_name: "sgemm_tiled16",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    },
    // Add new kernels here:
    // Kernel {
    //     name: "Tiled",
    //     shader_path: "shaders/tiled.metal",
    //     function_name: "sgemm_tiled",
    // },
];

// ============================================================================
// Benchmark configuration
// ============================================================================

const SIZES_TO_CHECK: &[usize] = &[8, 32, 64, 128, 256];
const SIZES_TO_BENCH: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
const WARMUP_ITERS: usize = 3;
const MIN_BENCH_DURATION_SECS: f64 = 2.0;
const MAX_BENCH_ITERS: usize = 100;
const TOLERANCE: f32 = 1e-3;

// ============================================================================
// Matrix utilities
// ============================================================================

fn generate_random_matrix(device: &Device, rows: usize, cols: usize) -> Buffer {
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_zero_buffer(device: &Device, len: usize) -> Buffer {
    let buffer = device.new_buffer(
        (len * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::write_bytes(buffer.contents(), 0, len * mem::size_of::<f32>());
    }
    buffer
}

fn buffer_to_vec(buffer: &Buffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, len).to_vec() }
}

fn clear_buffer(buffer: &Buffer, len: usize) {
    unsafe {
        std::ptr::write_bytes(buffer.contents(), 0, len * mem::size_of::<f32>());
    }
}

// ============================================================================
// CPU reference implementation
// ============================================================================

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

// ============================================================================
// Pipeline management
// ============================================================================

fn compile_kernel(device: &Device, kernel: &Kernel) -> Result<ComputePipelineState> {
    let shader_source = std::fs::read_to_string(kernel.shader_path)
        .map_err(|e| anyhow!("failed to read {}: {e}", kernel.shader_path))?;

    let compile_options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(&shader_source, &compile_options)
        .map_err(|e| anyhow!("failed to compile {}: {e}", kernel.shader_path))?;

    let function = library
        .get_function(kernel.function_name, None)
        .map_err(|e| anyhow!("failed to find function {}: {e}", kernel.function_name))?;

    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow!("failed to create pipeline for {}: {e}", kernel.name))
}

fn compile_all_kernels(device: &Device) -> Result<HashMap<&'static str, ComputePipelineState>> {
    let mut pipelines = HashMap::new();
    for kernel in KERNELS {
        let pipeline = compile_kernel(device, kernel)?;
        pipelines.insert(kernel.name, pipeline);
    }
    Ok(pipelines)
}

// ============================================================================
// Kernel execution
// ============================================================================

fn run_kernel(
    command_queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    buffer_a: &Buffer,
    buffer_b: &Buffer,
    buffer_c: &Buffer,
    dims_buffer: &Buffer,
    m: usize,
    n: usize,
    threads_per_threadgroup: MTLSize,
) {
    let threadgroups = MTLSize {
        width: (n as u64 + threads_per_threadgroup.width - 1) / threads_per_threadgroup.width,
        height: (m as u64 + threads_per_threadgroup.height - 1) / threads_per_threadgroup.height,
        depth: 1,
    };

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer_a), 0);
    encoder.set_buffer(1, Some(buffer_b), 0);
    encoder.set_buffer(2, Some(buffer_c), 0);
    encoder.set_buffer(3, Some(dims_buffer), 0);
    encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ============================================================================
// Correctness checking
// ============================================================================

fn check_kernel(
    device: &Device,
    command_queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    n: usize,
    threads_per_threadgroup: MTLSize,
) -> Result<f32> {
    let (m, k) = (n, n);

    let buffer_a = generate_random_matrix(device, m, k);
    let buffer_b = generate_random_matrix(device, k, n);
    let buffer_c = create_zero_buffer(device, m * n);

    let dims = MatrixDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let dims_buffer = device.new_buffer_with_data(
        &dims as *const _ as *const _,
        mem::size_of::<MatrixDims>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    run_kernel(
        command_queue,
        pipeline,
        &buffer_a,
        &buffer_b,
        &buffer_c,
        &dims_buffer,
        m,
        n,
        threads_per_threadgroup,
    );

    let gpu_result = buffer_to_vec(&buffer_c, m * n);
    let a_vec = buffer_to_vec(&buffer_a, m * k);
    let b_vec = buffer_to_vec(&buffer_b, k * n);
    let cpu_result = cpu_sgemm(m, n, k, &a_vec, &b_vec);

    let max_diff = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(g, c)| (g - c).abs())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);

    Ok(max_diff)
}

fn run_checks(
    device: &Device,
    command_queue: &metal::CommandQueue,
    pipelines: &HashMap<&'static str, ComputePipelineState>,
) -> Result<()> {
    println!("=== Correctness Checks ===\n");

    for kernel in KERNELS {
        let pipeline = pipelines.get(kernel.name).unwrap();
        print!("{:20} ", kernel.name);
        std::io::stdout().flush()?;

        let mut all_passed = true;
        for &sz in SIZES_TO_CHECK {
            let diff =
                check_kernel(device, command_queue, pipeline, sz, kernel.threads_per_threadgroup)?;
            if diff.is_nan() || diff > TOLERANCE {
                print!("FAIL({sz}:{diff:.2e}) ");
                all_passed = false;
            }
        }
        if all_passed {
            println!("OK (all sizes passed, tol={TOLERANCE:.0e})");
        } else {
            println!();
        }
    }
    println!();
    Ok(())
}

// ============================================================================
// Benchmarking
// ============================================================================

fn bench_kernel(
    device: &Device,
    command_queue: &metal::CommandQueue,
    pipeline: &ComputePipelineState,
    n: usize,
    threads_per_threadgroup: MTLSize,
) -> Result<f64> {
    let (m, k) = (n, n);

    let buffer_a = generate_random_matrix(device, m, k);
    let buffer_b = generate_random_matrix(device, k, n);
    let buffer_c = create_zero_buffer(device, m * n);

    let dims = MatrixDims {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let dims_buffer = device.new_buffer_with_data(
        &dims as *const _ as *const _,
        mem::size_of::<MatrixDims>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    for _ in 0..WARMUP_ITERS {
        run_kernel(
            command_queue,
            pipeline,
            &buffer_a,
            &buffer_b,
            &buffer_c,
            &dims_buffer,
            m,
            n,
            threads_per_threadgroup,
        );
    }

    // Time-based benchmarking
    let mut iterations = 0;
    let start = Instant::now();
    loop {
        clear_buffer(&buffer_c, m * n);
        run_kernel(
            command_queue,
            pipeline,
            &buffer_a,
            &buffer_b,
            &buffer_c,
            &dims_buffer,
            m,
            n,
            threads_per_threadgroup,
        );
        iterations += 1;

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= MIN_BENCH_DURATION_SECS || iterations >= MAX_BENCH_ITERS {
            let avg_time = elapsed / iterations as f64;
            let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
            let gflops = flops / avg_time / 1e9;
            return Ok(gflops);
        }
    }
}

fn run_benchmarks(
    device: &Device,
    command_queue: &metal::CommandQueue,
    pipelines: &HashMap<&'static str, ComputePipelineState>,
) -> Result<()> {
    println!("=== Benchmark Results (GFLOPS) ===\n");

    // Print header
    print!("{:20} ", "Kernel");
    for &sz in SIZES_TO_BENCH {
        print!("{:>7} ", sz);
    }
    println!();
    print!("{:-<20} ", "");
    for _ in SIZES_TO_BENCH {
        print!("{:->7} ", "");
    }
    println!();

    // Run benchmarks for each kernel
    for kernel in KERNELS {
        let pipeline = pipelines.get(kernel.name).unwrap();
        print!("{:20} ", kernel.name);
        std::io::stdout().flush()?;

        for &sz in SIZES_TO_BENCH {
            let gflops = bench_kernel(
                device,
                command_queue,
                pipeline,
                sz,
                kernel.threads_per_threadgroup,
            )?;
            print!("{:>7.0} ", gflops);
            std::io::stdout().flush()?;
        }
        println!();
    }

    println!();
    Ok(())
}

// ============================================================================
// Device info
// ============================================================================

fn print_device_info(device: &Device) {
    println!("=== Device Information ===\n");
    println!("Name:                       {}", device.name());
    println!("Registry ID:                {}", device.registry_id());
    println!("Is Low Power:               {}", device.is_low_power());
    println!(
        "Max Threadgroup Memory:     {} bytes",
        device.max_threadgroup_memory_length()
    );
    println!(
        "Max Threads per Threadgroup: {:?}",
        device.max_threads_per_threadgroup()
    );
    println!(
        "Max Buffer Length:          {} MB",
        device.max_buffer_length() / 1024 / 1024
    );
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;

    print_device_info(&device);

    let pipelines = compile_all_kernels(&device)?;
    let command_queue = device.new_command_queue();

    run_checks(&device, &command_queue, &pipelines)?;
    run_benchmarks(&device, &command_queue, &pipelines)?;

    Ok(())
}
