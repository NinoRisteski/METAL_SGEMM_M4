#![allow(unexpected_cfgs)]

use anyhow::{Result, anyhow};
use metal::NSUInteger;
use metal::{
    Buffer, CaptureDescriptor, CaptureManager, ComputePipelineState, Device, MTLCaptureDestination,
    MTLResourceOptions, MTLSize,
};
use objc::runtime::{NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use rand::Rng;
use std::collections::HashMap;
use std::io::Write;
use std::mem;
use std::time::Instant;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

const MPS_DATA_TYPE_FLOAT32: NSUInteger = 0x10000000 | 32;

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
    /// Output elements per threadgroup (width=N, height=M).
    /// If None, defaults to threads_per_threadgroup (1 output per thread).
    output_tile: Option<MTLSize>,
}

const KERNELS: &[Kernel] = &[
    Kernel {
        name: "Naive",
        shader_path: "shaders/naive.metal",
        function_name: "naive",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        output_tile: None,
    },
    Kernel {
        name: "Contiguous Global",
        shader_path: "shaders/contiguous_global.metal",
        function_name: "sgemm_v1_contig_global",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        output_tile: None,
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
        output_tile: None,
    },
    Kernel {
        name: "1D Microtile",
        shader_path: "shaders/microtile_1d.metal",
        function_name: "sgemm_microtile_1d",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=16 rows, BN=128 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 128,
            height: 16,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile",
        shader_path: "shaders/microtile_2d.metal",
        function_name: "sgemm_microtile_2d",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=64 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 64,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile Vec4",
        shader_path: "shaders/microtile_2d_vec4.metal",
        function_name: "sgemm_microtile_2d_vec4",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=64 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 64,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile Vec4 BK8",
        shader_path: "shaders/microtile_2d_vec4_bk8.metal",
        function_name: "sgemm_microtile_2d_vec4_bk8",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=64 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 64,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile Vec4 BK32",
        shader_path: "shaders/microtile_2d_vec4_bk32.metal",
        function_name: "sgemm_microtile_2d_vec4_bk32",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=64 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 64,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile Vec4 Padded",
        shader_path: "shaders/microtile_2d_vec4_padded.metal",
        function_name: "sgemm_microtile_2d_vec4_padded",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=64 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 64,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "2D Microtile Vec4 4x2",
        shader_path: "shaders/microtile_2d_vec4_tm4_tn2.metal",
        function_name: "sgemm_microtile_2d_vec4_tm4_tn2",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
        // BM=64 rows, BN=32 cols per threadgroup
        output_tile: Some(MTLSize {
            width: 32,
            height: 64,
            depth: 1,
        }),
    },
    Kernel {
        name: "Simdgroup 8x8",
        shader_path: "shaders/simdgroup_8x8.metal",
        function_name: "sgemm_simdgroup_8x8",
        threads_per_threadgroup: MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
        output_tile: Some(MTLSize {
            width: 8,
            height: 8,
            depth: 1,
        }),
    },
];

// ============================================================================
// Benchmark configuration
// ============================================================================

const SIZES_TO_CHECK: &[usize] = &[8, 31, 32, 64, 127, 128, 256, 257];
const SIZES_TO_BENCH: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096];
const WARMUP_ITERS: usize = 3;
const MIN_BENCH_DURATION_SECS: f64 = 2.0;
const MAX_BENCH_ITERS: usize = 100;
const TOLERANCE: f32 = 1e-3;

// ============================================================================
// Matrix utilities
// ============================================================================

fn generate_random_matrix(device: &Device, rows: usize, cols: usize) -> Buffer {
    let data = generate_random_vec(rows * cols);
    buffer_from_vec(device, &data)
}

fn generate_random_vec(len: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn buffer_from_vec(device: &Device, data: &[f32]) -> Buffer {
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
// MPS baseline helpers
// ============================================================================

struct MpsMatrix {
    object: *mut Object,
    buffer: Buffer,
    rows: usize,
    cols: usize,
    row_bytes: usize,
}

impl Drop for MpsMatrix {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.object, release];
        }
    }
}

struct MpsMatmul {
    object: *mut Object,
}

impl Drop for MpsMatmul {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.object, release];
        }
    }
}

fn mps_row_bytes(cols: usize) -> usize {
    unsafe {
        let descriptor_cls = class!(MPSMatrixDescriptor);
        let row_bytes: NSUInteger = msg_send![
            descriptor_cls,
            rowBytesFromColumns: cols as NSUInteger
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        row_bytes as usize
    }
}

fn create_mps_matrix_from_dense(
    device: &Device,
    rows: usize,
    cols: usize,
    dense: &[f32],
) -> Result<MpsMatrix> {
    if dense.len() != rows * cols {
        return Err(anyhow!(
            "dense matrix has {} elements, expected {}",
            dense.len(),
            rows * cols
        ));
    }

    let row_bytes = mps_row_bytes(cols);
    let row_stride = row_bytes / mem::size_of::<f32>();
    let mut padded = vec![0.0_f32; rows * row_stride];
    for row in 0..rows {
        let dense_start = row * cols;
        let padded_start = row * row_stride;
        padded[padded_start..padded_start + cols]
            .copy_from_slice(&dense[dense_start..dense_start + cols]);
    }

    let buffer = buffer_from_vec(device, &padded);
    create_mps_matrix_with_buffer(device, rows, cols, row_bytes, buffer)
}

fn create_zero_mps_matrix(device: &Device, rows: usize, cols: usize) -> Result<MpsMatrix> {
    let row_bytes = mps_row_bytes(cols);
    let row_stride = row_bytes / mem::size_of::<f32>();
    let buffer = create_zero_buffer(device, rows * row_stride);
    create_mps_matrix_with_buffer(device, rows, cols, row_bytes, buffer)
}

fn create_mps_matrix_with_buffer(
    _device: &Device,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    buffer: Buffer,
) -> Result<MpsMatrix> {
    unsafe {
        let descriptor_cls = class!(MPSMatrixDescriptor);
        let descriptor: *mut Object = msg_send![
            descriptor_cls,
            matrixDescriptorWithRows: rows as NSUInteger
            columns: cols as NSUInteger
            rowBytes: row_bytes as NSUInteger
            dataType: MPS_DATA_TYPE_FLOAT32
        ];
        if descriptor.is_null() {
            return Err(anyhow!("failed to create MPSMatrixDescriptor"));
        }

        let matrix_alloc: *mut Object = msg_send![class!(MPSMatrix), alloc];
        let matrix: *mut Object = msg_send![
            matrix_alloc,
            initWithBuffer: buffer.as_ref()
            descriptor: descriptor
        ];
        if matrix.is_null() {
            return Err(anyhow!("failed to create MPSMatrix"));
        }

        Ok(MpsMatrix {
            object: matrix,
            buffer,
            rows,
            cols,
            row_bytes,
        })
    }
}

fn create_mps_matmul(device: &Device, m: usize, n: usize, k: usize) -> Result<MpsMatmul> {
    unsafe {
        let matmul_alloc: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
        let matmul: *mut Object = msg_send![
            matmul_alloc,
            initWithDevice: device.as_ref()
            transposeLeft: NO
            transposeRight: NO
            resultRows: m as NSUInteger
            resultColumns: n as NSUInteger
            interiorColumns: k as NSUInteger
            alpha: 1.0_f64
            beta: 0.0_f64
        ];
        if matmul.is_null() {
            return Err(anyhow!("failed to create MPSMatrixMultiplication"));
        }
        Ok(MpsMatmul { object: matmul })
    }
}

fn run_mps_matmul(
    command_queue: &metal::CommandQueue,
    matmul: &MpsMatmul,
    a: &MpsMatrix,
    b: &MpsMatrix,
    c: &MpsMatrix,
) {
    let command_buffer = command_queue.new_command_buffer();
    unsafe {
        let _: () = msg_send![
            matmul.object,
            encodeToCommandBuffer: command_buffer
            leftMatrix: a.object
            rightMatrix: b.object
            resultMatrix: c.object
        ];
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

fn mps_matrix_to_dense(matrix: &MpsMatrix) -> Vec<f32> {
    let row_stride = matrix.row_bytes / mem::size_of::<f32>();
    let padded = buffer_to_vec(&matrix.buffer, matrix.rows * row_stride);
    let mut dense = vec![0.0_f32; matrix.rows * matrix.cols];
    for row in 0..matrix.rows {
        let dense_start = row * matrix.cols;
        let padded_start = row * row_stride;
        dense[dense_start..dense_start + matrix.cols]
            .copy_from_slice(&padded[padded_start..padded_start + matrix.cols]);
    }
    dense
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
    output_tile: Option<MTLSize>,
) {
    // Use output_tile for grid calculation if specified, otherwise use threads_per_threadgroup
    let tile = output_tile.unwrap_or(threads_per_threadgroup);
    let threadgroups = MTLSize {
        width: (n as u64 + tile.width - 1) / tile.width,
        height: (m as u64 + tile.height - 1) / tile.height,
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
    output_tile: Option<MTLSize>,
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
        output_tile,
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
            let diff = check_kernel(
                device,
                command_queue,
                pipeline,
                sz,
                kernel.threads_per_threadgroup,
                kernel.output_tile,
            )?;
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
    if metal::mps::mps_supports_device(device) {
        print!("{:20} ", "MPS");
        std::io::stdout().flush()?;

        let mut all_passed = true;
        for &sz in SIZES_TO_CHECK {
            let diff = check_mps(device, command_queue, sz)?;
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
    } else {
        println!("{:20} SKIP (MPS unsupported on this device)", "MPS");
    }
    println!();
    Ok(())
}

fn check_mps(device: &Device, command_queue: &metal::CommandQueue, n: usize) -> Result<f32> {
    let (m, k) = (n, n);

    let a_vec = generate_random_vec(m * k);
    let b_vec = generate_random_vec(k * n);
    let matrix_a = create_mps_matrix_from_dense(device, m, k, &a_vec)?;
    let matrix_b = create_mps_matrix_from_dense(device, k, n, &b_vec)?;
    let matrix_c = create_zero_mps_matrix(device, m, n)?;
    let matmul = create_mps_matmul(device, m, n, k)?;

    run_mps_matmul(command_queue, &matmul, &matrix_a, &matrix_b, &matrix_c);

    let gpu_result = mps_matrix_to_dense(&matrix_c);
    let cpu_result = cpu_sgemm(m, n, k, &a_vec, &b_vec);

    let max_diff = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(g, c)| (g - c).abs())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);

    Ok(max_diff)
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
    output_tile: Option<MTLSize>,
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
            output_tile,
        );
    }

    // Time-based benchmarking
    let mut iterations = 0;
    let start = Instant::now();
    loop {
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
            output_tile,
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

fn bench_mps(device: &Device, command_queue: &metal::CommandQueue, n: usize) -> Result<f64> {
    let (m, k) = (n, n);

    let a_vec = generate_random_vec(m * k);
    let b_vec = generate_random_vec(k * n);
    let matrix_a = create_mps_matrix_from_dense(device, m, k, &a_vec)?;
    let matrix_b = create_mps_matrix_from_dense(device, k, n, &b_vec)?;
    let matrix_c = create_zero_mps_matrix(device, m, n)?;
    let matmul = create_mps_matmul(device, m, n, k)?;

    for _ in 0..WARMUP_ITERS {
        run_mps_matmul(command_queue, &matmul, &matrix_a, &matrix_b, &matrix_c);
    }

    let mut iterations = 0;
    let start = Instant::now();
    loop {
        run_mps_matmul(command_queue, &matmul, &matrix_a, &matrix_b, &matrix_c);
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
                kernel.output_tile,
            )?;
            print!("{:>7.0} ", gflops);
            std::io::stdout().flush()?;
        }
        println!();
    }

    if metal::mps::mps_supports_device(device) {
        print!("{:20} ", "MPS");
        std::io::stdout().flush()?;

        for &sz in SIZES_TO_BENCH {
            let gflops = bench_mps(device, command_queue, sz)?;
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
// GPU profiling (Xcode / Metal capture)
// ============================================================================

fn profile_kernel(
    device: &Device,
    command_queue: &metal::CommandQueue,
    pipelines: &HashMap<&'static str, ComputePipelineState>,
) -> Result<()> {
    const N: usize = 4096;
    const DEFAULT_KERNEL_NAME: &str = "2D Microtile Vec4 4x2";
    // Optional: `--profile "<kernel name>"` overrides the default.
    let kernel_name = std::env::args()
        .skip_while(|a| a != "--profile")
        .nth(1)
        .filter(|a| !a.starts_with("--"))
        .unwrap_or_else(|| DEFAULT_KERNEL_NAME.to_string());
    let kernel_name = kernel_name.as_str();

    let kernel = KERNELS
        .iter()
        .find(|k| k.name == kernel_name)
        .ok_or_else(|| anyhow!("kernel not found"))?;

    let pipeline = pipelines
        .get(kernel.name)
        .ok_or_else(|| anyhow!("pipeline not found"))?;

    let buffer_a = generate_random_matrix(device, N, N);
    let buffer_b = generate_random_matrix(device, N, N);
    let buffer_c = create_zero_buffer(device, N * N);

    let dims = MatrixDims {
        m: N as u32,
        n: N as u32,
        k: N as u32,
    };

    let dims_buffer = device.new_buffer_with_data(
        &dims as *const _ as *const _,
        mem::size_of::<MatrixDims>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    println!("Profiling {kernel_name} at {N}x{N}x{N}");

    let run = || {
        run_kernel(
            command_queue,
            pipeline,
            &buffer_a,
            &buffer_b,
            &buffer_c,
            &dims_buffer,
            N,
            N,
            kernel.threads_per_threadgroup,
            kernel.output_tile,
        )
    };

    // Warmup
    for _ in 0..5 {
        run();
    }

    let capture_manager = CaptureManager::shared();

    let safe_name = kernel_name.replace(' ', "_").replace('×', "x").replace('/', "_");
    let trace_path = format!("/Users/nino/METAL_SGEMM_M4/{safe_name}.gputrace");
    let trace_path = trace_path.as_str();
    // Metal refuses to overwrite an existing trace document.
    let _ = std::fs::remove_dir_all(trace_path);

    let descriptor = CaptureDescriptor::new();
    descriptor.set_capture_command_queue(command_queue);
    descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
    descriptor.set_output_url(trace_path);

    println!("Starting GPU capture...");
    capture_manager
        .start_capture(&descriptor)
        .map_err(|e| anyhow!("failed to start capture: {e:?}"))?;
    run();
    capture_manager.stop_capture();
    println!("GPU capture complete: {trace_path}");

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let device = Device::system_default().ok_or_else(|| anyhow!("No Metal device found"))?;
    let args: Vec<String> = std::env::args().collect();
    let check_only = args.iter().any(|arg| arg == "--check-only");

    print_device_info(&device);

    let pipelines = compile_all_kernels(&device)?;
    let command_queue = device.new_command_queue();

    if args.iter().any(|arg| arg == "--profile") {
        profile_kernel(&device, &command_queue, &pipelines)?;
    } else {
        run_checks(&device, &command_queue, &pipelines)?;
        if !check_only {
            run_benchmarks(&device, &command_queue, &pipelines)?;
        }
    }

    Ok(())
}
