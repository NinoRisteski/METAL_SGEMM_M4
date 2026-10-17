# SGEMM Benchmark Framework

A comprehensive benchmarking framework for evaluating Metal SGEMM kernel performance on Apple Silicon.

## Overview

This benchmark measures single-precision general matrix multiplication (SGEMM) performance across multiple kernel implementations and matrix sizes, reports results in GFLOPS.

## Methodology

### FLOPS Calculation

For matrix multiplication C = A × B where A is M×K and B is K×N:
- Each output element requires K multiplications and K additions
- Total operations: `2 × M × N × K`
- GFLOPS = `(2 × M × N × K) / time_seconds / 1e9`

### Time-Based Benchmarking

Unlike fixed-iteration benchmarks, this framework uses **time-based measurement**:

```
MIN_BENCH_DURATION_SECS = 2.0
MAX_BENCH_ITERS = 100
```

The benchmark runs until either:
1. Minimum duration (2 seconds) is reached, OR
2. Maximum iterations (100) is reached

**Why time-based?**
- Small matrices complete in microseconds; fixed iterations would under-sample
- Large matrices take milliseconds; fixed iterations would over-sample
- Time-based ensures statistically meaningful measurements across all sizes

### Warmup Phase

Each kernel runs 3 warmup iterations before timing begins:
- Ensures GPU caches are primed
- Pipeline state is fully initialized
- Avoids cold-start penalties in measurements

### Random Input Data

Matrices are filled with random values in `[-1.0, 1.0]`:
- Prevents compiler optimizations for special values (zeros, ones)
- Ensures realistic floating-point behavior
- Avoids denormalized numbers that could slow computation

### Buffer Clearing

The output buffer C is cleared before each timed iteration:
- Ensures consistent memory state
- Prevents any accumulation artifacts
- Represents realistic usage patterns

## Correctness Verification

Before benchmarking, each kernel is verified against a CPU reference implementation:

```
SIZES_TO_CHECK = [8, 32, 64, 128, 256]
TOLERANCE = 1e-3
```

- Tests multiple sizes to catch edge cases
- Tolerance accounts for floating-point accumulation differences between GPU and CPU
- Fails fast if any kernel produces incorrect results

## Benchmark Configuration

### Matrix Sizes

```
SIZES_TO_BENCH = [64, 128, 256, 512, 1024, 2048, 4096]
```

This range captures three performance regimes:

| Size | Regime | Characteristics |
|------|--------|-----------------|
| 64-256 | Latency-bound | Kernel launch overhead dominates |
| 512-1024 | Transition | Cache effects significant |
| 2048-4096 | Bandwidth-bound | Memory throughput limits performance |

### Adding New Kernels

To benchmark a new kernel, add an entry to the `KERNELS` array:

```rust
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
        name: "My New Kernel",
        shader_path: "shaders/my_kernel.metal",
        function_name: "sgemm_my_kernel",
        threads_per_threadgroup: MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        },
    },
];
```

Requirements for new kernels:
- Same buffer layout: A (buffer 0), B (buffer 1), C (buffer 2), dims (buffer 3)
- Same `MatrixDims` struct with `m`, `n`, `k` as `uint`
- Thread grid: `gid.x` = column, `gid.y` = row
- Threadgroup size should match any tiling assumptions in the kernel

## Output Format

### Device Information

```
=== Device Information ===

Name:                       Apple M4
Max Threadgroup Memory:     32768 bytes
Max Threads per Threadgroup: MTLSize { width: 1024, height: 1024, depth: 1024 }
```

Useful for understanding hardware constraints when optimizing kernels.

### Correctness Results

```
=== Correctness Checks ===

Naive                OK (all sizes passed, tol=1e-3)
Contiguous Global    OK (all sizes passed, tol=1e-3)
```

### Performance Results

```
=== Benchmark Results (GFLOPS) ===

Kernel                    64     128     256     512    1024    2048    4096
-------------------- ------- ------- ------- ------- ------- ------- -------
Naive                      1      12      45     192     326     344     313
Contiguous Global          3      21     103     247     326     344     325
```

## Interpreting Results

### What Good Results Look Like

- **Correctness**: All kernels should pass with tolerance 1e-3
- **Scaling**: GFLOPS should increase with matrix size (until bandwidth-bound)
- **Peak**: On M4, naive implementations typically peak at 300-400 GFLOPS
- **Optimized**: Tiled/shared memory implementations can reach 1000+ GFLOPS

### Common Patterns

1. **Low GFLOPS at small sizes**: Normal; kernel launch overhead dominates
2. **Similar GFLOPS at large sizes**: Memory bandwidth is the bottleneck
3. **Peak at mid-sizes (2048)**: Sweet spot between latency and bandwidth
4. **Drop at very large sizes (4096+)**: Cache thrashing, TLB misses

### Comparing Kernels

When evaluating optimizations:
- Compare across ALL sizes, not just one
- Small-size improvements indicate reduced latency/overhead
- Large-size improvements indicate better memory access patterns
- Mid-size improvements indicate better cache utilization

## Running the Benchmark

```bash
# Release mode for accurate measurements
cargo run --release

# Debug mode (slower, but useful for development)
cargo run
```

Always use release mode for performance measurements. Debug mode includes bounds checking and unoptimized code that doesn't reflect real-world performance.
