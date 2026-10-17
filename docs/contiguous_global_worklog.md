# Contiguous Global Worklog

This log explains, from first principles, what we did to add the contiguous
global-memory SGEMM kernel and integrate it into the benchmark harness.

## 1) Added a contiguous-access kernel (GPU side)

We introduced `shaders/contiguous_global.metal` with a simple but deliberate
thread-to-data mapping:

- Each thread computes one output element C[row, col].
- The grid mapping uses `gid.x -> col` and `gid.y -> row`.
- For a fixed `kk`, threads that are adjacent in `x` read adjacent B elements:
  `B[kk * n + col]`.
- The write to `C[row * n + col]` is also contiguous across those threads.

This keeps global memory accesses coalesced across the threadgroup in the x
dimension, which makes better use of memory bandwidth.

The math is still the standard dot product:

```
C[row, col] = sum_{kk=0..k-1} A[row, kk] * B[kk, col]
```

The only change is how threads are mapped so that memory access is contiguous
for B and C across neighboring threads.

## 2) Registered the kernel (host side)

We added the kernel to the registry in `src/main.rs` so it is compiled, checked,
and benchmarked alongside the naive kernel:

- Name: "Contiguous Global"
- Shader path: `shaders/contiguous_global.metal`
- Function name: `sgemm_v1_contig_global`

This allows the same correctness checks and benchmarks to compare it directly
against the naive implementation under identical conditions.

## 3) Unified threadgroup size for benchmarking

To compare kernels on the same testing ground, we standardized the
threadgroup size to 16x16 for all kernels in `src/main.rs`. The contiguous
kernel does not require a specific size for correctness, so this change is
about fair and consistent performance measurement.

## 4) Result

Correctness matches the CPU reference, and performance improves over naive for
larger sizes due to more coalesced global memory access patterns.
