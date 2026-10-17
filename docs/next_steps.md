# Next Steps

Try these paths next, in this order.

## 1. Add an MPS baseline

Benchmark `MPSMatrixMultiplication` on the same sizes.

This gives the project a real target:

```text
this SGEMM vs Apple optimized SGEMM
```

Without this, we only know the kernels improved against earlier kernels. We do
not know how close they are to Apple-tuned performance.

## 2. Sweep tile shapes

The 2D microtile family is the best one so far, so test more shapes.

Thread microtiles:

```text
TM x TN:
4x4
4x8
8x4
8x8
8x16
16x8
```

Threadgroup tiles:

```text
64x64
64x128
128x64
128x128
```

The goal is to find the best balance between reuse, registers, and occupancy.

## 3. Measure register pressure

The Vec4, BK, and padding variants are starting to plateau. That may mean the
kernel is register-limited or occupancy-limited.

Check:

```text
threads per threadgroup
registers per thread
threadgroup memory usage
occupancy
```

Use Xcode GPU Frame Capture or Metal System Trace.

## 4. Try simdgroup-level SGEMM

This is probably the biggest unlock.

Look at kernels like:

```text
simdgroup tiled matmul
simdgroup_matrix multiply-accumulate
TiledSimd-style kernels
```

This is where public projects tend to jump from around `1.5 TFLOPS` into the
`2.5-5 TFLOPS` range.

## 5. Test rectangular blocking

Do not only test square matrices.

Try:

```text
M=N=K
M large, N small
M small, N large
K large
batch-style GEMM
```

This shows whether the kernel is generally useful or mainly tuned for square
SGEMM.

## 6. Add correctness stress tests

For every kernel variant, test:

```text
small sizes: 1, 2, 3, 7, 15, 31
non-multiple sizes: 65, 129, 257, 513
large sizes: 1024, 2048, 4096
```

Compare against CPU, Accelerate, or NumPy with a clear tolerance.

## 7. Compare against MLX, tinygrad, and PyTorch MPS

Use the same matrix sizes, dtype, warmup, and timing method:

```text
this kernel
MPS
MLX
PyTorch MPS
tinygrad
```

This makes the project much easier to judge from the outside.

## 8. Build a benchmark report

Track:

```text
GFLOPS
runtime ms
speedup vs naive
speedup vs MPS
best tile config
occupancy notes
correctness pass/fail
```

## Best Next Concrete Experiment

Implement simdgroup tiled SGEMM, then compare:

```text
2D Microtile BK32 vs Simdgroup SGEMM vs MPS
```

That is likely the path from around `1.5 TFLOPS` toward `2.5-3+ TFLOPS` on base
M4.
