# Threadgroup Tiling Worklog

This log explains, from first principles, what changed when we introduced the
threadgroup-tiling SGEMM kernel and updated the host code to run it correctly.

## 1) Added a tiled kernel (GPU side)

We introduced `shaders/threadgroup_tiling.metal` with a 16x16 tiling strategy.
The core idea is:

- Each threadgroup computes a 16x16 tile of the output matrix C.
- Threads cooperate to load a 16x16 tile from A and a 16x16 tile from B into
  fast threadgroup (shared) memory.
- Each thread accumulates one output element by looping over the K tiles.

This reduces repeated global-memory reads because each tile of A and B is reused
by 16x16 threads before moving to the next K tile.

## 2) Registered the kernel (host side)

We added a new kernel entry in `src/main.rs` so it is compiled, checked, and
benchmarked alongside the existing kernels:

- Name: "Threadgroup Tiling"
- Shader path: `shaders/threadgroup_tiling.metal`
- Function name: `sgemm_tiled16`

This lets the same correctness and benchmark harness exercise the new shader.

## 3) Correctness failure and root cause

Initial correctness checks failed for the tiled kernel. The reason:

- The kernel assumes a 16x16 threadgroup (because the shared tiles are 16x16).
- The host code was hard-coded to dispatch 8x8 threadgroups.
- That left the shared tiles half-populated, so the math was wrong.

In short, the kernel's assumptions about threadgroup size did not match the
dispatch configuration.

## 4) Host-side fix: per-kernel threadgroup size

We updated `src/main.rs` to make threadgroup size part of the kernel definition:

- Added `threads_per_threadgroup: MTLSize` to the `Kernel` struct.
- Set 8x8 for the naive and contiguous kernels (initially).
- Set 16x16 for `Threadgroup Tiling` (required by the 16x16 tiles).

Then we threaded that size through:

- `run_kernel(...)` (now takes threadgroup size and computes grid counts from it)
- `check_kernel(...)` and `bench_kernel(...)`
- `run_checks(...)` and `run_benchmarks(...)`

This ensures each kernel is dispatched with the geometry it expects.

## 5) Unified threadgroup size for fair comparison

To compare GFLOPS on the same testing ground, we later standardized the
threadgroup size to 16x16 for all kernels. This keeps the tiling kernel correct
and puts naive and contiguous kernels under the same launch geometry.

## 6) Result

Correctness checks now pass for the tiled kernel, and the benchmark shows a
clear speedup at larger sizes due to reuse of tiles in threadgroup memory.
