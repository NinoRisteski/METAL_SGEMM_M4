# 1D Microtile Worklog

This log explains the 1D microtiling optimization that builds on threadgroup
tiling to achieve higher throughput by having each thread compute multiple
output elements.

## 1) The core idea: register-level reuse

In threadgroup tiling (v2), each thread computes one output element. With 1D
microtiling, each thread computes a 1×TN strip (1 row, TN columns) of the
output matrix. We use TN=8.

The key insight: when computing C[row, col..col+7], we load A[row, k] once
and multiply it against B[k, col], B[k, col+1], ..., B[k, col+7]. This reuses
the A value 8 times, reducing global memory traffic.

## 2) Tile dimensions

```
BM = 16    // rows per threadgroup tile
BN = 128   // cols per threadgroup tile (16 threads × 8 outputs each)
BK = 16    // K dimension per tile
TN = 8     // outputs per thread in N dimension
```

With a 16×16 threadgroup (256 threads):
- Output tile: 16 rows × 128 cols = 2048 elements
- Each thread: 1 row × 8 cols = 8 elements
- Shared memory: As[16][16] + Bs[16][128] = 256 + 2048 = 2304 floats ≈ 9KB

## 3) Cooperative loading strategy

**Loading A (256 elements, 256 threads):**
- Thread (tid.y, tid.x) loads As[tid.y][tid.x]
- Maps directly: 16×16 threads load 16×16 tile

**Loading B (2048 elements, 256 threads):**
- Each thread loads 8 elements
- Linear thread id: `t = tid.y * 16 + tid.x`
- Thread t loads elements at indices t×8 through t×8+7
- Row/col computed as: `row = idx / 128`, `col = idx % 128`

## 4) Compute phase

```metal
for (uint kk = 0; kk < BK; ++kk) {
    float a_val = As[tid.y][kk];  // Load once
    for (uint i = 0; i < TN; ++i) {
        acc[i] = fma(a_val, Bs[kk][tid.x * TN + i], acc[i]);  // Reuse 8×
    }
}
```

Each thread maintains 8 accumulators in registers. The A value is loaded once
per K iteration and reused across all 8 B values.

## 5) Host-side changes: output_tile parameter

The previous dispatch logic assumed 1 output per thread:
```rust
threadgroups.width = ceil(n / threads_per_threadgroup.width)
```

With microtiling, each threadgroup produces 128 columns (not 16), so we added
an `output_tile` field to the Kernel struct:

```rust
struct Kernel {
    // ...existing fields...
    output_tile: Option<MTLSize>,  // Output elements per threadgroup
}
```

For 1D Microtile:
- `threads_per_threadgroup`: 16×16 (actual threads)
- `output_tile`: 128×16 (output coverage)

The dispatch logic now uses output_tile for grid calculation:
```rust
let tile = output_tile.unwrap_or(threads_per_threadgroup);
let threadgroups = MTLSize {
    width: (n + tile.width - 1) / tile.width,
    height: (m + tile.height - 1) / tile.height,
    ...
};
```

## 6) Performance results

```
Kernel                512    1024    2048    4096
Threadgroup Tiling    343     488     524     527
1D Microtile          404     707     765     752
```

**43% improvement at 4096×4096** (527 → 752 GFLOPS)

The gains come from:
1. 8× reuse of A values in registers (reduced memory traffic)
2. Larger output tile means fewer threadgroups and less dispatch overhead
3. Better arithmetic intensity (more FLOPs per byte loaded)

## 7) Why "1D"?

This is called 1D microtiling because each thread's micro-tile is one-dimensional:
1 row × 8 columns. The next optimization (2D microtiling) will have each thread
compute an 8×8 block, providing reuse in both dimensions.
