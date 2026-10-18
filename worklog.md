# SGEMM Worklog

- Chip: Apple M4
- All numbers below are from a single benchmark run (`cargo run --release`).

## GFLOPS

| Shader | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 19 | 105 | 253 | 332 | 352 | 320 |
| Contiguous Global | 3 | 19 | 106 | 256 | 334 | 353 | 320 |
| Threadgroup Tiling | 3 | 20 | 125 | 349 | 503 | 534 | 539 |
| 1D Microtile | 3 | 20 | 131 | 416 | 727 | 772 | 786 |
| 2D Microtile | 3 | 19 | 139 | 629 | 1255 | 1502 | 1546 |
| 2D Microtile Vec4 | 3 | 20 | 141 | 660 | 1343 | 1601 | 1638 |
| 2D Microtile Vec4 BK8 | 3 | 19 | 145 | 650 | 1291 | 1512 | 1536 |
| 2D Microtile Vec4 BK32 | 3 | 20 | 145 | 661 | 1325 | 1566 | 1605 |
| 2D Microtile Vec4 Padded | 3 | 21 | 142 | 660 | 1311 | 1540 | 1553 |
| 2D Microtile Vec4 4x2 | 3 | 21 | 132 | 600 | 1175 | 1391 | 1375 |
| Simdgroup 8x8 | 3 | 23 | 134 | 431 | 680 | 756 | 754 |
| Simdgroup 16x16 | 3 | 24 | 140 | 527 | 850 | 925 | 934 |
| Simdgroup 32x32 MultiFrag (BK8) | 3 | 23 | 160 | 732 | 1422 | 1602 | 1615 |
| Simdgroup 32x32 MultiFrag BK16 | 3 | 23 | 152 | 712 | 1449 | 1718 | 1740 |
| MPS | 3 | 21 | 138 | 814 | 2148 | 3036 | 3164 |

## Performance vs MPS (4096)

| Kernel | GFLOPS | vs MPS |
| --- | --- | --- |
| v0 Naive | 320 | 10% |
| v1 Contiguous global | 320 | 10% |
| v2 Threadgroup tiling | 539 | 17% |
| v3 1D microtile | 786 | 25% |
| v4 2D microtile | 1546 | 49% |
| v5 Vector loads | 1638 | 52% |
| v6 Pad TG tiles | 1553 | 49% |
| v6 BK sweep (BK32) | 1605 | 51% |
| v6 4x2 thread microtile | 1375 | 43% |
| v8 Simdgroup 8x8 | 754 | 24% |
| v8 Simdgroup 16x16 | 934 | 30% |
| v8 Simdgroup 32x32 MultiFrag (BK8) | 1615 | 51% |
| v8 Simdgroup 32x32 MultiFrag BK16 | 1740 | 55% |
| v7 Double buffer | ... | ...% |
| v9 Autotuned | ... | ...% |
| v10 Packed B | ... | ...% |
| MPS (reference) | 3164 | 100% |

## Experiments

### 4x2 thread microtile (BM=64, BN=32, BK=16)

Correct at all sizes, but ~13% slower than the 4x4 Vec4 kernel at 4096
(1375 vs 1638). Each thread does 8 FMAs per 6 shared-memory reads, versus
16 FMAs per 8 reads for 4x4, so less register-level reuse. The 4x4 kernel
profiles at 102 registers/thread and ~36% occupancy. The 4x2 trace has been
captured (`2D_Microtile_Vec4_4x2.gputrace`) but its register/occupancy numbers
are not recorded yet.

### Simdgroup kernels

- 8x8 (1 simdgroup per tile) -> 16x16 (4 simdgroups share staged A/B tiles):
  ~+24% at 4096, still far below the scalar microtile kernels. One 8x8x8 MMA
  per barrier pair leaves too little math between synchronizations.
- 32x32 MultiFrag: each simdgroup owns a 2x2 grid of 8x8 accumulators (4 MMAs
  per K-step). Jumps to ~1615 at 4096, on par with the best scalar kernel.
  Profile at 4096: 58 registers/thread (vs 102 for scalar 4x4), no spills,
  52.7% occupancy, instruction-throughput limiter 78.8%, F32 utilization
  44%, device read ~70 GiB/s (not bandwidth bound), threadgroup memory
  ~278 GiB/s read and write.
- BK8 -> BK16: 8 MMAs per barrier pair. +5% at 4096 (1615 -> 1740), now the
  best kernel at 55% of MPS. Not profiled yet.

### Fixes

- `Simdgroup 16x16`: the edge-output barrier was inside a per-simdgroup
  condition (undefined behavior); the condition is now threadgroup-uniform.

## Profiling

`METAL_CAPTURE_ENABLED=1 ./target/release/metal_sgemm_m4 --profile "<kernel name>"`
writes `<kernel_name_with_underscores>.gputrace` in the repo root (gitignored).
Without a name it profiles the default in `profile_kernel()`.

## Next

- Profile MultiFrag BK16 (registers, occupancy, threadgroup bandwidth).
- float4 loads into threadgroup memory for the simdgroup kernels.
- Larger per-simdgroup fragment grid (e.g. 4x2 fragments).
