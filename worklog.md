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
| Simdgroup 32x32 MultiFrag BK16 Vec4 | 4 | 25 | 156 | 849 | 1951 | 1966 | 2101 |
| Simdgroup 64x32 MultiFrag BK16 Vec4 | 3 | 23 | 151 | 839 | 1743 | 1789 | 1786 |
| Simdgroup 32x32 MultiFrag BK16 Vec4 DB | 3 | 24 | 160 | 840 | 1991 | 2181 | 2056 |
| MPS | 3 | 21 | 138 | 814 | 2148 | 3036 | 3164 |

Note: the Vec4 row is from the best of 3 full runs; its 4096 value ranged
1735-2101 across runs (clock/thermal drift, MPS also varied 2777-3161). Use the
512-2048 columns for stable comparison.

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
| v8 Simdgroup 32x32 MultiFrag BK16 Vec4 | ~1735-2101 | ~55-66% |
| v8 Simdgroup 64x32 MultiFrag BK16 Vec4 (negative) | 1786 | 56% |
| v7 Double buffer (Simdgroup 32x32 BK16 Vec4 DB, null result) | ~1830-2056 | ~60-65% |
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

- BK16 profile @ 4096 vs BK8: occupancy 52.7% -> 41.5% yet 5% faster.
  F32 util 44% -> 48%, instruction limiter 78.8% -> 82.8%, device read
  70 -> 53 GiB/s, threadgroup read/write 278 -> 304 GiB/s. Lesson: more math per
  barrier pair beats higher occupancy.
- BK16 Vec4 (float4 global->threadgroup loads, everything else unchanged):
  +35% at 1024, ~+20% at 4096. Profile @ 4096 vs BK16 scalar: occupancy
  unchanged (41.5%), F32 utilization 48% -> 62%, F32 limiter 62% -> 73%,
  instruction limiter 83% -> 68%, device read 53 -> 74 GiB/s, threadgroup read
  304 -> 387 GiB/s, threadgroup write 304 -> 194 GiB/s. The scalar loader was
  burning instruction issue slots; wider loads free them for MMA. The kernel is
  now more F32-limited than instruction-limited.

### 64x32 MultiFrag BK16 Vec4 (negative result)

Each simdgroup holds a 4x2 grid of accumulators (8 MMAs per K-step of 8, 64x32
tile) instead of 2x2. Correct, but ~10% slower than the 32x32 Vec4 kernel
(~1780 vs ~1900-2060 at 4096; stable across 3 runs). Profile @ 4096 vs 32x32
Vec4: occupancy 41.5% -> 20.8%, F32 utilization 61.5% -> 53.7%, ALU 40% ->
33.6%, instruction limiter 68% -> 44%, device read 73.5 -> 49.4 GiB/s,
threadgroup read 387 -> 253 GiB/s. The extra accumulator state halves residency
and the added reuse does not make up for it. Conclusion: 2x2 fragments per
simdgroup on a 32x32 tile is the better balance; stop increasing fragment count.
Kernel kept in the repo as a documented experiment.

### 32x32 MultiFrag BK16 Vec4 DB (double buffering, null result)

Two threadgroup buffers for A and B (8 KB); next tile's float4 loads are issued
into registers before the MMAs and stored into the other buffer afterwards, with
one barrier per K-tile instead of two. Correct, but within run-to-run noise of
the non-DB Vec4 kernel at every size (3 runs: 1024 1899-1991 vs 1959-2018,
2048 2109-2181 vs 2036-2130, 4096 1831-2056 vs 1859-2033).

Profile @ 4096 vs BK16 Vec4: occupancy 41.5% -> 24.95%, ALU utilization 40.0% ->
48.6%, F32 utilization 61.5% -> 68.6%, F32 limiter 72.8% -> 82.9%, instruction
limiter 67.7% -> 78.6%, device read 73.5 -> 77.4 GiB/s, threadgroup read 387 ->
432 GiB/s, threadgroup write 194 -> 216 GiB/s.

Double buffering improved compute utilization but reduced occupancy from ~41.5%
to ~25% (extra threadgroup buffers plus live prefetch registers). The increased
latency hiding was offset by the resource pressure, so there is no consistent
end-to-end speedup. Kernel kept in the repo as a documented experiment.

### Fixes

- `Simdgroup 16x16`: the edge-output barrier was inside a per-simdgroup
  condition (undefined behavior); the condition is now threadgroup-uniform.

## Profiling

`METAL_CAPTURE_ENABLED=1 ./target/release/metal_sgemm_m4 --profile "<kernel name>"`
writes `<kernel_name_with_underscores>.gputrace` in the repo root (gitignored).
Without a name it profiles the default in `profile_kernel()`.

## Next

- GEMM study is complete enough: best kernel is Simdgroup 32x32 MultiFrag BK16
  Vec4 (~1.9-2.1 TFLOPS at 4096, ~60-65% of MPS). Larger fragment grids and
  double buffering both lost to occupancy.
- Move on to reduction kernels.
- If revisiting GEMM: re-measure 4096 with repeated runs / cooldown for a
  stable figure.
