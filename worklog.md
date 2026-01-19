# SGEMM Worklog

- Chip: Apple M4

## Results (GFLOPS)

| Kernel | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 20 | 94 | 253 | 327 | 352 | 322 |
| Contiguous Global | 2 | 18 | 102 | 256 | 325 | 352 | 348 |
| Threadgroup Tiling | 4 | 24 | 129 | 376 | 478 | 532 | 534 |


| Kernel | GFLOPs/s | Performance vs MPS |
| --- | --- | --- |
| -v0 Naive | ... | ...% |
| -v1 Contiguous global | ... | ...% |
| -v2 Threadgroup tiling | ... | ...% |
| v3 1D microtile | ... | ...% |
| v4 2D microtile | ... | ...% |
| v5 Vector loads | ... | ...% |
| v6 Pad TG tiles | ... | ...% |
| v7 Double buffer | ... | ...% |
| v8 Simdgroup-aware | ... | ...% |
| v9 Autotuned | ... | ...% |
| v10 Packed B | ... | ...% |
| MPS (reference) | ... | 100% |
