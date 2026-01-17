# SGEMM Worklog

- Chip: Apple M4

## Results (GFLOPS)

| Kernel | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 11 | 64 | 208 | 307 | 301 | 271 |
| Contiguous Global | 3 | 18 | 99 | 243 | 315 | 342 | 305 |
| Threadgroup Tiling | 3 | 21 | 115 | 321 | 497 | 533 | 539 |


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
