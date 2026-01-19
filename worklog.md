# SGEMM Worklog

- Chip: Apple M4

## GFLOPS

| Kernel | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 13 | 39 | 217 | 332 | 353 | 350 |
| Contiguous Global | 2 | 17 | 102 | 247 | 332 | 353 | 355 |
| Threadgroup Tiling | 3 | 20 | 110 | 337 | 497 | 533 | 538 |


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
