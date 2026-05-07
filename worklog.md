# SGEMM Worklog

- Chip: Apple M4

## GFLOPS

| Shader | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 18 | 108 | 254 | 332 | 348 | 341 |
| Contiguous Global | 2 | 17 | 93 | 242 | 323 | 352 | 282 |
| Threadgroup Tiling | 2 | 18 | 93 | 321 | 454 | 514 | 526 |
| 1D Microtile | 3 | 18 | 121 | 391 | 712 | 761 | 769 |
| 2D Microtile | 3 | 18 | 139 | 608 | 1231 | 1474 | 1497 |


| Kernel | GFLOPs/s | Performance vs MPS |
| --- | --- | --- |
| -v0 Naive | ... | ...% |
| -v1 Contiguous global | ... | ...% |
| -v2 Threadgroup tiling | ... | ...% |
| v3 1D microtile | ... | ...% |
| v4 2D microtile | 1497 | ...% |
| v5 Vector loads | ... | ...% |
| v6 Pad TG tiles | ... | ...% |
| v7 Double buffer | ... | ...% |
| v8 Simdgroup-aware | ... | ...% |
| v9 Autotuned | ... | ...% |
| v10 Packed B | ... | ...% |
| MPS (reference) | ... | 100% |
