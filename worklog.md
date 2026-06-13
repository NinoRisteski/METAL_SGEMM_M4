# SGEMM Worklog

- Chip: Apple M4

## GFLOPS

| Shader | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 18 | 88 | 210 | 333 | 352 | 329 |
| Contiguous Global | 2 | 17 | 101 | 244 | 288 | 342 | 330 |
| Threadgroup Tiling | 3 | 21 | 129 | 336 | 501 | 529 | 540 |
| 1D Microtile | 4 | 24 | 154 | 467 | 708 | 785 | 778 |
| 2D Microtile | 2 | 17 | 136 | 611 | 1241 | 1501 | 1516 |
| 2D Microtile Vec4 | 3 | 18 | 139 | 659 | 1292 | 1520 | 1504 |
| 2D Microtile Vec4 BK8 | 3 | 17 | 121 | 586 | 1200 | 1388 | 1398 |
| 2D Microtile Vec4 BK32 | 3 | 18 | 125 | 617 | 1250 | 1462 | 1518 |
| 2D Microtile Vec4 Padded | 3 | 18 | 137 | 609 | 1267 | 1460 | 1474 |


| Kernel | GFLOPs/s | Performance vs MPS |
| --- | --- | --- |
| -v0 Naive | ... | ...% |
| -v1 Contiguous global | ... | ...% |
| -v2 Threadgroup tiling | ... | ...% |
| v3 1D microtile | ... | ...% |
| v4 2D microtile | 1516 | ...% |
| v5 Vector loads | 1504 | ...% |
| v6 Pad TG tiles | 1474 | ...% |
| v6 BK sweep | 1518 | ...% |
| v7 Double buffer | ... | ...% |
| v8 Simdgroup-aware | ... | ...% |
| v9 Autotuned | ... | ...% |
| v10 Packed B | ... | ...% |
| MPS (reference) | ... | 100% |
