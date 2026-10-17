# SGEMM Worklog

- Chip: Apple M4

## GFLOPS

| Shader | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 2 | 21 | 92 | 183 | 312 | 330 | 305 |
| Contiguous Global | 3 | 17 | 101 | 235 | 310 | 330 | 300 |
| Threadgroup Tiling | 3 | 19 | 120 | 331 | 472 | 502 | 508 |
| 1D Microtile | 3 | 18 | 122 | 400 | 683 | 741 | 726 |
| 2D Microtile | 3 | 14 | 146 | 608 | 1214 | 1422 | 1440 |
| 2D Microtile Vec4 | 3 | 19 | 125 | 598 | 1189 | 1460 | 1535 |
| 2D Microtile Vec4 BK8 | 3 | 18 | 132 | 611 | 1232 | 1405 | 1434 |
| 2D Microtile Vec4 BK32 | 3 | 18 | 136 | 613 | 1290 | 1482 | 1506 |
| 2D Microtile Vec4 Padded | 3 | 19 | 138 | 593 | 1252 | 1413 | 1422 |
| MPS | 2 | 20 | 133 | 758 | 2096 | 2861 | 2878 |


| Kernel | GFLOPs/s | Performance vs MPS |
| --- | --- | --- |
| -v0 Naive | 305 | 11% |
| -v1 Contiguous global | 300 | 10% |
| -v2 Threadgroup tiling | 508 | 18% |
| v3 1D microtile | 726 | 25% |
| v4 2D microtile | 1440 | 50% |
| v5 Vector loads | 1535 | 53% |
| v6 Pad TG tiles | 1422 | 49% |
| v6 BK sweep | 1506 | 52% |
| v7 Double buffer | ... | ...% |
| v8 Simdgroup-aware | ... | ...% |
| v9 Autotuned | ... | ...% |
| v10 Packed B | ... | ...% |
| MPS (reference) | 2878 | 100% |
