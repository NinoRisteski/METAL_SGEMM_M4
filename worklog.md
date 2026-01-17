Chip: Apple M4

| Shader | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 1 | 12 | 45 | 192 | 326 | 344 | 313 |
| Contiguous Global | 3 | 21 | 103 | 247 | 326 | 344 | 325 |
| Threadgroup Tiling | 3 | 22 | 119 | 339 | 496 | 533 | 538 |
TODO:
| Shader                | GFLOPs/s | Performance relative to MPS |
| --------------------- | -------: | --------------------------: |
| -v0 Naive              |        … |                          …% |
| -v1 Contiguous global  |        … |                          …% |
| v2 Threadgroup tiling |        … |                          …% |
| v3 1D microtile       |        … |                          …% |
| v4 2D microtile       |        … |                          …% |
| v5 Vector loads       |        … |                          …% |
| v6 Pad TG tiles       |        … |                          …% |
| v7 Double buffer      |        … |                          …% |
| v8 Simdgroup-aware    |        … |                          …% |
| v9 Autotuned          |        … |                          …% |
| v10 Packed B          |        … |                          …% |
| MPS (reference)       |        … |                        100% |
