70 segments of `tests/data/scottsbluff` (FABDEM, 1553 by 1311 cells), 445,270 library rows, with `dh` 0.5 m, `fldmn` 0.1 m, `fldmx` 10 m and spilling once a step. Median of 3 runs each.

| | Time (s) | Faster than MATLAB | Memory used (MB) | Peak memory (MB) |
| --- | ---: | ---: | ---: | ---: |
| MATLAB model | 137.7 | 1x | 184 | 784 |
| Python port of the MATLAB model | 53.8 | 3x | 41 | 202 |
| Curve2Flood, 1 thread | 0.5 | 289x | 82 | 326 |
| Curve2Flood, 4 threads | 0.4 | 376x | 156 | 409 |

All of them flood the same cells from each segment with the same DTFs. The port and Curve2Flood credit 2 of the rows to another stream pixel of the segment than MATLAB does, one that reaches the cell at the same depth, which changes no flood map.

On Intel(R) Core(TM) i7-14700 (20 cores, 28 threads) with 32 GB, Windows 11: MATLAB 25.2.0.3055257 (R2025b) Update 2, Python 3.14.4, NumPy 2.4.5, numba 0.65.1.
