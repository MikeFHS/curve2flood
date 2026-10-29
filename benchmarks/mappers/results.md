# Mapper benchmark

Made by `run_benchmark.py` (see its docstring for what is measured and how).

Curve2Flood 0.3.1, Python 3.14.4.

| Site | Event | Mapper | CSI | MCC | Wet km² (truth) | Depth bias (m) | Depth RMSE (m) | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| straight | 1 m deep | kernel-weighted | 0.784 | 0.883 | 0.092 (0.072) | 0.00 | 0.00 | 0.05 |
| straight | 1 m deep | multi-point | 0.845 | 0.918 | 0.085 (0.072) | -0.00 | 0.00 | 0.07 |
| straight | 1 m deep | FLDPLN | 1.000 | 1.000 | 0.072 (0.072) | 0.00 | 0.00 | 0.06 |
| straight | 3 m deep | kernel-weighted | 0.983 | 0.990 | 0.601 (0.600) | 0.00 | 0.03 | 0.05 |
| straight | 3 m deep | multi-point | 1.000 | 1.000 | 0.600 (0.600) | -0.00 | 0.00 | 0.08 |
| straight | 3 m deep | FLDPLN | 1.000 | 1.000 | 0.600 (0.600) | -0.00 | 0.00 | 0.07 |
| straight | 5 m deep | kernel-weighted | 0.974 | 0.978 | 1.562 (1.560) | 0.00 | 0.07 | 0.06 |
| straight | 5 m deep | multi-point | 1.000 | 1.000 | 1.560 (1.560) | -0.00 | 0.00 | 0.10 |
| straight | 5 m deep | FLDPLN | 1.000 | 1.000 | 1.560 (1.560) | -0.00 | 0.00 | 0.06 |
| straight | library build | FLDPLN |  |  |  |  |  | 0.88 |
| meander | 1 m deep | kernel-weighted | 0.937 | 0.967 | 0.130 (0.122) | -0.00 | 0.01 | 0.05 |
| meander | 1 m deep | multi-point | 0.929 | 0.963 | 0.132 (0.122) | 0.00 | 0.01 | 0.09 |
| meander | 1 m deep | FLDPLN | 0.999 | 1.000 | 0.122 (0.122) | -0.07 | 0.12 | 0.07 |
| meander | 3 m deep | kernel-weighted | 0.942 | 0.962 | 0.824 (0.813) | 0.00 | 0.05 | 0.05 |
| meander | 3 m deep | multi-point | 0.993 | 0.996 | 0.818 (0.813) | 0.00 | 0.02 | 0.09 |
| meander | 3 m deep | FLDPLN | 0.942 | 0.963 | 0.766 (0.813) | -0.07 | 0.12 | 0.07 |
| meander | 5 m deep | kernel-weighted | 0.927 | 0.918 | 2.073 (2.056) | 0.01 | 0.17 | 0.06 |
| meander | 5 m deep | multi-point | 0.990 | 0.989 | 2.061 (2.056) | 0.00 | 0.09 | 0.11 |
| meander | 5 m deep | FLDPLN | 0.980 | 0.978 | 2.018 (2.056) | -0.07 | 0.11 | 0.07 |
| meander | library build | FLDPLN |  |  |  |  |  | 0.07 |
| confluence | 1 m deep | kernel-weighted | 0.815 | 0.899 | 0.152 (0.123) | -0.00 | 0.00 | 0.05 |
| confluence | 1 m deep | multi-point | 0.856 | 0.923 | 0.144 (0.123) | -0.00 | 0.00 | 0.08 |
| confluence | 1 m deep | FLDPLN | 0.999 | 1.000 | 0.124 (0.123) | -0.00 | 0.01 | 0.07 |
| confluence | 3 m deep | kernel-weighted | 0.963 | 0.975 | 0.890 (0.894) | -0.00 | 0.04 | 0.06 |
| confluence | 3 m deep | multi-point | 0.994 | 0.996 | 0.897 (0.894) | -0.00 | 0.01 | 0.10 |
| confluence | 3 m deep | FLDPLN | 0.996 | 0.997 | 0.890 (0.894) | -0.00 | 0.01 | 0.07 |
| confluence | 5 m deep | kernel-weighted | 0.971 | 0.969 | 2.033 (2.011) | 0.00 | 0.10 | 0.06 |
| confluence | 5 m deep | multi-point | 0.998 | 0.998 | 2.013 (2.011) | -0.01 | 0.05 | 0.11 |
| confluence | 5 m deep | FLDPLN | 1.000 | 0.999 | 2.010 (2.011) | -0.00 | 0.01 | 0.07 |
| confluence | library build | FLDPLN |  |  |  |  |  | 0.11 |
| levee_pond | 1 m deep | kernel-weighted | 0.784 | 0.883 | 0.092 (0.072) | 0.00 | 0.00 | 0.05 |
| levee_pond | 1 m deep | multi-point | 0.845 | 0.918 | 0.085 (0.072) | -0.00 | 0.00 | 0.07 |
| levee_pond | 1 m deep | FLDPLN | 1.000 | 1.000 | 0.072 (0.072) | 0.00 | 0.00 | 0.07 |
| levee_pond | 3 m deep | kernel-weighted | 0.983 | 0.990 | 0.601 (0.600) | 0.00 | 0.03 | 0.05 |
| levee_pond | 3 m deep | multi-point | 1.000 | 1.000 | 0.600 (0.600) | -0.00 | 0.00 | 0.09 |
| levee_pond | 3 m deep | FLDPLN | 1.000 | 1.000 | 0.600 (0.600) | -0.00 | 0.00 | 0.07 |
| levee_pond | 5 m deep | kernel-weighted | 0.982 | 0.987 | 1.105 (1.104) | 0.00 | 0.06 | 0.05 |
| levee_pond | 5 m deep | multi-point | 1.000 | 1.000 | 1.104 (1.104) | -0.00 | 0.00 | 0.11 |
| levee_pond | 5 m deep | FLDPLN | 1.000 | 1.000 | 1.104 (1.104) | -0.00 | 0.00 | 0.07 |
| levee_pond | library build | FLDPLN |  |  |  |  |  | 0.06 |
| steep | 1 m deep | kernel-weighted | 0.522 | 0.719 | 0.046 (0.024) | 0.00 | 0.00 | 0.05 |
| steep | 1 m deep | multi-point | 0.857 | 0.925 | 0.028 (0.024) | 0.00 | 0.00 | 0.08 |
| steep | 1 m deep | FLDPLN | 0.984 | 0.992 | 0.024 (0.024) | 0.00 | 0.00 | 0.06 |
| steep | 3 m deep | kernel-weighted | 0.873 | 0.930 | 0.245 (0.217) | 0.00 | 0.11 | 0.06 |
| steep | 3 m deep | multi-point | 0.978 | 0.988 | 0.220 (0.217) | -0.02 | 0.06 | 0.09 |
| steep | 3 m deep | FLDPLN | 0.994 | 0.997 | 0.217 (0.217) | 0.00 | 0.02 | 0.07 |
| steep | 5 m deep | kernel-weighted | 0.907 | 0.946 | 0.437 (0.409) | 0.02 | 0.28 | 0.06 |
| steep | 5 m deep | multi-point | 0.986 | 0.992 | 0.412 (0.409) | -0.04 | 0.10 | 0.08 |
| steep | 5 m deep | FLDPLN | 0.995 | 0.997 | 0.409 (0.409) | 0.00 | 0.02 | 0.07 |
| steep | library build | FLDPLN |  |  |  |  |  | 0.05 |

Means over every site and event:

| Mapper | CSI | MCC | Seconds |
| --- | --- | --- | --- |
| FLDPLN | 0.993 | 0.995 | 0.07 |
| kernel-weighted | 0.890 | 0.933 | 0.05 |
| multi-point | 0.952 | 0.974 | 0.09 |
