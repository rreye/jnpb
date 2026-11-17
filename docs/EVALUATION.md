# Performance evaluation

The goal of the evaluation is the efficiency of the Java code, not of the communications. Every result is
therefore measured with **one process**, so the same computation runs in Java and in Fortran/C without the
influence of a particular MPJ or MPI implementation, and with the time given by the benchmark itself.

* **Machine**: Intel Core i7-12700, 31 GB, Linux 6.12 (Debian 13). Each run is restricted with `taskset` to four
  performance cores (one hardware thread each).
* **Java**: OpenJDK 25.0.5 (classes compiled with `--release 8`), default JVM options (G1) and `-Xmx8g` (class
  B) or `-Xmx20g` (class C). MPJ versions run on FastMPJ with the `smdev` device.
* **Reference**: the official NPB 3.4.4 built with GCC/gfortran 14.2.0, `-O3 -mcmodel=medium`; the serial
  reference is the OpenMP version built without `-fopenmp`, the MPJ reference is the MPI version with Open MPI
  5.0.7 (one process).
* **Statistics**: median of 5 repetitions (the coefficient of variation is at most 5.9 %); every run verified successfully.
  The ratio is Java time over reference time, so values below 1 mean that Java is faster.

## Class B (time in seconds)

| Kernel | Java SER | OpenMP (no OpenMP) | ratio | Java MPJ | MPI | ratio |
|---|--:|--:|--:|--:|--:|--:|
| BT | 103.03 | 88.37 | 1.17 | 106.64 | 84.09 | 1.27 |
| CG | 14.96 | 16.49 | 0.91 | 14.82 | 15.42 | 0.96 |
| EP | 8.92 | 8.75 | 1.02 | 9.03 | 8.72 | 1.04 |
| FT | 13.60 | 13.56 | 1.00 | 13.08 | 16.21 | 0.81 |
| IS | 0.62 | 0.65 | 0.95 | 0.73 | 0.70 | 1.04 |
| LU | 72.50 | 52.29 | 1.39 | 79.18 | 55.28 | 1.43 |
| MG | 2.77 | 1.82 | 1.52 | 2.99 | 2.05 | 1.46 |
| SP | 52.01 | 51.92 | 1.00 | 75.30 | 63.37 | 1.19 |

## Class C (time in seconds)

| Kernel | Java SER | OpenMP (no OpenMP) | ratio | Java MPJ | MPI | ratio |
|---|--:|--:|--:|--:|--:|--:|
| BT | 427.03 | 372.39 | 1.15 | 434.68 | 344.09 | 1.26 |
| CG | 55.90 | 63.38 | 0.88 | 56.65 | 64.22 | 0.88 |
| EP | 35.68 | 34.88 | 1.02 | 36.11 | 34.82 | 1.04 |
| FT | 59.27 | 59.43 | 1.00 | 58.87 | 73.42 | 0.80 |
| IS | 2.47 | 2.58 | 0.96 | 2.91 | 2.81 | 1.04 |
| LU | 299.75 | 216.57 | 1.38 | 325.90 | 227.44 | 1.43 |
| MG | 21.77 | 15.19 | 1.43 | 23.01 | 16.43 | 1.40 |
| SP | 225.61 | 232.37 | 0.97 | 306.17 | 267.32 | 1.15 |

Serial and MPJ FT share the same FFT core.

## Class D (time in seconds)

Class D is evaluated for the kernels that fit on the test machine (31 GB of RAM, no swap) with a single process: BT, SP,
LU, CG, EP and MG (IS and FT do not fit). Because of the run time, BT, SP, LU, CG and MG were run once; EP is the median
of 3 repetitions. Java used `-Xmx26g` (`-Xmx28g` for MG, which peaks at 27.2 GB of resident memory and leaves about 2 GB free;
`-Xmx2g` for EP); all 31 runs verified successfully. MG class D with the official MPI code is not reported: the unmodified
NPB 3.4.4 binary crashes with a segmentation fault (in `zero3`) as soon as it starts with one process.

| Kernel | Java SER | OpenMP (no OpenMP) | ratio | Java MPJ | MPI | ratio |
|---|--:|--:|--:|--:|--:|--:|
| BT | 8803.16 | 7486.42 | 1.18 | 9009.06 | 7136.13 | 1.26 |
| SP | 4832.01 | 4975.03 | 0.97 | 6154.20 | 5639.26 | 1.09 |
| LU | 5893.63 | 4255.72 | 1.38 | 6335.53 | 4552.63 | 1.39 |
| CG | 2724.48 | 2887.87 | 0.94 | 2623.21 | 2645.91 | 0.99 |
| EP | 561.40 | 557.49 | 1.01 | 566.15 | 557.14 | 1.02 |
| MG | 433.37 | 359.23 | 1.21 | 457.01 | - | - |
