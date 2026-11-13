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
| BT | 103.03 | 88.37 | 1.17 | 107.04 | 84.09 | 1.27 |
| CG | 14.96 | 16.49 | 0.91 | 14.82 | 15.42 | 0.96 |
| EP | 8.92 | 8.75 | 1.02 | 9.03 | 8.72 | 1.04 |
| FT | 13.60 | 13.56 | 1.00 | 13.08 | 16.21 | 0.81 |
| IS | 0.62 | 0.65 | 0.95 | 0.73 | 0.70 | 1.04 |
| LU | 72.50 | 52.29 | 1.39 | 79.70 | 55.28 | 1.44 |
| MG | 2.77 | 1.82 | 1.52 | 2.99 | 2.05 | 1.46 |
| SP | 52.01 | 51.92 | 1.00 | 78.73 | 63.37 | 1.24 |

## Class C (time in seconds)

| Kernel | Java SER | OpenMP (no OpenMP) | ratio | Java MPJ | MPI | ratio |
|---|--:|--:|--:|--:|--:|--:|
| BT | 427.03 | 372.39 | 1.15 | 483.63 | 344.09 | 1.41 |
| CG | 55.90 | 63.38 | 0.88 | 56.65 | 64.22 | 0.88 |
| EP | 35.68 | 34.88 | 1.02 | 36.11 | 34.82 | 1.04 |
| FT | 59.27 | 59.43 | 1.00 | 58.87 | 73.42 | 0.80 |
| IS | 2.47 | 2.58 | 0.96 | 2.91 | 2.81 | 1.04 |
| LU | 299.75 | 216.57 | 1.38 | 386.65 | 227.44 | 1.70 |
| MG | 21.77 | 15.19 | 1.43 | 23.01 | 16.43 | 1.40 |
| SP | 225.61 | 232.37 | 0.97 | 382.74 | 267.32 | 1.43 |

Serial and MPJ FT share the same FFT core.

## Class D

Class D is evaluated for the kernels that fit on the test machine (BT, SP, LU, CG and EP); results pending.

| Kernel | Java SER | OpenMP (no OpenMP) | ratio | Java MPJ | MPI | ratio |
|---|--:|--:|--:|--:|--:|--:|
| BT | TBD | TBD | TBD | TBD | TBD | TBD |
| SP | TBD | TBD | TBD | TBD | TBD | TBD |
| LU | TBD | TBD | TBD | TBD | TBD | TBD |
| CG | TBD | TBD | TBD | TBD | TBD | TBD |
| EP | TBD | TBD | TBD | TBD | TBD | TBD |
