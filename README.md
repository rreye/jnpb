# JNPB

[![CI](https://github.com/rreye/jnpb/actions/workflows/ci.yml/badge.svg)](https://github.com/rreye/jnpb/actions/workflows/ci.yml)

NAS Parallel Benchmarks for Java: serial and MPJ (message-passing) versions.

JNPB is an **unofficial** Java port of the [NAS Parallel Benchmarks](https://www.nas.nasa.gov/software/npb.html)
(NPB), developed at the Universidade da Coruña. It is not affiliated with or endorsed by NASA. Every kernel
reproduces the verification of the official NPB 3.4.4 (Fortran/C) and is checked value by value against it.

* **`NPB.SER`**: serial versions of CG, EP, FT, IS, MG, SP, BT and LU. They follow the NPB 3.4.4 OpenMP
  version built without OpenMP, the serial reference of the NPB 3.4 series. They need nothing but a JDK.
* **`NPB.MPJ`**: message-passing versions of CG, EP, FT, IS, MG, SP, BT, LU and DT, for any
  MPJ library that implements the mpiJava 1.2 API (see Requirements). They follow the NPB 3.4.4 MPI
  version. BT is the base version, without the BTIO variants. DT exists only in MPI, so it has no serial
  counterpart.
* **`NPB.common`**: the random number generator (bit-exact with NPB's), timers, argument parsing and result
  printing shared by all benchmarks.

DC and UA (only available in the OpenMP version) are not included.

## Requirements

* Java 8 or later (the code is compiled with `--release 8`) and Maven 3.
* For `NPB.MPJ` only: a Java message-passing library that implements the **mpiJava 1.2 API** (package `mpi`),
  such as [FastMPJ](https://gac.udc.es/~rober/fastmpj), [MPJ Express](https://mpjexpress.sourceforge.net/)
  or the original mpiJava. Other Java MPI bindings with a different API (Open MPI's Java bindings,
  MPJ/Ibis) are not compatible. The MPJ versions have only been tested with FastMPJ, so the build and run
  examples below use it. FastMPJ is not published in Maven Central, so the `mpj` profile reads its `mpj.jar`
  from `$FMPJ_HOME/lib/mpj.jar`; with another library, point `fmpj.home` to a directory whose `lib/mpj.jar` is
  the library jar, or compile the sources of `NPB.MPJ` with that jar in the class path.

## Build

```
mvn compile                                   # serial versions + common, no MPJ library needed
export FMPJ_HOME=/path/to/fastmpj
mvn compile                                   # also the MPJ versions (profile "mpj" is activated by FMPJ_HOME)
mvn compile -Pmpj -Dfmpj.home=/path/to/fastmpj   # same, without the environment variable
mvn package -DskipTests                       # builds target/jnpb-<version>.jar
mvn clean
```

Classes are written to `bin/` instead of `target/classes`. The jar built with the `mpj` profile contains both
flavours; running the MPJ classes needs the MPJ library at run time (the launch command depends on the
library: `fmpjrun` for FastMPJ, `mpjrun.sh` for MPJ Express).

## Run

Every benchmark has its own `main`. Arguments are `KEY=value` pairs, case-insensitive in the key:

| Argument | Meaning |
|---|---|
| `CLASS={S\|W\|A\|B\|C\|D\|E}` | problem class (a kernel rejects the classes it does not implement with `BadClassException`) |
| `DTIMING={true\|false}` | detailed timing (default `false`) |
| `GRAPH={BH\|WH\|SH}` | DT only: the graph. As in the official DT, the bare name (`DT BH CLASS=S`) is also accepted |

```
# serial
java -cp bin NPB.SER.CG CLASS=S

# MPJ, with FastMPJ, the library used for testing (hostfile "machines" lists the hosts)
fmpjrun -np 4 -hostfile machines -class NPB.MPJ.CG CLASS=A
fmpjrun -np 43 -dev niodev -hostfile machines -class NPB.MPJ.DT BH CLASS=B
```

A run is correct if the output ends with `Verification = SUCCESSFUL`. Large classes need a suitable `-Xmx`.

**Detailed timing.** `DTIMING=true` prints, in the serial versions, per-section timers after the verification
(as the official `timer.flag` does) and, in the MPJ versions, splits the time into computation and
communication (`Time in seconds = 0.11 (COMP/COMM: 0.05/0.06)`, maximum over the processes). It slightly
penalizes performance. Like the official NPB, it can also be switched on with the environment variable
`NPB_TIMER_FLAG` (empty, starting with `1`-`9`, or `on`/`yes`/`true`) or with a file `timer.flag` in the working
directory; `DTIMING=` on the command line overrides both.

**Input files.** As in the official NPB, the optional files `inputbt.data`, `inputsp.data` and `inputlu.data`
(BT, SP, LU), `mg.input` (MG) and `inputft.data` (FT, MPJ only) are read from the working directory;
without them the compiled defaults are used.

**Number of processes (MPJ).**
* CG and IS need a power of two. IS follows NPB-MPI 3.4: classes D and E need at least 4 and 64 processes,
  class S at most 128 and the rest at most 1024.
* SP and BT need a square number of processes. LU needs `n1*n2` processes with `n2 <= n1 <= 2*n2`
  (1, 2, 4, 6, 8, 9, 12, 16, ...) and at least 3 points per process and dimension.
* An invalid number of processes is rejected, unless the environment variable `NPB_NPROCS_STRICT` is set to
  `0`, `off`, `no` or `false`: the excess processes are then masked out, as in NPB-MPI 3.4.
* DT needs at least as many processes as nodes in the graph (BH/WH: 5, 11, 21, 43, 85, 171 for S, W, A, B, C,
  D; SH: 12, 32, 80, 192, 448, 1024); extra processes stay idle.

## Classes

| | Serial | MPJ |
|---|---|---|
| CG, MG, SP, BT, LU | S, W, A, B, C verified; D implemented, see below | S-E |
| EP | S-E | S-E |
| FT | S, W, A, B, C verified; D implemented, see below | S-E |
| IS | S-C verified; D verified (about 16.5 GB, `-Xmx20g`) | S-E |
| DT | - | S, W, A, B verified; C and D run but are not verified, as in the official code |

Notes: the serial FT and IS split their large arrays into chunks so that class D fits in Java arrays (chunk
sizes can be changed with `-Dnpb.ft.chunkelems=N` and `-Dnpb.is.chunklog=N`). Serial FT class D needs about
86 GB; only its geometry is tested. Classes D and E of the MPJ versions need many processes and a lot of memory;
only EP class E can be run on a single machine. DT class B is verified here although the official code leaves
it unverified.

## Verification and tests

Both flavours were compared value by value with the official NPB 3.4.4 for classes S, W, A and B, including
intermediate values and not only the final verification. The tests are run with `mvn test` (serial versions
and `NPB.common`) or `FMPJ_HOME=/path/to/fastmpj mvn test` (also the MPJ versions, which need FastMPJ).
See [docs/TESTING.md](docs/TESTING.md) for the details, the test groups and how the golden files are
regenerated.

## Performance evaluation

The goal of the evaluation is the efficiency of the Java code, not of the communications, so every result is
measured with one process and compared with the official NPB 3.4.4 (OpenMP without OpenMP for the serial
versions, MPI for the MPJ versions) built with GCC/gfortran 14.2.0 `-O3`. Values below are medians of 5
repetitions, in seconds; the ratio is Java time over reference time. Machine, JVM flags, methodology, class C
and class D results are in [docs/EVALUATION.md](docs/EVALUATION.md).

### Class B (time in seconds)

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

## Differences from the official NPB

* The class is a run-time argument (`CLASS=`) instead of a compile-time one, and the optional arguments
  `DTIMING=` and `GRAPH=` are specific to JNPB (see above).
* Performance-oriented changes with respect to a literal translation (loop structure, fused FFT stages,
  hot loops in small methods, etc.) are allowed as long as the results stay bit-identical with the
  reference; the verification is not modified.
* Verification code is NaN-safe: a NaN result fails the verification.
* IS prints an additional line, ` Sorted keys digest = ...`, because the official verification of IS does not
  notice a change in the keys that preserves their order.
* DT class B is verified, and an unknown class raises `BadClassException`.

## Origin and license

The serial versions are translations of the NPB 3.4.4 OpenMP version. The MPJ versions are based on the
NPB-MPJ implementation by Damián Álvarez Mallón (CG, EP, FT, IS, MG, SP, DT, and the common classes),
and on the NPB 3.4.4 MPI version (BT, LU). Every source file states which one it is based on. The algorithms,
problem classes and verification values belong to the NAS Parallel Benchmarks; see the
[NPB site](https://www.nas.nasa.gov/software/npb.html) for the original authors and credits.

JNPB is released under the MIT License, see `LICENSE`.
