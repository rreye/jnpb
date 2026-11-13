# Verification and tests

Both flavours were compared value by value with the official NPB 3.4.4 for classes S, W, A and B (serial with
the OpenMP version without OpenMP, MPJ with the MPI version), including the intermediate values (norms,
checksums, per-iteration values) and not only the final verification. Reference values are stored in
`src/test/resources/golden` (serial) and `golden/mpi` (MPJ, one value per number of processes).

```
mvn test                                  # serial versions and common (JUnit 5)
FMPJ_HOME=/path/to/fastmpj mvn test       # also the MPJ tests, which launch the kernels with fmpjrun
mvn test -DexcludedGroups=classC -Dgroups=classB      # class B golden tests (excluded by default)
mvn test -DexcludedGroups= -Dgroups=classC -Dtest.xmx=24g   # class C (excluded by default)
```

The golden files are regenerated with `scripts/gen_golden.py` (serial), `scripts/gen_golden_mpi.py` (MPJ) and
`scripts/gen_random_golden.py` (random number generator). They need the official NPB 3.4.4 distribution,
unpacked as `NPB3.4.4` at the repository root (it is not included in this repository), and `gfortran`
(and an MPI installation for the MPJ ones).

Floating-point results are bit-exact with the reference wherever the arithmetic is the same: Java does not fuse
multiply-adds and the association of every Fortran expression was preserved. The only remaining differences
are in FT (at most 4.5e-16), caused by `Math.exp` versus glibc's `exp`, and by the order of the reduction.
