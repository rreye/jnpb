# Changelog

## 1.0.0

First release.

* Serial versions of CG, EP, FT, IS, MG, SP, BT and LU, following the NPB 3.4.4 OpenMP version built without
  OpenMP.
* MPJ versions of CG, EP, FT, IS, MG, SP, BT, LU and DT, following the NPB 3.4.4 MPI version.
* Classes S, W, A, B and C verified in both flavours (classes D and E implemented where the class table says
  so, see the README).
* Results checked value by value against the official NPB 3.4.4, with golden fixtures and JUnit 5 tests.
* Optimized compute code in all kernels, bit-identical with the reference results.
* Command line compatible with the official NPB conventions where they apply: bare graph name for DT,
  `NPB_TIMER_FLAG` / `timer.flag` for detailed timing, `NPB_NPROCS_STRICT` for the number of processes.
