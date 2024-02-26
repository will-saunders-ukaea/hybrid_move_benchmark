Hybrid Move Benchmark
=====================

Benchmarks the time taken to move particles over a 2D domain on which domain decomposition is applied. 
This benchmark does no useful work and is a benchmark of the particle bookkeeping.

Requires NESO-Particles `NP`_.
See the `example_outputs` directory for cmake configuration and example launching.

.. _NP: https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles



Program arguments are as follows:

```
  mpirun -n <nproc> ./bin/hybrid_move_bench <mode> <number of particles> <number of cells> <number of warmup steps> <number of timed steps>
```

where "mode=1" places all particles within a single block of memory per MPI rank and disables moving particles between mesh cells. "mode=0" is the previous behaviour where particles are binned into cells and particle data is moved between cells. The number of cells is the number of fine Cartesian cells per direction (not total number of cells).


