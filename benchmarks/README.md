# Benchmarks

This directory contains the executable experiment entrypoints for the SOKO artifact.

Files:
- `soko_token_benchmark.c`: main offline/online/verify benchmark comparing CPU, CPUMT, and SOKO.
- `soko_refill_simulator.c`: consume-only refill simulator used to measure token service capacity under demand.

Design note:
- Benchmark-specific reporting, CLI parsing, and experiment control stay here.
- Reusable implementation logic is intentionally kept out of this directory and lives under `../src/backends/`.

Paper mapping:
- `soko_token_benchmark.c` supports the main token-bank latency and throughput measurements.
- `soko_refill_simulator.c` supports the consume-only refill-pressure analysis.