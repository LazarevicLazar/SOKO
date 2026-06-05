# Benchmarks

This directory contains the executable experiment entrypoints for the SOKO artifact.

Files:

- `soko_token_benchmark.c`: main offline/online/verify benchmark comparing CPU, CPUMT, and SOKO.
- `soko_refill_simulator.c`: consume-only refill simulator used to measure token service capacity under demand.
- `bench_cert_dilithium.c`: certificate-chain benchmark for ML-DSA.
- `bench_cert_oodilithium.c`: certificate-chain benchmark for OO-ML-DSA.
- `bench_cert_falcon.c`: certificate-chain benchmark for FN-DSA.
- `bench_cert_oofalcon.c`: certificate-chain benchmark for OO-FN-DSA.
- `bench_cert_ed25519.c`: certificate-chain benchmark for Ed25519.
- `bench_cert_ooed25519.c`: certificate-chain benchmark for OO-Ed25519.

Design note:

- Benchmark-specific reporting, CLI parsing, and experiment control stay here.
- Reusable implementation logic is intentionally kept out of this directory and lives under `../src/backends/`.

Paper mapping:

- `soko_token_benchmark.c` supports the main token-bank latency and throughput measurements.
- `soko_refill_simulator.c` supports the consume-only refill-pressure analysis.
