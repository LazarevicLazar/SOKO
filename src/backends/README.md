# Backends

This directory contains the reusable SOKO backend modules.

Files:
- `soko_falcon_common.c`: shared Falcon key-material preparation and common helpers.
- `soko_cpu_backend.c`: single-threaded CPU offline presign generation.
- `soko_cpumt_backend.c`: OpenMP-based multi-threaded CPU offline presign generation.
- `soko_gpu_backend.c`: dynamic loader and wrapper layer for the GPU helper DLL.
- `soko_falcon_pipeline.c`: backend-owned ring buffer and async refill pipeline used by the SOKO steady-state path.
- `oo_ed25519.c`: OO-Ed25519 helper implementation used by the certificate-chain benchmark.

Design note:
- The goal of this directory is to separate implementation from measurement harnesses.
- The refill queue and refill worker now live here as features of SOKO rather than as benchmark-local code.