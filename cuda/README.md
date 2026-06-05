# CUDA

This directory contains the CUDA implementation used by the SOKO GPU path.

Files:
- `soko_gpu_offline.cu`: CUDA kernels, device helpers, and exported host wrappers for GPU presign generation and GPU verification helpers.
- `soko_gpu_offline.h`: public header for the CUDA-side exported API.

Scope:
- The CUDA code accelerates the message-independent offline phase.
- The latency-sensitive online signing path remains on the CPU.

Artifact note:
- The CUDA implementation is intentionally kept compact for reviewer inspection.
- The backend loader that consumes the produced DLL is under `../src/backends/soko_gpu_backend.c`.