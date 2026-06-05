# Include

This directory contains the public C interface used across the SOKO artifact.

Files:
- `soko_falcon_backends.h`: shared declarations for key preparation, CPU and CPUMT presign generation, GPU loading/generation helpers, and the backend-owned refill pipeline.
- `oo_ed25519.h`: public interface for the OO-Ed25519 certificate benchmark helper implementation.

Purpose:
- Keep benchmark entrypoints decoupled from backend implementation details.
- Provide one common contract for the CPU, CPUMT, GPU, and refill-pipeline code paths.