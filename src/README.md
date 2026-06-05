# Source

This directory contains reusable implementation code for the SOKO artifact.

Layout:
- `backends/`: shared backend modules used by multiple benchmarks.

Separation rationale:
- `benchmarks/` owns experiment flow and reporting.
- `src/` owns reusable implementation pieces that represent the actual SOKO system components.