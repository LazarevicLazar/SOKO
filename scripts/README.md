# Scripts

This directory contains the artifact build, run, and cleanup helpers.

Files:
- `build_soko_token_benchmark.sh`: builds the main SOKO token benchmark.
- `run_soko_token_benchmark.sh`: builds and runs the main token benchmark with GPU-required mode.
- `build_soko_refill_simulator.sh`: builds the consume-only refill simulator.
- `clean_soko_artifact.sh`: removes generated build outputs and common artifact CSV/log files.

Design note:
- The scripts use root-relative paths so they can be invoked from any current working directory.
- Generated binaries, objects, DLLs, and CSV outputs are written under `../build/`.