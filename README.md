# SOKO Artifact

SOKO is a reviewer-facing artifact for the paper "High-Throughput Post-Quantum Signature Framework for Time-Critical Communications." It packages the FN-DSA online/offline implementation, the CPU and CPUMT baselines, the GPU-backed SOKO path, and the benchmark programs used to evaluate offline refill cost, online cost, verification cost, and sustained token service under demand.

At a high level, SOKO moves message-independent FN-DSA precomputation off the latency-critical path. CPU and CPUMT generate offline presign material on the CPU, while SOKO generates the same presign material with a CUDA backend and then completes the online phase on the CPU.

## What This Artifact Contains

- `CPU`: single-threaded offline presign generation with CPU online signing.
- `CPUMT`: OpenMP-based multi-threaded offline presign generation with CPU online signing.
- `SOKO`: GPU offline presign generation with CPU online signing and GPU-backed refill experiments.
- `benchmarks/soko_token_benchmark.c`: the main benchmark for offline latency, online latency, verification, and end-to-end throughput.
- `benchmarks/soko_refill_simulator.c`: the consume-only refill simulator used to stress token regeneration capacity.
- `src/backends/`: shared backend implementations so the benchmark entrypoints stay thin and reviewer-readable.

## Repository Layout

```text
SOKO/
├── benchmarks/
│   ├── soko_token_benchmark.c
│   └── soko_refill_simulator.c
├── cuda/
│   ├── soko_gpu_offline.cu
│   └── soko_gpu_offline.h
├── include/
│   └── soko_falcon_backends.h
├── scripts/
│   ├── build_soko_token_benchmark.sh
│   ├── run_soko_token_benchmark.sh
│   ├── build_soko_refill_simulator.sh
│   └── clean_soko_artifact.sh
├── src/
│   └── backends/
│       ├── soko_falcon_common.c
│       ├── soko_cpu_backend.c
│       ├── soko_cpumt_backend.c
│       └── soko_gpu_backend.c
└── build/
```

`build/` is intentionally the only generated-output directory. Object files, executables, helper DLLs, and CSV outputs are kept there so the artifact root remains source-oriented.

## External Dependency

The build scripts expect dependency sources under this in-repo location:

```text
/submodules/OO-FN-DSA/falcon-lazy
```

Primary dependency roots used by the benchmark suite:

```text
/submodules/FN-DSA
/submodules/ML-DSA
/submodules/OO-FN-DSA
/submodules/OO-ML-DSA
```

These paths are referenced directly by scripts under `scripts/` and by include paths in `benchmarks/`.

## Platform Notes

- CPU and CPUMT builds require `gcc` or a compatible C compiler.
- CPUMT is enabled when the compiler accepts `-fopenmp`.
- The packaged GPU build flow is currently Windows/MSYS2-oriented and expects both `nvcc` and `cl.exe` to be available in order to build the helper DLL.
- The paper reports measurements on Ubuntu 22.04 with an Intel Core i9-13900K and an NVIDIA GeForce RTX 4070. This artifact keeps the current engineering build flow used in this workspace while preserving the same benchmark logic and implementation split.

## Quick Start

From the `SOKO/` directory:

```sh
git submodule update --init --recursive
sh scripts/build_soko_token_benchmark.sh --check-gpu-env
sh scripts/run_soko_token_benchmark.sh
sh scripts/build_soko_refill_simulator.sh
./build/soko_refill_simulator.exe --rates 1024,2048,4096 --consume-target 327680
```

The first command initializes nested dependency submodules (required for all benchmark variants, including OO-Ed25519). The second command only checks tool availability. The third builds and runs the main benchmark in GPU-required mode and writes a CSV under `build/` by default.

## Reproducing The Main Results

### 1. Offline, Online, Verify, and Throughput Benchmark

This benchmark corresponds to the main CPU vs CPUMT vs SOKO token-bank comparison and is the source for the artifact's main latency and throughput tables.

Build only:

```sh
sh scripts/build_soko_token_benchmark.sh
```

Build and run with default CSV output:

```sh
sh scripts/run_soko_token_benchmark.sh
```

Run manually from the build directory:

```sh
./build/soko_token_benchmark.exe --csv ./build/soko_token_benchmark_results.csv --max-attempts 4096
```

Useful flags:

- `--cpu-mt-threads <n>`: fixes the CPUMT thread count instead of using auto detection.
- `--no-verify`: skips `falcon_verify()` during the online stage.
- `--verify-on-gpu` or `--verify-full-gpu`: enables GPU-backed verification modes when available.
- `--pipeline`: enables the ring-buffer plus asynchronous GPU refill pipeline.

### 2. Consume-Only Refill Sweep

This simulator removes the online signing stage and directly measures how well each backend keeps up with token demand. It is the cleanest way to isolate refill capacity.

Build:

```sh
sh scripts/build_soko_refill_simulator.sh
```

Example run:

```sh
./build/soko_refill_simulator.exe \
	--csv ./build/multisigner_refill.csv \
	--rates 1024,2048,4096,8192,16384,32768,65536 \
	--consume-target 327680
```

Useful flags:

- `--queue <tokens>`: queue capacity used for the refill model.
- `--consume-target <tokens>`: stop after this many tokens have been successfully served.
- `--rates <comma-list>`: demand sweep in tokens per second.
- `--gpu-batch <tokens>`: refill batch size for the SOKO path.
- `--cpu-mt-threads <count>`: CPUMT thread count override.

## Mapping To The Paper

- `Table I`: offline, online, verify, and throughput comparisons come from the main benchmark plus the other scheme baselines in the full workspace.
- `Fig. 3`: offline per-token latency from the token-bank benchmark.
- `Fig. 4`: online per-signature completion latency from the token-bank benchmark.
- `Fig. 5`: consume-only service percentage under rising token demand from `soko_refill_simulator.c`.
- `Fig. 6`: end-to-end SOKO throughput versus token-bank size from the token-bank benchmark.

## Code Organization Rationale

- `benchmarks/` contains experiment entrypoints only.
- `src/backends/` contains reusable implementation code shared across experiments.
- `include/soko_falcon_backends.h` is the small common interface between the entrypoints and the backend implementations.
- `cuda/` contains the GPU offline-generation implementation so CUDA-specific code is isolated from the C benchmark drivers.
- `scripts/` are written around root-relative paths so reviewers can invoke them from any current directory.

This split is deliberate: the artifact is easier to inspect when benchmark logic, reusable backend logic, and CUDA-specific implementation are separate.

## Cleaning Generated Outputs

To remove built executables, object files, DLLs, and generated CSV outputs:

```sh
sh scripts/clean_soko_artifact.sh
```

## Artifact Scope

This folder focuses on the SOKO implementation and the experiments most relevant to the paper's FN-DSA online/offline pipeline. The larger workspace contains additional schemes, older benchmark variants, plotting helpers, and paper sources that are intentionally left outside this artifact-facing directory.
