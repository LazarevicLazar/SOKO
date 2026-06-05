#!/bin/sh
set -e

# Dedicated build script for the offline/online benchmark artifact.

CC=gcc
USER_CFLAGS="${CFLAGS:-}"
CFLAGS="-Wall -Wextra -O3 -fomit-frame-pointer ${USER_CFLAGS}"
LDFLAGS=""
LIBS="-lm -ladvapi32"
OMP_CFLAGS=""

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
BENCH_DIR="$ROOT_DIR/benchmarks"
INCLUDE_DIR="$ROOT_DIR/include"
BACKEND_DIR="$ROOT_DIR/src/backends"
CUDA_DIR="$ROOT_DIR/cuda"
FALCON_DIR="$ROOT_DIR/submodules/OO-FN-DSA/falcon-lazy"
TARGET="bench_oofalcon_gpu_tokens"

GPU_DEFS=""
GPU_REQUIRED=0
NVCC=""
CL_DIR=""
CL_DIR_WIN=""
NVCC_HOST_FLAGS_MSVC="/W3 /O2"
GPU_ENABLED=0
GPU_DLL="oofalcon_gpu_offline.dll"
NVCC_CUDA_ARCH_FLAGS="-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89"

mkdir -p "$BUILD_DIR"

detect_openmp() {
    test_bin="$BUILD_DIR/oofalcon_openmp_test_$$.exe"
    if printf 'int main(void){return 0;}\n' | "$CC" -x c - -fopenmp -o "$test_bin" >/dev/null 2>&1; then
        rm -f "$test_bin"
        return 0
    fi
    rm -f "$test_bin"
    return 1
}

detect_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi

    if [ -n "${CUDA_PATH:-}" ] && [ -x "$CUDA_PATH/bin/nvcc.exe" ]; then
        echo "$CUDA_PATH/bin/nvcc.exe"
        return 0
    fi

    for d in /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/*; do
        if [ -x "$d/bin/nvcc.exe" ]; then
            NVCC="$d/bin/nvcc.exe"
        fi
    done

    if [ -n "$NVCC" ]; then
        echo "$NVCC"
        return 0
    fi

    return 1
}

detect_cl_dir() {
    if command -v cl >/dev/null 2>&1; then
        local_dir=$(dirname "$(command -v cl)")
        if [ -x "$local_dir/cl.exe" ] || [ -x "$local_dir/cl" ]; then
            echo "$local_dir"
            return 0
        fi
    fi

    if [ -n "${VCToolsInstallDir:-}" ] && [ -x "$VCToolsInstallDir/bin/Hostx64/x64/cl.exe" ]; then
        echo "$VCToolsInstallDir/bin/Hostx64/x64"
        return 0
    fi

    for d in /c/Program\ Files/Microsoft\ Visual\ Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64; do
        if [ -x "$d/cl.exe" ]; then
            CL_DIR="$d"
        fi
    done

    if [ -n "$CL_DIR" ]; then
        echo "$CL_DIR"
        return 0
    fi

    return 1
}

to_windows_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$1"
    else
        echo "$1"
    fi
}

for arg in "$@"; do
    case "$arg" in
        --gpu-required)
            GPU_REQUIRED=1
            ;;
        --check-gpu-env)
            echo "===== OO-Falcon GPU Environment Check ====="
            if NVCC_PATH=$(detect_nvcc); then
                echo "nvcc: FOUND ($NVCC_PATH)"
                "$NVCC_PATH" --version | sed -n '1,3p'
            else
                echo "nvcc: MISSING"
                echo "Hint: install CUDA Toolkit or add nvcc to PATH/CUDA_PATH"
            fi

            if CL_PATH=$(detect_cl_dir); then
                echo "cl.exe: FOUND ($CL_PATH/cl.exe)"
            else
                echo "cl.exe: MISSING"
                echo "Hint: install Visual Studio C++ Build Tools (MSVC)"
            fi

            if command -v nvidia-smi >/dev/null 2>&1; then
                echo
                echo "nvidia-smi: FOUND"
                nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi
            else
                echo "nvidia-smi: MISSING"
            fi
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--gpu-required] [--check-gpu-env]"
            echo "  --gpu-required  Fail build if nvcc is not available"
            echo "  --check-gpu-env Print GPU toolchain/runtime availability and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

echo "===== Building OO-Falcon Hybrid GPU/CPU Token Benchmark ====="
echo

if detect_openmp; then
    OMP_CFLAGS="-fopenmp"
    CFLAGS="$CFLAGS $OMP_CFLAGS"
    echo "OpenMP support: ENABLED ($OMP_CFLAGS)"
else
    echo "OpenMP support: DISABLED (compiler does not accept -fopenmp)"
fi

echo

if NVCC=$(detect_nvcc); then
    echo "CUDA toolchain detected ($NVCC). Preparing GPU offline DLL..."

    if CL_DIR=$(detect_cl_dir); then
        echo "MSVC host compiler detected ($CL_DIR/cl.exe)."
        CL_DIR_WIN=$(to_windows_path "$CL_DIR")
        GPU_DEFS="-DOOFALCON_USE_CUDA"
        GPU_ENABLED=1

        CU_FILE_WIN=$(to_windows_path "$CUDA_DIR/oofalcon_gpu_offline.cu")
        DLL_FILE_WIN=$(to_windows_path "$BUILD_DIR/$GPU_DLL")
        MSYS2_ARG_CONV_EXCL='*' "$NVCC" -O3 $NVCC_CUDA_ARCH_FLAGS -Xcompiler "$NVCC_HOST_FLAGS_MSVC" \
            -ccbin "$CL_DIR_WIN" -DOOFALCON_GPU_BUILD_DLL --shared \
            "$CU_FILE_WIN" -o "$DLL_FILE_WIN"

        echo "Built GPU helper DLL: $BUILD_DIR/$GPU_DLL"
    else
        if [ "$GPU_REQUIRED" -eq 1 ]; then
            echo "MSVC host compiler (cl.exe) not found and --gpu-required was set. Aborting." >&2
            echo "Tip: install Visual Studio C++ Build Tools or run from Developer Command Prompt." >&2
            exit 1
        fi
        echo "cl.exe not detected. Building CPU-only benchmark binary."
    fi
else
    if [ "$GPU_REQUIRED" -eq 1 ]; then
        echo "nvcc not found and --gpu-required was set. Aborting." >&2
        echo "Tip: run '$0 --check-gpu-env' to inspect CUDA visibility." >&2
        exit 1
    fi
    echo "nvcc not found. Building CPU-only benchmark binary (GPU path disabled at compile time)."
fi

echo

if [ "$GPU_ENABLED" -eq 0 ] && [ "$GPU_REQUIRED" -eq 1 ]; then
    echo "GPU-required mode requested, but GPU backend DLL could not be built." >&2
    exit 1
fi

echo "Compiling benchmark source..."
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BENCH_DIR/bench_oofalcon_gpu_tokens.c" -o "$BUILD_DIR/bench_oofalcon_gpu_tokens.o"

echo "Compiling shared SOKO backends..."
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BACKEND_DIR/soko_falcon_common.c" -o "$BUILD_DIR/soko_falcon_common.o"
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BACKEND_DIR/soko_cpu_backend.c" -o "$BUILD_DIR/soko_cpu_backend.o"
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BACKEND_DIR/soko_cpumt_backend.c" -o "$BUILD_DIR/soko_cpumt_backend.o"
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BACKEND_DIR/soko_gpu_backend.c" -o "$BUILD_DIR/soko_gpu_backend.o"
"$CC" $CFLAGS $GPU_DEFS -I"$INCLUDE_DIR" -I"$FALCON_DIR" -I"$ROOT_DIR" \
    -c "$BACKEND_DIR/soko_falcon_pipeline.c" -o "$BUILD_DIR/soko_falcon_pipeline.o"

echo "Compiling OO-Falcon sources..."
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/codec.c" -o "$BUILD_DIR/oofal_gpu_codec.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/common.c" -o "$BUILD_DIR/oofal_gpu_common.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/falcon.c" -o "$BUILD_DIR/oofal_gpu_falcon.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/fft.c" -o "$BUILD_DIR/oofal_gpu_fft.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/fpr.c" -o "$BUILD_DIR/oofal_gpu_fpr.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/keygen.c" -o "$BUILD_DIR/oofal_gpu_keygen.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/rng.c" -o "$BUILD_DIR/oofal_gpu_rng.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/shake.c" -o "$BUILD_DIR/oofal_gpu_shake.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/sign.c" -o "$BUILD_DIR/oofal_gpu_sign.o"
"$CC" $CFLAGS -I"$FALCON_DIR" -c "$FALCON_DIR/vrfy.c" -o "$BUILD_DIR/oofal_gpu_vrfy.o"

echo
echo "Linking $TARGET..."
"$CC" $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
    "$BUILD_DIR/bench_oofalcon_gpu_tokens.o" "$BUILD_DIR/soko_falcon_common.o" \
    "$BUILD_DIR/soko_cpu_backend.o" "$BUILD_DIR/soko_cpumt_backend.o" \
    "$BUILD_DIR/soko_gpu_backend.o" "$BUILD_DIR/soko_falcon_pipeline.o" \
    "$BUILD_DIR/oofal_gpu_codec.o" "$BUILD_DIR/oofal_gpu_common.o" \
    "$BUILD_DIR/oofal_gpu_falcon.o" "$BUILD_DIR/oofal_gpu_fft.o" \
    "$BUILD_DIR/oofal_gpu_fpr.o" "$BUILD_DIR/oofal_gpu_keygen.o" \
    "$BUILD_DIR/oofal_gpu_rng.o" "$BUILD_DIR/oofal_gpu_shake.o" \
    "$BUILD_DIR/oofal_gpu_sign.o" "$BUILD_DIR/oofal_gpu_vrfy.o" \
    $LIBS

echo
echo "===== Build Complete ====="
if [ "$GPU_ENABLED" -eq 1 ]; then
    echo "GPU backend DLL: $BUILD_DIR/$GPU_DLL"
fi
echo "Run with: $BUILD_DIR/$TARGET.exe"
