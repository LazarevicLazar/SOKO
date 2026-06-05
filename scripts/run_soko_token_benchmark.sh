#!/bin/sh
set -e

# Build and run the SOKO token benchmark in GPU-required mode.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
CALL_DIR=$(pwd)

CSV_PATH="${1:-$BUILD_DIR/soko_token_benchmark_results.csv}"
MAX_ATTEMPTS="${2:-4096}"
CPU_MT_THREADS="${3:-}"
VERIFY_MODE="${4:-verify}"
VERIFY_FLAG=""

case "$CSV_PATH" in
	/*|?:/*)
		;;
	*)
		CSV_PATH="$CALL_DIR/$CSV_PATH"
		;;
esac

case "$VERIFY_MODE" in
	verify|on|1|true)
		VERIFY_FLAG=""
		VERIFY_LABEL="enabled"
		;;
	noverify|off|0|false)
		VERIFY_FLAG="--no-verify"
		VERIFY_LABEL="disabled"
		;;
	*)
		echo "Invalid verify_mode: $VERIFY_MODE"
		echo "Use one of: verify|on|1|true|noverify|off|0|false"
		exit 1
		;;
esac

echo "===== SOKO Token Benchmark Run ====="
echo "CSV output: $CSV_PATH"
echo "Max attempts/token: $MAX_ATTEMPTS"
echo "Verification: $VERIFY_LABEL"
if [ -n "$CPU_MT_THREADS" ]; then
	echo "CPU-MT threads: $CPU_MT_THREADS"
else
	echo "CPU-MT threads: auto"
fi
echo

mkdir -p "$BUILD_DIR"
sh "$SCRIPT_DIR/build_soko_token_benchmark.sh" --gpu-required

if command -v nvcc >/dev/null 2>&1; then
	CUDA_BIN_DIR=$(dirname "$(command -v nvcc)")
	export PATH="$CUDA_BIN_DIR:$PATH"
	echo "CUDA runtime PATH includes: $CUDA_BIN_DIR"
fi

cd "$BUILD_DIR"

if [ -n "$CPU_MT_THREADS" ]; then
	if [ -n "$VERIFY_FLAG" ]; then
		./soko_token_benchmark.exe --csv "$CSV_PATH" --max-attempts "$MAX_ATTEMPTS" --cpu-mt-threads "$CPU_MT_THREADS" "$VERIFY_FLAG"
	else
		./soko_token_benchmark.exe --csv "$CSV_PATH" --max-attempts "$MAX_ATTEMPTS" --cpu-mt-threads "$CPU_MT_THREADS"
	fi
else
	if [ -n "$VERIFY_FLAG" ]; then
		./soko_token_benchmark.exe --csv "$CSV_PATH" --max-attempts "$MAX_ATTEMPTS" "$VERIFY_FLAG"
	else
		./soko_token_benchmark.exe --csv "$CSV_PATH" --max-attempts "$MAX_ATTEMPTS"
	fi
fi
