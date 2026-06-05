#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"

rm -rf "$BUILD_DIR"
rm -f "$ROOT_DIR/soko_token_benchmark_results.csv" "$ROOT_DIR/soko_token_benchmark_coldsteady.csv"
rm -f "$ROOT_DIR/build_soko_token_benchmark.log" "$ROOT_DIR/build_soko_refill_simulator.log"
mkdir -p "$BUILD_DIR"

echo "Cleaned SOKO build artifacts under $BUILD_DIR."