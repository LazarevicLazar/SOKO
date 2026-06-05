#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"

rm -rf "$BUILD_DIR"
rm -f "$ROOT_DIR/oofalcon_gpu_tokens_results.csv" "$ROOT_DIR/oofalcon_gpu_coldsteady.csv"
rm -f "$ROOT_DIR/build_oofalcongpu.log" "$ROOT_DIR/build_oofalcon_gpu_tokens.log"
mkdir -p "$BUILD_DIR"

echo "Cleaned SOKO build artifacts under $BUILD_DIR."