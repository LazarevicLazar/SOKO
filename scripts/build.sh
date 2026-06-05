#!/bin/sh
# =============================================================================
# Master Build Script for SOKO Benchmarks
# Usage:  ./scripts/build.sh [target]    Build a specific benchmark
#         ./scripts/build.sh all         Build all supported benchmarks
#         ./scripts/build.sh             Show interactive menu
# =============================================================================

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
BENCH_DIR="$ROOT_DIR/benchmarks"

CC=gcc
USER_CFLAGS="${CFLAGS:-}"
USER_LDFLAGS="${LDFLAGS:-}"
CFLAGS="-Wall -Wextra -O3 -fomit-frame-pointer ${USER_CFLAGS}"
LDFLAGS="${USER_LDFLAGS}"

DILITHIUM_STD_DIR="$ROOT_DIR/submodules/ML-DSA/ref"
DILITHIUM_OO_DIR="$ROOT_DIR/submodules/OO-ML-DSA/dilithium/ref"
FALCON_STD_DIR="$ROOT_DIR/submodules/FN-DSA"
OO_FALCON_DIR="$ROOT_DIR/submodules/OO-FN-DSA/falcon-lazy"

ensure_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1" >&2
        return 1
    fi
}

ensure_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing required directory: $1" >&2
        return 1
    fi
}

build_dilithium() {
    TARGET="bench_cert_dilithium"
    SRC="$BENCH_DIR/$TARGET.c"

    echo "===== Building: Dilithium Certificate Chain Benchmark ====="
    ensure_file "$SRC"
    ensure_dir "$DILITHIUM_STD_DIR"

    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$ROOT_DIR" -I"$DILITHIUM_STD_DIR" \
        -c "$SRC" -o "$BUILD_DIR/$TARGET.o"

    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/sign.c" -o "$BUILD_DIR/std_dil_sign.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/packing.c" -o "$BUILD_DIR/std_dil_packing.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/polyvec.c" -o "$BUILD_DIR/std_dil_polyvec.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/poly.c" -o "$BUILD_DIR/std_dil_poly.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/ntt.c" -o "$BUILD_DIR/std_dil_ntt.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/reduce.c" -o "$BUILD_DIR/std_dil_reduce.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/rounding.c" -o "$BUILD_DIR/std_dil_rounding.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/randombytes.c" -o "$BUILD_DIR/std_dil_randombytes.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/fips202.c" -o "$BUILD_DIR/std_dil_fips202.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_STD_DIR" \
        -c "$DILITHIUM_STD_DIR/symmetric-shake.c" -o "$BUILD_DIR/std_dil_symmetric.o"

    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
        "$BUILD_DIR/$TARGET.o" \
        "$BUILD_DIR/std_dil_sign.o" "$BUILD_DIR/std_dil_packing.o" \
        "$BUILD_DIR/std_dil_polyvec.o" "$BUILD_DIR/std_dil_poly.o" \
        "$BUILD_DIR/std_dil_ntt.o" "$BUILD_DIR/std_dil_reduce.o" \
        "$BUILD_DIR/std_dil_rounding.o" "$BUILD_DIR/std_dil_randombytes.o" \
        "$BUILD_DIR/std_dil_fips202.o" \
        "$BUILD_DIR/std_dil_symmetric.o" -lm

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_ed25519() {
    TARGET="bench_cert_ed25519"
    SRC="$BENCH_DIR/$TARGET.c"

    echo "===== Building: Ed25519 Certificate Chain Benchmark ====="
    ensure_file "$SRC"

    $CC $CFLAGS -I"$ROOT_DIR" -c "$SRC" -o "$BUILD_DIR/$TARGET.o"
    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" "$BUILD_DIR/$TARGET.o" -lsodium

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_ooed25519() {
    TARGET="bench_cert_ooed25519"
    SRC="$BENCH_DIR/$TARGET.c"
    ED25519_DIR="$ROOT_DIR/submodules/OO-FN-DSA/ed25519/src"

    echo "===== Building: Online-Offline Ed25519 Certificate Chain Benchmark ====="
    ensure_file "$SRC"
    ensure_dir "$ED25519_DIR"

    if [ ! -f "$ROOT_DIR/oo_ed25519.c" ] || [ ! -f "$ROOT_DIR/oo_ed25519.h" ]; then
        echo "Missing oo_ed25519.c/oo_ed25519.h in repository root." >&2
        echo "This target cannot be built until OO-Ed25519 implementation files are added." >&2
        return 1
    fi

    $CC $CFLAGS -I"$ROOT_DIR" -I"$ED25519_DIR" -c "$SRC" -o "$BUILD_DIR/$TARGET.o"
    $CC $CFLAGS -I"$ROOT_DIR" -I"$ED25519_DIR" -c "$ROOT_DIR/oo_ed25519.c" -o "$BUILD_DIR/oo_ed25519.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/fe.c" -o "$BUILD_DIR/ooed_fe.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/ge.c" -o "$BUILD_DIR/ooed_ge.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/sc.c" -o "$BUILD_DIR/ooed_sc.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/sha512.c" -o "$BUILD_DIR/ooed_sha512.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/keypair.c" -o "$BUILD_DIR/ooed_keypair.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/sign.c" -o "$BUILD_DIR/ooed_sign.o"
    $CC $CFLAGS -I"$ED25519_DIR" -c "$ED25519_DIR/verify.c" -o "$BUILD_DIR/ooed_verify.o"
    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
        "$BUILD_DIR/$TARGET.o" "$BUILD_DIR/oo_ed25519.o" \
        "$BUILD_DIR/ooed_fe.o" "$BUILD_DIR/ooed_ge.o" "$BUILD_DIR/ooed_sc.o" \
        "$BUILD_DIR/ooed_sha512.o" "$BUILD_DIR/ooed_keypair.o" \
        "$BUILD_DIR/ooed_sign.o" "$BUILD_DIR/ooed_verify.o"

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_falcon() {
    TARGET="bench_cert_falcon"
    SRC="$BENCH_DIR/$TARGET.c"

    echo "===== Building: Falcon Certificate Chain Benchmark ====="
    ensure_file "$SRC"
    ensure_dir "$FALCON_STD_DIR"

    $CC $CFLAGS -I"$ROOT_DIR" -I"$FALCON_STD_DIR" -c "$SRC" -o "$BUILD_DIR/$TARGET.o"

    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/codec.c" -o "$BUILD_DIR/fal_codec.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/common.c" -o "$BUILD_DIR/fal_common.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/falcon.c" -o "$BUILD_DIR/fal_falcon.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/fft.c" -o "$BUILD_DIR/fal_fft.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/fpr.c" -o "$BUILD_DIR/fal_fpr.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/keygen.c" -o "$BUILD_DIR/fal_keygen.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/rng.c" -o "$BUILD_DIR/fal_rng.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/shake.c" -o "$BUILD_DIR/fal_shake.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/sign.c" -o "$BUILD_DIR/fal_sign.o"
    $CC $CFLAGS -I"$FALCON_STD_DIR" -c "$FALCON_STD_DIR/vrfy.c" -o "$BUILD_DIR/fal_vrfy.o"

    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
        "$BUILD_DIR/$TARGET.o" \
        "$BUILD_DIR/fal_codec.o" "$BUILD_DIR/fal_common.o" "$BUILD_DIR/fal_falcon.o" \
        "$BUILD_DIR/fal_fft.o" "$BUILD_DIR/fal_fpr.o" "$BUILD_DIR/fal_keygen.o" \
        "$BUILD_DIR/fal_rng.o" "$BUILD_DIR/fal_shake.o" "$BUILD_DIR/fal_sign.o" \
        "$BUILD_DIR/fal_vrfy.o" -lm

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_oodilithium() {
    TARGET="bench_cert_oodilithium"
    SRC="$BENCH_DIR/$TARGET.c"

    echo "===== Building: OO-Dilithium Certificate Chain Benchmark ====="
    ensure_file "$SRC"
    ensure_dir "$DILITHIUM_OO_DIR"

    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$ROOT_DIR" -I"$DILITHIUM_OO_DIR" \
        -c "$SRC" -o "$BUILD_DIR/$TARGET.o"

    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/sign.c" -o "$BUILD_DIR/ood_sign.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/packing.c" -o "$BUILD_DIR/ood_packing.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/polyvec.c" -o "$BUILD_DIR/ood_polyvec.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/poly.c" -o "$BUILD_DIR/ood_poly.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/ntt.c" -o "$BUILD_DIR/ood_ntt.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/reduce.c" -o "$BUILD_DIR/ood_reduce.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/rounding.c" -o "$BUILD_DIR/ood_rounding.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/randombytes.c" -o "$BUILD_DIR/ood_randombytes.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/fips202.c" -o "$BUILD_DIR/ood_fips202.o"
    $CC $CFLAGS -DDILITHIUM_MODE=2 -I"$DILITHIUM_OO_DIR" \
        -c "$DILITHIUM_OO_DIR/symmetric-shake.c" -o "$BUILD_DIR/ood_symmetric.o"

    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
        "$BUILD_DIR/$TARGET.o" \
        "$BUILD_DIR/ood_sign.o" "$BUILD_DIR/ood_packing.o" "$BUILD_DIR/ood_polyvec.o" \
        "$BUILD_DIR/ood_poly.o" "$BUILD_DIR/ood_ntt.o" "$BUILD_DIR/ood_reduce.o" \
        "$BUILD_DIR/ood_rounding.o" "$BUILD_DIR/ood_randombytes.o" \
        "$BUILD_DIR/ood_fips202.o" "$BUILD_DIR/ood_symmetric.o" -lm

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_oofalcon() {
    TARGET="bench_cert_oofalcon"
    SRC="$BENCH_DIR/$TARGET.c"

    echo "===== Building: OO-Falcon Certificate Chain Benchmark ====="
    ensure_file "$SRC"
    ensure_dir "$OO_FALCON_DIR"

    $CC $CFLAGS -I"$ROOT_DIR" -I"$OO_FALCON_DIR" -c "$SRC" -o "$BUILD_DIR/$TARGET.o"

    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/codec.c" -o "$BUILD_DIR/oofal_codec.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/common.c" -o "$BUILD_DIR/oofal_common.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/falcon.c" -o "$BUILD_DIR/oofal_falcon.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/fft.c" -o "$BUILD_DIR/oofal_fft.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/fpr.c" -o "$BUILD_DIR/oofal_fpr.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/keygen.c" -o "$BUILD_DIR/oofal_keygen.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/rng.c" -o "$BUILD_DIR/oofal_rng.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/shake.c" -o "$BUILD_DIR/oofal_shake.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/sign.c" -o "$BUILD_DIR/oofal_sign.o"
    $CC $CFLAGS -I"$OO_FALCON_DIR" -c "$OO_FALCON_DIR/vrfy.c" -o "$BUILD_DIR/oofal_vrfy.o"

    $CC $CFLAGS $LDFLAGS -o "$BUILD_DIR/$TARGET.exe" \
        "$BUILD_DIR/$TARGET.o" \
        "$BUILD_DIR/oofal_codec.o" "$BUILD_DIR/oofal_common.o" "$BUILD_DIR/oofal_falcon.o" \
        "$BUILD_DIR/oofal_fft.o" "$BUILD_DIR/oofal_fpr.o" "$BUILD_DIR/oofal_keygen.o" \
        "$BUILD_DIR/oofal_rng.o" "$BUILD_DIR/oofal_shake.o" "$BUILD_DIR/oofal_sign.o" \
        "$BUILD_DIR/oofal_vrfy.o" -lm -ladvapi32

    echo "  -> $BUILD_DIR/$TARGET.exe"
}

build_oofalcongpu() {
    echo "===== Building: OO-Falcon Hybrid GPU/CPU Token Benchmark ====="
    sh "$SCRIPT_DIR/build_oofalcon_gpu_tokens.sh"
}

build_soko_token() {
    echo "===== Building: SOKO Token Benchmark ====="
    sh "$SCRIPT_DIR/build_soko_token_benchmark.sh"
}

build_soko_refill() {
    echo "===== Building: SOKO Refill Simulator ====="
    sh "$SCRIPT_DIR/build_soko_refill_simulator.sh"
}

show_menu() {
    echo ""
    echo "============================================"
    echo "  Benchmark Build Menu"
    echo "============================================"
    echo "  1) ml-dsa       Certificate benchmark"
    echo "  2) ed25519      Certificate benchmark"
    echo "  3) ooed25519    Certificate benchmark (requires oo_ed25519.c/h)"
    echo "  4) falcon       Certificate benchmark"
    echo "  5) oo-ml-dsa    Certificate benchmark"
    echo "  6) oofalcon     Certificate benchmark"
    echo "  7) oofalcongpu  GPU token benchmark"
    echo "  8) soko-token   Main SOKO token benchmark"
    echo "  9) soko-refill  Refill simulator benchmark"
    echo " 10) all          Build all supported targets"
    echo "  0) quit"
    echo "============================================"
    printf "  Select [0-10]: "
    read choice
    case "$choice" in
        1) TARGETS="ml-dsa" ;;
        2) TARGETS="ed25519" ;;
        3) TARGETS="ooed25519" ;;
        4) TARGETS="falcon" ;;
        5) TARGETS="oo-ml-dsa" ;;
        6) TARGETS="oofalcon" ;;
        7) TARGETS="oofalcongpu" ;;
        8) TARGETS="soko-token" ;;
        9) TARGETS="soko-refill" ;;
       10) TARGETS="all" ;;
        0) echo "Bye."; exit 0 ;;
        *) echo "Invalid choice."; exit 1 ;;
    esac
}

run_target() {
    case "$1" in
        ml-dsa)         (set -e; build_dilithium) ;;
        ed25519)        (set -e; build_ed25519) ;;
        ooed25519)      (set -e; build_ooed25519) ;;
        falcon)         (set -e; build_falcon) ;;
        oo-ml-dsa)      (set -e; build_oodilithium) ;;
        oofalcon)       (set -e; build_oofalcon) ;;
        oofalcongpu)    (set -e; build_oofalcongpu) ;;
        soko-token)     (set -e; build_soko_token) ;;
        soko-refill)    (set -e; build_soko_refill) ;;
        *)
            echo "Unknown target: $1" >&2
            echo "Valid targets: ml-dsa ed25519 ooed25519 falcon oo-ml-dsa oofalcon oofalcongpu soko-token soko-refill all" >&2
            return 1
            ;;
    esac
}

mkdir -p "$BUILD_DIR"

if [ $# -eq 0 ]; then
    show_menu
else
    TARGETS="$1"
fi

if [ "$TARGETS" = "all" ]; then
    TARGETS="ml-dsa ed25519 falcon oo-ml-dsa oofalcon oofalcongpu soko-token soko-refill"
fi

BUILT=0
FAILED=0
for t in $TARGETS; do
    if run_target "$t"; then
        BUILT=$((BUILT + 1))
        echo ""
    else
        echo "ERROR: Failed to build $t" >&2
        FAILED=$((FAILED + 1))
        echo ""
    fi
done

echo "============================================"
echo "  Done: $BUILT built, $FAILED failed"
echo "============================================"

[ "$FAILED" -eq 0 ]
