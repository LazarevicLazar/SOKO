#ifndef SOKO_GPU_OFFLINE_H
#define SOKO_GPU_OFFLINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(OOFALCON_GPU_BUILD_DLL)
#define OOFALCON_GPU_API __declspec(dllexport)
#elif defined(OOFALCON_GPU_USE_DLL)
#define OOFALCON_GPU_API __declspec(dllimport)
#else
#define OOFALCON_GPU_API
#endif
#else
#define OOFALCON_GPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Generate an OO-Falcon offline pre-sign bank on GPU.
 *
 * Inputs:
 *   h_coeffs      Public key polynomial in coefficient domain (mod q)
 *   logn          Falcon degree exponent (e.g. 9 for Falcon-512)
 *   seed          Seed for GPU-side pseudo-random sampling
 *   tokens        Number of pre-sign tokens to generate
 *
 * Outputs (flattened token-major arrays, length tokens*(1<<logn)):
 *   sample1_bank  int8 Gaussian-like sample vector #1
 *   sample2_bank  int8 Gaussian-like sample vector #2
 *   target_bank   uint16 target = sample1 - h*sample2 mod q
 *
 * Return:
 *   0 on success, non-zero on error.
 */
OOFALCON_GPU_API int oofalcon_gpu_generate_presign_bank(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank);

/*
 * Verify a batch of Falcon signatures with the raw verification equation on GPU.
 *
 * Inputs:
 *   h_ntt_monty   Public key polynomial already converted to NTT + Montgomery form
 *   logn          Falcon degree exponent
 *   tokens        Number of signatures in the batch
 *   hm_bank       Hashed message points, flattened token-major, length tokens*(1<<logn)
 *   s2_bank       Decoded signature vectors, flattened token-major, length tokens*(1<<logn)
 *
 * Outputs:
 *   result_bank   One byte per signature: 1 = accepted, 0 = rejected
 *
 * Return:
 *   0 on success, non-zero on error.
 */
OOFALCON_GPU_API int oofalcon_gpu_verify_raw_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank);

/*
 * Benchmark-oriented full GPU verification path.
 *
 * Inputs:
 *   h_ntt_monty   Public key polynomial already converted to NTT + Montgomery form
 *   logn          Falcon degree exponent
 *   tokens        Number of signatures in the batch
 *   sig_bank      Token-major compressed signatures
 *   sig_stride    Byte stride between signatures in sig_bank
 *   sig_lens      Actual length of each compressed signature
 *
 * Outputs:
 *   result_bank   One byte per signature: 1 = accepted, 0 = rejected
 *
 * Notes:
 *   This path is benchmark-specific: it assumes the same message generation rule
 *   as soko_token_benchmark.c and processes compressed Falcon signatures
 *   entirely on the GPU after transfer.
 */
OOFALCON_GPU_API int oofalcon_gpu_verify_compressed_benchmark_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank);

/* Reset CUDA device/context to force cold-start behavior on next call. */
OOFALCON_GPU_API int oofalcon_gpu_reset_device(void);

#ifdef __cplusplus
}
#endif

#endif
