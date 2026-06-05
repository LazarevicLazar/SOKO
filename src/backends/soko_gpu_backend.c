#include <string.h>

#include "soko_falcon_backends.h"

#if defined(OOFALCON_USE_CUDA) && defined(_WIN32)
#include <windows.h>

typedef int (*gpu_presign_fn_t)(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank);

typedef int (*gpu_reset_fn_t)(void);

typedef int (*gpu_verify_raw_batch_fn_t)(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank);

typedef int (*gpu_verify_compressed_bench_batch_fn_t)(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank);

static HMODULE gpu_module = NULL;
static gpu_presign_fn_t gpu_presign_fn = NULL;
static gpu_reset_fn_t gpu_reset_fn = NULL;
static gpu_verify_raw_batch_fn_t gpu_verify_raw_batch_fn = NULL;
static gpu_verify_compressed_bench_batch_fn_t gpu_verify_compressed_bench_batch_fn = NULL;
static int gpu_loader_state = 0;
#endif

void soko_copy_presign_bank_to_tokens(
    int tokens,
    const int8_t *sample1_bank,
    const int8_t *sample2_bank,
    const uint16_t *target_bank,
    soko_falcon_token_t *token_bank)
{
    int i;

    for (i = 0; i < tokens; i++) {
        memcpy(token_bank[i].s1, sample1_bank + (size_t)i * SOKO_FALCON_N, SOKO_FALCON_N);
        memcpy(token_bank[i].s2, sample2_bank + (size_t)i * SOKO_FALCON_N, SOKO_FALCON_N);
        memcpy(token_bank[i].target, target_bank + (size_t)i * SOKO_FALCON_N,
            (size_t)SOKO_FALCON_N * sizeof(uint16_t));
    }
}

int soko_gpu_backend_load(void)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    return -100;
#else
    if (gpu_loader_state != 0) {
        return (gpu_loader_state > 0) ? 0 : gpu_loader_state;
    }

    gpu_module = LoadLibraryA("soko_gpu_offline.dll");
    if (gpu_module == NULL) {
        gpu_loader_state = -101;
        return gpu_loader_state;
    }

    gpu_presign_fn = (gpu_presign_fn_t)(uintptr_t)GetProcAddress(
        gpu_module, "oofalcon_gpu_generate_presign_bank");
    if (gpu_presign_fn == NULL) {
        FreeLibrary(gpu_module);
        gpu_module = NULL;
        gpu_loader_state = -102;
        return gpu_loader_state;
    }

    gpu_reset_fn = (gpu_reset_fn_t)(uintptr_t)GetProcAddress(
        gpu_module, "oofalcon_gpu_reset_device");
    if (gpu_reset_fn == NULL) {
        FreeLibrary(gpu_module);
        gpu_module = NULL;
        gpu_loader_state = -104;
        return gpu_loader_state;
    }

    gpu_verify_raw_batch_fn = (gpu_verify_raw_batch_fn_t)(uintptr_t)GetProcAddress(
        gpu_module, "oofalcon_gpu_verify_raw_batch");
    if (gpu_verify_raw_batch_fn == NULL) {
        FreeLibrary(gpu_module);
        gpu_module = NULL;
        gpu_loader_state = -105;
        return gpu_loader_state;
    }

    gpu_verify_compressed_bench_batch_fn =
        (gpu_verify_compressed_bench_batch_fn_t)(uintptr_t)GetProcAddress(
            gpu_module, "oofalcon_gpu_verify_compressed_benchmark_batch");
    if (gpu_verify_compressed_bench_batch_fn == NULL) {
        FreeLibrary(gpu_module);
        gpu_module = NULL;
        gpu_loader_state = -106;
        return gpu_loader_state;
    }

    gpu_loader_state = 1;
    return 0;
#endif
}

int soko_gpu_generate_presign_bank_dynamic(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    (void)h_coeffs;
    (void)logn;
    (void)seed;
    (void)tokens;
    (void)sample1_bank;
    (void)sample2_bank;
    (void)target_bank;
    return -100;
#else
    int status = soko_gpu_backend_load();
    if (status != 0) {
        return status;
    }
    return gpu_presign_fn(h_coeffs, logn, seed, tokens,
        sample1_bank, sample2_bank, target_bank);
#endif
}

int soko_gpu_reset_device_dynamic(void)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    return -100;
#else
    int status = soko_gpu_backend_load();
    if (status != 0) {
        return status;
    }
    return gpu_reset_fn();
#endif
}

int soko_gpu_verify_raw_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    (void)h_ntt_monty;
    (void)logn;
    (void)tokens;
    (void)hm_bank;
    (void)s2_bank;
    (void)result_bank;
    return -100;
#else
    int status = soko_gpu_backend_load();
    if (status != 0) {
        return status;
    }
    return gpu_verify_raw_batch_fn(h_ntt_monty, logn, tokens,
        hm_bank, s2_bank, result_bank);
#endif
}

int soko_gpu_verify_compressed_benchmark_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    (void)h_ntt_monty;
    (void)logn;
    (void)tokens;
    (void)sig_bank;
    (void)sig_stride;
    (void)sig_lens;
    (void)result_bank;
    return -100;
#else
    int status = soko_gpu_backend_load();
    if (status != 0) {
        return status;
    }
    return gpu_verify_compressed_bench_batch_fn(h_ntt_monty, logn, tokens,
        sig_bank, sig_stride, sig_lens, result_bank);
#endif
}
