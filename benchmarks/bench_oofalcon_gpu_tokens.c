/*
 * OO-Falcon Hybrid Benchmark
 *
 * Goal:
 * - Generate offline pre-sign bank for token batches (8/128/1024/8192/65536)
 * - Compare CPU-offline bank generation vs GPU-offline bank generation
 * - Keep online signing on CPU
 *
 * Notes:
 * - This benchmark is intentionally separate from bench_cert_oofalcon.c.
 * - It uses existing OO-Falcon primitives from falcon-lazy/sign.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>

#if defined(__STDC_NO_ATOMICS__)
typedef volatile int atomic_int;
static inline int atomic_load(const atomic_int *p)
{
    return *p;
}
static inline void atomic_store(atomic_int *p, int v)
{
    *p = v;
}
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <process.h>
static inline double get_time_seconds(void)
{
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
#include <pthread.h>
static inline double get_time_seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}
#endif

#include "soko_falcon_backends.h"

#if defined(OOFALCON_USE_CUDA) && defined(_WIN32)
static int gpu_generate_presign_bank_dynamic(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank);
static int gpu_reset_device_dynamic(void);
#endif

/* Offline/online split primitives in falcon-lazy/sign.c */
extern int sign_dyn_lazy_online(
    int8_t *sample1, int8_t *sample2, uint16_t *sample_target,
    int16_t *s2,
    const fpr *restrict f_fft, const fpr *restrict g_fft,
    const fpr *restrict F_fft, const fpr *restrict G_fft,
    const uint16_t *hm, unsigned logn, fpr *restrict tmp);

#define LOGN SOKO_FALCON_LOGN
#define DEFAULT_TOKEN_STEPS 6
#define MSG_SIZE SOKO_FALCON_MSG_SIZE
#define DEFAULT_MAX_SIGN_ATTEMPTS 4096

enum {
    VERIFY_BACKEND_CPU = 0,
    VERIFY_BACKEND_GPU_RAW = 1,
    VERIFY_BACKEND_GPU_FULL = 2
};

static const char *verify_backend_name(int mode)
{
    switch (mode) {
    case VERIFY_BACKEND_GPU_RAW:
        return "gpu_raw";
    case VERIFY_BACKEND_GPU_FULL:
        return "gpu_full";
    default:
        return "cpu";
    }
}

typedef struct {
    const char *csv_path;
    int max_sign_attempts;
    int cpu_mt_threads;
    int verify_signatures;
    int verify_backend_mode;
    int pipeline_mode;
} bench_config_t;

static void smallints_to_fpr_local(fpr *out, const int8_t *in, unsigned logn)
{
    size_t n = (size_t)1 << logn;
    for (size_t i = 0; i < n; i++) {
        out[i] = fpr_of(in[i]);
    }
}


typedef struct {
    int cpu_mt_threads_used;

    double offline_cpu_cold_ms;
    double offline_cpu_steady_ms;
    double offline_cpumt_cold_ms;
    double offline_cpumt_steady_ms;
    double offline_gpu_cold_ms;
    double offline_gpu_steady_ms;

    double online_cpu_from_cpu_cold_ms;
    double online_cpu_from_cpu_steady_ms;
    double online_cpu_from_cpumt_cold_ms;
    double online_cpu_from_cpumt_steady_ms;
    double online_cpu_from_gpu_cold_ms;
    double online_cpu_from_gpu_steady_ms;

    double verify_cpu_from_cpu_cold_ms;
    double verify_cpu_from_cpu_steady_ms;
    double verify_cpu_from_cpumt_cold_ms;
    double verify_cpu_from_cpumt_steady_ms;
    double verify_cpu_from_gpu_cold_ms;
    double verify_cpu_from_gpu_steady_ms;

    double avg_attempts_cpu_cold;
    double avg_attempts_cpu_steady;
    double avg_attempts_cpumt_cold;
    double avg_attempts_cpumt_steady;
    double avg_attempts_gpu_cold;
    double avg_attempts_gpu_steady;

    int ok_cpu_cold;
    int fail_cpu_cold;
    int ok_cpu_steady;
    int fail_cpu_steady;
    int ok_cpumt_cold;
    int fail_cpumt_cold;
    int ok_cpumt_steady;
    int fail_cpumt_steady;
    int ok_gpu_cold;
    int fail_gpu_cold;
    int ok_gpu_steady;
    int fail_gpu_steady;

    int gpu_available_cold;
    int gpu_available_steady;
    int gpu_error_code_cold;
    int gpu_error_code_steady;
    int verify_backend_mode;
} token_result_t;

static int gpu_generate_presign_bank_dynamic(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    return soko_gpu_generate_presign_bank_dynamic(h_coeffs, logn, seed, tokens,
        sample1_bank, sample2_bank, target_bank);
}

static int gpu_reset_device_dynamic(void)
{
    return soko_gpu_reset_device_dynamic();
}

static int gpu_verify_raw_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank)
{
    return soko_gpu_verify_raw_batch_dynamic(h_ntt_monty, logn, tokens,
        hm_bank, s2_bank, result_bank);
}

static int gpu_verify_compressed_benchmark_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank)
{
    return soko_gpu_verify_compressed_benchmark_batch_dynamic(h_ntt_monty, logn, tokens,
        sig_bank, sig_stride, sig_lens, result_bank);
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [--csv <path>] [--max-attempts <n>] [--cpu-mt-threads <n>] [--no-verify] [--verify-on-gpu] [--verify-full-gpu] [--pipeline]\n", prog);
    printf("  --csv <path>          Write results table as CSV\n");
    printf("  --max-attempts <n>    Max online signing retries per token (default: %d)\n",
        DEFAULT_MAX_SIGN_ATTEMPTS);
    printf("  --cpu-mt-threads <n>  CPU threads for multithread offline pipeline (default: auto)\n");
    printf("  --no-verify           Skip falcon_verify() during online stage\n");
    printf("  --verify-on-gpu       Use hybrid GPU verify (CPU decode/hash + GPU raw check)\n");
    printf("  --verify-full-gpu     Use full GPU verify benchmark (decode + hash + check on GPU)\n");
    printf("  --pipeline            Use ring-buffer + async GPU refill pipeline\n");
}

static int parse_positive_int(const char *s, int *out)
{
    char *end = NULL;
    long v;

    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0 || v > 1000000L) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_args(int argc, char **argv, bench_config_t *cfg)
{
    int i;

    cfg->csv_path = NULL;
    cfg->max_sign_attempts = DEFAULT_MAX_SIGN_ATTEMPTS;
    cfg->cpu_mt_threads = 0;
    cfg->verify_signatures = 1;
    cfg->verify_backend_mode = VERIFY_BACKEND_CPU;
    cfg->pipeline_mode = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --csv\n");
                return 0;
            }
            cfg->csv_path = argv[++i];
        } else if (strcmp(argv[i], "--max-attempts") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --max-attempts\n");
                return 0;
            }
            if (!parse_positive_int(argv[++i], &cfg->max_sign_attempts)) {
                fprintf(stderr, "Invalid --max-attempts value\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--cpu-mt-threads") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for --cpu-mt-threads\n");
                return 0;
            }
            if (!parse_positive_int(argv[++i], &cfg->cpu_mt_threads)) {
                fprintf(stderr, "Invalid --cpu-mt-threads value\n");
                return 0;
            }
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            cfg->verify_signatures = 0;
        } else if (strcmp(argv[i], "--verify-on-gpu") == 0) {
            cfg->verify_backend_mode = VERIFY_BACKEND_GPU_RAW;
        } else if (strcmp(argv[i], "--verify-full-gpu") == 0) {
            cfg->verify_backend_mode = VERIFY_BACKEND_GPU_FULL;
        } else if (strcmp(argv[i], "--pipeline") == 0) {
            cfg->pipeline_mode = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 0;
        }
    }

    return 1;
}

static void write_csv_header(FILE *f)
{
    fprintf(f,
    "tokens,"
    "cpu_mt_threads_used,"
    "cpu_offline_cold_ms,cpu_offline_steady_ms,"
    "cpumt_offline_cold_ms,cpumt_offline_steady_ms,"
    "gpu_available_cold,gpu_error_code_cold,gpu_offline_cold_ms,"
    "gpu_available_steady,gpu_error_code_steady,gpu_offline_steady_ms,"
    "cpu_online_from_cpu_cold_ms,cpu_online_from_cpu_steady_ms,"
    "cpu_online_from_cpumt_cold_ms,cpu_online_from_cpumt_steady_ms,"
    "cpu_online_from_gpu_cold_ms,cpu_online_from_gpu_steady_ms,"
    "cpu_verify_from_cpu_cold_ms,cpu_verify_from_cpu_steady_ms,"
    "cpu_verify_from_cpumt_cold_ms,cpu_verify_from_cpumt_steady_ms,"
    "cpu_verify_from_gpu_cold_ms,cpu_verify_from_gpu_steady_ms,"
    "cpu_ok_cold,cpu_fail_cold,cpu_ok_steady,cpu_fail_steady,"
    "cpumt_ok_cold,cpumt_fail_cold,cpumt_ok_steady,cpumt_fail_steady,"
    "gpu_ok_cold,gpu_fail_cold,gpu_ok_steady,gpu_fail_steady,"
    "avg_attempts_cpu_cold,avg_attempts_cpu_steady,"
    "avg_attempts_cpumt_cold,avg_attempts_cpumt_steady,"
    "avg_attempts_gpu_cold,avg_attempts_gpu_steady,"
    "cpu_sign_throughput_cold,cpu_sign_throughput_steady,"
    "cpu_verify_throughput_cold,cpu_verify_throughput_steady,"
    "cpu_sign_and_verify_throughput_cold,cpu_sign_and_verify_throughput_steady,"
    "cpumt_sign_throughput_cold,cpumt_sign_throughput_steady,"
    "cpumt_verify_throughput_cold,cpumt_verify_throughput_steady,"
    "cpumt_sign_and_verify_throughput_cold,cpumt_sign_and_verify_throughput_steady,"
    "gpu_sign_throughput_cold,gpu_sign_throughput_steady,"
    "gpu_verify_throughput_cold,gpu_verify_throughput_steady,"
    "gpu_sign_and_verify_throughput_cold,gpu_sign_and_verify_throughput_steady,"
    "cpu_total_cold_ms,cpumt_total_cold_ms,gpu_total_cold_ms,"
    "cpu_to_cpumt_cold_speedup,cpu_to_gpu_cold_speedup,cpumt_to_gpu_cold_speedup,"
    "cpu_total_steady_ms,cpumt_total_steady_ms,gpu_total_steady_ms,"
    "cpu_to_cpumt_steady_speedup,cpu_to_gpu_steady_speedup,cpumt_to_gpu_steady_speedup,"
    "cpu_offline_cold_us_per_token,cpu_offline_steady_us_per_token,"
    "cpumt_offline_cold_us_per_token,cpumt_offline_steady_us_per_token,"
    "gpu_offline_cold_us_per_token,gpu_offline_steady_us_per_token,"
    "verify_backend\n");
}

static double throughput_from_ms(int tokens, double elapsed_ms)
{
    if (tokens <= 0 || elapsed_ms <= 0.0) {
        return -1.0;
    }
    return ((double)tokens * 1000.0) / elapsed_ms;
}

static void write_csv_row(FILE *f, int tokens, const token_result_t *r)
{
    double cpu_sign_cold_ms = (r->verify_cpu_from_cpu_cold_ms >= 0.0)
        ? (r->online_cpu_from_cpu_cold_ms - r->verify_cpu_from_cpu_cold_ms)
        : r->online_cpu_from_cpu_cold_ms;
    double cpu_sign_steady_ms = (r->verify_cpu_from_cpu_steady_ms >= 0.0)
        ? (r->online_cpu_from_cpu_steady_ms - r->verify_cpu_from_cpu_steady_ms)
        : r->online_cpu_from_cpu_steady_ms;
    double cpumt_sign_cold_ms = (r->verify_cpu_from_cpumt_cold_ms >= 0.0)
        ? (r->online_cpu_from_cpumt_cold_ms - r->verify_cpu_from_cpumt_cold_ms)
        : r->online_cpu_from_cpumt_cold_ms;
    double cpumt_sign_steady_ms = (r->verify_cpu_from_cpumt_steady_ms >= 0.0)
        ? (r->online_cpu_from_cpumt_steady_ms - r->verify_cpu_from_cpumt_steady_ms)
        : r->online_cpu_from_cpumt_steady_ms;
    double gpu_sign_cold_ms = (r->verify_cpu_from_gpu_cold_ms >= 0.0)
        ? (r->online_cpu_from_gpu_cold_ms - r->verify_cpu_from_gpu_cold_ms)
        : r->online_cpu_from_gpu_cold_ms;
    double gpu_sign_steady_ms = (r->verify_cpu_from_gpu_steady_ms >= 0.0)
        ? (r->online_cpu_from_gpu_steady_ms - r->verify_cpu_from_gpu_steady_ms)
        : r->online_cpu_from_gpu_steady_ms;
    double cpu_total_cold = r->offline_cpu_cold_ms + r->online_cpu_from_cpu_cold_ms;
    double cpu_total_steady = r->offline_cpu_steady_ms + r->online_cpu_from_cpu_steady_ms;
    double cpumt_total_cold = r->offline_cpumt_cold_ms + r->online_cpu_from_cpumt_cold_ms;
    double cpumt_total_steady = r->offline_cpumt_steady_ms + r->online_cpu_from_cpumt_steady_ms;
    double gpu_total_cold = -1.0;
    double gpu_total_steady = -1.0;
    double cpu_to_cpumt_cold_speedup = -1.0;
    double cpu_to_cpumt_steady_speedup = -1.0;
    double cpu_to_gpu_cold_speedup = -1.0;
    double cpu_to_gpu_steady_speedup = -1.0;
    double cpumt_to_gpu_cold_speedup = -1.0;
    double cpumt_to_gpu_steady_speedup = -1.0;
    double cpu_offline_cold_us_per_token = -1.0;
    double cpu_offline_steady_us_per_token = -1.0;
    double cpumt_offline_cold_us_per_token = -1.0;
    double cpumt_offline_steady_us_per_token = -1.0;
    double gpu_offline_cold_us_per_token = -1.0;
    double gpu_offline_steady_us_per_token = -1.0;
    double cpu_sign_throughput_cold = throughput_from_ms(tokens, cpu_sign_cold_ms);
    double cpu_sign_throughput_steady = throughput_from_ms(tokens, cpu_sign_steady_ms);
    double cpu_verify_throughput_cold = throughput_from_ms(tokens, r->verify_cpu_from_cpu_cold_ms);
    double cpu_verify_throughput_steady = throughput_from_ms(tokens, r->verify_cpu_from_cpu_steady_ms);
    double cpu_sign_verify_throughput_cold = throughput_from_ms(tokens,
        (r->verify_cpu_from_cpu_cold_ms >= 0.0) ? r->online_cpu_from_cpu_cold_ms : -1.0);
    double cpu_sign_verify_throughput_steady = throughput_from_ms(tokens,
        (r->verify_cpu_from_cpu_steady_ms >= 0.0) ? r->online_cpu_from_cpu_steady_ms : -1.0);
    double cpumt_sign_throughput_cold = throughput_from_ms(tokens, cpumt_sign_cold_ms);
    double cpumt_sign_throughput_steady = throughput_from_ms(tokens, cpumt_sign_steady_ms);
    double cpumt_verify_throughput_cold = throughput_from_ms(tokens, r->verify_cpu_from_cpumt_cold_ms);
    double cpumt_verify_throughput_steady = throughput_from_ms(tokens, r->verify_cpu_from_cpumt_steady_ms);
    double cpumt_sign_verify_throughput_cold = throughput_from_ms(tokens,
        (r->verify_cpu_from_cpumt_cold_ms >= 0.0) ? r->online_cpu_from_cpumt_cold_ms : -1.0);
    double cpumt_sign_verify_throughput_steady = throughput_from_ms(tokens,
        (r->verify_cpu_from_cpumt_steady_ms >= 0.0) ? r->online_cpu_from_cpumt_steady_ms : -1.0);
    double gpu_sign_throughput_cold = throughput_from_ms(tokens, gpu_sign_cold_ms);
    double gpu_sign_throughput_steady = throughput_from_ms(tokens, gpu_sign_steady_ms);
    double gpu_verify_throughput_cold = throughput_from_ms(tokens, r->verify_cpu_from_gpu_cold_ms);
    double gpu_verify_throughput_steady = throughput_from_ms(tokens, r->verify_cpu_from_gpu_steady_ms);
    double gpu_sign_verify_throughput_cold = throughput_from_ms(tokens,
        (r->verify_cpu_from_gpu_cold_ms >= 0.0) ? r->online_cpu_from_gpu_cold_ms : -1.0);
    double gpu_sign_verify_throughput_steady = throughput_from_ms(tokens,
        (r->verify_cpu_from_gpu_steady_ms >= 0.0) ? r->online_cpu_from_gpu_steady_ms : -1.0);

    if (cpumt_total_cold > 0.0) {
        cpu_to_cpumt_cold_speedup = cpu_total_cold / cpumt_total_cold;
    }
    if (cpumt_total_steady > 0.0) {
        cpu_to_cpumt_steady_speedup = cpu_total_steady / cpumt_total_steady;
    }

    if (tokens > 0) {
        cpu_offline_cold_us_per_token = (r->offline_cpu_cold_ms * 1000.0) / (double)tokens;
        cpu_offline_steady_us_per_token = (r->offline_cpu_steady_ms * 1000.0) / (double)tokens;
        cpumt_offline_cold_us_per_token = (r->offline_cpumt_cold_ms * 1000.0) / (double)tokens;
        cpumt_offline_steady_us_per_token = (r->offline_cpumt_steady_ms * 1000.0) / (double)tokens;
    }

    if (r->gpu_available_cold && r->offline_gpu_cold_ms >= 0.0 && r->online_cpu_from_gpu_cold_ms >= 0.0) {
        gpu_total_cold = r->offline_gpu_cold_ms + r->online_cpu_from_gpu_cold_ms;
        if (gpu_total_cold > 0.0) {
            cpu_to_gpu_cold_speedup = cpu_total_cold / gpu_total_cold;
            cpumt_to_gpu_cold_speedup = cpumt_total_cold / gpu_total_cold;
        }
        if (tokens > 0) {
            gpu_offline_cold_us_per_token = (r->offline_gpu_cold_ms * 1000.0) / (double)tokens;
        }
    }

    if (r->gpu_available_steady && r->offline_gpu_steady_ms >= 0.0 && r->online_cpu_from_gpu_steady_ms >= 0.0) {
        gpu_total_steady = r->offline_gpu_steady_ms + r->online_cpu_from_gpu_steady_ms;
        if (gpu_total_steady > 0.0) {
            cpu_to_gpu_steady_speedup = cpu_total_steady / gpu_total_steady;
            cpumt_to_gpu_steady_speedup = cpumt_total_steady / gpu_total_steady;
        }
        if (tokens > 0) {
            gpu_offline_steady_us_per_token = (r->offline_gpu_steady_ms * 1000.0) / (double)tokens;
        }
    }

    fprintf(f, "%d", tokens);
    fprintf(f, ",%d", r->cpu_mt_threads_used);
    fprintf(f, ",%.6f,%.6f", r->offline_cpu_cold_ms, r->offline_cpu_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->offline_cpumt_cold_ms, r->offline_cpumt_steady_ms);
    fprintf(f, ",%d,%d,%.6f", r->gpu_available_cold, r->gpu_error_code_cold, r->offline_gpu_cold_ms);
    fprintf(f, ",%d,%d,%.6f", r->gpu_available_steady, r->gpu_error_code_steady, r->offline_gpu_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->online_cpu_from_cpu_cold_ms, r->online_cpu_from_cpu_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->online_cpu_from_cpumt_cold_ms, r->online_cpu_from_cpumt_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->online_cpu_from_gpu_cold_ms, r->online_cpu_from_gpu_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->verify_cpu_from_cpu_cold_ms, r->verify_cpu_from_cpu_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->verify_cpu_from_cpumt_cold_ms, r->verify_cpu_from_cpumt_steady_ms);
    fprintf(f, ",%.6f,%.6f", r->verify_cpu_from_gpu_cold_ms, r->verify_cpu_from_gpu_steady_ms);
    fprintf(f, ",%d,%d,%d,%d", r->ok_cpu_cold, r->fail_cpu_cold, r->ok_cpu_steady, r->fail_cpu_steady);
    fprintf(f, ",%d,%d,%d,%d", r->ok_cpumt_cold, r->fail_cpumt_cold, r->ok_cpumt_steady, r->fail_cpumt_steady);
    fprintf(f, ",%d,%d,%d,%d", r->ok_gpu_cold, r->fail_gpu_cold, r->ok_gpu_steady, r->fail_gpu_steady);
    fprintf(f, ",%.6f,%.6f", r->avg_attempts_cpu_cold, r->avg_attempts_cpu_steady);
    fprintf(f, ",%.6f,%.6f", r->avg_attempts_cpumt_cold, r->avg_attempts_cpumt_steady);
    fprintf(f, ",%.6f,%.6f", r->avg_attempts_gpu_cold, r->avg_attempts_gpu_steady);
    fprintf(f, ",%.6f,%.6f", cpu_sign_throughput_cold, cpu_sign_throughput_steady);
    fprintf(f, ",%.6f,%.6f", cpu_verify_throughput_cold, cpu_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f", cpu_sign_verify_throughput_cold, cpu_sign_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f", cpumt_sign_throughput_cold, cpumt_sign_throughput_steady);
    fprintf(f, ",%.6f,%.6f", cpumt_verify_throughput_cold, cpumt_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f", cpumt_sign_verify_throughput_cold, cpumt_sign_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f", gpu_sign_throughput_cold, gpu_sign_throughput_steady);
    fprintf(f, ",%.6f,%.6f", gpu_verify_throughput_cold, gpu_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f", gpu_sign_verify_throughput_cold, gpu_sign_verify_throughput_steady);
    fprintf(f, ",%.6f,%.6f,%.6f", cpu_total_cold, cpumt_total_cold, gpu_total_cold);
    fprintf(f, ",%.6f,%.6f,%.6f", cpu_to_cpumt_cold_speedup, cpu_to_gpu_cold_speedup, cpumt_to_gpu_cold_speedup);
    fprintf(f, ",%.6f,%.6f,%.6f", cpu_total_steady, cpumt_total_steady, gpu_total_steady);
    fprintf(f, ",%.6f,%.6f,%.6f", cpu_to_cpumt_steady_speedup, cpu_to_gpu_steady_speedup, cpumt_to_gpu_steady_speedup);
    fprintf(f, ",%.6f,%.6f", cpu_offline_cold_us_per_token, cpu_offline_steady_us_per_token);
    fprintf(f, ",%.6f,%.6f", cpumt_offline_cold_us_per_token, cpumt_offline_steady_us_per_token);
    fprintf(f, ",%.6f,%.6f", gpu_offline_cold_us_per_token, gpu_offline_steady_us_per_token);
    fprintf(f, ",%s\n", verify_backend_name(r->verify_backend_mode));
}

static void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "Allocation failed (%lu bytes)\n", (unsigned long)size);
        exit(1);
    }
    return p;
}

static void fill_message(uint8_t *msg, size_t msg_len, int token_index)
{
    for (size_t i = 0; i < msg_len; i++) {
        msg[i] = (uint8_t)((token_index * 131 + (int)i * 17) & 0xFF);
    }
}

static void derive_nonce_from_token(uint8_t nonce[40], int token_index)
{
    uint64_t x = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)(token_index + 1) * 0xD6E8FEB86659FD93ULL);
    for (int i = 0; i < 40; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        nonce[i] = (uint8_t)((x * 2685821657736338717ULL) >> 56);
    }
}

static int sign_token_to_signature(
    unsigned logn,
    const uint8_t *message,
    size_t message_len,
    const uint8_t nonce[40],
    const int8_t *sample1,
    const int8_t *sample2,
    const uint16_t *sample_target,
    const fpr *f_fft,
    const fpr *g_fft,
    const fpr *F_fft,
    const fpr *G_fft,
    uint16_t *hm,
    uint16_t *target_work,
    int16_t *sv,
    fpr *ftmp,
    uint8_t *sig_buf,
    size_t sig_buf_len,
    size_t *sig_len)
{
    size_t n = (size_t)1 << logn;
    shake256_context hash_data;

    shake256_init(&hash_data);
    shake256_inject(&hash_data, nonce, 40);
    shake256_inject(&hash_data, message, message_len);
    shake256_flip(&hash_data);
    Zf(hash_to_point_vartime)((inner_shake256_context *)&hash_data, hm, logn);

    memcpy(target_work, sample_target, n * sizeof(uint16_t));
    sign_dyn_lazy_online((int8_t *)sample1, (int8_t *)sample2, target_work,
        sv, f_fft, g_fft, F_fft, G_fft, hm, logn, ftmp);

    if (sig_len != NULL) {
        *sig_len = 0;
    }
    if (sig_buf == NULL || sig_len == NULL) {
        return 1;
    }

    sig_buf[0] = 0x30 + logn;
    memcpy(sig_buf + 1, nonce, 40);
    {
        size_t v = Zf(comp_encode)(sig_buf + 41, sig_buf_len - 41, sv, logn);
        if (v == 0) {
            return 0;
        }
        *sig_len = 41 + v;
        return 1;
    }
}

static int verify_token_signature(
    unsigned logn,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint8_t *message,
    size_t message_len,
    const uint8_t *sig,
    size_t sig_len,
    uint8_t *tmp_verify,
    size_t tmp_verify_len)
{
    (void)logn;
    return falcon_verify(sig, sig_len,
        FALCON_SIG_COMPRESSED,
        pubkey, pubkey_len,
        message, message_len,
        tmp_verify, tmp_verify_len) == 0;
}

static int prepare_verify_raw_inputs_from_signature(
    unsigned logn,
    const uint8_t *message,
    size_t message_len,
    const uint8_t *sig,
    size_t sig_len,
    uint16_t *hm_out,
    int16_t *s2_out)
{
    shake256_context hash_data;
    size_t v;

    if (sig_len < 41) {
        return 0;
    }
    if ((sig[0] & 0xF0) != 0x30 || (unsigned)(sig[0] & 0x0F) != logn) {
        return 0;
    }

    v = Zf(comp_decode)(s2_out, logn, sig + 41, sig_len - 41);
    if (v == 0 || (41 + v) != sig_len) {
        return 0;
    }

    shake256_init(&hash_data);
    shake256_inject(&hash_data, sig + 1, 40);
    shake256_inject(&hash_data, message, message_len);
    shake256_flip(&hash_data);
    Zf(hash_to_point_vartime)((inner_shake256_context *)&hash_data, hm_out, logn);
    return 1;
}

static void run_verify_batch_serial(
    unsigned logn,
    int tokens,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    double *verify_ms,
    int *ok_count,
    int *fail_count)
{
    size_t tmp_verify_len = FALCON_TMPSIZE_VERIFY(logn);
    uint8_t *tmp_verify = (uint8_t *)xmalloc(tmp_verify_len);
    uint8_t message[MSG_SIZE];
    int ok = 0;
    int fail = 0;

    double start = get_time_seconds();
    for (int t = 0; t < tokens; t++) {
        const uint8_t *sig = sig_bank + (size_t)t * sig_stride;
        if (sig_lens[t] == 0) {
            continue;
        }
        fill_message(message, sizeof(message), t);
        if (verify_token_signature(logn,
                pubkey, pubkey_len,
                message, sizeof(message),
                sig, sig_lens[t],
                tmp_verify, tmp_verify_len)) {
            ok++;
        } else {
            fail++;
        }
    }
    double end = get_time_seconds();

    *verify_ms = (end - start) * 1000.0;
    *ok_count = ok;
    *fail_count = fail;
    free(tmp_verify);
}

static void run_verify_batch_mt(
    unsigned logn,
    int tokens,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    int verify_threads,
    double *verify_ms,
    int *ok_count,
    int *fail_count)
{
#ifndef _OPENMP
    (void)verify_threads;
    run_verify_batch_serial(logn, tokens,
        pubkey, pubkey_len,
        sig_bank, sig_stride, sig_lens,
        verify_ms, ok_count, fail_count);
#else
    if (verify_threads <= 1) {
        run_verify_batch_serial(logn, tokens,
            pubkey, pubkey_len,
            sig_bank, sig_stride, sig_lens,
            verify_ms, ok_count, fail_count);
        return;
    }

    size_t tmp_verify_len = FALCON_TMPSIZE_VERIFY(logn);
    int ok = 0;
    int fail = 0;

    double start = get_time_seconds();
#pragma omp parallel num_threads(verify_threads)
    {
        uint8_t *tmp_verify = (uint8_t *)xmalloc(tmp_verify_len);
        uint8_t message[MSG_SIZE];
        int local_ok = 0;
        int local_fail = 0;

#pragma omp for schedule(static)
        for (int t = 0; t < tokens; t++) {
            const uint8_t *sig = sig_bank + (size_t)t * sig_stride;
            if (sig_lens[t] == 0) {
                continue;
            }
            fill_message(message, sizeof(message), t);
            if (verify_token_signature(logn,
                    pubkey, pubkey_len,
                    message, sizeof(message),
                    sig, sig_lens[t],
                    tmp_verify, tmp_verify_len)) {
                local_ok++;
            } else {
                local_fail++;
            }
        }

        free(tmp_verify);

#pragma omp atomic
        ok += local_ok;
#pragma omp atomic
        fail += local_fail;
    }
    double end = get_time_seconds();

    *verify_ms = (end - start) * 1000.0;
    *ok_count = ok;
    *fail_count = fail;
#endif
}

static void run_verify_batch_gpu_raw(
    unsigned logn,
    const uint16_t *h_monty,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    double *verify_ms,
    int *ok_count,
    int *fail_count)
{
#if defined(OOFALCON_USE_CUDA) && defined(_WIN32)
    size_t n = (size_t)1 << logn;
    uint16_t *hm_bank = (uint16_t *)xmalloc((size_t)tokens * n * sizeof(uint16_t));
    int16_t *s2_bank = (int16_t *)xmalloc((size_t)tokens * n * sizeof(int16_t));
    uint8_t *result_bank = (uint8_t *)xmalloc((size_t)tokens);
    uint8_t message[MSG_SIZE];
    int active = 0;
    int ok = 0;
    int fail = 0;

    double start = get_time_seconds();
    for (int t = 0; t < tokens; t++) {
        const uint8_t *sig = sig_bank + (size_t)t * sig_stride;
        if (sig_lens[t] == 0) {
            continue;
        }

        fill_message(message, sizeof(message), t);
        if (prepare_verify_raw_inputs_from_signature(logn,
                message, sizeof(message),
                sig, sig_lens[t],
                hm_bank + (size_t)active * n,
                s2_bank + (size_t)active * n)) {
            active++;
        } else {
            fail++;
        }
    }

    if (active > 0) {
        int gr = gpu_verify_raw_batch_dynamic(h_monty, logn, active,
            hm_bank, s2_bank, result_bank);
        if (gr != 0) {
            fprintf(stderr, "GPU verify backend failed (err=%d)\n", gr);
            exit(1);
        }
        for (int i = 0; i < active; i++) {
            if (result_bank[i] != 0) {
                ok++;
            } else {
                fail++;
            }
        }
    }
    double end = get_time_seconds();

    *verify_ms = (end - start) * 1000.0;
    *ok_count = ok;
    *fail_count = fail;

    free(hm_bank);
    free(s2_bank);
    free(result_bank);
#else
    (void)logn;
    (void)h_monty;
    (void)tokens;
    (void)sig_bank;
    (void)sig_stride;
    (void)sig_lens;
    (void)verify_ms;
    (void)ok_count;
    (void)fail_count;
    fprintf(stderr, "GPU verify backend requested, but this benchmark was built without CUDA support.\n");
    exit(1);
#endif
}

static void run_verify_batch_gpu_full(
    unsigned logn,
    const uint16_t *h_monty,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    double *verify_ms,
    int *ok_count,
    int *fail_count)
{
#if defined(OOFALCON_USE_CUDA) && defined(_WIN32)
    uint8_t *result_bank = (uint8_t *)xmalloc((size_t)tokens);
    int ok = 0;
    int fail = 0;

    double start = get_time_seconds();
    {
        int gr = gpu_verify_compressed_benchmark_batch_dynamic(
            h_monty, logn, tokens,
            sig_bank, sig_stride, sig_lens,
            result_bank);
        if (gr != 0) {
            fprintf(stderr, "Full GPU verify backend failed (err=%d)\n", gr);
            exit(1);
        }
    }
    double end = get_time_seconds();

    for (int i = 0; i < tokens; i++) {
        if (result_bank[i] != 0) {
            ok++;
        } else {
            fail++;
        }
    }

    *verify_ms = (end - start) * 1000.0;
    *ok_count = ok;
    *fail_count = fail;
    free(result_bank);
#else
    (void)logn;
    (void)h_monty;
    (void)tokens;
    (void)sig_bank;
    (void)sig_stride;
    (void)sig_lens;
    (void)verify_ms;
    (void)ok_count;
    (void)fail_count;
    fprintf(stderr, "Full GPU verify backend requested, but this benchmark was built without CUDA support.\n");
    exit(1);
#endif
}

static int prepare_key_and_state(
    unsigned logn,
    uint8_t *privkey,
    size_t privkey_len,
    uint8_t *pubkey,
    size_t pubkey_len,
    uint8_t *tmp,
    size_t tmp_len,
    int8_t *f,
    int8_t *g,
    int8_t *F,
    int8_t *G,
    uint16_t *h,
    uint16_t *h_monty,
    fpr *f_fft,
    fpr *g_fft,
    fpr *F_fft,
    fpr *G_fft)
{
    return soko_prepare_falcon_key_material(
        logn,
        privkey, privkey_len,
        pubkey, pubkey_len,
        tmp, tmp_len,
        f, g, F, G,
        h, h_monty,
        f_fft, g_fft, F_fft, G_fft);
}

static void generate_bank_cpu_original(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    soko_cpu_generate_presign_bank_raw(logn, tokens, h_monty,
        sample1_bank, sample2_bank, target_bank);
}

static int resolve_cpu_mt_threads(int requested)
{
    return soko_resolve_cpu_mt_threads(requested);
}

static void generate_bank_cpu_mt(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank,
    int cpu_mt_threads)
{
    soko_cpumt_generate_presign_bank_raw(logn, tokens, h_monty,
        sample1_bank, sample2_bank, target_bank, cpu_mt_threads);
}

static void run_online_cpu(
    unsigned logn,
    int tokens,
    int max_sign_attempts,
    int verify_signatures,
    int verify_backend_mode,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint16_t *h_monty,
    const int8_t *sample1_bank,
    const int8_t *sample2_bank,
    const uint16_t *target_bank,
    const fpr *f_fft,
    const fpr *g_fft,
    const fpr *F_fft,
    const fpr *G_fft,
    double *online_ms,
    double *verify_ms,
    double *avg_attempts,
    int *ok_count,
    int *fail_count)
{
    size_t n = (size_t)1 << logn;
    uint16_t *hm = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    uint16_t *target_work = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    int16_t *sv = (int16_t *)xmalloc(n * sizeof(int16_t));

    size_t tmp_sign_len = FALCON_TMPSIZE_SIGNDYN(logn);
    uint8_t *tmp_sign = (uint8_t *)xmalloc(tmp_sign_len);
    fpr *ftmp = (fpr *)tmp_sign;
    size_t sig_max = verify_signatures ? FALCON_SIG_COMPRESSED_MAXSIZE(logn) : 0;
    uint8_t *sig_bank = verify_signatures ? (uint8_t *)xmalloc((size_t)tokens * sig_max) : NULL;
    size_t *sig_lens = verify_signatures ? (size_t *)xmalloc((size_t)tokens * sizeof(size_t)) : NULL;

    uint8_t message[MSG_SIZE];

    int ok = 0;
    int fail = 0;
    int total_attempts_ok = 0;
    int sign_success = 0;
    max_sign_attempts = 1;

    double start = get_time_seconds();
    for (int t = 0; t < tokens; t++) {
        const int8_t *s1 = sample1_bank + (size_t)t * n;
        const int8_t *s2 = sample2_bank + (size_t)t * n;
        const uint16_t *tg = target_bank + (size_t)t * n;

        fill_message(message, sizeof(message), t);

        int token_ok = 0;
        int used_attempts = 0;

        for (int a = 0; a < max_sign_attempts; a++) {
            used_attempts++;

            uint8_t nonce[40];
            derive_nonce_from_token(nonce, t);
            token_ok = sign_token_to_signature(logn,
                message, sizeof(message),
                nonce,
                s1, s2, tg,
                f_fft, g_fft, F_fft, G_fft,
                hm, target_work, sv,
                ftmp,
                verify_signatures ? (sig_bank + (size_t)t * sig_max) : NULL,
                sig_max,
                verify_signatures ? &sig_lens[t] : NULL);
            break;
        }

        if (token_ok) {
            total_attempts_ok += used_attempts;
            sign_success++;
            if (!verify_signatures) {
                ok++;
            }
        } else {
            fail++;
            if (verify_signatures) {
                sig_lens[t] = 0;
            }
        }
    }
    double end = get_time_seconds();

    *online_ms = (end - start) * 1000.0;
    *verify_ms = -1.0;
    if (verify_signatures) {
        int verify_ok = 0;
        int verify_fail = 0;
        if (verify_backend_mode == VERIFY_BACKEND_GPU_RAW) {
            run_verify_batch_gpu_raw(logn, h_monty, tokens,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else if (verify_backend_mode == VERIFY_BACKEND_GPU_FULL) {
            run_verify_batch_gpu_full(logn, h_monty, tokens,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else {
            run_verify_batch_serial(logn, tokens,
                pubkey, pubkey_len,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        }
        *online_ms += *verify_ms;
        ok += verify_ok;
        fail += verify_fail;
    }
    *avg_attempts = (sign_success > 0) ? ((double)total_attempts_ok / (double)sign_success) : 0.0;
    *ok_count = ok;
    *fail_count = fail;
    free(hm);
    free(target_work);
    free(sv);
    free(sig_bank);
    free(sig_lens);
    free(tmp_sign);
}

static void run_online_cpu_mt(
    unsigned logn,
    int tokens,
    int max_sign_attempts,
    int verify_signatures,
    int verify_backend_mode,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint16_t *h_monty,
    const int8_t *sample1_bank,
    const int8_t *sample2_bank,
    const uint16_t *target_bank,
    const fpr *f_fft,
    const fpr *g_fft,
    const fpr *F_fft,
    const fpr *G_fft,
    int online_threads,
    double *online_ms,
    double *verify_ms,
    double *avg_attempts,
    int *ok_count,
    int *fail_count)
{
    (void)max_sign_attempts;

#ifndef _OPENMP
    (void)online_threads;
    run_online_cpu(logn, tokens, 1, verify_signatures, verify_backend_mode,
        pubkey, pubkey_len,
        h_monty,
        sample1_bank, sample2_bank, target_bank,
        f_fft, g_fft, F_fft, G_fft,
        online_ms, verify_ms, avg_attempts, ok_count, fail_count);
    return;
#else
    if (online_threads <= 1) {
        run_online_cpu(logn, tokens, 1, verify_signatures, verify_backend_mode,
            pubkey, pubkey_len,
            h_monty,
            sample1_bank, sample2_bank, target_bank,
            f_fft, g_fft, F_fft, G_fft,
            online_ms, verify_ms, avg_attempts, ok_count, fail_count);
        return;
    }

    size_t n = (size_t)1 << logn;
    int ok = 0;
    int fail = 0;
    int total_attempts_ok = 0;
    int sign_success = 0;
    size_t sig_max = verify_signatures ? FALCON_SIG_COMPRESSED_MAXSIZE(logn) : 0;
    uint8_t *sig_bank = verify_signatures ? (uint8_t *)xmalloc((size_t)tokens * sig_max) : NULL;
    size_t *sig_lens = verify_signatures ? (size_t *)xmalloc((size_t)tokens * sizeof(size_t)) : NULL;

    double start = get_time_seconds();
#pragma omp parallel num_threads(online_threads)
    {
        uint16_t *hm = (uint16_t *)xmalloc(n * sizeof(uint16_t));
        uint16_t *target_work = (uint16_t *)xmalloc(n * sizeof(uint16_t));
        int16_t *sv = (int16_t *)xmalloc(n * sizeof(int16_t));
        size_t tmp_sign_len = FALCON_TMPSIZE_SIGNDYN(logn);
        uint8_t *tmp_sign = (uint8_t *)xmalloc(tmp_sign_len);
        fpr *ftmp = (fpr *)tmp_sign;
        uint8_t message[MSG_SIZE];

        int local_sign_success = 0;
        int local_fail = 0;
        int local_attempts_ok = 0;

#pragma omp for schedule(static)
        for (int t = 0; t < tokens; t++) {
            const int8_t *s1 = sample1_bank + (size_t)t * n;
            const int8_t *s2 = sample2_bank + (size_t)t * n;
            const uint16_t *tg = target_bank + (size_t)t * n;
            uint8_t nonce[40];
            int token_ok;

            fill_message(message, sizeof(message), t);
            derive_nonce_from_token(nonce, t);
            token_ok = sign_token_to_signature(logn,
                message, sizeof(message),
                nonce,
                s1, s2, tg,
                f_fft, g_fft, F_fft, G_fft,
                hm, target_work, sv,
                ftmp,
                verify_signatures ? (sig_bank + (size_t)t * sig_max) : NULL,
                sig_max,
                verify_signatures ? &sig_lens[t] : NULL);

            if (token_ok) {
                local_sign_success++;
                local_attempts_ok += 1;
            } else {
                local_fail++;
                if (verify_signatures) {
                    sig_lens[t] = 0;
                }
            }
        }

        free(hm);
        free(target_work);
        free(sv);
        free(tmp_sign);

#pragma omp atomic
        sign_success += local_sign_success;
#pragma omp atomic
        fail += local_fail;
#pragma omp atomic
        total_attempts_ok += local_attempts_ok;
    }
    double end = get_time_seconds();

    *online_ms = (end - start) * 1000.0;
    *verify_ms = -1.0;
    if (verify_signatures) {
        int verify_ok = 0;
        int verify_fail = 0;
        if (verify_backend_mode == VERIFY_BACKEND_GPU_RAW) {
            run_verify_batch_gpu_raw(logn, h_monty, tokens,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else if (verify_backend_mode == VERIFY_BACKEND_GPU_FULL) {
            run_verify_batch_gpu_full(logn, h_monty, tokens,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else {
            run_verify_batch_mt(logn, tokens,
                pubkey, pubkey_len,
                sig_bank, sig_max, sig_lens,
                online_threads,
                verify_ms, &verify_ok, &verify_fail);
        }
        *online_ms += *verify_ms;
        ok = verify_ok;
        fail += verify_fail;
    } else {
        ok = sign_success;
    }
    *avg_attempts = (sign_success > 0) ? ((double)total_attempts_ok / (double)sign_success) : 0.0;
    *ok_count = ok;
    *fail_count = fail;
    free(sig_bank);
    free(sig_lens);
#endif
}

#ifdef OOFALCON_USE_CUDA
static int run_online_cpu_pipeline(
    unsigned logn,
    int tokens_to_sign,
    int tokens_per_bank,
    int max_sign_attempts,
    int verify_signatures,
    int verify_backend_mode,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint16_t *h_monty,
    const fpr *f_fft,
    const fpr *g_fft,
    const fpr *F_fft,
    const fpr *G_fft,
    soko_falcon_token_bank_t *banks,
    const uint16_t *h,
    uint32_t seed_base,
    int threshold,
    int verify_threads,
    double *online_ms,
    double *verify_ms,
    double *avg_attempts,
    int *ok_count,
    int *fail_count,
    double *refill_wait_ms,
    int *gpu_error)
{
    soko_falcon_refill_pipeline_t *pipeline = NULL;

    if (tokens_to_sign <= tokens_per_bank) {
        *refill_wait_ms = 0.0;
        *gpu_error = 0;
        run_online_cpu_mt(logn, tokens_to_sign,
            max_sign_attempts,
            verify_signatures,
            verify_backend_mode,
            pubkey, pubkey_len,
            h_monty,
            banks[0].s1, banks[0].s2, banks[0].target,
            f_fft, g_fft, F_fft, G_fft,
            verify_threads,
            online_ms,
            verify_ms,
            avg_attempts,
            ok_count,
            fail_count);
        return 0;
    }

    size_t n = (size_t)1 << logn;
    uint16_t *hm = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    uint16_t *target_work = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    int16_t *sv = (int16_t *)xmalloc(n * sizeof(int16_t));

    size_t sig_max = verify_signatures ? FALCON_SIG_COMPRESSED_MAXSIZE(logn) : 0;
    uint8_t *sig_bank = verify_signatures ? (uint8_t *)xmalloc((size_t)tokens_to_sign * sig_max) : NULL;
    size_t *sig_lens = verify_signatures ? (size_t *)xmalloc((size_t)tokens_to_sign * sizeof(size_t)) : NULL;

    size_t tmp_sign_len = FALCON_TMPSIZE_SIGNDYN(logn);
    uint8_t *tmp_sign = (uint8_t *)xmalloc(tmp_sign_len);
    fpr *ftmp = (fpr *)tmp_sign;

    shake256_context rng_nonce;
    uint8_t seed[48];
    memset(seed, 0, sizeof(seed));
    seed[0] = 0x5A;
    shake256_init_prng_from_seed(&rng_nonce, seed, sizeof(seed));

    uint8_t message[MSG_SIZE];

    int ok = 0;
    int fail = 0;
    int total_attempts_ok = 0;
    int sign_success = 0;

    max_sign_attempts = 1;

    *gpu_error = soko_falcon_refill_pipeline_init(&pipeline,
        h, logn, seed_base, tokens_per_bank, threshold, banks, 2);
    if (*gpu_error != 0) {
        goto cleanup;
    }

    double wait_ms = 0.0;
    double start = get_time_seconds();
    for (int t = 0; t < tokens_to_sign; t++) {
        soko_falcon_token_view_t token_view;
        double token_wait_ms = 0.0;
        int pipeline_status = soko_falcon_refill_pipeline_acquire(pipeline, &token_view, &token_wait_ms);
        if (pipeline_status != 0) {
            *gpu_error = pipeline_status;
            goto cleanup;
        }
        wait_ms += token_wait_ms;

        const int8_t *s1 = token_view.s1;
        const int8_t *s2 = token_view.s2;
        const uint16_t *tg = token_view.target;

        fill_message(message, sizeof(message), t);

        int token_ok = 0;
        int used_attempts = 0;

        for (int a = 0; a < max_sign_attempts; a++) {
            used_attempts++;

            uint8_t nonce[40];
            shake256_extract(&rng_nonce, nonce, 40);
            token_ok = sign_token_to_signature(logn,
                message, sizeof(message),
                nonce,
                s1, s2, tg,
                f_fft, g_fft, F_fft, G_fft,
                hm, target_work, sv,
                ftmp,
                verify_signatures ? (sig_bank + (size_t)t * sig_max) : NULL,
                sig_max,
                verify_signatures ? &sig_lens[t] : NULL);
            break;
        }

        if (token_ok) {
            total_attempts_ok += used_attempts;
            sign_success++;
            if (!verify_signatures) {
                ok++;
            }
        } else {
            fail++;
            if (verify_signatures) {
                sig_lens[t] = 0;
            }
        }
    }
    double end = get_time_seconds();

    *online_ms = (end - start) * 1000.0;
    *verify_ms = -1.0;
    if (verify_signatures) {
        int verify_ok = 0;
        int verify_fail = 0;
        if (verify_backend_mode == VERIFY_BACKEND_GPU_RAW) {
            run_verify_batch_gpu_raw(logn, h_monty, tokens_to_sign,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else if (verify_backend_mode == VERIFY_BACKEND_GPU_FULL) {
            run_verify_batch_gpu_full(logn, h_monty, tokens_to_sign,
                sig_bank, sig_max, sig_lens,
                verify_ms, &verify_ok, &verify_fail);
        } else {
            run_verify_batch_mt(logn, tokens_to_sign,
                pubkey, pubkey_len,
                sig_bank, sig_max, sig_lens,
                verify_threads,
                verify_ms, &verify_ok, &verify_fail);
        }
        *online_ms += *verify_ms;
        ok = verify_ok;
        fail += verify_fail;
    }
    *avg_attempts = (sign_success > 0) ? ((double)total_attempts_ok / (double)sign_success) : 0.0;
    *ok_count = ok;
    *fail_count = fail;
    *refill_wait_ms = wait_ms;
    *gpu_error = 0;

cleanup:
    soko_falcon_refill_pipeline_destroy(pipeline);
    free(hm);
    free(target_work);
    free(sv);
    free(sig_bank);
    free(sig_lens);
    free(tmp_sign);
    return (*gpu_error == 0) ? 0 : -1;
}
#endif

static token_result_t run_token_step(
    unsigned logn,
    int tokens,
    double keygen_ms,
    int max_sign_attempts,
    int verify_signatures,
    int verify_backend_mode,
    const uint8_t *pubkey,
    size_t pubkey_len,
    const uint16_t *h,
    const uint16_t *h_monty,
    const fpr *f_fft,
    const fpr *g_fft,
    const fpr *F_fft,
    const fpr *G_fft,
    int cpu_mt_threads,
    int pipeline_mode)
{
    token_result_t r;
    memset(&r, 0, sizeof(r));
    r.verify_cpu_from_cpu_cold_ms = -1.0;
    r.verify_cpu_from_cpu_steady_ms = -1.0;
    r.verify_cpu_from_cpumt_cold_ms = -1.0;
    r.verify_cpu_from_cpumt_steady_ms = -1.0;
    r.verify_cpu_from_gpu_cold_ms = -1.0;
    r.verify_cpu_from_gpu_steady_ms = -1.0;
    r.gpu_error_code_cold = 0;
    r.gpu_error_code_steady = 0;
    r.cpu_mt_threads_used = resolve_cpu_mt_threads(cpu_mt_threads);
    r.verify_backend_mode = verify_backend_mode;

    size_t n = (size_t)1 << logn;
    size_t bank_i8 = (size_t)tokens * n * sizeof(int8_t);
    size_t bank_u16 = (size_t)tokens * n * sizeof(uint16_t);

    int8_t *cpu_s1_cold = (int8_t *)xmalloc(bank_i8);
    int8_t *cpu_s2_cold = (int8_t *)xmalloc(bank_i8);
    uint16_t *cpu_tg_cold = (uint16_t *)xmalloc(bank_u16);

    int8_t *cpu_s1_steady = (int8_t *)xmalloc(bank_i8);
    int8_t *cpu_s2_steady = (int8_t *)xmalloc(bank_i8);
    uint16_t *cpu_tg_steady = (uint16_t *)xmalloc(bank_u16);

    int8_t *cpumt_s1_cold = (int8_t *)xmalloc(bank_i8);
    int8_t *cpumt_s2_cold = (int8_t *)xmalloc(bank_i8);
    uint16_t *cpumt_tg_cold = (uint16_t *)xmalloc(bank_u16);

    int8_t *cpumt_s1_steady = (int8_t *)xmalloc(bank_i8);
    int8_t *cpumt_s2_steady = (int8_t *)xmalloc(bank_i8);
    uint16_t *cpumt_tg_steady = (uint16_t *)xmalloc(bank_u16);

    uint32_t bank_seed = 0xC0FFEEu + (uint32_t)tokens;

    double t0 = get_time_seconds();
    generate_bank_cpu_original(logn, tokens, h_monty, cpu_s1_cold, cpu_s2_cold, cpu_tg_cold);
    double t1 = get_time_seconds();
    r.offline_cpu_cold_ms = (t1 - t0) * 1000.0;

    t0 = get_time_seconds();
    generate_bank_cpu_original(logn, tokens, h_monty, cpu_s1_steady, cpu_s2_steady, cpu_tg_steady);
    t1 = get_time_seconds();
    r.offline_cpu_steady_ms = (t1 - t0) * 1000.0;

    /* Add one-time keygen cost so offline figures include keygen amortized over tokens. */
    r.offline_cpu_cold_ms += keygen_ms;
    r.offline_cpu_steady_ms += keygen_ms;

    t0 = get_time_seconds();
    generate_bank_cpu_mt(logn, tokens, h_monty,
        cpumt_s1_cold, cpumt_s2_cold, cpumt_tg_cold,
        r.cpu_mt_threads_used);
    t1 = get_time_seconds();
    r.offline_cpumt_cold_ms = (t1 - t0) * 1000.0;

    t0 = get_time_seconds();
    generate_bank_cpu_mt(logn, tokens, h_monty,
        cpumt_s1_steady, cpumt_s2_steady, cpumt_tg_steady,
        r.cpu_mt_threads_used);
    t1 = get_time_seconds();
    r.offline_cpumt_steady_ms = (t1 - t0) * 1000.0;

    r.offline_cpumt_cold_ms += keygen_ms;
    r.offline_cpumt_steady_ms += keygen_ms;

    run_online_cpu(logn, tokens,
        max_sign_attempts,
        verify_signatures,
        verify_backend_mode,
        pubkey, pubkey_len,
        h_monty,
        cpu_s1_cold, cpu_s2_cold, cpu_tg_cold,
        f_fft, g_fft, F_fft, G_fft,
        &r.online_cpu_from_cpu_cold_ms,
        &r.verify_cpu_from_cpu_cold_ms,
        &r.avg_attempts_cpu_cold,
        &r.ok_cpu_cold,
        &r.fail_cpu_cold);

    run_online_cpu(logn, tokens,
        max_sign_attempts,
        verify_signatures,
        verify_backend_mode,
        pubkey, pubkey_len,
        h_monty,
        cpu_s1_steady, cpu_s2_steady, cpu_tg_steady,
        f_fft, g_fft, F_fft, G_fft,
        &r.online_cpu_from_cpu_steady_ms,
        &r.verify_cpu_from_cpu_steady_ms,
        &r.avg_attempts_cpu_steady,
        &r.ok_cpu_steady,
        &r.fail_cpu_steady);

    run_online_cpu_mt(logn, tokens,
        max_sign_attempts,
        verify_signatures,
        verify_backend_mode,
        pubkey, pubkey_len,
        h_monty,
        cpumt_s1_cold, cpumt_s2_cold, cpumt_tg_cold,
        f_fft, g_fft, F_fft, G_fft,
        r.cpu_mt_threads_used,
        &r.online_cpu_from_cpumt_cold_ms,
        &r.verify_cpu_from_cpumt_cold_ms,
        &r.avg_attempts_cpumt_cold,
        &r.ok_cpumt_cold,
        &r.fail_cpumt_cold);

    run_online_cpu_mt(logn, tokens,
        max_sign_attempts,
        verify_signatures,
        verify_backend_mode,
        pubkey, pubkey_len,
        h_monty,
        cpumt_s1_steady, cpumt_s2_steady, cpumt_tg_steady,
        f_fft, g_fft, F_fft, G_fft,
        r.cpu_mt_threads_used,
        &r.online_cpu_from_cpumt_steady_ms,
        &r.verify_cpu_from_cpumt_steady_ms,
        &r.avg_attempts_cpumt_steady,
        &r.ok_cpumt_steady,
        &r.fail_cpumt_steady);

#ifdef OOFALCON_USE_CUDA
    int8_t *gpu_s1_cold = (int8_t *)xmalloc(bank_i8);
    int8_t *gpu_s2_cold = (int8_t *)xmalloc(bank_i8);
    uint16_t *gpu_tg_cold = (uint16_t *)xmalloc(bank_u16);

    int8_t *gpu_s1_steady = (int8_t *)xmalloc(bank_i8);
    int8_t *gpu_s2_steady = (int8_t *)xmalloc(bank_i8);
    uint16_t *gpu_tg_steady = (uint16_t *)xmalloc(bank_u16);

    int gr_cold;
    int gr_steady;
#ifdef _WIN32
    int rr = gpu_reset_device_dynamic();
    if (rr == 0) {
        t0 = get_time_seconds();
        gr_cold = gpu_generate_presign_bank_dynamic(h, logn, bank_seed,
            tokens, gpu_s1_cold, gpu_s2_cold, gpu_tg_cold);
        t1 = get_time_seconds();
        r.offline_gpu_cold_ms = (t1 - t0) * 1000.0 + keygen_ms;
    } else {
        gr_cold = rr;
        r.offline_gpu_cold_ms = -1.0;
    }

    t0 = get_time_seconds();
    gr_steady = gpu_generate_presign_bank_dynamic(h, logn, bank_seed,
        tokens, gpu_s1_steady, gpu_s2_steady, gpu_tg_steady);
    t1 = get_time_seconds();
    r.offline_gpu_steady_ms = (gr_steady == 0) ? ((t1 - t0) * 1000.0 + keygen_ms) : -1.0;
#else
    gr_cold = -103;
    gr_steady = -103;
#endif

    if (gr_cold == 0) {
        r.gpu_available_cold = 1;
        run_online_cpu_mt(logn, tokens,
            max_sign_attempts,
            verify_signatures,
            verify_backend_mode,
            pubkey, pubkey_len,
            h_monty,
            gpu_s1_cold, gpu_s2_cold, gpu_tg_cold,
            f_fft, g_fft, F_fft, G_fft,
            r.cpu_mt_threads_used,
            &r.online_cpu_from_gpu_cold_ms,
            &r.verify_cpu_from_gpu_cold_ms,
            &r.avg_attempts_gpu_cold,
            &r.ok_gpu_cold,
            &r.fail_gpu_cold);
    } else {
        r.gpu_available_cold = 0;
        r.gpu_error_code_cold = gr_cold;
        r.offline_gpu_cold_ms = -1.0;
        r.online_cpu_from_gpu_cold_ms = -1.0;
        r.avg_attempts_gpu_cold = -1.0;
    }

    if (gr_steady == 0) {
        r.gpu_available_steady = 1;
        if (pipeline_mode) {
            double refill_wait_ms = 0.0;
            int gpu_err = 0;
            double t_sign_ms = 0.0;
            if (r.online_cpu_from_cpu_steady_ms > 0.0 && tokens > 0) {
                t_sign_ms = r.online_cpu_from_cpu_steady_ms / (double)tokens;
            } else if (r.online_cpu_from_cpu_cold_ms > 0.0 && tokens > 0) {
                t_sign_ms = r.online_cpu_from_cpu_cold_ms / (double)tokens;
            }
            int threshold = tokens;
            if (t_sign_ms > 0.0) {
                double ratio = 1.2 * r.offline_gpu_steady_ms / t_sign_ms;
                if (ratio < 1.0) {
                    ratio = 1.0;
                }
                threshold = (int)ceil(ratio);
            }
            if (threshold > tokens * 2) {
                threshold = tokens * 2;
            }
            if (threshold < 1) {
                threshold = 1;
            }

            soko_falcon_token_bank_t banks[2];
            banks[0].s1 = gpu_s1_steady;
            banks[0].s2 = gpu_s2_steady;
            banks[0].target = gpu_tg_steady;
            banks[0].available = 0;
            banks[1].s1 = gpu_s1_cold;
            banks[1].s2 = gpu_s2_cold;
            banks[1].target = gpu_tg_cold;
            banks[1].available = 1;

            int pipeline_ok = run_online_cpu_pipeline(logn,
                tokens,
                tokens,
                max_sign_attempts,
                verify_signatures,
                verify_backend_mode,
                pubkey, pubkey_len,
                h_monty,
                f_fft, g_fft, F_fft, G_fft,
                banks,
                h,
                bank_seed + 1u,
                threshold,
                r.cpu_mt_threads_used,
                &r.online_cpu_from_gpu_steady_ms,
                &r.verify_cpu_from_gpu_steady_ms,
                &r.avg_attempts_gpu_steady,
                &r.ok_gpu_steady,
                &r.fail_gpu_steady,
                &refill_wait_ms,
                &gpu_err);

            if (pipeline_ok != 0) {
                r.gpu_available_steady = 0;
                r.gpu_error_code_steady = gpu_err;
                r.offline_gpu_steady_ms = -1.0;
                r.online_cpu_from_gpu_steady_ms = -1.0;
                r.avg_attempts_gpu_steady = -1.0;
            }
        } else {
            run_online_cpu_mt(logn, tokens,
                max_sign_attempts,
                verify_signatures,
                verify_backend_mode,
                pubkey, pubkey_len,
                h_monty,
                gpu_s1_steady, gpu_s2_steady, gpu_tg_steady,
                f_fft, g_fft, F_fft, G_fft,
                r.cpu_mt_threads_used,
                &r.online_cpu_from_gpu_steady_ms,
                &r.verify_cpu_from_gpu_steady_ms,
                &r.avg_attempts_gpu_steady,
                &r.ok_gpu_steady,
                &r.fail_gpu_steady);
        }
    } else {
        r.gpu_available_steady = 0;
        r.gpu_error_code_steady = gr_steady;
        r.offline_gpu_steady_ms = -1.0;
        r.online_cpu_from_gpu_steady_ms = -1.0;
        r.avg_attempts_gpu_steady = -1.0;
    }

    free(gpu_s1_cold);
    free(gpu_s2_cold);
    free(gpu_tg_cold);
    free(gpu_s1_steady);
    free(gpu_s2_steady);
    free(gpu_tg_steady);
#else
    (void)h;
    r.gpu_available_cold = 0;
    r.gpu_available_steady = 0;
    r.gpu_error_code_cold = -103;
    r.gpu_error_code_steady = -103;
    r.offline_gpu_cold_ms = -1.0;
    r.offline_gpu_steady_ms = -1.0;
    r.online_cpu_from_gpu_cold_ms = -1.0;
    r.online_cpu_from_gpu_steady_ms = -1.0;
    r.avg_attempts_gpu_cold = -1.0;
    r.avg_attempts_gpu_steady = -1.0;
#endif

    free(cpu_s1_cold);
    free(cpu_s2_cold);
    free(cpu_tg_cold);
    free(cpu_s1_steady);
    free(cpu_s2_steady);
    free(cpu_tg_steady);
    free(cpumt_s1_cold);
    free(cpumt_s2_cold);
    free(cpumt_tg_cold);
    free(cpumt_s1_steady);
    free(cpumt_s2_steady);
    free(cpumt_tg_steady);

    return r;
}

int main(int argc, char **argv)
{
    bench_config_t cfg;

    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.verify_backend_mode != VERIFY_BACKEND_CPU && !cfg.verify_signatures) {
        fprintf(stderr, "GPU verification backends require verification to be enabled. Remove --no-verify.\n");
        return 1;
    }

#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    if (cfg.verify_backend_mode != VERIFY_BACKEND_CPU) {
        fprintf(stderr, "GPU verify backend requested, but this benchmark was built without CUDA support.\n");
        return 1;
    }
#endif

    const unsigned logn = LOGN;
    const size_t n = (size_t)1 << logn;

    const int token_steps[DEFAULT_TOKEN_STEPS] = {8, 128, 1024, 8192, 65536, 131072};

    size_t privkey_len = FALCON_PRIVKEY_SIZE(logn);
    size_t pubkey_len = FALCON_PUBKEY_SIZE(logn);
    size_t tmp_len = FALCON_TMPSIZE_KEYGEN(logn);
    if (tmp_len < FALCON_TMPSIZE_SIGNDYN(logn)) {
        tmp_len = FALCON_TMPSIZE_SIGNDYN(logn);
    }

    uint8_t *privkey = (uint8_t *)xmalloc(privkey_len);
    uint8_t *pubkey = (uint8_t *)xmalloc(pubkey_len);
    uint8_t *tmp = (uint8_t *)xmalloc(tmp_len);

    int8_t *f = (int8_t *)xmalloc(n);
    int8_t *g = (int8_t *)xmalloc(n);
    int8_t *F = (int8_t *)xmalloc(n);
    int8_t *G = (int8_t *)xmalloc(n);

    uint16_t *h = (uint16_t *)xmalloc(n * sizeof(uint16_t));
    uint16_t *h_monty = (uint16_t *)xmalloc(n * sizeof(uint16_t));

    fpr *f_fft = (fpr *)xmalloc(n * sizeof(fpr));
    fpr *g_fft = (fpr *)xmalloc(n * sizeof(fpr));
    fpr *F_fft = (fpr *)xmalloc(n * sizeof(fpr));
    fpr *G_fft = (fpr *)xmalloc(n * sizeof(fpr));

    double keygen_start = get_time_seconds();
    int prep = prepare_key_and_state(logn,
        privkey, privkey_len,
        pubkey, pubkey_len,
        tmp, tmp_len,
        f, g, F, G,
        h,
        h_monty,
        f_fft, g_fft, F_fft, G_fft);
    double keygen_end = get_time_seconds();
    double keygen_ms = (keygen_end - keygen_start) * 1000.0;

    if (prep != 0) {
        fprintf(stderr, "Failed to prepare key state (err=%d)\n", prep);
        return 1;
    }

    FILE *csv = NULL;
    if (cfg.csv_path != NULL) {
        csv = fopen(cfg.csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "Failed to open CSV output: %s\n", cfg.csv_path);
            return 1;
        }
        write_csv_header(csv);
    }

    printf("===============================================================\n");
    printf("OO-FALCON HYBRID OFFLINE BENCHMARK (GPU offline, CPU online)\n");
    printf("Degree: Falcon-%u   (logn=%u, n=%lu)\n", (unsigned)(1u << logn), logn, (unsigned long)n);
#ifdef OOFALCON_USE_CUDA
    printf("GPU path: ENABLED\n");
#else
    printf("GPU path: DISABLED (binary built without CUDA)\n");
#endif
    printf("Max sign attempts/token: %d\n", cfg.max_sign_attempts);
    if (cfg.cpu_mt_threads > 0) {
        printf("CPU-MT threads: %d (configured)\n", cfg.cpu_mt_threads);
    } else {
        printf("CPU-MT threads: auto\n");
    }
    printf("One-time keygen: %.3f ms (amortized into offline metrics)\n", keygen_ms);
    if (csv != NULL) {
        printf("CSV output: %s\n", cfg.csv_path);
    }
    printf("Pipeline A: CPU-1T generated offline bank -> CPU online signing\n");
    printf("Pipeline B: CPU-MT generated offline bank -> CPU online signing\n");
    printf("Pipeline C: GPU generated offline bank -> CPU online signing\n");
    printf("Fairness mode: all pipelines generate the same number of presigns (tokens).\n");
    printf("Verification mode: %s.\n", cfg.verify_signatures ? "enabled" : "disabled");
    if (cfg.verify_signatures) {
        printf("Verification backend: %s.\n", verify_backend_name(cfg.verify_backend_mode));
    }
    printf("Online mode: single attempt (no rejection sampling).\n");
    if (cfg.pipeline_mode) {
        printf("GPU pipeline: ring-buffer + async refill enabled for steady GPU online stage.\n");
    }
    printf("Cold mode: each token level enforces a fresh GPU context reset before cold offline timing.\n");
    printf("Steady mode: immediate second run at same token level (no GPU reset).\n");
    printf("Token sets: 8, 128, 1024, 8192, 65536, 131072\n");
    printf("===============================================================\n\n");

    printf("%-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s\n",
        "Tokens",
        "CPU1T cold",
        "CPU1T steady",
        "CPUMT cold",
        "CPUMT steady",
        "GPU cold",
        "GPU steady");
    printf("---------------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < DEFAULT_TOKEN_STEPS; i++) {
        int tokens = token_steps[i];
        token_result_t r = run_token_step(logn, tokens,
            keygen_ms,
            cfg.max_sign_attempts,
            cfg.verify_signatures,
            cfg.verify_backend_mode,
            pubkey, pubkey_len,
            h,
            h_monty,
            f_fft, g_fft, F_fft, G_fft,
            cfg.cpu_mt_threads,
            cfg.pipeline_mode);

        if (csv != NULL) {
            write_csv_row(csv, tokens, &r);
        }

        if (r.gpu_available_cold || r.gpu_available_steady) {
            printf("%-8d | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12.3f\n",
                tokens,
                r.offline_cpu_cold_ms,
                r.offline_cpu_steady_ms,
                r.offline_cpumt_cold_ms,
                r.offline_cpumt_steady_ms,
                r.offline_gpu_cold_ms,
                r.offline_gpu_steady_ms);

            printf("          CPU-MT threads used: %d\n", r.cpu_mt_threads_used);
            printf("          Offline amortized (us/token): CPU1T cold=%.3f steady=%.3f | CPUMT cold=%.3f steady=%.3f",
                (r.offline_cpu_cold_ms * 1000.0) / (double)tokens,
                (r.offline_cpu_steady_ms * 1000.0) / (double)tokens,
                (r.offline_cpumt_cold_ms * 1000.0) / (double)tokens,
                (r.offline_cpumt_steady_ms * 1000.0) / (double)tokens);
            if (r.gpu_available_cold || r.gpu_available_steady) {
                printf(" | GPU cold=%.3f steady=%.3f\n",
                    (r.offline_gpu_cold_ms * 1000.0) / (double)tokens,
                    (r.offline_gpu_steady_ms * 1000.0) / (double)tokens);
            } else {
                printf("\n");
            }

            if (cfg.verify_signatures) {
                printf("          Verify cold:   CPU1T ok=%d fail=%d | CPUMT ok=%d fail=%d | GPU ok=%d fail=%d\n",
                    r.ok_cpu_cold, r.fail_cpu_cold,
                    r.ok_cpumt_cold, r.fail_cpumt_cold,
                    r.ok_gpu_cold, r.fail_gpu_cold);
                printf("          Verify steady: CPU1T ok=%d fail=%d | CPUMT ok=%d fail=%d | GPU ok=%d fail=%d\n",
                    r.ok_cpu_steady, r.fail_cpu_steady,
                    r.ok_cpumt_steady, r.fail_cpumt_steady,
                    r.ok_gpu_steady, r.fail_gpu_steady);
            } else {
                printf("          Sign cold:     CPU1T ok=%d fail=%d | CPUMT ok=%d fail=%d | GPU ok=%d fail=%d\n",
                    r.ok_cpu_cold, r.fail_cpu_cold,
                    r.ok_cpumt_cold, r.fail_cpumt_cold,
                    r.ok_gpu_cold, r.fail_gpu_cold);
                printf("          Sign steady:   CPU1T ok=%d fail=%d | CPUMT ok=%d fail=%d | GPU ok=%d fail=%d\n",
                    r.ok_cpu_steady, r.fail_cpu_steady,
                    r.ok_cpumt_steady, r.fail_cpumt_steady,
                    r.ok_gpu_steady, r.fail_gpu_steady);
            }
            printf("          Avg attempts/token: CPU1T cold=%.3f steady=%.3f | CPUMT cold=%.3f steady=%.3f | GPU cold=%.3f steady=%.3f\n",
                r.avg_attempts_cpu_cold,
                r.avg_attempts_cpu_steady,
                r.avg_attempts_cpumt_cold,
                r.avg_attempts_cpumt_steady,
                r.avg_attempts_gpu_cold,
                r.avg_attempts_gpu_steady);

            if (cfg.verify_signatures) {
                double cpu_sign_cold_ms = r.online_cpu_from_cpu_cold_ms - r.verify_cpu_from_cpu_cold_ms;
                double cpu_sign_steady_ms = r.online_cpu_from_cpu_steady_ms - r.verify_cpu_from_cpu_steady_ms;
                double cpumt_sign_cold_ms = r.online_cpu_from_cpumt_cold_ms - r.verify_cpu_from_cpumt_cold_ms;
                double cpumt_sign_steady_ms = r.online_cpu_from_cpumt_steady_ms - r.verify_cpu_from_cpumt_steady_ms;
                double gpu_sign_cold_ms = r.online_cpu_from_gpu_cold_ms - r.verify_cpu_from_gpu_cold_ms;
                double gpu_sign_steady_ms = r.online_cpu_from_gpu_steady_ms - r.verify_cpu_from_gpu_steady_ms;

                printf("          Throughput cold (sig/s): CPU1T sign=%.0f verify=%.0f sign+verify=%.0f | CPUMT sign=%.0f verify=%.0f sign+verify=%.0f | GPU sign=%.0f verify=%.0f sign+verify=%.0f\n",
                    throughput_from_ms(tokens, cpu_sign_cold_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpu_cold_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpu_cold_ms),
                    throughput_from_ms(tokens, cpumt_sign_cold_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpumt_cold_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpumt_cold_ms),
                    throughput_from_ms(tokens, gpu_sign_cold_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_gpu_cold_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_gpu_cold_ms));
                printf("          Throughput steady (sig/s): CPU1T sign=%.0f verify=%.0f sign+verify=%.0f | CPUMT sign=%.0f verify=%.0f sign+verify=%.0f | GPU sign=%.0f verify=%.0f sign+verify=%.0f\n",
                    throughput_from_ms(tokens, cpu_sign_steady_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpu_steady_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpu_steady_ms),
                    throughput_from_ms(tokens, cpumt_sign_steady_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpumt_steady_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpumt_steady_ms),
                    throughput_from_ms(tokens, gpu_sign_steady_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_gpu_steady_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_gpu_steady_ms));
            }

            {
                double cpu_total_cold = r.offline_cpu_cold_ms + r.online_cpu_from_cpu_cold_ms;
                double cpu_total_steady = r.offline_cpu_steady_ms + r.online_cpu_from_cpu_steady_ms;
                double cpumt_total_cold = r.offline_cpumt_cold_ms + r.online_cpu_from_cpumt_cold_ms;
                double cpumt_total_steady = r.offline_cpumt_steady_ms + r.online_cpu_from_cpumt_steady_ms;
                double gpu_total_cold = r.offline_gpu_cold_ms + r.online_cpu_from_gpu_cold_ms;
                double gpu_total_steady = r.offline_gpu_steady_ms + r.online_cpu_from_gpu_steady_ms;

                printf("          Cold total: CPU1T=%.3f ms | CPUMT=%.3f ms",
                    cpu_total_cold, cpumt_total_cold);
                if (cpumt_total_cold > 0.0) {
                    printf(" | CPU1T->CPUMT speedup=%.3fx", cpu_total_cold / cpumt_total_cold);
                }
                if (r.gpu_available_cold && gpu_total_cold > 0.0) {
                    printf(" | GPU=%.3f ms | CPU1T->GPU=%.3fx | CPUMT->GPU=%.3fx\n",
                        gpu_total_cold,
                        cpu_total_cold / gpu_total_cold,
                        cpumt_total_cold / gpu_total_cold);
                } else {
                    printf(" | GPU unavailable (err=%d)\n", r.gpu_error_code_cold);
                }

                printf("          Steady total: CPU1T=%.3f ms | CPUMT=%.3f ms",
                    cpu_total_steady, cpumt_total_steady);
                if (cpumt_total_steady > 0.0) {
                    printf(" | CPU1T->CPUMT speedup=%.3fx", cpu_total_steady / cpumt_total_steady);
                }
                if (r.gpu_available_steady && gpu_total_steady > 0.0) {
                    printf(" | GPU=%.3f ms | CPU1T->GPU=%.3fx | CPUMT->GPU=%.3fx\n",
                        gpu_total_steady,
                        cpu_total_steady / gpu_total_steady,
                        cpumt_total_steady / gpu_total_steady);
                } else {
                    printf(" | GPU unavailable (err=%d)\n", r.gpu_error_code_steady);
                }
            }
        } else {
            printf("%-8d | %-12.3f | %-12.3f | %-12.3f | %-12.3f | %-12s | %-12s\n",
                tokens,
                r.offline_cpu_cold_ms,
                r.offline_cpu_steady_ms,
                r.offline_cpumt_cold_ms,
                r.offline_cpumt_steady_ms,
                "N/A",
                "N/A");
            printf("          CPU-MT threads used: %d\n", r.cpu_mt_threads_used);
            printf("          GPU unavailable: cold err=%d, steady err=%d\n",
                r.gpu_error_code_cold, r.gpu_error_code_steady);
            printf("          Offline amortized (us/token): CPU1T cold=%.3f steady=%.3f | CPUMT cold=%.3f steady=%.3f\n",
                (r.offline_cpu_cold_ms * 1000.0) / (double)tokens,
                (r.offline_cpu_steady_ms * 1000.0) / (double)tokens,
                (r.offline_cpumt_cold_ms * 1000.0) / (double)tokens,
                (r.offline_cpumt_steady_ms * 1000.0) / (double)tokens);
            if (cfg.verify_signatures) {
                printf("          Verify: CPU1T cold ok=%d fail=%d | CPU1T steady ok=%d fail=%d\n",
                    r.ok_cpu_cold, r.fail_cpu_cold, r.ok_cpu_steady, r.fail_cpu_steady);
                printf("                  CPUMT cold ok=%d fail=%d | CPUMT steady ok=%d fail=%d\n",
                    r.ok_cpumt_cold, r.fail_cpumt_cold, r.ok_cpumt_steady, r.fail_cpumt_steady);
            } else {
                printf("          Sign:   CPU1T cold ok=%d fail=%d | CPU1T steady ok=%d fail=%d\n",
                    r.ok_cpu_cold, r.fail_cpu_cold, r.ok_cpu_steady, r.fail_cpu_steady);
                printf("                  CPUMT cold ok=%d fail=%d | CPUMT steady ok=%d fail=%d\n",
                    r.ok_cpumt_cold, r.fail_cpumt_cold, r.ok_cpumt_steady, r.fail_cpumt_steady);
            }
            printf("          Avg attempts/token: CPU1T cold=%.3f steady=%.3f | CPUMT cold=%.3f steady=%.3f\n",
                r.avg_attempts_cpu_cold,
                r.avg_attempts_cpu_steady,
                r.avg_attempts_cpumt_cold,
                r.avg_attempts_cpumt_steady);
            if (cfg.verify_signatures) {
                double cpu_sign_cold_ms = r.online_cpu_from_cpu_cold_ms - r.verify_cpu_from_cpu_cold_ms;
                double cpu_sign_steady_ms = r.online_cpu_from_cpu_steady_ms - r.verify_cpu_from_cpu_steady_ms;
                double cpumt_sign_cold_ms = r.online_cpu_from_cpumt_cold_ms - r.verify_cpu_from_cpumt_cold_ms;
                double cpumt_sign_steady_ms = r.online_cpu_from_cpumt_steady_ms - r.verify_cpu_from_cpumt_steady_ms;

                printf("          Throughput cold (sig/s): CPU1T sign=%.0f verify=%.0f sign+verify=%.0f | CPUMT sign=%.0f verify=%.0f sign+verify=%.0f\n",
                    throughput_from_ms(tokens, cpu_sign_cold_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpu_cold_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpu_cold_ms),
                    throughput_from_ms(tokens, cpumt_sign_cold_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpumt_cold_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpumt_cold_ms));
                printf("          Throughput steady (sig/s): CPU1T sign=%.0f verify=%.0f sign+verify=%.0f | CPUMT sign=%.0f verify=%.0f sign+verify=%.0f\n",
                    throughput_from_ms(tokens, cpu_sign_steady_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpu_steady_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpu_steady_ms),
                    throughput_from_ms(tokens, cpumt_sign_steady_ms),
                    throughput_from_ms(tokens, r.verify_cpu_from_cpumt_steady_ms),
                    throughput_from_ms(tokens, r.online_cpu_from_cpumt_steady_ms));
            }
        }
    }

    if (csv != NULL) {
        fclose(csv);
        printf("\nCSV written.\n");
    }

    printf("\nDone.\n");

    free(privkey);
    free(pubkey);
    free(tmp);
    free(f);
    free(g);
    free(F);
    free(G);
    free(h);
    free(h_monty);
    free(f_fft);
    free(g_fft);
    free(F_fft);
    free(G_fft);

    return 0;
}
