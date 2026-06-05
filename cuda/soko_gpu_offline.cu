#include "soko_gpu_offline.h"

#ifndef __CUDACC__

int oofalcon_gpu_generate_presign_bank(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    (void)h_coeffs;
    (void)logn;
    (void)seed;
    (void)tokens;
    (void)sample1_bank;
    (void)sample2_bank;
    (void)target_bank;
    return 99;
}

int oofalcon_gpu_reset_device(void)
{
    return 99;
}

int oofalcon_gpu_verify_raw_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank)
{
    (void)h_ntt_monty;
    (void)logn;
    (void)tokens;
    (void)hm_bank;
    (void)s2_bank;
    (void)result_bank;
    return 99;
}

int oofalcon_gpu_verify_compressed_benchmark_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank)
{
    (void)h_ntt_monty;
    (void)logn;
    (void)tokens;
    (void)sig_bank;
    (void)sig_stride;
    (void)sig_lens;
    (void)result_bank;
    return 99;
}

#else

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#define OOFALCON_Q 12289
#define OOFALCON_Q0I 12287
#define OOFALCON_R 4091
#define OOFALCON_R2 10952
#define OOFALCON_G 7
#define OOFALCON_GINV 8778
#define OOFALCON_GMB_SIZE 1024
#define OOFALCON_VERIFY_BENCH_MSG_SIZE 64

static uint32_t mod_pow_u32(uint32_t a, uint32_t e)
{
    uint64_t r = 1;
    uint64_t b = a % OOFALCON_Q;
    while (e > 0) {
        if (e & 1U) {
            r = (r * b) % OOFALCON_Q;
        }
        b = (b * b) % OOFALCON_Q;
        e >>= 1U;
    }
    return (uint32_t)r;
}

static uint32_t mod_inv_u32(uint32_t a)
{
    return mod_pow_u32(a, (uint32_t)(OOFALCON_Q - 2U));
}

static inline uint32_t mq_add_u32(uint32_t x, uint32_t y)
{
    uint32_t d;

    d = x + y - OOFALCON_Q;
    d += OOFALCON_Q & -(d >> 31);
    return d;
}

static inline uint32_t mq_sub_u32(uint32_t x, uint32_t y)
{
    uint32_t d;

    d = x - y;
    d += OOFALCON_Q & -(d >> 31);
    return d;
}

static inline uint32_t mq_rshift1_u32(uint32_t x)
{
    x += OOFALCON_Q & -(x & 1U);
    return (x >> 1);
}

static inline uint32_t mq_montymul_u32(uint32_t x, uint32_t y)
{
    uint32_t z, w;

    z = x * y;
    w = ((z * OOFALCON_Q0I) & 0xFFFFU) * OOFALCON_Q;
    z = (z + w) >> 16;
    z -= OOFALCON_Q;
    z += OOFALCON_Q & -(z >> 31);
    return z;
}

static uint32_t bitrev_u32(uint32_t x, int bits)
{
    uint32_t r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1U);
        x >>= 1U;
    }
    return r;
}

static uint64_t hash_h_coeffs(const uint16_t *h, int n)
{
    uint64_t hsh = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        hsh ^= (uint64_t)h[i];
        hsh *= 1099511628211ULL;
    }
    return hsh;
}

typedef struct {
    int logn;
    int n;
    uint32_t n_inv;
    uint16_t *gmb;
    uint16_t *igmb;
} ntt_tables_host;

static ntt_tables_host g_ntt_host = {0};
static uint16_t *g_d_gmb = NULL;
static uint16_t *g_d_igmb = NULL;
static uint16_t *g_d_h_hat = NULL;
static uint16_t *g_d_h_verify = NULL;
static uint64_t g_h_hash = 0;
static uint64_t g_h_verify_hash = 0;
static int g_cached_logn = 0;
static int g_cached_verify_logn = 0;
static cudaStream_t g_stream = NULL;
static cudaEvent_t g_kernel_done = NULL;

static int ensure_ntt_tables(int logn)
{
    int n = 1 << logn;

    if (g_ntt_host.gmb == NULL || g_ntt_host.igmb == NULL) {
        uint16_t *gmb = (uint16_t *)malloc(OOFALCON_GMB_SIZE * sizeof(uint16_t));
        uint16_t *igmb = (uint16_t *)malloc(OOFALCON_GMB_SIZE * sizeof(uint16_t));
        if (gmb == NULL || igmb == NULL) {
            free(gmb);
            free(igmb);
            return -1;
        }
        for (uint32_t i = 0; i < OOFALCON_GMB_SIZE; i++) {
            uint32_t e = bitrev_u32(i, 10);
            uint32_t v = mod_pow_u32(OOFALCON_G, e);
            uint32_t vinv = mod_pow_u32(OOFALCON_GINV, e);
            gmb[i] = (uint16_t)(((uint64_t)v * OOFALCON_R) % OOFALCON_Q);
            igmb[i] = (uint16_t)(((uint64_t)vinv * OOFALCON_R) % OOFALCON_Q);
        }
        free(g_ntt_host.gmb);
        free(g_ntt_host.igmb);
        g_ntt_host.gmb = gmb;
        g_ntt_host.igmb = igmb;
    }

    uint32_t n_inv = OOFALCON_R;
    for (int m = n; m > 1; m >>= 1) {
        n_inv = mq_rshift1_u32(n_inv);
    }

    g_ntt_host.logn = logn;
    g_ntt_host.n = n;
    g_ntt_host.n_inv = n_inv;
    g_cached_logn = 0;
    return 0;
}

static int ensure_device_tables(int logn)
{
    if (ensure_ntt_tables(logn) != 0) {
        return -1;
    }
    if (g_d_gmb != NULL && g_d_igmb != NULL) {
        return 0;
    }

    size_t bytes = OOFALCON_GMB_SIZE * sizeof(uint16_t);

    if (g_d_gmb == NULL) {
        cudaMalloc((void **)&g_d_gmb, bytes);
    }
    if (g_d_igmb == NULL) {
        cudaMalloc((void **)&g_d_igmb, bytes);
    }

    cudaMemcpy(g_d_gmb, g_ntt_host.gmb, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_igmb, g_ntt_host.igmb, bytes, cudaMemcpyHostToDevice);

    return 0;
}

static void mq_ntt_host(uint16_t *a, int logn)
{
    size_t n = (size_t)1 << logn;
    size_t t = n;
    for (size_t m = 1; m < n; m <<= 1) {
        size_t ht = t >> 1;
        for (size_t i = 0, j1 = 0; i < m; i++, j1 += t) {
            size_t j2 = j1 + ht;
            uint32_t s = g_ntt_host.gmb[m + i];
            for (size_t j = j1; j < j2; j++) {
                uint32_t u = a[j];
                uint32_t v = mq_montymul_u32(a[j + ht], s);
                a[j] = (uint16_t)mq_add_u32(u, v);
                a[j + ht] = (uint16_t)mq_sub_u32(u, v);
            }
        }
        t = ht;
    }
}

static void mq_poly_tomonty_host(uint16_t *f, int logn)
{
    size_t n = (size_t)1 << logn;
    for (size_t u = 0; u < n; u++) {
        f[u] = (uint16_t)mq_montymul_u32(f[u], OOFALCON_R2);
    }
}

static int ensure_h_hat(const uint16_t *h_coeffs, int logn)
{
    int n = 1 << logn;
    uint64_t h_hash = hash_h_coeffs(h_coeffs, n);

    if (g_d_h_hat == NULL) {
        cudaMalloc((void **)&g_d_h_hat, (size_t)n * sizeof(uint16_t));
        g_h_hash = 0;
    }

    if (g_h_hash == h_hash && g_cached_logn == logn) {
        return 0;
    }

    uint16_t *h_hat = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    if (h_hat == NULL) {
        return -1;
    }
    memcpy(h_hat, h_coeffs, (size_t)n * sizeof(uint16_t));
    mq_ntt_host(h_hat, logn);
    mq_poly_tomonty_host(h_hat, logn);

    cudaMemcpy(g_d_h_hat, h_hat, (size_t)n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    g_h_hash = h_hash;
    g_cached_logn = logn;

    free(h_hat);
    return 0;
}

static int ensure_h_verify(const uint16_t *h_ntt_monty, int logn)
{
    int n = 1 << logn;
    uint64_t h_hash = hash_h_coeffs(h_ntt_monty, n);

    if (g_d_h_verify == NULL) {
        cudaMalloc((void **)&g_d_h_verify, (size_t)n * sizeof(uint16_t));
        g_h_verify_hash = 0;
    }

    if (g_h_verify_hash == h_hash && g_cached_verify_logn == logn) {
        return 0;
    }

    cudaMemcpy(g_d_h_verify, h_ntt_monty,
        (size_t)n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    g_h_verify_hash = h_hash;
    g_cached_verify_logn = logn;
    return 0;
}

static __device__ __forceinline__ uint32_t verify_l2bound_device(int logn)
{
    switch (logn) {
    case 1: return 101498u;
    case 2: return 208714u;
    case 3: return 428865u;
    case 4: return 892039u;
    case 5: return 1852696u;
    case 6: return 3842630u;
    case 7: return 7959734u;
    case 8: return 16468416u;
    case 9: return 34034726u;
    case 10: return 70265242u;
    default: return 0u;
    }
}

__device__ __constant__ uint64_t keccakf_rndc_device[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __constant__ int keccakf_rotc_device[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __constant__ int keccakf_piln_device[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

typedef struct {
    uint64_t st[25];
    int dptr;
} shake256_device_ctx;

static __device__ __forceinline__ uint64_t rotl64_device(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

static __device__ void keccakf1600_device(uint64_t st[25])
{
    uint64_t bc[5];

    for (int round = 0; round < 24; round++) {
        uint64_t t;

        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64_device(bc[(i + 1) % 5], 1);
            st[i] ^= t;
            st[i + 5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln_device[i];
            uint64_t tmp = st[j];
            st[j] = rotl64_device(t, keccakf_rotc_device[i]);
            t = tmp;
        }

        for (int j = 0; j < 25; j += 5) {
            uint64_t row0 = st[j + 0];
            uint64_t row1 = st[j + 1];
            uint64_t row2 = st[j + 2];
            uint64_t row3 = st[j + 3];
            uint64_t row4 = st[j + 4];
            st[j + 0] = row0 ^ ((~row1) & row2);
            st[j + 1] = row1 ^ ((~row2) & row3);
            st[j + 2] = row2 ^ ((~row3) & row4);
            st[j + 3] = row3 ^ ((~row4) & row0);
            st[j + 4] = row4 ^ ((~row0) & row1);
        }

        st[0] ^= keccakf_rndc_device[round];
    }
}

static __device__ void shake256_device_init(shake256_device_ctx *sc)
{
    for (int i = 0; i < 25; i++) {
        sc->st[i] = 0;
    }
    sc->dptr = 0;
}

static __device__ void shake256_device_absorb_bench_input(
    shake256_device_ctx *sc, const uint8_t *nonce, int token)
{
    uint8_t block[136];

    for (int i = 0; i < 136; i++) {
        block[i] = 0;
    }
    for (int i = 0; i < 40; i++) {
        block[i] = nonce[i];
    }
    for (int i = 0; i < OOFALCON_VERIFY_BENCH_MSG_SIZE; i++) {
        block[40 + i] = (uint8_t)((token * 131 + i * 17) & 0xFF);
    }
    block[40 + OOFALCON_VERIFY_BENCH_MSG_SIZE] = 0x1F;
    block[135] |= 0x80;

    for (int i = 0; i < 17; i++) {
        uint64_t lane = 0;
        for (int j = 0; j < 8; j++) {
            lane |= (uint64_t)block[(i << 3) + j] << (j << 3);
        }
        sc->st[i] ^= lane;
    }
    keccakf1600_device(sc->st);
    sc->dptr = 0;
}

static __device__ __forceinline__ uint8_t shake256_device_extract_byte(shake256_device_ctx *sc)
{
    if (sc->dptr == 136) {
        keccakf1600_device(sc->st);
        sc->dptr = 0;
    }
    {
        int idx = sc->dptr++;
        return (uint8_t)(sc->st[idx >> 3] >> ((idx & 7) << 3));
    }
}

static __device__ void hash_to_point_vartime_bench_device(
    int token, const uint8_t *nonce, unsigned logn, uint32_t *hm_out)
{
    shake256_device_ctx sc;
    size_t n;

    shake256_device_init(&sc);
    shake256_device_absorb_bench_input(&sc, nonce, token);

    n = (size_t)1 << logn;
    while (n > 0) {
        uint32_t w;

        w = ((uint32_t)shake256_device_extract_byte(&sc) << 8)
            | (uint32_t)shake256_device_extract_byte(&sc);
        if (w < 61445u) {
            while (w >= OOFALCON_Q) {
                w -= OOFALCON_Q;
            }
            *hm_out++ = w;
            n--;
        }
    }
}

static __device__ int comp_decode_device(
    int16_t *x, unsigned logn,
    const uint8_t *buf, size_t max_in_len,
    size_t *used_len)
{
    size_t n, u, v;
    uint32_t acc;
    unsigned acc_len;

    n = (size_t)1 << logn;
    acc = 0;
    acc_len = 0;
    v = 0;
    for (u = 0; u < n; u++) {
        unsigned b, s, m;

        if (v >= max_in_len) {
            return 0;
        }
        acc = (acc << 8) | (uint32_t)buf[v++];
        b = acc >> acc_len;
        s = b & 128u;
        m = b & 127u;

        for (;;) {
            if (acc_len == 0) {
                if (v >= max_in_len) {
                    return 0;
                }
                acc = (acc << 8) | (uint32_t)buf[v++];
                acc_len = 8;
            }
            acc_len--;
            if (((acc >> acc_len) & 1u) != 0) {
                break;
            }
            m += 128u;
            if (m > 2047u) {
                return 0;
            }
        }

        if (s && m == 0) {
            return 0;
        }
        x[u] = (int16_t)(s ? -(int)m : (int)m);
    }

    if ((acc & ((1u << acc_len) - 1u)) != 0) {
        return 0;
    }
    if (used_len != NULL) {
        *used_len = v;
    }
    return 1;
}

static __device__ uint64_t splitmix64(uint64_t x)
{
    uint64_t z = x + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static __device__ uint64_t exp_scaled_device(uint64_t x)
{
    uint64_t r;

    r = ((809438661408ULL) * x) >> 28;
    r = ((869506949331ULL - r) * x) >> 28;
    r = ((640044208952ULL - r) * x) >> 27;
    r = ((793458686015ULL - r) * x) >> 27;
    r = ((839743192604ULL - r) * x) >> 27;
    r = ((740389683060ULL - r) * x) >> 26;
    r = ((1044449863563ULL - r) * x) >> 27;
    r = ((552517269260ULL - r) * x) >> 25;
    r = ((779422325990ULL - r) * x) >> 23;
    r = (2199023255552ULL - r);

    return r;
}

static __device__ __forceinline__ uint32_t mq_add_device(uint32_t x, uint32_t y)
{
    uint32_t d = x + y - OOFALCON_Q;
    d += OOFALCON_Q & -(d >> 31);
    return d;
}

static __device__ __forceinline__ uint32_t mq_sub_device(uint32_t x, uint32_t y)
{
    uint32_t d = x - y;
    d += OOFALCON_Q & -(d >> 31);
    return d;
}

static __device__ __forceinline__ uint32_t mq_montymul_device(uint32_t x, uint32_t y)
{
    uint32_t z = x * y;
    uint32_t w = ((z * OOFALCON_Q0I) & 0xFFFFU) * OOFALCON_Q;
    z = (z + w) >> 16;
    z -= OOFALCON_Q;
    z += OOFALCON_Q & -(z >> 31);
    return z;
}

static __device__ uint32_t sample_berncdt_device(uint64_t utop, uint64_t ubot)
{
    uint32_t x = 0;
    uint32_t b = 1;
    uint32_t r, s, t;

    t = (utop > 11881272476311950404ULL);
    r = (utop == 11881272476311950404ULL);
    s = (ubot > 2232598800125794762ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 17729174351313943813ULL);
    r = (utop == 17729174351313943813ULL);
    s = (ubot > 17046599807202850264ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 18426461144592799266ULL);
    r = (utop == 18426461144592799266ULL);
    s = (ubot > 9031501729263515114ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 18446602887327906610ULL);
    r = (utop == 18446602887327906610ULL);
    s = (ubot > 11817852927693963396ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 18446743834670245612ULL);
    r = (utop == 18446743834670245612ULL);
    s = (ubot > 7306021935394802834ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 18446744073611412414ULL);
    r = (utop == 18446744073611412414ULL);
    s = (ubot > 17880792342251759005ULL);
    b = (r && s) || t;
    x += b;

    t = (utop > 18446744073709541852ULL);
    r = (utop == 18446744073709541852ULL);
    s = (ubot > 14689009182029885173ULL);
    b = (r && s) || t;
    x += b;

    r = (utop == 18446744073709551615ULL);
    s = (ubot > 14106032229701791861ULL);
    b = (r && s);
    x += b;

    s = (ubot > 18446718728838181855ULL);
    b &= s;
    x += b;

    s = (ubot > 18446744073673701140ULL);
    b &= s;
    x += b;

    return x;
}

static __device__ uint32_t div16404853_device(uint32_t x)
{
    uint32_t y, z;
    y = (uint32_t)((97488647ULL * (uint64_t)x) >> 32);
    z = (x - y) >> 1;
    return (z + y) >> 23;
}

typedef struct {
    uint32_t a;
    uint32_t b;
} bern_prng_state;

static __device__ uint8_t bern_rand_byte(bern_prng_state *st)
{
    st->a += st->b;
    st->b += st->a;
    return (uint8_t)((st->a >> 24) ^ (st->b >> 16));
}

static __device__ uint64_t bern_rand_u64(bern_prng_state *st)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= ((uint64_t)bern_rand_byte(st)) << (8 * i);
    }
    return v;
}

static __device__ int8_t sample_gaussian_bern_device(bern_prng_state *st)
{
    for (;;) {
        uint64_t utop = bern_rand_u64(st);
        uint64_t ubot = bern_rand_u64(st);
        uint64_t w = bern_rand_u64(st) >> 1;

        uint32_t x = sample_berncdt_device(utop, ubot) << 8;
        uint32_t yb = (uint32_t)bern_rand_byte(st);
        uint32_t z = x + yb;

        uint32_t y = (yb * (yb + 2U * x)) << 8;
        uint32_t k = div16404853_device(y);
        uint32_t t = y - 16404853U * k;
        uint64_t v = exp_scaled_device(t) << (22 - k);

        uint8_t c = bern_rand_byte(st);
        if ((w > v) || (c & (z == 0))) {
            continue;
        }

        c >>= 1;
        int32_t zi = (int32_t)z;
        int32_t coeff = (c & 1U) ? -zi : zi;
        return (int8_t)coeff;
    }
}

static __global__ void kernel_generate_samples_and_targets(
    const uint16_t *h_hat,
    const uint16_t *gmb,
    const uint16_t *igmb,
    uint32_t n_inv,
    int tokens,
    int n,
    uint32_t seed,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    int token = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    if (token >= tokens) {
        return;
    }

    extern __shared__ unsigned char smem[];
    uint32_t *h_shared = (uint32_t *)smem;
    uint32_t *u0_shared = h_shared + n;
    uint32_t *u1_shared = u0_shared + n;

    for (int i = tid; i < n; i += (int)blockDim.x) {
        h_shared[i] = (uint32_t)h_hat[i];
    }

    const int base = token * n;

    /* Sample s1 and s2 once. */
    for (int j = tid; j < n; j += (int)blockDim.x) {
        uint64_t mix = ((uint64_t)seed << 32)
            ^ ((uint64_t)(token + 1) * 0xD6E8FEB86659FD93ULL)
            ^ ((uint64_t)(j + 1) * 0xA0761D6478BD642FULL);
        bern_prng_state st1;
        bern_prng_state st2;

        st1.a = (uint32_t)splitmix64(mix);
        st1.b = (uint32_t)splitmix64(mix ^ 0xA5A5A5A5A5A5A5A5ULL);
        st2.a = (uint32_t)splitmix64(mix ^ 0x9E3779B97F4A7C15ULL);
        st2.b = (uint32_t)splitmix64(mix ^ 0xC3A5C85C97CB3127ULL);

        if (st1.a == 0 && st1.b == 0) {
            st1.a = 1;
        }
        if (st2.a == 0 && st2.b == 0) {
            st2.a = 1;
        }

        int8_t s1 = sample_gaussian_bern_device(&st1);
        int8_t s2 = sample_gaussian_bern_device(&st2);
        uint32_t u0 = (uint32_t)s1;
        uint32_t u1 = (uint32_t)s2;
        u0 += OOFALCON_Q & -(u0 >> 31);
        u1 += OOFALCON_Q & -(u1 >> 31);
        sample1_bank[base + j] = s1;
        sample2_bank[base + j] = s2;
        u0_shared[j] = u0;
        u1_shared[j] = u1;
    }

    __syncthreads();

    /* NTT(u1) with Falcon GMb table. */
    {
        int t = n;
        for (int m = 1; m < n; m <<= 1) {
            int ht = t >> 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / t;
                int j1 = i * t;
                int j2 = j1 + ht;
                if (j >= j2) {
                    continue;
                }
                uint32_t s = gmb[m + i];
                uint32_t u = u1_shared[j];
                uint32_t v = u1_shared[j + ht];
                v = mq_montymul_device(v, s);
                u1_shared[j] = mq_add_device(u, v);
                u1_shared[j + ht] = mq_sub_device(u, v);
            }
            __syncthreads();
            t = ht;
        }
    }

    /* Pointwise multiply with H_hat */
    for (int j = tid; j < n; j += (int)blockDim.x) {
        u1_shared[j] = mq_montymul_device(u1_shared[j], h_shared[j]);
    }
    __syncthreads();

    /* iNTT */
    {
        int t = 1;
        for (int m = n; m > 1; m >>= 1) {
            int hm = m >> 1;
            int dt = t << 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / dt;
                int j1 = i * dt;
                int j2 = j1 + t;
                if (j >= j2) {
                    continue;
                }
                uint32_t u = u1_shared[j];
                uint32_t v = u1_shared[j + t];
                u1_shared[j] = mq_add_device(u, v);
                uint32_t w = mq_sub_device(u, v);
                uint32_t s = igmb[hm + i];
                u1_shared[j + t] = mq_montymul_device(w, s);
            }
            __syncthreads();
            t = dt;
        }
    }

    for (int j = tid; j < n; j += (int)blockDim.x) {
        u1_shared[j] = mq_montymul_device(u1_shared[j], n_inv);
    }
    __syncthreads();

    /* Compute target = u0 - h*u1 mod q */
    for (int j = tid; j < n; j += (int)blockDim.x) {
        uint32_t u0 = u0_shared[j];
        uint32_t conv = u1_shared[j];
        uint32_t t = (u0 + OOFALCON_Q - conv);
        if (t >= OOFALCON_Q) {
            t -= OOFALCON_Q;
        }
        target_bank[base + j] = (uint16_t)t;
    }
}

static __global__ void kernel_verify_raw_batch(
    const uint16_t *h_ntt_monty,
    const uint16_t *gmb,
    const uint16_t *igmb,
    uint32_t n_inv,
    unsigned logn,
    int tokens,
    int n,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank)
{
    int token = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    if (token >= tokens) {
        return;
    }

    extern __shared__ unsigned char smem[];
    uint32_t *hm_shared = (uint32_t *)smem;
    uint32_t *tt_shared = hm_shared + n;
    unsigned long long *sum_shared = (unsigned long long *)(tt_shared + n);
    const int base = token * n;

    for (int j = tid; j < n; j += (int)blockDim.x) {
        uint32_t w;

        hm_shared[j] = (uint32_t)hm_bank[base + j];
        w = (uint32_t)s2_bank[base + j];
        w += OOFALCON_Q & -(w >> 31);
        tt_shared[j] = w;
    }
    __syncthreads();

    {
        int t = n;
        for (int m = 1; m < n; m <<= 1) {
            int ht = t >> 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / t;
                int j1 = i * t;
                int j2 = j1 + ht;
                if (j >= j2) {
                    continue;
                }
                uint32_t s = gmb[m + i];
                uint32_t u = tt_shared[j];
                uint32_t v = mq_montymul_device(tt_shared[j + ht], s);
                tt_shared[j] = mq_add_device(u, v);
                tt_shared[j + ht] = mq_sub_device(u, v);
            }
            __syncthreads();
            t = ht;
        }
    }

    for (int j = tid; j < n; j += (int)blockDim.x) {
        tt_shared[j] = mq_montymul_device(tt_shared[j], h_ntt_monty[j]);
    }
    __syncthreads();

    {
        int t = 1;
        for (int m = n; m > 1; m >>= 1) {
            int hm = m >> 1;
            int dt = t << 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / dt;
                int j1 = i * dt;
                int j2 = j1 + t;
                if (j >= j2) {
                    continue;
                }
                uint32_t u = tt_shared[j];
                uint32_t v = tt_shared[j + t];
                tt_shared[j] = mq_add_device(u, v);
                uint32_t w = mq_sub_device(u, v);
                uint32_t s = igmb[hm + i];
                tt_shared[j + t] = mq_montymul_device(w, s);
            }
            __syncthreads();
            t = dt;
        }
    }

    unsigned long long local_sum = 0;
    for (int j = tid; j < n; j += (int)blockDim.x) {
        uint32_t w = mq_montymul_device(tt_shared[j], n_inv);
        int32_t s1n;
        int32_t s2v;

        w = mq_sub_device(w, hm_shared[j]);
        s1n = (w > (OOFALCON_Q >> 1)) ? ((int32_t)w - OOFALCON_Q) : (int32_t)w;
        s2v = (int32_t)s2_bank[base + j];
        local_sum += (unsigned long long)((int64_t)s1n * (int64_t)s1n);
        local_sum += (unsigned long long)((int64_t)s2v * (int64_t)s2v);
    }

    sum_shared[tid] = local_sum;
    __syncthreads();

    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long bound = 3ULL * (unsigned long long)verify_l2bound_device((int)logn);
        result_bank[token] = (sum_shared[0] <= bound) ? 1u : 0u;
    }
}

static __global__ void kernel_verify_compressed_bench_batch(
    const uint16_t *h_ntt_monty,
    const uint16_t *gmb,
    const uint16_t *igmb,
    uint32_t n_inv,
    unsigned logn,
    int tokens,
    int n,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank)
{
    int token = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    __shared__ int decode_ok;

    if (token >= tokens) {
        return;
    }

    extern __shared__ unsigned char smem[];
    uint32_t *hm_shared = (uint32_t *)smem;
    uint32_t *tt_shared = hm_shared + n;
    unsigned long long *sum_shared = (unsigned long long *)(tt_shared + n);
    int16_t *s2_shared = (int16_t *)(sum_shared + blockDim.x);

    if (tid == 0) {
        const uint8_t *sig = sig_bank + (size_t)token * sig_stride;
        size_t sig_len = sig_lens[token];
        size_t used_len = 0;

        if (sig_len < 41
            || (sig[0] & 0xF0) != 0x30
            || (unsigned)(sig[0] & 0x0F) != logn)
        {
            decode_ok = 0;
        } else if (!comp_decode_device(s2_shared, logn,
                sig + 41, sig_len - 41, &used_len)
            || (41 + used_len) != sig_len)
        {
            decode_ok = 0;
        } else {
            hash_to_point_vartime_bench_device(token, sig + 1, logn, hm_shared);
            decode_ok = 1;
        }
    }
    __syncthreads();

    if (!decode_ok) {
        if (tid == 0) {
            result_bank[token] = 0;
        }
        return;
    }

    for (int j = tid; j < n; j += (int)blockDim.x) {
        uint32_t w = (uint32_t)s2_shared[j];
        w += OOFALCON_Q & -(w >> 31);
        tt_shared[j] = w;
    }
    __syncthreads();

    {
        int t = n;
        for (int m = 1; m < n; m <<= 1) {
            int ht = t >> 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / t;
                int j1 = i * t;
                int j2 = j1 + ht;
                if (j >= j2) {
                    continue;
                }
                uint32_t s = gmb[m + i];
                uint32_t u = tt_shared[j];
                uint32_t v = mq_montymul_device(tt_shared[j + ht], s);
                tt_shared[j] = mq_add_device(u, v);
                tt_shared[j + ht] = mq_sub_device(u, v);
            }
            __syncthreads();
            t = ht;
        }
    }

    for (int j = tid; j < n; j += (int)blockDim.x) {
        tt_shared[j] = mq_montymul_device(tt_shared[j], h_ntt_monty[j]);
    }
    __syncthreads();

    {
        int t = 1;
        for (int m = n; m > 1; m >>= 1) {
            int hm = m >> 1;
            int dt = t << 1;
            for (int j = tid; j < n; j += (int)blockDim.x) {
                int i = j / dt;
                int j1 = i * dt;
                int j2 = j1 + t;
                if (j >= j2) {
                    continue;
                }
                uint32_t u = tt_shared[j];
                uint32_t v = tt_shared[j + t];
                tt_shared[j] = mq_add_device(u, v);
                uint32_t w = mq_sub_device(u, v);
                uint32_t s = igmb[hm + i];
                tt_shared[j + t] = mq_montymul_device(w, s);
            }
            __syncthreads();
            t = dt;
        }
    }

    {
        unsigned long long local_sum = 0;
        for (int j = tid; j < n; j += (int)blockDim.x) {
            uint32_t w = mq_montymul_device(tt_shared[j], n_inv);
            int32_t s1n;
            int32_t s2v;

            w = mq_sub_device(w, hm_shared[j]);
            s1n = (w > (OOFALCON_Q >> 1)) ? ((int32_t)w - OOFALCON_Q) : (int32_t)w;
            s2v = (int32_t)s2_shared[j];
            local_sum += (unsigned long long)((int64_t)s1n * (int64_t)s1n);
            local_sum += (unsigned long long)((int64_t)s2v * (int64_t)s2v);
        }
        sum_shared[tid] = local_sum;
    }
    __syncthreads();

    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        unsigned long long bound = 3ULL * (unsigned long long)verify_l2bound_device((int)logn);
        result_bank[token] = (sum_shared[0] <= bound) ? 1u : 0u;
    }
}

int oofalcon_gpu_generate_presign_bank(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    if (h_coeffs == NULL || sample1_bank == NULL || sample2_bank == NULL || target_bank == NULL) {
        return 1;
    }
    if (logn < 1 || logn > 10 || tokens <= 0) {
        return 2;
    }

    const int n = 1 << logn;
    const size_t poly_bytes_u16 = (size_t)n * sizeof(uint16_t);
    const size_t bank_bytes_i8 = (size_t)tokens * (size_t)n * sizeof(int8_t);
    const size_t bank_bytes_u16 = (size_t)tokens * (size_t)n * sizeof(uint16_t);

    int8_t *d_s1 = NULL;
    int8_t *d_s2 = NULL;
    uint16_t *d_t = NULL;
    int s1_registered = 0;
    int s2_registered = 0;
    int t_registered = 0;

    cudaError_t err;

    if (g_stream == NULL) {
        err = cudaStreamCreate(&g_stream);
        if (err != cudaSuccess) {
            return 9;
        }
    }

    if (g_kernel_done == NULL) {
        err = cudaEventCreate(&g_kernel_done);
        if (err != cudaSuccess) {
            return 9;
        }
    }

    if (ensure_device_tables((int)logn) != 0) {
        return 9;
    }

    if (ensure_h_hat(h_coeffs, (int)logn) != 0) {
        return 9;
    }

    err = cudaMalloc((void **)&d_s1, bank_bytes_i8);
    if (err != cudaSuccess) {
        return 11;
    }
    err = cudaMalloc((void **)&d_s2, bank_bytes_i8);
    if (err != cudaSuccess) {
        cudaFree(d_s1);
        return 12;
    }
    err = cudaMalloc((void **)&d_t, bank_bytes_u16);
    if (err != cudaSuccess) {
        cudaFree(d_s1);
        cudaFree(d_s2);
        return 13;
    }

    const int block = 256;
    dim3 grid((unsigned int)tokens, 1U, 1U);
    const size_t shared_bytes = (size_t)(3 * n) * sizeof(uint32_t);
    kernel_generate_samples_and_targets<<<grid, block, shared_bytes, g_stream>>>(
        g_d_h_hat,
        g_d_gmb,
        g_d_igmb,
        g_ntt_host.n_inv,
        tokens, n, seed, d_s1, d_s2, d_t);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_s1);
        cudaFree(d_s2);
        cudaFree(d_t);
        return 15;
    }

    // Record event to mark kernel completion
    err = cudaEventRecord(g_kernel_done, g_stream);
    if (err != cudaSuccess) {
        cudaFree(d_s1);
        cudaFree(d_s2);
        cudaFree(d_t);
        return 15;
    }

    if (cudaHostRegister(sample1_bank, bank_bytes_i8, cudaHostRegisterDefault) == cudaSuccess) {
        s1_registered = 1;
    }
    if (cudaHostRegister(sample2_bank, bank_bytes_i8, cudaHostRegisterDefault) == cudaSuccess) {
        s2_registered = 1;
    }
    if (cudaHostRegister(target_bank, bank_bytes_u16, cudaHostRegisterDefault) == cudaSuccess) {
        t_registered = 1;
    }

    // Ensure kernel is complete before starting D2H transfers
    err = cudaEventSynchronize(g_kernel_done);
    if (err != cudaSuccess) {
        if (s1_registered) {
            cudaHostUnregister(sample1_bank);
        }
        if (s2_registered) {
            cudaHostUnregister(sample2_bank);
        }
        if (t_registered) {
            cudaHostUnregister(target_bank);
        }
        cudaFree(d_s1);
        cudaFree(d_s2);
        cudaFree(d_t);
        return 16;
    }

    err = cudaMemcpyAsync(sample1_bank, d_s1, bank_bytes_i8, cudaMemcpyDeviceToHost, g_stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(sample2_bank, d_s2, bank_bytes_i8, cudaMemcpyDeviceToHost, g_stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(target_bank, d_t, bank_bytes_u16, cudaMemcpyDeviceToHost, g_stream);
    }
    if (err != cudaSuccess) {
        if (s1_registered) {
            cudaHostUnregister(sample1_bank);
        }
        if (s2_registered) {
            cudaHostUnregister(sample2_bank);
        }
        if (t_registered) {
            cudaHostUnregister(target_bank);
        }
        cudaFree(d_s1);
        cudaFree(d_s2);
        cudaFree(d_t);
        return 16;
    }

    err = cudaStreamSynchronize(g_stream);

    if (s1_registered) {
        cudaHostUnregister(sample1_bank);
    }
    if (s2_registered) {
        cudaHostUnregister(sample2_bank);
    }
    if (t_registered) {
        cudaHostUnregister(target_bank);
    }

    if (err != cudaSuccess) {
        cudaFree(d_s1);
        cudaFree(d_s2);
        cudaFree(d_t);
        return 17;
    }

    cudaFree(d_s1);
    cudaFree(d_s2);
    cudaFree(d_t);
    return 0;
}

int oofalcon_gpu_verify_raw_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank)
{
    if (h_ntt_monty == NULL || hm_bank == NULL || s2_bank == NULL || result_bank == NULL) {
        return 41;
    }
    if (logn < 1 || logn > 10 || tokens <= 0) {
        return 42;
    }

    const int n = 1 << logn;
    const size_t bank_bytes_u16 = (size_t)tokens * (size_t)n * sizeof(uint16_t);
    const size_t bank_bytes_i16 = (size_t)tokens * (size_t)n * sizeof(int16_t);
    const size_t result_bytes = (size_t)tokens * sizeof(uint8_t);
    uint16_t *d_hm = NULL;
    int16_t *d_s2 = NULL;
    uint8_t *d_res = NULL;
    int hm_registered = 0;
    int s2_registered = 0;
    int res_registered = 0;
    cudaError_t err;

    if (g_stream == NULL) {
        err = cudaStreamCreate(&g_stream);
        if (err != cudaSuccess) {
            return 49;
        }
    }

    if (g_kernel_done == NULL) {
        err = cudaEventCreate(&g_kernel_done);
        if (err != cudaSuccess) {
            return 49;
        }
    }

    if (ensure_device_tables((int)logn) != 0) {
        return 49;
    }
    if (ensure_h_verify(h_ntt_monty, (int)logn) != 0) {
        return 49;
    }

    err = cudaMalloc((void **)&d_hm, bank_bytes_u16);
    if (err != cudaSuccess) {
        return 51;
    }
    err = cudaMalloc((void **)&d_s2, bank_bytes_i16);
    if (err != cudaSuccess) {
        cudaFree(d_hm);
        return 52;
    }
    err = cudaMalloc((void **)&d_res, result_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_hm);
        cudaFree(d_s2);
        return 53;
    }

    if (cudaHostRegister((void *)hm_bank, bank_bytes_u16, cudaHostRegisterDefault) == cudaSuccess) {
        hm_registered = 1;
    }
    if (cudaHostRegister((void *)s2_bank, bank_bytes_i16, cudaHostRegisterDefault) == cudaSuccess) {
        s2_registered = 1;
    }
    if (cudaHostRegister(result_bank, result_bytes, cudaHostRegisterDefault) == cudaSuccess) {
        res_registered = 1;
    }

    err = cudaMemcpyAsync(d_hm, hm_bank, bank_bytes_u16, cudaMemcpyHostToDevice, g_stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(d_s2, s2_bank, bank_bytes_i16, cudaMemcpyHostToDevice, g_stream);
    }
    if (err != cudaSuccess) {
        if (hm_registered) {
            cudaHostUnregister((void *)hm_bank);
        }
        if (s2_registered) {
            cudaHostUnregister((void *)s2_bank);
        }
        if (res_registered) {
            cudaHostUnregister(result_bank);
        }
        cudaFree(d_hm);
        cudaFree(d_s2);
        cudaFree(d_res);
        return 54;
    }

    {
        const int block = 256;
        dim3 grid((unsigned int)tokens, 1U, 1U);
        size_t shared_bytes = (size_t)(2 * n) * sizeof(uint32_t)
            + (size_t)block * sizeof(unsigned long long);
        kernel_verify_raw_batch<<<grid, block, shared_bytes, g_stream>>>(
            g_d_h_verify,
            g_d_gmb,
            g_d_igmb,
            g_ntt_host.n_inv,
            logn,
            tokens,
            n,
            d_hm,
            d_s2,
            d_res);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (hm_registered) {
            cudaHostUnregister((void *)hm_bank);
        }
        if (s2_registered) {
            cudaHostUnregister((void *)s2_bank);
        }
        if (res_registered) {
            cudaHostUnregister(result_bank);
        }
        cudaFree(d_hm);
        cudaFree(d_s2);
        cudaFree(d_res);
        return 55;
    }

    err = cudaEventRecord(g_kernel_done, g_stream);
    if (err != cudaSuccess) {
        if (hm_registered) {
            cudaHostUnregister((void *)hm_bank);
        }
        if (s2_registered) {
            cudaHostUnregister((void *)s2_bank);
        }
        if (res_registered) {
            cudaHostUnregister(result_bank);
        }
        cudaFree(d_hm);
        cudaFree(d_s2);
        cudaFree(d_res);
        return 56;
    }

    err = cudaEventSynchronize(g_kernel_done);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(result_bank, d_res, result_bytes, cudaMemcpyDeviceToHost, g_stream);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(g_stream);
    }

    if (hm_registered) {
        cudaHostUnregister((void *)hm_bank);
    }
    if (s2_registered) {
        cudaHostUnregister((void *)s2_bank);
    }
    if (res_registered) {
        cudaHostUnregister(result_bank);
    }

    cudaFree(d_hm);
    cudaFree(d_s2);
    cudaFree(d_res);

    return (err == cudaSuccess) ? 0 : 57;
}

int oofalcon_gpu_verify_compressed_benchmark_batch(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank)
{
    if (h_ntt_monty == NULL || sig_bank == NULL || sig_lens == NULL || result_bank == NULL) {
        return 61;
    }
    if (logn < 1 || logn > 10 || tokens <= 0 || sig_stride == 0) {
        return 62;
    }

    {
        const int n = 1 << logn;
        const size_t sig_bytes = (size_t)tokens * sig_stride;
        const size_t sig_lens_bytes = (size_t)tokens * sizeof(size_t);
        const size_t result_bytes = (size_t)tokens * sizeof(uint8_t);
        uint8_t *d_sig = NULL;
        size_t *d_sig_lens = NULL;
        uint8_t *d_res = NULL;
        int sig_registered = 0;
        int sig_lens_registered = 0;
        int res_registered = 0;
        cudaError_t err;

        if (g_stream == NULL) {
            err = cudaStreamCreate(&g_stream);
            if (err != cudaSuccess) {
                return 69;
            }
        }
        if (g_kernel_done == NULL) {
            err = cudaEventCreate(&g_kernel_done);
            if (err != cudaSuccess) {
                return 69;
            }
        }
        if (ensure_device_tables((int)logn) != 0) {
            return 69;
        }
        if (ensure_h_verify(h_ntt_monty, (int)logn) != 0) {
            return 69;
        }

        err = cudaMalloc((void **)&d_sig, sig_bytes);
        if (err != cudaSuccess) {
            return 71;
        }
        err = cudaMalloc((void **)&d_sig_lens, sig_lens_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_sig);
            return 72;
        }
        err = cudaMalloc((void **)&d_res, result_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_sig);
            cudaFree(d_sig_lens);
            return 73;
        }

        if (cudaHostRegister((void *)sig_bank, sig_bytes, cudaHostRegisterDefault) == cudaSuccess) {
            sig_registered = 1;
        }
        if (cudaHostRegister((void *)sig_lens, sig_lens_bytes, cudaHostRegisterDefault) == cudaSuccess) {
            sig_lens_registered = 1;
        }
        if (cudaHostRegister(result_bank, result_bytes, cudaHostRegisterDefault) == cudaSuccess) {
            res_registered = 1;
        }

        err = cudaMemcpyAsync(d_sig, sig_bank, sig_bytes, cudaMemcpyHostToDevice, g_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(d_sig_lens, sig_lens, sig_lens_bytes, cudaMemcpyHostToDevice, g_stream);
        }
        if (err != cudaSuccess) {
            if (sig_registered) {
                cudaHostUnregister((void *)sig_bank);
            }
            if (sig_lens_registered) {
                cudaHostUnregister((void *)sig_lens);
            }
            if (res_registered) {
                cudaHostUnregister(result_bank);
            }
            cudaFree(d_sig);
            cudaFree(d_sig_lens);
            cudaFree(d_res);
            return 74;
        }

        {
            const int block = 256;
            dim3 grid((unsigned int)tokens, 1U, 1U);
            size_t shared_bytes = (size_t)(2 * n) * sizeof(uint32_t)
                + (size_t)block * sizeof(unsigned long long)
                + (size_t)n * sizeof(int16_t);
            kernel_verify_compressed_bench_batch<<<grid, block, shared_bytes, g_stream>>>(
                g_d_h_verify,
                g_d_gmb,
                g_d_igmb,
                g_ntt_host.n_inv,
                logn,
                tokens,
                n,
                d_sig,
                sig_stride,
                d_sig_lens,
                d_res);
        }

        err = cudaGetLastError();
        if (err == cudaSuccess) {
            err = cudaEventRecord(g_kernel_done, g_stream);
        }
        if (err == cudaSuccess) {
            err = cudaEventSynchronize(g_kernel_done);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(result_bank, d_res, result_bytes, cudaMemcpyDeviceToHost, g_stream);
        }
        if (err == cudaSuccess) {
            err = cudaStreamSynchronize(g_stream);
        }

        if (sig_registered) {
            cudaHostUnregister((void *)sig_bank);
        }
        if (sig_lens_registered) {
            cudaHostUnregister((void *)sig_lens);
        }
        if (res_registered) {
            cudaHostUnregister(result_bank);
        }

        cudaFree(d_sig);
        cudaFree(d_sig_lens);
        cudaFree(d_res);

        return (err == cudaSuccess) ? 0 : 75;
    }
}

int oofalcon_gpu_reset_device(void)
{
    if (g_kernel_done != NULL) {
        cudaEventDestroy(g_kernel_done);
        g_kernel_done = NULL;
    }
    if (g_stream != NULL) {
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
    }
    cudaError_t err = cudaDeviceReset();
    g_d_gmb = NULL;
    g_d_igmb = NULL;
    g_d_h_hat = NULL;
    g_d_h_verify = NULL;
    g_h_hash = 0;
    g_h_verify_hash = 0;
    g_cached_logn = 0;
    g_cached_verify_logn = 0;
    return (err == cudaSuccess) ? 0 : 30;
}

#endif
