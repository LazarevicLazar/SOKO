#ifndef SOKO_FALCON_BACKENDS_H
#define SOKO_FALCON_BACKENDS_H

#include <stddef.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "falcon.h"
#include "inner.h"

#define SOKO_FALCON_LOGN 9
#define SOKO_FALCON_N (1 << SOKO_FALCON_LOGN)
#define SOKO_FALCON_MSG_SIZE 64

typedef struct {
    int8_t s1[SOKO_FALCON_N];
    int8_t s2[SOKO_FALCON_N];
    uint16_t target[SOKO_FALCON_N];
} soko_falcon_token_t;

typedef struct {
    unsigned logn;
    size_t privkey_len;
    size_t pubkey_len;
    size_t tmp_len;
    size_t sig_max;
    uint8_t *privkey;
    uint8_t *pubkey;
    uint8_t *tmp;
    int8_t *f;
    int8_t *g;
    int8_t *F;
    int8_t *G;
    uint16_t *h;
    uint16_t *h_monty;
    fpr *f_fft;
    fpr *g_fft;
    fpr *F_fft;
    fpr *G_fft;
} soko_falcon_shared_t;

typedef struct {
    uint16_t hm[SOKO_FALCON_N];
    uint16_t target_work[SOKO_FALCON_N];
    int16_t sv[SOKO_FALCON_N];
    uint8_t nonce[40];
    uint8_t message[SOKO_FALCON_MSG_SIZE];
    uint8_t *sig_buf;
    uint8_t *tmp_sign;
    fpr *ftmp;
    size_t sig_buf_len;
} soko_falcon_signer_local_t;

typedef struct {
    int8_t *s1;
    int8_t *s2;
    uint16_t *target;
    int available;
} soko_falcon_token_bank_t;

typedef struct {
    int bank_id;
    int token_index;
    const int8_t *s1;
    const int8_t *s2;
    const uint16_t *target;
} soko_falcon_token_view_t;

typedef struct soko_falcon_refill_pipeline soko_falcon_refill_pipeline_t;

int soko_resolve_cpu_mt_threads(int requested);

int soko_prepare_falcon_key_material(
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
    fpr *G_fft);

int soko_prepare_falcon_key_state(soko_falcon_shared_t *state);
void soko_destroy_falcon_key_state(soko_falcon_shared_t *state);

int soko_falcon_init_signer_local(const soko_falcon_shared_t *state, soko_falcon_signer_local_t *local);
void soko_falcon_destroy_signer_local(soko_falcon_signer_local_t *local);
int soko_falcon_sign_token(
    const soko_falcon_shared_t *state,
    soko_falcon_signer_local_t *local,
    const soko_falcon_token_t *token,
    const uint8_t *message,
    size_t message_len,
    const uint8_t nonce[40],
    size_t *sig_len);

void soko_cpu_generate_presign_bank_raw(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank);

int soko_cpu_generate_presign_tokens(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    soko_falcon_token_t *token_bank);

void soko_cpumt_generate_presign_bank_raw(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank,
    int cpu_mt_threads);

int soko_cpumt_generate_presign_tokens(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    soko_falcon_token_t *token_bank,
    int cpu_mt_threads);

void soko_copy_presign_bank_to_tokens(
    int tokens,
    const int8_t *sample1_bank,
    const int8_t *sample2_bank,
    const uint16_t *target_bank,
    soko_falcon_token_t *token_bank);

int soko_gpu_backend_load(void);
int soko_gpu_generate_presign_bank_dynamic(
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank);
int soko_gpu_reset_device_dynamic(void);
int soko_gpu_verify_raw_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint16_t *hm_bank,
    const int16_t *s2_bank,
    uint8_t *result_bank);
int soko_gpu_verify_compressed_benchmark_batch_dynamic(
    const uint16_t *h_ntt_monty,
    unsigned logn,
    int tokens,
    const uint8_t *sig_bank,
    size_t sig_stride,
    const size_t *sig_lens,
    uint8_t *result_bank);

int soko_falcon_refill_pipeline_init(
    soko_falcon_refill_pipeline_t **pipeline_out,
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed_base,
    int tokens_per_bank,
    int refill_threshold,
    soko_falcon_token_bank_t *banks,
    int bank_count);
int soko_falcon_refill_pipeline_acquire(
    soko_falcon_refill_pipeline_t *pipeline,
    soko_falcon_token_view_t *token,
    double *wait_ms);
void soko_falcon_refill_pipeline_destroy(soko_falcon_refill_pipeline_t *pipeline);

#endif
