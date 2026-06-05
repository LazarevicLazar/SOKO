#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "soko_falcon_backends.h"

extern int sign_dyn_lazy_online(
    int8_t *sample1, int8_t *sample2, uint16_t *sample_target,
    int16_t *s2,
    const fpr *restrict f_fft, const fpr *restrict g_fft,
    const fpr *restrict F_fft, const fpr *restrict G_fft,
    const uint16_t *hm, unsigned logn, fpr *restrict tmp);

static void *backend_xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Allocation failed (%lu bytes)\n", (unsigned long)size);
        exit(1);
    }
    return ptr;
}

static void smallints_to_fpr_local(fpr *out, const int8_t *in, unsigned logn)
{
    size_t n = (size_t)1 << logn;
    size_t i;

    for (i = 0; i < n; i++) {
        out[i] = fpr_of(in[i]);
    }
}

int soko_resolve_cpu_mt_threads(int requested)
{
#ifdef _OPENMP
    if (requested > 0) {
        return requested;
    }
    {
        int threads = omp_get_max_threads();
        return (threads > 0) ? threads : 1;
    }
#else
    (void)requested;
    return 1;
#endif
}

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
    fpr *G_fft)
{
    shake256_context rng;
    const uint8_t *sk;
    size_t u;
    size_t v;

    if (shake256_init_prng_from_system(&rng) != 0) {
        return -1;
    }

    falcon_keygen_make(&rng, logn,
        privkey, privkey_len,
        pubkey, pubkey_len,
        tmp, tmp_len);

    sk = privkey;
    u = 1;

    v = Zf(trim_i8_decode)(f, logn, Zf(max_fg_bits)[logn], sk + u, privkey_len - u);
    if (v == 0) {
        return -2;
    }
    u += v;

    v = Zf(trim_i8_decode)(g, logn, Zf(max_fg_bits)[logn], sk + u, privkey_len - u);
    if (v == 0) {
        return -3;
    }
    u += v;

    v = Zf(trim_i8_decode)(F, logn, Zf(max_FG_bits)[logn], sk + u, privkey_len - u);
    if (v == 0) {
        return -4;
    }
    u += v;

    if (u != privkey_len) {
        return -5;
    }

    if (!Zf(complete_private)(G, f, g, F, logn, tmp)) {
        return -6;
    }

    if (Zf(modq_decode)(h, logn, pubkey + 1, pubkey_len - 1) != pubkey_len - 1) {
        return -7;
    }

    memcpy(h_monty, h, ((size_t)1 << logn) * sizeof(uint16_t));
    falcon_inner_to_ntt_monty(h_monty, logn);

    smallints_to_fpr_local(f_fft, f, logn);
    smallints_to_fpr_local(g_fft, g, logn);
    smallints_to_fpr_local(F_fft, F, logn);
    smallints_to_fpr_local(G_fft, G, logn);
    Zf(FFT)(f_fft, logn);
    Zf(FFT)(g_fft, logn);
    Zf(FFT)(F_fft, logn);
    Zf(FFT)(G_fft, logn);

    return 0;
}

int soko_prepare_falcon_key_state(soko_falcon_shared_t *state)
{
    memset(state, 0, sizeof(*state));
    state->logn = SOKO_FALCON_LOGN;
    state->privkey_len = FALCON_PRIVKEY_SIZE(state->logn);
    state->pubkey_len = FALCON_PUBKEY_SIZE(state->logn);
    state->tmp_len = FALCON_TMPSIZE_KEYGEN(state->logn);
    if (FALCON_TMPSIZE_SIGNDYN(state->logn) > state->tmp_len) {
        state->tmp_len = FALCON_TMPSIZE_SIGNDYN(state->logn);
    }
    if (FALCON_TMPSIZE_VERIFY(state->logn) > state->tmp_len) {
        state->tmp_len = FALCON_TMPSIZE_VERIFY(state->logn);
    }
    state->sig_max = FALCON_SIG_COMPRESSED_MAXSIZE(state->logn);

    state->privkey = (uint8_t *)backend_xmalloc(state->privkey_len);
    state->pubkey = (uint8_t *)backend_xmalloc(state->pubkey_len);
    state->tmp = (uint8_t *)backend_xmalloc(state->tmp_len);
    state->f = (int8_t *)backend_xmalloc(SOKO_FALCON_N);
    state->g = (int8_t *)backend_xmalloc(SOKO_FALCON_N);
    state->F = (int8_t *)backend_xmalloc(SOKO_FALCON_N);
    state->G = (int8_t *)backend_xmalloc(SOKO_FALCON_N);
    state->h = (uint16_t *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(uint16_t));
    state->h_monty = (uint16_t *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(uint16_t));
    state->f_fft = (fpr *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(fpr));
    state->g_fft = (fpr *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(fpr));
    state->F_fft = (fpr *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(fpr));
    state->G_fft = (fpr *)backend_xmalloc((size_t)SOKO_FALCON_N * sizeof(fpr));

    if (soko_prepare_falcon_key_material(state->logn,
            state->privkey, state->privkey_len,
            state->pubkey, state->pubkey_len,
            state->tmp, state->tmp_len,
            state->f, state->g, state->F, state->G,
            state->h, state->h_monty,
            state->f_fft, state->g_fft, state->F_fft, state->G_fft) != 0) {
        soko_destroy_falcon_key_state(state);
        return 0;
    }

    return 1;
}

void soko_destroy_falcon_key_state(soko_falcon_shared_t *state)
{
    free(state->privkey);
    free(state->pubkey);
    free(state->tmp);
    free(state->f);
    free(state->g);
    free(state->F);
    free(state->G);
    free(state->h);
    free(state->h_monty);
    free(state->f_fft);
    free(state->g_fft);
    free(state->F_fft);
    free(state->G_fft);
    memset(state, 0, sizeof(*state));
}

int soko_falcon_init_signer_local(const soko_falcon_shared_t *state, soko_falcon_signer_local_t *local)
{
    memset(local, 0, sizeof(*local));
    local->sig_buf_len = state->sig_max;
    local->sig_buf = (uint8_t *)malloc(local->sig_buf_len);
    if (local->sig_buf == NULL) {
        return 0;
    }
    local->tmp_sign = (uint8_t *)malloc(FALCON_TMPSIZE_SIGNDYN(state->logn));
    if (local->tmp_sign == NULL) {
        free(local->sig_buf);
        local->sig_buf = NULL;
        local->sig_buf_len = 0;
        return 0;
    }
    local->ftmp = (fpr *)local->tmp_sign;
    return 1;
}

void soko_falcon_destroy_signer_local(soko_falcon_signer_local_t *local)
{
    free(local->sig_buf);
    free(local->tmp_sign);
    memset(local, 0, sizeof(*local));
}

int soko_falcon_sign_token(
    const soko_falcon_shared_t *state,
    soko_falcon_signer_local_t *local,
    const soko_falcon_token_t *token,
    const uint8_t *message,
    size_t message_len,
    const uint8_t nonce[40],
    size_t *sig_len)
{
    shake256_context hash_data;
    size_t encoded_len;

    shake256_init(&hash_data);
    shake256_inject(&hash_data, nonce, 40);
    shake256_inject(&hash_data, message, message_len);
    shake256_flip(&hash_data);
    Zf(hash_to_point_vartime)((inner_shake256_context *)&hash_data, local->hm, state->logn);

    memcpy(local->target_work, token->target, sizeof(token->target));
    sign_dyn_lazy_online((int8_t *)token->s1, (int8_t *)token->s2, local->target_work,
        local->sv,
        state->f_fft, state->g_fft, state->F_fft, state->G_fft,
        local->hm, state->logn, local->ftmp);

    local->sig_buf[0] = 0x30 + state->logn;
    memcpy(local->sig_buf + 1, nonce, 40);
    encoded_len = Zf(comp_encode)(local->sig_buf + 41, local->sig_buf_len - 41, local->sv, state->logn);
    if (encoded_len == 0) {
        if (sig_len != NULL) {
            *sig_len = 0;
        }
        return 0;
    }

    if (sig_len != NULL) {
        *sig_len = 41 + encoded_len;
    }
    return 1;
}
