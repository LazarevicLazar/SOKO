#include "soko_falcon_backends.h"

extern void sample_gaussian_poly_bern(int8_t *sample1, int8_t *sample2, size_t n);
extern void compute_target(const uint16_t *h_monty,
    const int8_t *x0, const int8_t *x1, uint16_t *res, unsigned logn);

void soko_cpu_generate_presign_bank_raw(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank)
{
    size_t n = (size_t)1 << logn;
    int t;

    for (t = 0; t < tokens; t++) {
        int8_t *s1 = sample1_bank + (size_t)t * n;
        int8_t *s2 = sample2_bank + (size_t)t * n;
        uint16_t *tg = target_bank + (size_t)t * n;
        sample_gaussian_poly_bern(s1, s2, n);
        compute_target(h_monty, s1, s2, tg, logn);
    }
}

int soko_cpu_generate_presign_tokens(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    soko_falcon_token_t *token_bank)
{
    int t;

    for (t = 0; t < tokens; t++) {
        sample_gaussian_poly_bern(token_bank[t].s1, token_bank[t].s2, (size_t)1 << logn);
        compute_target(h_monty, token_bank[t].s1, token_bank[t].s2, token_bank[t].target, logn);
    }
    return tokens;
}
