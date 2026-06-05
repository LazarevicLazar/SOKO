#include "soko_falcon_backends.h"

extern void sample_gaussian_poly_bern(int8_t *sample1, int8_t *sample2, size_t n);
extern void compute_target(const uint16_t *h_monty,
    const int8_t *x0, const int8_t *x1, uint16_t *res, unsigned logn);

void soko_cpumt_generate_presign_bank_raw(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    int8_t *sample1_bank,
    int8_t *sample2_bank,
    uint16_t *target_bank,
    int cpu_mt_threads)
{
    size_t n = (size_t)1 << logn;

#ifdef _OPENMP
    {
        int threads = soko_resolve_cpu_mt_threads(cpu_mt_threads);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < tokens; t++) {
            int8_t *s1 = sample1_bank + (size_t)t * n;
            int8_t *s2 = sample2_bank + (size_t)t * n;
            uint16_t *tg = target_bank + (size_t)t * n;
            sample_gaussian_poly_bern(s1, s2, n);
            compute_target(h_monty, s1, s2, tg, logn);
        }
    }
#else
    (void)cpu_mt_threads;
    soko_cpu_generate_presign_bank_raw(logn, tokens, h_monty, sample1_bank, sample2_bank, target_bank);
#endif
}

int soko_cpumt_generate_presign_tokens(
    unsigned logn,
    int tokens,
    const uint16_t *h_monty,
    soko_falcon_token_t *token_bank,
    int cpu_mt_threads)
{
#ifdef _OPENMP
    {
        int threads = soko_resolve_cpu_mt_threads(cpu_mt_threads);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int t = 0; t < tokens; t++) {
            sample_gaussian_poly_bern(token_bank[t].s1, token_bank[t].s2, (size_t)1 << logn);
            compute_target(h_monty, token_bank[t].s1, token_bank[t].s2, token_bank[t].target, logn);
        }
    }
#else
    (void)cpu_mt_threads;
    soko_cpu_generate_presign_tokens(logn, tokens, h_monty, token_bank);
#endif
    return tokens;
}
