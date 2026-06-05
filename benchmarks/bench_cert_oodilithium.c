/*
 * Certificate Chain Benchmark: Precomputed Dilithium2
 *
 * Measures:
 * - Key generation time per chain level
 * - Offline precomputation time per chain level
 * - Online signing time per chain level
 * - Verification time per chain level
 * - Total chain operations
 *
 * Supports chain lengths: 1 (client only) through 10 (root CA + 8 intermediates + client)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
static inline double get_time() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#define TICK() get_time()
#define TOCK(start) (get_time() - (start))
#else
#define TICK() ((double)clock() / CLOCKS_PER_SEC)
#define TOCK(start) (((double)clock() / CLOCKS_PER_SEC) - (start))
#endif

#include "submodules/OO-ML-DSA/ref/sign.h"
#include "submodules/OO-ML-DSA/ref/params.h"
#include "submodules/OO-ML-DSA/ref/randombytes.h"

/* Configuration */
#define MSG_SIZE 512
#define DEFAULT_ITERATIONS 10
#define MAX_CHAIN_LENGTH 10

/* Certificate structure for chain */
typedef struct {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t cert_sig[CRYPTO_BYTES];  /* Signature from parent (empty for root) */
    size_t cert_siglen;
    char name[32];
} Certificate;

/* Message buffer */
uint8_t message[MSG_SIZE];

static int resolve_verify_threads(int requested) {
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

static void fill_verify_message(uint8_t *msg, int index) {
    for (int i = 0; i < MSG_SIZE; i++) {
        msg[i] = (uint8_t)((index * 131 + i * 17) & 0xFF);
    }
}

/* Helper: get display name for a chain level */
static void get_cert_name(char *buf, size_t buflen, int level, int chain_length) {
    if (chain_length == 1) {
        snprintf(buf, buflen, "Client (self-signed)");
    } else if (level == 0) {
        snprintf(buf, buflen, "Root CA");
    } else if (level == chain_length - 1) {
        snprintf(buf, buflen, "Client");
    } else if (chain_length <= 3) {
        snprintf(buf, buflen, "Intermediate CA");
    } else {
        snprintf(buf, buflen, "Intermediate CA %d", level);
    }
}

/* Function to build and benchmark a certificate chain.
 *
 * Structured in 3 phases to avoid memory exhaustion from the precomputing cache:
 *   Phase 1 - Keygen timing: averaged over `iterations` using temporary keys.
 *             precomputing() is NOT called here, so sk_params stays empty.
 *   Phase 2 - Setup (once): generate real chain keys, call precomputing() exactly
 *             once per level, build chain signatures.
 *   Phase 3 - Online sign + verify loop: iterate over precomputed nonce pool.
 *
 * Why this matters: each call to precomputing() mallocates ~120 MB in the
 * add_one_param() helper and never frees it. The original per-iteration call
 * pattern (iterations * chain_length calls) caused OOM -> segfault for large
 * chain lengths. Here precomputing() is called exactly chain_length times total.
 */
void benchmark_cert_chain(int chain_length, int iterations) {
    Certificate chain[MAX_CHAIN_LENGTH];

    /* keygen_times: summed over `iterations` in Phase 1 -> divide by iterations */
    /* offline_times: single measurement in Phase 2 -> printed as-is            */
    /* online_times, verify_times: summed over `iterations` in Phase 3           */
    double keygen_times[MAX_CHAIN_LENGTH] = {0};
    double offline_times[MAX_CHAIN_LENGTH] = {0};
    double online_times[MAX_CHAIN_LENGTH] = {0};
    double verify_times[MAX_CHAIN_LENGTH] = {0};

    int iter, level;
    double start;

    printf("\n================================================================================\n");
    printf("CERTIFICATE CHAIN LENGTH: %d\n", chain_length);
    printf("================================================================================\n");

    /* Print chain structure */
    printf("Chain structure: ");
    if (chain_length == 1) {
        printf("Client (self-signed)\n");
    } else {
        char name_buf[32];
        for (int i = 0; i < chain_length; i++) {
            get_cert_name(name_buf, sizeof(name_buf), i, chain_length);
            printf("%s%s", name_buf, (i < chain_length - 1) ? " -> " : "\n");
        }
    }
    printf("================================================================================\n\n");

    /* =========================================================
     * PHASE 1: Average keygen time using temporary throw-away keys.
     * precomputing() is NOT called, so the global sk_params table
     * does not grow.
     * ========================================================= */
    for (iter = 0; iter < iterations; iter++) {
        for (level = 0; level < chain_length; level++) {
            uint8_t tmp_pk[CRYPTO_PUBLICKEYBYTES];
            uint8_t tmp_sk[CRYPTO_SECRETKEYBYTES];
            start = TICK();
            crypto_sign_keypair(tmp_pk, tmp_sk);
            keygen_times[level] += TOCK(start);
        }
    }

    /* =========================================================
     * PHASE 2: Generate real chain keys, precompute ONCE per level,
     * and build the certificate chain (parent signs child PK).
     * group_index=0 is reserved for chain construction signings.
     * ========================================================= */
    for (level = 0; level < chain_length; level++) {
        get_cert_name(chain[level].name, sizeof(chain[level].name), level, chain_length);

        crypto_sign_keypair(chain[level].pk, chain[level].sk);

        start = TICK();
        precomputing(chain[level].sk, PARAM_NUM);
        offline_times[level] = TOCK(start);  /* single measurement */

        if (level > 0) {
            chain[level].cert_siglen = CRYPTO_BYTES;
            crypto_sign_signature_with_precomputing(
                chain[level].cert_sig,
                &chain[level].cert_siglen,
                chain[level].pk,
                CRYPTO_PUBLICKEYBYTES,
                chain[level - 1].sk,
                0   /* group_index=0: chain construction */
            );
        }
    }

    /* =========================================================
     * PHASE 3: Online signing + verification loop.
     * group_index starts at 1 (group 0 used above) and cycles so
     * each iteration draws from a fresh group of precomputed nonces.
     * With PARAM_NUM=10000 and GROUP_NUM=50, there are 200 groups,
     * enough for far more iterations than DEFAULT_ITERATIONS.
     * ========================================================= */
    for (iter = 0; iter < iterations; iter++) {
        int client_idx = chain_length - 1;
        int group = 1 + iter;  /* avoid group 0, used for chain construction */

        /* Each CA (levels 0..chain_length-2) signs the next level's PK */
        for (level = 0; level < chain_length - 1; level++) {
            uint8_t tmp_sig[CRYPTO_BYTES];
            size_t tmp_siglen = CRYPTO_BYTES;
            start = TICK();
            crypto_sign_signature_with_precomputing(
                tmp_sig, &tmp_siglen,
                chain[level + 1].pk, CRYPTO_PUBLICKEYBYTES,
                chain[level].sk, group
            );
            online_times[level] += TOCK(start);
        }

        /* Client signs the actual message */
        uint8_t msg_sig[CRYPTO_BYTES];
        size_t msg_siglen = CRYPTO_BYTES;
        start = TICK();
        crypto_sign_signature_with_precomputing(
            msg_sig, &msg_siglen,
            message, MSG_SIZE,
            chain[client_idx].sk, group
        );
        online_times[client_idx] += TOCK(start);

        /* Verify message signature */
        start = TICK();
        int verify_result = crypto_sign_verify(
            msg_sig, msg_siglen,
            message, MSG_SIZE,
            chain[client_idx].pk
        );
        verify_times[client_idx] += TOCK(start);
        if (verify_result != 0) {
            fprintf(stderr, "Message signature verification failed!\n");
        }

        /* Verify certificate chain (bottom-up, using Phase 2 cert_sigs) */
        for (level = chain_length - 1; level > 0; level--) {
            start = TICK();
            verify_result = crypto_sign_verify(
                chain[level].cert_sig, chain[level].cert_siglen,
                chain[level].pk, CRYPTO_PUBLICKEYBYTES,
                chain[level - 1].pk
            );
            verify_times[level - 1] += TOCK(start);
            if (verify_result != 0) {
                fprintf(stderr, "Certificate verification failed at level %d!\n", level);
            }
        }
    }

    /* =========================================================
     * RESULTS
     * keygen / offline: Phase 1 averaged over iterations / Phase 2 single run
     * online / verify:  Phase 3 averaged over iterations
     * ========================================================= */
    printf("Results (Average Times per Operation):\n");
    printf("--------------------------------------------------------------------------------\n");

    for (level = 0; level < chain_length; level++) {
        char name_buf[32];
        get_cert_name(name_buf, sizeof(name_buf), level, chain_length);
        printf("%s:\n", name_buf);
        printf("  Key Generation:          %10.3f ms  (avg over %d runs)\n",
               (keygen_times[level] / iterations) * 1000.0, iterations);
        printf("  Offline Precomputation:  %10.3f ms  (single run)\n",
               offline_times[level] * 1000.0);
        printf("  Online Signing:          %10.3f us  (avg over %d runs)\n",
               (online_times[level] / iterations) * 1000000.0, iterations);
        printf("  Total Sign Time:         %10.3f ms\n",
               offline_times[level] * 1000.0 + (online_times[level] / iterations) * 1000.0);
        printf("  Verification:            %10.3f us  (avg over %d runs)\n",
               (verify_times[level] / iterations) * 1000000.0, iterations);
        printf("\n");
    }

    /* Calculate totals */
    double total_keygen = 0, total_offline = 0, total_online = 0, total_verify = 0;
    for (level = 0; level < chain_length; level++) {
        total_keygen  += keygen_times[level];
        total_offline += offline_times[level];
        total_online  += online_times[level];
        total_verify  += verify_times[level];
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Total Chain Operations:\n");
    printf("  Total Key Generation:          %10.3f ms\n",
           (total_keygen / iterations) * 1000.0);
    printf("  Total Offline Precomputation:  %10.3f ms\n",
           total_offline * 1000.0);
    printf("  Total Online Signing:          %10.3f us\n",
           (total_online / iterations) * 1000000.0);
    printf("  Total Sign Time:               %10.3f ms\n",
           total_offline * 1000.0 + (total_online / iterations) * 1000.0);
    printf("  Total Verification:            %10.3f us\n",
           (total_verify / iterations) * 1000000.0);
    printf("================================================================================\n");
}

static void benchmark_verify_batch_mt(int batch_count, int verify_threads) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t *messages = NULL;
    uint8_t *signatures = NULL;
    size_t *sig_lens = NULL;
    int failures = 0;
    double start;
    double elapsed;

    if (batch_count <= 0) {
        return;
    }

    messages = (uint8_t *)malloc((size_t)batch_count * MSG_SIZE);
    signatures = (uint8_t *)malloc((size_t)batch_count * CRYPTO_BYTES);
    sig_lens = (size_t *)malloc((size_t)batch_count * sizeof(size_t));
    if (messages == NULL || signatures == NULL || sig_lens == NULL) {
        fprintf(stderr, "Failed to allocate OO-Dilithium MT verify batch buffers\n");
        goto cleanup;
    }

    if (crypto_sign_keypair(pk, sk) != 0) {
        fprintf(stderr, "OO-Dilithium key generation failed for MT verify batch\n");
        goto cleanup;
    }

    precomputing(sk, PARAM_NUM);

    for (int i = 0; i < batch_count; i++) {
        uint8_t group_index = (uint8_t)(i % MSG_NUM);
        int sign_status;
        fill_verify_message(messages + (size_t)i * MSG_SIZE, i);
        sig_lens[i] = CRYPTO_BYTES;
        sign_status = crypto_sign_signature_with_precomputing(
                signatures + (size_t)i * CRYPTO_BYTES,
                &sig_lens[i],
                messages + (size_t)i * MSG_SIZE,
                MSG_SIZE,
                sk,
                group_index);
        if (sign_status < 0) {
            fprintf(stderr, "OO-Dilithium signing failed for MT verify batch item %d\n", i);
            goto cleanup;
        }
    }

    verify_threads = resolve_verify_threads(verify_threads);
    start = TICK();
#ifdef _OPENMP
    if (verify_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(verify_threads) reduction(+:failures)
        for (int i = 0; i < batch_count; i++) {
            if (crypto_sign_verify(
                    signatures + (size_t)i * CRYPTO_BYTES,
                    sig_lens[i],
                    messages + (size_t)i * MSG_SIZE,
                    MSG_SIZE,
                    pk) != 0) {
                failures++;
            }
        }
    } else
#endif
    {
        for (int i = 0; i < batch_count; i++) {
            if (crypto_sign_verify(
                    signatures + (size_t)i * CRYPTO_BYTES,
                    sig_lens[i],
                    messages + (size_t)i * MSG_SIZE,
                    MSG_SIZE,
                    pk) != 0) {
                failures++;
            }
        }
    }
    elapsed = TOCK(start);

    printf("MT verify batch:\n");
    printf("  Threads:         %10d\n", verify_threads);
    printf("  Batch size:      %10d\n", batch_count);
    printf("  Total verify:    %10.3f ms\n", elapsed * 1000.0);
    printf("  Avg verify:      %10.3f us\n", (elapsed * 1000000.0) / (double)batch_count);
    printf("  Throughput:      %10.0f sig/s\n", (double)batch_count / elapsed);
    printf("  Verify failures: %10d\n", failures);
    printf("================================================================================\n");

cleanup:
    free(messages);
    free(signatures);
    free(sig_lens);
}

int main(int argc, char *argv[]) {
    int iterations = DEFAULT_ITERATIONS;
    int chain_length = 3;  /* Default to 3-level chain */
    int verify_threads = 0;
    int verify_batch = 0;
    int i;
    
    /* Parse command line arguments */
    if (argc > 1) {
        chain_length = atoi(argv[1]);
        if (chain_length < 1 || chain_length > MAX_CHAIN_LENGTH) {
            fprintf(stderr, "Invalid chain length. Must be between 1 and %d. Using default: 3\n", MAX_CHAIN_LENGTH);
            chain_length = 3;
        }
    }
    
    if (argc > 2) {
        iterations = atoi(argv[2]);
        if (iterations <= 0) {
            fprintf(stderr, "Invalid iteration count. Using default: %d\n", DEFAULT_ITERATIONS);
            iterations = DEFAULT_ITERATIONS;
        }
    }

    if (argc > 3) {
        verify_threads = atoi(argv[3]);
        if (verify_threads < 0) {
            fprintf(stderr, "Invalid verify thread count. Disabling MT verify batch.\n");
            verify_threads = 0;
        }
    }

    if (argc > 4) {
        verify_batch = atoi(argv[4]);
        if (verify_batch <= 0) {
            fprintf(stderr, "Invalid MT verify batch size. Using iterations instead.\n");
            verify_batch = 0;
        }
    }
    
    /* Initialize message */
    for (i = 0; i < MSG_SIZE; i++) {
        message[i] = (uint8_t)(rand() % 256);
    }
    
    printf("================================================================================\n");
    printf("PRECOMPUTED DILITHIUM2 CERTIFICATE CHAIN BENCHMARK\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Algorithm:      Dilithium2 (precomputed)\n");
    printf("  Message size:   %d bytes\n", MSG_SIZE);
    printf("  Iterations:     %d\n", iterations);
    printf("  Chain length:   %d\n", chain_length);
    if (verify_threads > 0) {
        printf("  MT verify:      enabled (%d threads, batch=%d)\n", resolve_verify_threads(verify_threads), verify_batch > 0 ? verify_batch : iterations);
    } else {
        printf("  MT verify:      disabled\n");
    }
    printf("  Public key:     %lu bytes\n", (unsigned long)CRYPTO_PUBLICKEYBYTES);
    printf("  Signature:      %lu bytes\n", (unsigned long)CRYPTO_BYTES);
    printf("================================================================================\n");
    
    /* Run benchmark for specified chain length */
    benchmark_cert_chain(chain_length, iterations);

    if (verify_threads > 0) {
        benchmark_verify_batch_mt(verify_batch > 0 ? verify_batch : iterations, verify_threads);
    }
    
    return 0;
}