/*
 * Certificate Chain Benchmark: Online-Offline Ed25519
 *
 * Measures:
 * - Key generation time per chain level
 * - Offline phase (token generation) time per chain level
 * - Online signing time per chain level
 * - Verification time per chain level
 * - Total chain operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

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

#include "oo_ed25519.h"

/* Configuration */
#define MSG_SIZE 512
#define DEFAULT_ITERATIONS 100
#define MAX_CHAIN_LENGTH 10

/* Certificate structure for chain */
typedef struct {
    uint8_t pk[OOED25519_PK_SIZE];
    uint8_t sk[OOED25519_SK_SIZE];
    uint8_t a[OOED25519_SCALAR_SIZE];
    uint8_t cert_sig[OOED25519_SIG_SIZE];
    size_t cert_siglen;
    ooed25519_token token;
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

void benchmark_cert_chain_ooed25519(int chain_length, int iterations) {
    Certificate chain[MAX_CHAIN_LENGTH];

    double keygen_times[MAX_CHAIN_LENGTH] = {0};
    double offline_times[MAX_CHAIN_LENGTH] = {0};
    double online_times[MAX_CHAIN_LENGTH] = {0};
    double verify_times[MAX_CHAIN_LENGTH] = {0};

    int iter, level;
    double start;

    printf("\n================================================================================\n");
    printf("CERTIFICATE CHAIN LENGTH: %d (Online-Offline Ed25519)\n", chain_length);
    printf("================================================================================\n");

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

    for (iter = 0; iter < iterations; iter++) {
        /* Keygen + offline token generation */
        for (level = 0; level < chain_length; level++) {
            get_cert_name(chain[level].name, sizeof(chain[level].name), level, chain_length);

            start = TICK();
            ooed25519_keypair(chain[level].pk, chain[level].sk);
            keygen_times[level] += TOCK(start);

            if (ooed25519_expand_secret(chain[level].sk, chain[level].a) != 0) {
                fprintf(stderr, "Failed to expand secret at level %d\n", level);
                return;
            }

            start = TICK();
            if (ooed25519_token_generate(&chain[level].token) != 0) {
                fprintf(stderr, "Failed to generate token at level %d\n", level);
                return;
            }
            offline_times[level] += TOCK(start);
        }

        /* Online signing: parent signs child PK, client signs message */
        for (level = 0; level < chain_length - 1; level++) {
            chain[level + 1].cert_siglen = OOED25519_SIG_SIZE;
            start = TICK();
            if (ooed25519_sign_online(chain[level + 1].cert_sig,
                    chain[level + 1].pk, OOED25519_PK_SIZE,
                    chain[level].pk, chain[level].a,
                    &chain[level].token) != 0) {
                fprintf(stderr, "Online sign failed at level %d\n", level);
                return;
            }
            online_times[level] += TOCK(start);
        }

        /* Client signs the actual message */
        {
            int client_idx = chain_length - 1;
            uint8_t msg_sig[OOED25519_SIG_SIZE];
            start = TICK();
            if (ooed25519_sign_online(msg_sig, message, MSG_SIZE,
                    chain[client_idx].pk, chain[client_idx].a,
                    &chain[client_idx].token) != 0) {
                fprintf(stderr, "Client sign failed\n");
                return;
            }
            online_times[client_idx] += TOCK(start);

            start = TICK();
            if (ooed25519_verify(msg_sig, message, MSG_SIZE, chain[client_idx].pk) != 0) {
                fprintf(stderr, "Message signature verification failed!\n");
            }
            verify_times[client_idx] += TOCK(start);
        }

        /* Verify certificate chain (client -> root) */
        for (level = chain_length - 1; level > 0; level--) {
            start = TICK();
            if (ooed25519_verify(chain[level].cert_sig,
                    chain[level].pk, OOED25519_PK_SIZE,
                    chain[level - 1].pk) != 0) {
                fprintf(stderr, "Certificate verification failed at level %d!\n", level);
            }
            verify_times[level - 1] += TOCK(start);
        }
    }

    printf("Results (Average Times per Operation):\n");
    printf("--------------------------------------------------------------------------------\n");

    for (level = 0; level < chain_length; level++) {
        char name_buf[32];
        get_cert_name(name_buf, sizeof(name_buf), level, chain_length);
        printf("%s:\n", name_buf);
        printf("  Key Generation:  %10.3f us\n", (keygen_times[level] / iterations) * 1000000.0);
        printf("  Offline Phase:   %10.3f us\n", (offline_times[level] / iterations) * 1000000.0);
        printf("  Online Phase:    %10.3f us\n", (online_times[level] / iterations) * 1000000.0);
        printf("  Verification:    %10.3f us\n", (verify_times[level] / iterations) * 1000000.0);
        printf("\n");
    }

    double total_keygen = 0, total_offline = 0, total_online = 0, total_verify = 0;
    for (level = 0; level < chain_length; level++) {
        total_keygen += keygen_times[level];
        total_offline += offline_times[level];
        total_online += online_times[level];
        total_verify += verify_times[level];
    }

    printf("--------------------------------------------------------------------------------\n");
    printf("Total Chain Operations (Average):\n");
    printf("  Total Key Generation:  %10.3f us\n", (total_keygen / iterations) * 1000000.0);
    printf("  Total Offline Phase:   %10.3f us\n", (total_offline / iterations) * 1000000.0);
    printf("  Total Online Phase:    %10.3f us\n", (total_online / iterations) * 1000000.0);
    printf("  Total Verification:    %10.3f us\n", (total_verify / iterations) * 1000000.0);
    printf("================================================================================\n");
}

static void benchmark_verify_batch_mt(int batch_count, int verify_threads) {
    uint8_t pk[OOED25519_PK_SIZE];
    uint8_t sk[OOED25519_SK_SIZE];
    uint8_t a[OOED25519_SCALAR_SIZE];
    uint8_t *messages = NULL;
    uint8_t *signatures = NULL;
    ooed25519_token *tokens = NULL;
    int failures = 0;
    double start;
    double elapsed;

    if (batch_count <= 0) {
        return;
    }

    messages = (uint8_t *)malloc((size_t)batch_count * MSG_SIZE);
    signatures = (uint8_t *)malloc((size_t)batch_count * OOED25519_SIG_SIZE);
    tokens = (ooed25519_token *)malloc((size_t)batch_count * sizeof(ooed25519_token));
    if (messages == NULL || signatures == NULL || tokens == NULL) {
        fprintf(stderr, "Failed to allocate OO-Ed25519 MT verify batch buffers\n");
        goto cleanup;
    }

    if (ooed25519_keypair(pk, sk) != 0) {
        fprintf(stderr, "OO-Ed25519 key generation failed for MT verify batch\n");
        goto cleanup;
    }
    if (ooed25519_expand_secret(sk, a) != 0) {
        fprintf(stderr, "OO-Ed25519 secret expansion failed for MT verify batch\n");
        goto cleanup;
    }

    for (int i = 0; i < batch_count; i++) {
        fill_verify_message(messages + (size_t)i * MSG_SIZE, i);
        if (ooed25519_token_generate(&tokens[i]) != 0) {
            fprintf(stderr, "OO-Ed25519 token generation failed for batch item %d\n", i);
            goto cleanup;
        }
        if (ooed25519_sign_online(
                signatures + (size_t)i * OOED25519_SIG_SIZE,
                messages + (size_t)i * MSG_SIZE,
                MSG_SIZE,
                pk,
                a,
                &tokens[i]) != 0) {
            fprintf(stderr, "OO-Ed25519 signing failed for batch item %d\n", i);
            goto cleanup;
        }
    }

    verify_threads = resolve_verify_threads(verify_threads);
    start = TICK();
#ifdef _OPENMP
    if (verify_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(verify_threads) reduction(+:failures)
        for (int i = 0; i < batch_count; i++) {
            if (ooed25519_verify(
                    signatures + (size_t)i * OOED25519_SIG_SIZE,
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
            if (ooed25519_verify(
                    signatures + (size_t)i * OOED25519_SIG_SIZE,
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
    free(tokens);
}

int main(int argc, char *argv[]) {
    int iterations = DEFAULT_ITERATIONS;
    int chain_length = 3;
    int verify_threads = 0;
    int verify_batch = 0;
    int i;

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

    if (ooed25519_init() != 0) {
        fprintf(stderr, "libsodium initialization failed\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    for (i = 0; i < MSG_SIZE; i++) {
        message[i] = (uint8_t)(rand() & 0xFF);
    }

    printf("================================================================================\n");
    printf("ED25519 ONLINE-OFFLINE CERTIFICATE CHAIN BENCHMARK\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Algorithm:      Ed25519 (Online-Offline)\n");
    printf("  Message size:   %d bytes\n", MSG_SIZE);
    printf("  Iterations:     %d\n", iterations);
    printf("  Chain length:   %d\n", chain_length);
    if (verify_threads > 0) {
        printf("  MT verify:      enabled (%d threads, batch=%d)\n", resolve_verify_threads(verify_threads), verify_batch > 0 ? verify_batch : iterations);
    } else {
        printf("  MT verify:      disabled\n");
    }
    printf("  Public key:     %d bytes\n", OOED25519_PK_SIZE);
    printf("  Signature:      %d bytes\n", OOED25519_SIG_SIZE);
    printf("================================================================================\n");

    benchmark_cert_chain_ooed25519(chain_length, iterations);

    if (verify_threads > 0) {
        benchmark_verify_batch_mt(verify_batch > 0 ? verify_batch : iterations, verify_threads);
    }

    return 0;
}
