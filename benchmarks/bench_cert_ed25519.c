/*
 * Certificate Chain Benchmark: ED25519
 *
 * Measures:
 * - Key generation time per chain level
 * - Signing time per chain level
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

#include <sodium.h>

/* Configuration */
#define MSG_SIZE 512
#define DEFAULT_ITERATIONS 100
#define MAX_CHAIN_LENGTH 10
#define ED25519_SIG_SIZE 64
#define ED25519_PK_SIZE 32

/* Certificate structure for chain */
typedef struct {
    unsigned char pk[ED25519_PK_SIZE];
    unsigned char sk[crypto_sign_SECRETKEYBYTES];
    unsigned char *cert_sig;    /* Signature from parent (empty for root) */
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

/* Function to build and benchmark a certificate chain */
void benchmark_cert_chain(int chain_length, int iterations) {
    Certificate chain[MAX_CHAIN_LENGTH];
    
    double keygen_times[MAX_CHAIN_LENGTH] = {0};
    double sign_times[MAX_CHAIN_LENGTH] = {0};
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
    
    /* Initialize certificate structures */
    for (level = 0; level < chain_length; level++) {
        chain[level].cert_sig = malloc(ED25519_SIG_SIZE);
        if (!chain[level].cert_sig) {
            fprintf(stderr, "Failed to allocate memory for certificate %d\n", level);
            /* Cleanup */
            for (int j = 0; j < level; j++) {
                free(chain[j].cert_sig);
            }
            return;
        }
    }
    
    /* Run iterations */
    for (iter = 0; iter < iterations; iter++) {
        /* Build certificate chain from root to client */
        for (level = 0; level < chain_length; level++) {
            unsigned long long sig_len = ED25519_SIG_SIZE;

            get_cert_name(chain[level].name, sizeof(chain[level].name), level, chain_length);
            
            /* Step 1: Generate keypair for this level */
            start = TICK();
            if (crypto_sign_keypair(chain[level].pk, chain[level].sk) != 0) {
                fprintf(stderr, "Failed to generate Ed25519 keypair for certificate %d\n", level);
                return;
            }
            keygen_times[level] += TOCK(start);
            
            /* Step 2: Get certificate signed by parent (if not root) */
            if (level > 0) {
                /* Parent signs this level's public key */
                chain[level].cert_siglen = ED25519_SIG_SIZE;
                
                start = TICK();
                if (crypto_sign_detached(
                        chain[level].cert_sig,
                        &sig_len,
                        chain[level].pk,
                        ED25519_PK_SIZE,
                        chain[level - 1].sk) != 0) {
                    fprintf(stderr, "Failed to sign certificate for level %d\n", level);
                    return;
                }
                chain[level].cert_siglen = (size_t)sig_len;
                sign_times[level - 1] += TOCK(start);  /* Attribute to parent */
            }
        }
        
        /* Step 3: Client signs the actual message */
        unsigned char *msg_sig = malloc(ED25519_SIG_SIZE);
        if (!msg_sig) {
            fprintf(stderr, "Failed to allocate message signature buffer\n");
            break;
        }
        
        unsigned long long msg_siglen = ED25519_SIG_SIZE;
        int client_idx = chain_length - 1;
        
        start = TICK();
        if (crypto_sign_detached(
                msg_sig,
                &msg_siglen,
                message,
                MSG_SIZE,
                chain[client_idx].sk) != 0) {
            fprintf(stderr, "Failed to sign client message\n");
            free(msg_sig);
            return;
        }
        sign_times[client_idx] += TOCK(start);
        
        /* Step 4: Verify the entire chain (bottom-up) */
        /* First verify the message signature */
        start = TICK();
        int verify_result = crypto_sign_verify_detached(msg_sig, message, MSG_SIZE, chain[client_idx].pk);
        verify_times[client_idx] += TOCK(start);
        
        if (verify_result != 0) {
            fprintf(stderr, "Message signature verification failed!\n");
        }
        
        free(msg_sig);
        
        /* Then verify each certificate in the chain (from client up to root) */
        for (level = chain_length - 1; level > 0; level--) {
            start = TICK();
            verify_result = crypto_sign_verify_detached(
                chain[level].cert_sig,
                chain[level].pk,
                ED25519_PK_SIZE,
                chain[level - 1].pk);
            verify_times[level - 1] += TOCK(start);  /* Attribute to parent */

            if (verify_result != 0) {
                fprintf(stderr, "Certificate verification failed at level %d!\n", level);
            }
        }
    }
    
    /* Print results */
    printf("Results (Average Times per Operation):\n");
    printf("--------------------------------------------------------------------------------\n");
    
    for (level = 0; level < chain_length; level++) {
        char name_buf[32];
        get_cert_name(name_buf, sizeof(name_buf), level, chain_length);
        printf("%s:\n", name_buf);
        printf("  Key Generation:  %10.3f us\n", (keygen_times[level] / iterations) * 1000000.0);
        printf("  Signing:         %10.3f us\n", (sign_times[level] / iterations) * 1000000.0);
        printf("  Verification:    %10.3f us\n", (verify_times[level] / iterations) * 1000000.0);
        printf("\n");
    }
    
    /* Calculate totals */
    double total_keygen = 0, total_sign = 0, total_verify = 0;
    for (level = 0; level < chain_length; level++) {
        total_keygen += keygen_times[level];
        total_sign += sign_times[level];
        total_verify += verify_times[level];
    }
    
    printf("--------------------------------------------------------------------------------\n");
    printf("Total Chain Operations (Average):\n");
    printf("  Total Key Generation:  %10.3f us\n", (total_keygen / iterations) * 1000000.0);
    printf("  Total Signing:         %10.3f us\n", (total_sign / iterations) * 1000000.0);
    printf("  Total Verification:    %10.3f us\n", (total_verify / iterations) * 1000000.0);
    printf("================================================================================\n");
    
    /* Final cleanup */
    for (level = 0; level < chain_length; level++) {
        free(chain[level].cert_sig);
    }
}

static void benchmark_verify_batch_mt(int batch_count, int verify_threads) {
    unsigned char pk_bytes[ED25519_PK_SIZE];
    unsigned char sk_bytes[crypto_sign_SECRETKEYBYTES];
    unsigned char *messages = NULL;
    unsigned char *signatures = NULL;
    int failures = 0;
    double start;
    double elapsed;

    if (batch_count <= 0) {
        return;
    }

    if (crypto_sign_keypair(pk_bytes, sk_bytes) != 0) {
        fprintf(stderr, "Failed to generate Ed25519 keypair for MT verify benchmark\n");
        goto cleanup;
    }

    messages = (unsigned char *)malloc((size_t)batch_count * MSG_SIZE);
    signatures = (unsigned char *)malloc((size_t)batch_count * ED25519_SIG_SIZE);
    if (messages == NULL || signatures == NULL) {
        fprintf(stderr, "Failed to allocate Ed25519 MT verify batch buffers\n");
        goto cleanup;
    }

    for (int i = 0; i < batch_count; i++) {
        fill_verify_message(messages + (size_t)i * MSG_SIZE, i);
        if (crypto_sign_detached(
                signatures + (size_t)i * ED25519_SIG_SIZE,
                NULL,
                messages + (size_t)i * MSG_SIZE,
                MSG_SIZE,
                sk_bytes) != 0) {
            fprintf(stderr, "Failed to sign Ed25519 batch item %d\n", i);
            goto cleanup;
        }
    }

    verify_threads = resolve_verify_threads(verify_threads);
    start = TICK();
#ifdef _OPENMP
    if (verify_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(verify_threads) reduction(+:failures)
        for (int i = 0; i < batch_count; i++) {
            if (crypto_sign_verify_detached(
                    signatures + (size_t)i * ED25519_SIG_SIZE,
                    messages + (size_t)i * MSG_SIZE,
                    MSG_SIZE,
                    pk_bytes) != 0) {
                failures++;
            }
        }
    } else
#endif
    {
        for (int i = 0; i < batch_count; i++) {
            if (crypto_sign_verify_detached(
                    signatures + (size_t)i * ED25519_SIG_SIZE,
                    messages + (size_t)i * MSG_SIZE,
                    MSG_SIZE,
                    pk_bytes) != 0) {
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
    
    if (sodium_init() < 0) {
        fprintf(stderr, "libsodium initialization failed\n");
        return 1;
    }

    /* Initialize message */
    for (i = 0; i < MSG_SIZE; i++) {
        message[i] = (uint8_t)(rand() % 256);
    }
    
    printf("================================================================================\n");
    printf("ED25519 CERTIFICATE CHAIN BENCHMARK\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Algorithm:      ED25519 (EdDSA)\n");
    printf("  Chain backend:  libsodium detached sign/verify\n");
    printf("  Message size:   %d bytes\n", MSG_SIZE);
    printf("  Iterations:     %d\n", iterations);
    printf("  Chain length:   %d\n", chain_length);
    if (verify_threads > 0) {
        printf("  MT verify:      enabled (%d threads, batch=%d)\n", resolve_verify_threads(verify_threads), verify_batch > 0 ? verify_batch : iterations);
        printf("  MT backend:     libsodium detached verify\n");
    } else {
        printf("  MT verify:      disabled\n");
    }
    printf("  Public key:     %d bytes\n", ED25519_PK_SIZE);
    printf("  Signature:      %d bytes\n", ED25519_SIG_SIZE);
    printf("================================================================================\n");
    
    /* Run benchmark for specified chain length */
    benchmark_cert_chain(chain_length, iterations);

    if (verify_threads > 0) {
        benchmark_verify_batch_mt(verify_batch > 0 ? verify_batch : iterations, verify_threads);
    }
    
    return 0;
}