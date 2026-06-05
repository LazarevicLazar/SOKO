/*
 * Certificate Chain Benchmark: Online-Offline Falcon-512
 *
 * Measures:
 * - Key generation time per chain level
 * - Offline Phase (Precomputation) time per chain level
 * - Online Phase (Signing) time per chain level
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

/* Specific headers for Online-Offline Falcon */
#include "submodules/OO-FN-DSA/falcon-lazy/falcon.h"
#include "submodules/OO-FN-DSA/falcon-lazy/inner.h"

/* External declarations for Lazy/Online-Offline functions */
extern int sign_dyn_lazy_online(
    int8_t* sample1, int8_t* sample2, uint16_t* sample_target,
    int16_t *s2,
    const fpr *restrict f_fft, const fpr *restrict g_fft,
    const fpr *restrict F_fft, const fpr *restrict G_fft,
    const uint16_t *hm, unsigned logn, fpr *restrict tmp);

extern void sign_dyn_lazy_offline(
    inner_shake256_context *rng,
    const int8_t *restrict f, const int8_t *restrict g,
    const int8_t *restrict F, const int8_t *restrict G,
    const uint16_t *h,
    unsigned logn,
    int8_t* sample1, int8_t* sample2, uint16_t* sample_target,
    fpr *restrict f_fft, fpr *restrict g_fft,
    fpr *restrict F_fft, fpr *restrict G_fft);

/* Configuration */
#define MSG_SIZE 512
#define DEFAULT_ITERATIONS 100
#define FALCON_LOGN 9
#define MAX_CHAIN_LENGTH 10

/* Certificate structure for chain */
typedef struct {
    uint8_t *pk;
    uint8_t *sk;
    uint8_t *cert_sig;      /* Signature from parent (empty for root) */
    size_t cert_siglen;
    char name[32];
} Certificate;

/* Message buffer */
uint8_t message[MSG_SIZE];

static inline size_t maxsz(size_t a, size_t b) {
    return a > b ? a : b;
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

/* Function to build and benchmark the OO-Falcon certificate chain */
void benchmark_cert_chain_oo(int chain_length, int iterations, unsigned logn, shake256_context *rng) {
    Certificate chain[MAX_CHAIN_LENGTH];
    
    size_t pk_len = FALCON_PUBKEY_SIZE(logn);
    size_t sk_len = FALCON_PRIVKEY_SIZE(logn);
    size_t sig_len = FALCON_SIG_COMPRESSED_MAXSIZE(logn);
    size_t tmp_len = FALCON_TMPSIZE_KEYGEN(logn);
    tmp_len = maxsz(tmp_len, FALCON_TMPSIZE_SIGNDYN(logn));
    tmp_len = maxsz(tmp_len, FALCON_TMPSIZE_VERIFY(logn));
    
    uint8_t *tmp = malloc(tmp_len);
    if (!tmp) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return;
    }
    
    double keygen_times[MAX_CHAIN_LENGTH] = {0};
    double offline_times[MAX_CHAIN_LENGTH] = {0};
    double online_times[MAX_CHAIN_LENGTH] = {0};
    double verify_times[MAX_CHAIN_LENGTH] = {0};
    
    int iter, level;
    double start;
    
    printf("\n================================================================================\n");
    printf("CERTIFICATE CHAIN LENGTH: %d (Online-Offline Mode)\n", chain_length);
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
    
    /* Allocate memory for all certificates */
    for (level = 0; level < chain_length; level++) {
        chain[level].pk = malloc(pk_len);
        chain[level].sk = malloc(sk_len);
        chain[level].cert_sig = malloc(sig_len);
        
        if (!chain[level].pk || !chain[level].sk || !chain[level].cert_sig) {
            fprintf(stderr, "Failed to allocate memory for certificate %d\n", level);
            /* Cleanup */
            for (int j = 0; j <= level; j++) {
                free(chain[j].pk);
                free(chain[j].sk);
                free(chain[j].cert_sig);
            }
            free(tmp);
            return;
        }
    }
    
    /* Run iterations */
    for (iter = 0; iter < iterations; iter++) {
        /* 1. Generate Keypairs for the whole chain */
        for (level = 0; level < chain_length; level++) {
            get_cert_name(chain[level].name, sizeof(chain[level].name), level, chain_length);
            
            start = TICK();
            falcon_keygen_make(rng, logn, 
                             chain[level].sk, sk_len,
                             chain[level].pk, pk_len,
                             tmp, tmp_len);
            keygen_times[level] += TOCK(start);
        }
        
        /* 2. Sign Chain (Bottom-Up or Top-Down doesn't matter for independent timings, 
              but we simulate chain construction logic: Client signed by CA, CA signed by Root) */
        
        /* We iterate backwards from Client up to Root to match the signing flow logic 
           (though in OO benchmark we often just measure the ops independently) */
        for (level = chain_length - 1; level >= 0; level--) {
            size_t n = (size_t)1 << logn;
            int8_t f[n], g[n], F[n], G[n];
            uint16_t h[n], hm[n];
            int16_t sv[n];
            int8_t sample1[n], sample2[n];
            uint16_t sample_target[n];
            fpr f_fft[n], g_fft[n], F_fft[n], G_fft[n];
            fpr *ftmp = (fpr *)tmp;

            /* OFFLINE PHASE: key decode + modq_decode + precompute
             * (everything that can be done before the message is known) */
            start = TICK();
            {
                const uint8_t *sk = chain[level].sk;
                size_t u = 1, v;
                v = Zf(trim_i8_decode)(f, logn, Zf(max_fg_bits)[logn], sk + u, sk_len - u); u += v;
                v = Zf(trim_i8_decode)(g, logn, Zf(max_fg_bits)[logn], sk + u, sk_len - u); u += v;
                v = Zf(trim_i8_decode)(F, logn, Zf(max_FG_bits)[logn], sk + u, sk_len - u); u += v;
                Zf(complete_private)(G, f, g, F, logn, (uint8_t *)tmp);
                Zf(modq_decode)(h, logn, chain[level].pk + 1, pk_len - 1);
                sign_dyn_lazy_offline((inner_shake256_context *)rng,
                    f, g, F, G, h, logn,
                    sample1, sample2, sample_target,
                    f_fft, g_fft, F_fft, G_fft);
            }
            offline_times[level] += TOCK(start);

            /* ONLINE PHASE: nonce + hash + hash_to_point + online sign + encode
             * (everything that depends on the message) */
            uint8_t *target_sig_buf;
            size_t *target_sig_len;
            /* Use a correctly sized heap buffer for the client message signature */
            uint8_t *temp_msg_sig = NULL;
            size_t temp_msg_len = 0;
            if (level == chain_length - 1) {
                temp_msg_sig = malloc(FALCON_SIG_COMPRESSED_MAXSIZE(logn));
                target_sig_buf = temp_msg_sig;
                target_sig_len = &temp_msg_len;
            } else {
                target_sig_buf = chain[level + 1].cert_sig;
                target_sig_len = &chain[level + 1].cert_siglen;
            }

            start = TICK();
            {
                shake256_context hash_data;
                uint8_t nonce[40];
                shake256_extract(rng, nonce, 40);
                shake256_init(&hash_data);
                shake256_inject(&hash_data, nonce, 40);
                if (level == chain_length - 1) {
                    shake256_inject(&hash_data, message, MSG_SIZE);
                } else {
                    shake256_inject(&hash_data, chain[level + 1].pk, pk_len);
                }
                shake256_flip(&hash_data);
                Zf(hash_to_point_vartime)((inner_shake256_context *)&hash_data, hm, logn);

                sign_dyn_lazy_online(sample1, sample2, sample_target,
                    sv, f_fft, g_fft, F_fft, G_fft,
                    hm, logn, ftmp);

                target_sig_buf[0] = 0x30 + logn;
                memcpy(target_sig_buf + 1, nonce, 40);
                size_t v = Zf(comp_encode)(target_sig_buf + 41,
                    FALCON_SIG_COMPRESSED_MAXSIZE(logn) - 41, sv, logn);
                if (v == 0) fprintf(stderr, "comp_encode failed at level %d\n", level);
                *target_sig_len = 41 + v;
            }
            online_times[level] += TOCK(start);

            /* VERIFICATION */
            start = TICK();
            int verify_result = falcon_verify(target_sig_buf, *target_sig_len,
                FALCON_SIG_COMPRESSED,
                chain[level].pk, pk_len,
                (level == chain_length - 1) ? (const void *)message : (const void *)chain[level + 1].pk,
                (level == chain_length - 1) ? MSG_SIZE : pk_len,
                tmp, tmp_len);
            verify_times[level] += TOCK(start);
            free(temp_msg_sig);
            temp_msg_sig = NULL;

            if (verify_result != 0) {
                 fprintf(stderr, "Verification failed at level %d! Result is %d\n", level, verify_result);
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
        printf("  Key Generation:  %10.3f ms\n", (keygen_times[level] / iterations) * 1000.0);
        printf("  Offline Phase:   %10.3f us\n", (offline_times[level] / iterations) * 1000000.0);
        printf("  Online Phase:    %10.3f us\n", (online_times[level] / iterations) * 1000000.0);
        printf("  Verification:    %10.3f us\n", (verify_times[level] / iterations) * 1000000.0);
        printf("\n");
    }
    
    /* Calculate totals */
    double total_keygen = 0, total_offline = 0, total_online = 0, total_verify = 0;
    for (level = 0; level < chain_length; level++) {
        total_keygen += keygen_times[level];
        total_offline += offline_times[level];
        total_online += online_times[level];
        total_verify += verify_times[level];
    }
    
    printf("--------------------------------------------------------------------------------\n");
    printf("Total Chain Operations (Average):\n");
    printf("  Total Key Generation:  %10.3f ms\n", (total_keygen / iterations) * 1000.0);
    printf("  Total Offline Phase:   %10.3f ms\n", (total_offline / iterations) * 1000.0);
    printf("  Total Online Phase:    %10.3f us\n", (total_online / iterations) * 1000000.0);
    printf("  Total Verification:    %10.3f us\n", (total_verify / iterations) * 1000000.0);
    printf("================================================================================\n");
    
    /* Cleanup */
    for (level = 0; level < chain_length; level++) {
        free(chain[level].pk);
        free(chain[level].sk);
        free(chain[level].cert_sig);
    }
    free(tmp);
}

int main(int argc, char *argv[]) {
    int iterations = DEFAULT_ITERATIONS;
    int chain_length = 3;  /* Default to 3-level chain */
    unsigned logn = FALCON_LOGN;
    shake256_context rng;
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
    
    /* Initialize RNG */
    if (shake256_init_prng_from_system(&rng) != 0) {
        fprintf(stderr, "RNG initialization failed\n");
        return 1;
    }
    
    /* Initialize message */
    for (i = 0; i < MSG_SIZE; i++) {
        message[i] = (uint8_t)(rand() % 256);
    }
    
    size_t pk_len = FALCON_PUBKEY_SIZE(logn);
    size_t sig_len = FALCON_SIG_PADDED_SIZE(logn) + 256;
    
    printf("================================================================================\n");
    printf("FALCON-512 ONLINE-OFFLINE BENCHMARK\n");
    printf("================================================================================\n");
    printf("Configuration:\n");
    printf("  Algorithm:      Falcon-512 (Online-Offline)\n");
    printf("  Message size:   %d bytes\n", MSG_SIZE);
    printf("  Iterations:     %d\n", iterations);
    printf("  Chain length:   %d\n", chain_length);
    printf("  Public key:     %lu bytes\n", (unsigned long)pk_len);
    printf("  Signature:      %lu bytes\n", (unsigned long)sig_len);
    printf("================================================================================\n");
    
    /* Run benchmark */
    benchmark_cert_chain_oo(chain_length, iterations, logn, &rng);
    
    return 0;
}