#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "soko_falcon_backends.h"

#define SIM_FALCON_LOGN SOKO_FALCON_LOGN
#define SIM_FALCON_N SOKO_FALCON_N
#define DEFAULT_QUEUE_CAPACITY 2048
#define DEFAULT_CONSUME_TARGET_TOKENS 327680
#define DEFAULT_GPU_BATCH_TOKENS 256
#define MAX_RATE_POINTS 32

static const int g_default_token_rates[] = {1024, 2048, 4096, 8192, 16384, 32768};

#ifdef _WIN32
static double get_time_seconds(void)
{
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}
#endif

static void *xmalloc(unsigned long long size)
{
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "Allocation failed (%llu bytes)\n", size);
        exit(1);
    }
    return p;
}

typedef struct {
    const char *csv_path;
    int queue_capacity;
    int consume_target_tokens;
    int gpu_batch_tokens;
    int cpu_mt_threads;
    int token_rates[MAX_RATE_POINTS];
    int token_rate_len;
} sim_config_t;

typedef struct {
    const char *algorithm;
    int token_rate;
    int queue_capacity;
    int consume_target_tokens;
    int refill_threshold_tokens;
    double offered_load;
    double measured_seconds;
    uint64_t produced_tokens;
    uint64_t served_tokens;
    uint64_t starved_requests;
    double refill_throughput;
    double served_token_rate;
    double service_ratio;
} sim_result_t;

static void print_usage(const char *prog)
{
    printf("Usage: %s [--csv <path>] [--queue <tokens>] [--consume-target <tokens>] [--rates <comma-list>] [--gpu-batch <tokens>] [--cpu-mt-threads <count>]\n", prog);
    printf("Simulates consume-only token demand against threshold-based refill for cpu, cpumt, and soko.\n");
    printf("Defaults: queue=%d consume-target=%d rates=1024,2048,4096,8192,16384,32768 gpu-batch=%d cpu-mt-threads=max\n",
        DEFAULT_QUEUE_CAPACITY, DEFAULT_CONSUME_TARGET_TOKENS, DEFAULT_GPU_BATCH_TOKENS);
}

static int parse_positive_int(const char *s, int *out)
{
    char *end = NULL;
    long v;

    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v <= 0 || v > 100000000L) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_rate_list(const char *s, sim_config_t *cfg)
{
    unsigned long long len = (unsigned long long)strlen(s) + 1ULL;
    char *buf = (char *)malloc(len);
    char *tok;
    int count = 0;

    if (buf == NULL) {
        return 0;
    }

    memcpy(buf, s, len);

    tok = strtok(buf, ",");
    while (tok != NULL) {
        int v;
        if (count >= MAX_RATE_POINTS || !parse_positive_int(tok, &v)) {
            free(buf);
            return 0;
        }
        cfg->token_rates[count++] = v;
        tok = strtok(NULL, ",");
    }

    free(buf);
    cfg->token_rate_len = count;
    return count > 0;
}

static int parse_args(int argc, char **argv, sim_config_t *cfg)
{
    int i;

    cfg->csv_path = NULL;
    cfg->queue_capacity = DEFAULT_QUEUE_CAPACITY;
    cfg->consume_target_tokens = DEFAULT_CONSUME_TARGET_TOKENS;
    cfg->gpu_batch_tokens = DEFAULT_GPU_BATCH_TOKENS;
    cfg->cpu_mt_threads = 0;
    cfg->token_rate_len = (int)(sizeof(g_default_token_rates) / sizeof(g_default_token_rates[0]));
    memcpy(cfg->token_rates, g_default_token_rates, sizeof(g_default_token_rates));

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            if (i + 1 >= argc) {
                return 0;
            }
            cfg->csv_path = argv[++i];
        } else if (strcmp(argv[i], "--queue") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg->queue_capacity) || cfg->queue_capacity < 2) {
                return 0;
            }
        } else if (strcmp(argv[i], "--consume-target") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg->consume_target_tokens)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--rates") == 0) {
            if (i + 1 >= argc || !parse_rate_list(argv[++i], cfg)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--gpu-batch") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg->gpu_batch_tokens)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--cpu-mt-threads") == 0) {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], &cfg->cpu_mt_threads)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            return 0;
        }
    }

    return 1;
}

typedef soko_falcon_token_t falcon_token_t;

typedef soko_falcon_shared_t falcon_shared_t;

typedef struct {
    falcon_shared_t falcon;
    double refill_batch_seconds;
    int batch_tokens;
} oofalcon_cpu_state_t;

typedef struct {
    falcon_shared_t falcon;
    double refill_batch_seconds;
    int batch_tokens;
    int cpu_mt_threads;
} oofalcon_cpumt_state_t;

typedef struct {
    falcon_shared_t falcon;
    uint32_t next_seed;
    int gpu_batch_tokens;
    double refill_batch_seconds;
    int8_t *sample1_bank;
    int8_t *sample2_bank;
    uint16_t *target_bank;
} soko_state_t;

typedef struct sim_algorithm sim_algorithm_t;

struct sim_algorithm {
    const char *name;
    unsigned long long state_size;
    unsigned long long token_size;
    int preferred_batch_tokens;
    int (*init)(void *state, const sim_config_t *cfg);
    void (*destroy)(void *state);
    int (*produce_batch)(void *state, void *token_buf, int max_tokens);
    void (*configure_refill)(void *state, const sim_config_t *cfg, int token_rate, int *batch_tokens, int *refill_threshold_tokens);
};

static int resolve_cpu_mt_threads(int requested)
{
    return soko_resolve_cpu_mt_threads(requested);
}

static int rate_based_refill_threshold(double refill_batch_seconds, int token_rate, int queue_capacity)
{
    int threshold;

    threshold = (int)ceil(1.2 * refill_batch_seconds * (double)token_rate);
    if (threshold < 1) {
        threshold = 1;
    }
    if (threshold >= queue_capacity) {
        threshold = queue_capacity - 1;
    }
    return threshold;
}

static int oofalcon_cpu_produce_batch(void *state, void *token_buf, int max_tokens);
static int oofalcon_cpumt_produce_batch(void *state, void *token_buf, int max_tokens);
static void soko_configure_refill(void *state, const sim_config_t *cfg, int token_rate, int *batch_tokens, int *refill_threshold_tokens);

static int prepare_falcon_key_state(falcon_shared_t *s)
{
    return soko_prepare_falcon_key_state(s);
}

static void destroy_falcon_key_state(falcon_shared_t *s)
{
    soko_destroy_falcon_key_state(s);
}

static int oofalcon_cpu_init(void *state, const sim_config_t *cfg)
{
    oofalcon_cpu_state_t *s = (oofalcon_cpu_state_t *)state;
    falcon_token_t *tokens;
    double start;

    memset(s, 0, sizeof(*s));
    s->batch_tokens = (cfg->gpu_batch_tokens > 0) ? cfg->gpu_batch_tokens : DEFAULT_GPU_BATCH_TOKENS;
    if (!prepare_falcon_key_state(&s->falcon)) {
        return 0;
    }

    tokens = (falcon_token_t *)xmalloc((unsigned long long)s->batch_tokens * sizeof(*tokens));
    (void)oofalcon_cpu_produce_batch(s, tokens, s->batch_tokens);

    start = get_time_seconds();
    (void)oofalcon_cpu_produce_batch(s, tokens, s->batch_tokens);
    s->refill_batch_seconds = get_time_seconds() - start;

    free(tokens);
    return 1;
}

static void oofalcon_cpu_destroy(void *state)
{
    destroy_falcon_key_state(&((oofalcon_cpu_state_t *)state)->falcon);
}

static int oofalcon_cpu_produce_batch(void *state, void *token_buf, int max_tokens)
{
    oofalcon_cpu_state_t *s = (oofalcon_cpu_state_t *)state;
    return soko_cpu_generate_presign_tokens(s->falcon.logn, max_tokens,
        s->falcon.h_monty, (falcon_token_t *)token_buf);
}

static void oofalcon_cpumt_configure_refill(void *state, const sim_config_t *cfg, int token_rate, int *batch_tokens, int *refill_threshold_tokens)
{
    oofalcon_cpumt_state_t *s = (oofalcon_cpumt_state_t *)state;
    *batch_tokens = s->batch_tokens;
    *refill_threshold_tokens = rate_based_refill_threshold(s->refill_batch_seconds, token_rate, cfg->queue_capacity);
}

static int oofalcon_cpumt_init(void *state, const sim_config_t *cfg)
{
    oofalcon_cpumt_state_t *s = (oofalcon_cpumt_state_t *)state;
    falcon_token_t *tokens;
    double start;

    memset(s, 0, sizeof(*s));
    s->batch_tokens = (cfg->gpu_batch_tokens > 0) ? cfg->gpu_batch_tokens : DEFAULT_GPU_BATCH_TOKENS;
    s->cpu_mt_threads = resolve_cpu_mt_threads(cfg->cpu_mt_threads);
    if (!prepare_falcon_key_state(&s->falcon)) {
        return 0;
    }

    tokens = (falcon_token_t *)xmalloc((unsigned long long)s->batch_tokens * sizeof(*tokens));
    (void)oofalcon_cpumt_produce_batch(s, tokens, s->batch_tokens);

    start = get_time_seconds();
    (void)oofalcon_cpumt_produce_batch(s, tokens, s->batch_tokens);
    s->refill_batch_seconds = get_time_seconds() - start;

    free(tokens);
    return 1;
}

static void oofalcon_cpumt_destroy(void *state)
{
    destroy_falcon_key_state(&((oofalcon_cpumt_state_t *)state)->falcon);
}

static int oofalcon_cpumt_produce_batch(void *state, void *token_buf, int max_tokens)
{
    oofalcon_cpumt_state_t *s = (oofalcon_cpumt_state_t *)state;
    return soko_cpumt_generate_presign_tokens(s->falcon.logn, max_tokens,
        s->falcon.h_monty, (falcon_token_t *)token_buf, s->cpu_mt_threads);
}

static void oofalcon_cpu_configure_refill(void *state, const sim_config_t *cfg, int token_rate, int *batch_tokens, int *refill_threshold_tokens)
{
    oofalcon_cpu_state_t *s = (oofalcon_cpu_state_t *)state;
    *batch_tokens = s->batch_tokens;
    *refill_threshold_tokens = rate_based_refill_threshold(s->refill_batch_seconds, token_rate, cfg->queue_capacity);
}

static int load_gpu_backend_once(void)
{
    return soko_gpu_backend_load();
}

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

static int soko_init(void *state, const sim_config_t *cfg)
{
    soko_state_t *s = (soko_state_t *)state;
    int warmup_tokens;
    memset(s, 0, sizeof(*s));
    if (!prepare_falcon_key_state(&s->falcon)) {
        return 0;
    }
    s->gpu_batch_tokens = cfg->gpu_batch_tokens;
    s->next_seed = 0x13579BDFu;
    s->sample1_bank = (int8_t *)xmalloc((unsigned long long)s->gpu_batch_tokens * SIM_FALCON_N);
    s->sample2_bank = (int8_t *)xmalloc((unsigned long long)s->gpu_batch_tokens * SIM_FALCON_N);
    s->target_bank = (uint16_t *)xmalloc((unsigned long long)s->gpu_batch_tokens * SIM_FALCON_N * sizeof(uint16_t));
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    return 0;
#else
    if (load_gpu_backend_once() != 0) {
        return 0;
    }

    warmup_tokens = (s->gpu_batch_tokens > 0) ? s->gpu_batch_tokens : DEFAULT_GPU_BATCH_TOKENS;
    {
        if (gpu_generate_presign_bank_dynamic(s->falcon.h, s->falcon.logn, s->next_seed,
                warmup_tokens, s->sample1_bank, s->sample2_bank, s->target_bank) != 0) {
            return 0;
        }
        s->next_seed += (uint32_t)(0x9E3779B1u + (uint32_t)warmup_tokens);
    }

    {
        double start = get_time_seconds();
        if (gpu_generate_presign_bank_dynamic(s->falcon.h, s->falcon.logn, s->next_seed,
                warmup_tokens, s->sample1_bank, s->sample2_bank, s->target_bank) != 0) {
            return 0;
        }
        s->refill_batch_seconds = get_time_seconds() - start;
        s->next_seed += (uint32_t)(0x9E3779B1u + (uint32_t)warmup_tokens);
    }
    return 1;
#endif
}

static void soko_configure_refill(void *state, const sim_config_t *cfg, int token_rate, int *batch_tokens, int *refill_threshold_tokens)
{
    soko_state_t *s = (soko_state_t *)state;
    *batch_tokens = s->gpu_batch_tokens;
    *refill_threshold_tokens = rate_based_refill_threshold(s->refill_batch_seconds, token_rate, cfg->queue_capacity);
}

static void soko_destroy(void *state)
{
    soko_state_t *s = (soko_state_t *)state;
    free(s->sample1_bank);
    free(s->sample2_bank);
    free(s->target_bank);
    destroy_falcon_key_state(&s->falcon);
    memset(s, 0, sizeof(*s));
}

static int soko_produce_batch(void *state, void *token_buf, int max_tokens)
{
#if !defined(OOFALCON_USE_CUDA) || !defined(_WIN32)
    (void)state;
    (void)token_buf;
    (void)max_tokens;
    return 0;
#else
    soko_state_t *s = (soko_state_t *)state;
    int batch = max_tokens;
    if (batch > s->gpu_batch_tokens) {
        batch = s->gpu_batch_tokens;
    }
    if (batch <= 0) {
        return 0;
    }
    if (gpu_generate_presign_bank_dynamic(s->falcon.h, s->falcon.logn, s->next_seed, batch,
            s->sample1_bank, s->sample2_bank, s->target_bank) != 0) {
        return 0;
    }
    s->next_seed += (uint32_t)(0x9E3779B1u + (uint32_t)batch);
    soko_copy_presign_bank_to_tokens(batch,
        s->sample1_bank, s->sample2_bank, s->target_bank,
        (falcon_token_t *)token_buf);
    return batch;
#endif
}

static const sim_algorithm_t g_algorithms[] = {
    {
        "cpu",
        sizeof(oofalcon_cpu_state_t),
        sizeof(falcon_token_t),
        DEFAULT_GPU_BATCH_TOKENS,
        oofalcon_cpu_init,
        oofalcon_cpu_destroy,
        oofalcon_cpu_produce_batch,
        oofalcon_cpu_configure_refill
    },
    {
        "cpumt",
        sizeof(oofalcon_cpumt_state_t),
        sizeof(falcon_token_t),
        DEFAULT_GPU_BATCH_TOKENS,
        oofalcon_cpumt_init,
        oofalcon_cpumt_destroy,
        oofalcon_cpumt_produce_batch,
        oofalcon_cpumt_configure_refill
    },
    {
        "soko",
        sizeof(soko_state_t),
        sizeof(falcon_token_t),
        DEFAULT_GPU_BATCH_TOKENS,
        soko_init,
        soko_destroy,
        soko_produce_batch,
        soko_configure_refill
    }
};
static double get_algorithm_refill_batch_seconds(const sim_algorithm_t *alg, const void *alg_state)
{
    if (strcmp(alg->name, "cpu") == 0) {
        const oofalcon_cpu_state_t *state = (const oofalcon_cpu_state_t *)alg_state;
        return state->refill_batch_seconds;
    }
    if (strcmp(alg->name, "cpumt") == 0) {
        const oofalcon_cpumt_state_t *state = (const oofalcon_cpumt_state_t *)alg_state;
        return state->refill_batch_seconds;
    }
    if (strcmp(alg->name, "soko") == 0) {
        const soko_state_t *state = (const soko_state_t *)alg_state;
        return state->refill_batch_seconds;
    }
    return 0.0;
}

static void maybe_start_refill(double now, int queue_capacity, int batch_tokens,
    int refill_threshold_tokens, double refill_batch_seconds, int *queued_tokens,
    int *refill_in_flight, int *refill_batch_size, double *refill_complete_time)
{
    int free_slots;
    int want;

    if (*refill_in_flight || *queued_tokens > refill_threshold_tokens) {
        return;
    }

    free_slots = queue_capacity - *queued_tokens;
    if (free_slots <= 0) {
        return;
    }

    want = batch_tokens;
    if (want > free_slots) {
        want = free_slots;
    }
    if (want <= 0) {
        return;
    }

    *refill_in_flight = 1;
    *refill_batch_size = want;
    *refill_complete_time = now + refill_batch_seconds;
}

static void settle_refills(double now, int queue_capacity, int batch_tokens,
    int refill_threshold_tokens, double refill_batch_seconds, int *queued_tokens,
    int *refill_in_flight, int *refill_batch_size, double *refill_complete_time,
    uint64_t *produced_tokens)
{
    while (*refill_in_flight && *refill_complete_time <= now) {
        *queued_tokens += *refill_batch_size;
        *produced_tokens += (uint64_t)*refill_batch_size;
        *refill_in_flight = 0;
        *refill_batch_size = 0;

        maybe_start_refill(*refill_complete_time, queue_capacity, batch_tokens,
            refill_threshold_tokens, refill_batch_seconds, queued_tokens,
            refill_in_flight, refill_batch_size, refill_complete_time);
    }
}

static int run_simulation_case(const sim_algorithm_t *alg, const sim_config_t *cfg, int token_rate, sim_result_t *out)
{
    uint8_t *alg_state = (uint8_t *)xmalloc(alg->state_size);
    int batch_tokens;
    int refill_threshold_tokens;
    int queued_tokens;
    int refill_in_flight;
    int refill_batch_size;
    double refill_complete_time;
    double refill_batch_seconds;
    double interval;
    double next_arrival_time;
    double simulated_end_time;
    uint64_t produced_tokens;
    uint64_t served_tokens;
    uint64_t starved_requests;

    memset(out, 0, sizeof(*out));
    memset(alg_state, 0, alg->state_size);

    if (!alg->init(alg_state, cfg)) {
        free(alg_state);
        return 0;
    }

    batch_tokens = alg->preferred_batch_tokens;
    refill_threshold_tokens = 0;
    alg->configure_refill(alg_state, cfg, token_rate, &batch_tokens, &refill_threshold_tokens);
    refill_batch_seconds = get_algorithm_refill_batch_seconds(alg, alg_state);

    if (refill_batch_seconds < 0.0) {
        refill_batch_seconds = 0.0;
    }

    interval = 1.0 / (double)token_rate;
    queued_tokens = cfg->queue_capacity;
    refill_in_flight = 0;
    refill_batch_size = 0;
    refill_complete_time = 0.0;
    next_arrival_time = 0.0;
    simulated_end_time = 0.0;
    produced_tokens = 0ULL;
    served_tokens = 0ULL;
    starved_requests = 0ULL;

    while (served_tokens < (uint64_t)cfg->consume_target_tokens) {
        double attempt_time = next_arrival_time;

        settle_refills(attempt_time, cfg->queue_capacity, batch_tokens,
            refill_threshold_tokens, refill_batch_seconds, &queued_tokens,
            &refill_in_flight, &refill_batch_size, &refill_complete_time,
            &produced_tokens);

        if (queued_tokens > 0) {
            queued_tokens--;
            served_tokens++;
            maybe_start_refill(attempt_time, cfg->queue_capacity, batch_tokens,
                refill_threshold_tokens, refill_batch_seconds, &queued_tokens,
                &refill_in_flight, &refill_batch_size, &refill_complete_time);
            simulated_end_time = attempt_time;
        } else {
            starved_requests++;
            maybe_start_refill(attempt_time, cfg->queue_capacity, batch_tokens,
                refill_threshold_tokens, refill_batch_seconds, &queued_tokens,
                &refill_in_flight, &refill_batch_size, &refill_complete_time);
            simulated_end_time = attempt_time;
        }

        next_arrival_time += interval;
    }

    settle_refills(simulated_end_time, cfg->queue_capacity, batch_tokens,
        refill_threshold_tokens, refill_batch_seconds, &queued_tokens,
        &refill_in_flight, &refill_batch_size, &refill_complete_time,
        &produced_tokens);

    out->algorithm = alg->name;
    out->token_rate = token_rate;
    out->queue_capacity = cfg->queue_capacity;
    out->consume_target_tokens = cfg->consume_target_tokens;
    out->refill_threshold_tokens = refill_threshold_tokens;
    out->offered_load = (double)token_rate;
    out->measured_seconds = simulated_end_time;
    out->produced_tokens = produced_tokens;
    out->served_tokens = served_tokens;
    out->starved_requests = starved_requests;
    out->refill_throughput = (out->measured_seconds > 0.0)
        ? ((double)out->produced_tokens / out->measured_seconds) : 0.0;
    out->served_token_rate = (out->measured_seconds > 0.0)
        ? ((double)out->served_tokens / out->measured_seconds) : 0.0;
    out->service_ratio = (out->offered_load > 0.0)
        ? (out->served_token_rate / out->offered_load) : 0.0;

    alg->destroy(alg_state);
    free(alg_state);
    return 1;
}

static void write_csv_header(FILE *f)
{
    fprintf(f,
    "algorithm,token_rate,queue_capacity,consume_target_tokens,refill_threshold_tokens,offered_load,measured_seconds,"
        "produced_tokens,served_tokens,starved_requests,refill_throughput,served_token_rate,service_ratio\n");
}

static void write_csv_row(FILE *f, const sim_result_t *r)
{
    fprintf(f, "%s,%d,%d,%d,%d,%.6f,%.6f,%llu,%llu,%llu,%.6f,%.6f,%.6f\n",
        r->algorithm,
        r->token_rate,
        r->queue_capacity,
        r->consume_target_tokens,
        r->refill_threshold_tokens,
        r->offered_load,
        r->measured_seconds,
        (unsigned long long)r->produced_tokens,
        (unsigned long long)r->served_tokens,
        (unsigned long long)r->starved_requests,
        r->refill_throughput,
        r->served_token_rate,
        r->service_ratio);
}

static void print_result(const sim_result_t *r)
{
    printf("%-11s rate=%-9d target=%-7d thr=%-6d produced=%-10llu served=%-10llu refill=%.2f/s served_rate=%.2f/s served=%.1f%% starved=%llu\n",
        r->algorithm,
        r->token_rate,
        r->consume_target_tokens,
        r->refill_threshold_tokens,
        (unsigned long long)r->produced_tokens,
        (unsigned long long)r->served_tokens,
        r->refill_throughput,
        r->served_token_rate,
        r->service_ratio * 100.0,
        (unsigned long long)r->starved_requests);
}

int main(int argc, char **argv)
{
    sim_config_t cfg;
    FILE *csv = NULL;
    unsigned long long alg_count = sizeof(g_algorithms) / sizeof(g_algorithms[0]);
    unsigned long long a;
    int i;

    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    printf("===============================================================\n");
    printf("CONSUME-ONLY REFILL SIMULATOR\n");
    printf("Queue: %d  Consume target: %d  GPU batch: %d  CPU MT threads: %d\n",
        cfg.queue_capacity, cfg.consume_target_tokens, cfg.gpu_batch_tokens,
        resolve_cpu_mt_threads(cfg.cpu_mt_threads));
    printf("Rates:");
    for (i = 0; i < cfg.token_rate_len; i++) {
        printf(" %d", cfg.token_rates[i]);
    }
    printf("\n");
    printf("Algorithms: cpu, cpumt, soko\n");
    printf("Mode: steady prefill, threshold-based refill, fixed served-token stop\n");
    printf("===============================================================\n");

    if (cfg.csv_path != NULL) {
        csv = fopen(cfg.csv_path, "w");
        if (csv == NULL) {
            fprintf(stderr, "Failed to open CSV path: %s\n", cfg.csv_path);
            return 1;
        }
        write_csv_header(csv);
    }

    for (a = 0; a < alg_count; a++) {
        for (i = 0; i < cfg.token_rate_len; i++) {
            sim_result_t r;
            if (!run_simulation_case(&g_algorithms[a], &cfg, cfg.token_rates[i], &r)) {
                fprintf(stderr, "Skipping %s: initialization failed\n", g_algorithms[a].name);
                break;
            }
            print_result(&r);
            if (csv != NULL) {
                write_csv_row(csv, &r);
            }
        }
    }

    if (csv != NULL) {
        fclose(csv);
        printf("CSV written: %s\n", cfg.csv_path);
    }

    return 0;
}