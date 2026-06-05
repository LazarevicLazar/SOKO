#include <stdlib.h>
#include <string.h>

#include "soko_falcon_backends.h"

#if defined(__STDC_NO_ATOMICS__)
typedef volatile int atomic_int;
static int atomic_load(const atomic_int *p)
{
    return *p;
}
static void atomic_store(atomic_int *p, int v)
{
    *p = v;
}
#else
#include <stdatomic.h>
#endif

#if defined(OOFALCON_USE_CUDA) && defined(_WIN32)
#include <process.h>
#include <windows.h>

typedef struct {
    int bank_id;
    int token_index;
} soko_token_ref_t;

typedef struct {
    soko_token_ref_t *items;
    int capacity;
    int head;
    int tail;
    int count;
} soko_token_queue_t;

typedef struct {
    atomic_int in_flight;
    atomic_int done;
    atomic_int started;
    int bank_id;
    int tokens;
    unsigned logn;
    uint32_t seed;
    const uint16_t *h_coeffs;
    soko_falcon_token_bank_t *bank;
    int result;
    HANDLE thread;
} soko_refill_worker_t;

struct soko_falcon_refill_pipeline {
    const uint16_t *h_coeffs;
    unsigned logn;
    uint32_t next_seed;
    int tokens_per_bank;
    int refill_threshold;
    soko_falcon_token_bank_t *banks;
    int bank_count;
    int *bank_remaining;
    soko_token_queue_t queue;
    soko_refill_worker_t worker;
};

static int soko_queue_init(soko_token_queue_t *queue, int capacity)
{
    queue->items = (soko_token_ref_t *)malloc((size_t)capacity * sizeof(*queue->items));
    if (queue->items == NULL) {
        return 0;
    }
    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    return 1;
}

static void soko_queue_destroy(soko_token_queue_t *queue)
{
    free(queue->items);
    queue->items = NULL;
    queue->capacity = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
}

static int soko_queue_push(soko_token_queue_t *queue, soko_token_ref_t ref)
{
    if (queue->count >= queue->capacity) {
        return 0;
    }
    queue->items[queue->tail] = ref;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    return 1;
}

static int soko_queue_pop(soko_token_queue_t *queue, soko_token_ref_t *ref)
{
    if (queue->count <= 0) {
        return 0;
    }
    *ref = queue->items[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    return 1;
}

static int soko_enqueue_bank_tokens(soko_token_queue_t *queue, int bank_id, int tokens)
{
    int token_index;

    for (token_index = 0; token_index < tokens; token_index++) {
        soko_token_ref_t ref;

        ref.bank_id = bank_id;
        ref.token_index = token_index;
        if (!soko_queue_push(queue, ref)) {
            return 0;
        }
    }
    return 1;
}

static unsigned __stdcall soko_refill_thread_proc(void *arg)
{
    soko_refill_worker_t *worker = (soko_refill_worker_t *)arg;

    worker->result = soko_gpu_generate_presign_bank_dynamic(
        worker->h_coeffs, worker->logn, worker->seed, worker->tokens,
        worker->bank->s1, worker->bank->s2, worker->bank->target);
    atomic_store(&worker->done, 1);
    atomic_store(&worker->in_flight, 0);
    return 0;
}

static int soko_start_refill_worker(
    soko_refill_worker_t *worker,
    int bank_id,
    soko_falcon_token_bank_t *bank,
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed,
    int tokens)
{
    if (atomic_load(&worker->in_flight)) {
        return 0;
    }

    worker->bank_id = bank_id;
    worker->bank = bank;
    worker->h_coeffs = h_coeffs;
    worker->logn = logn;
    worker->seed = seed;
    worker->tokens = tokens;
    worker->result = 0;
    atomic_store(&worker->done, 0);
    atomic_store(&worker->in_flight, 1);
    atomic_store(&worker->started, 1);

    worker->thread = (HANDLE)_beginthreadex(NULL, 0, soko_refill_thread_proc, worker, 0, NULL);
    if (worker->thread == NULL) {
        atomic_store(&worker->in_flight, 0);
        atomic_store(&worker->started, 0);
        return 0;
    }

    return 1;
}

static void soko_join_refill_worker(soko_refill_worker_t *worker)
{
    if (!atomic_load(&worker->started)) {
        return;
    }

    WaitForSingleObject(worker->thread, INFINITE);
    CloseHandle(worker->thread);
    worker->thread = NULL;
    atomic_store(&worker->started, 0);
}

static int soko_finish_completed_refill(soko_falcon_refill_pipeline_t *pipeline, int enqueue_error)
{
    if (!atomic_load(&pipeline->worker.done)) {
        return 0;
    }

    soko_join_refill_worker(&pipeline->worker);
    atomic_store(&pipeline->worker.done, 0);
    if (pipeline->worker.result != 0) {
        return pipeline->worker.result;
    }
    if (!soko_enqueue_bank_tokens(&pipeline->queue, pipeline->worker.bank_id, pipeline->tokens_per_bank)) {
        return enqueue_error;
    }

    pipeline->bank_remaining[pipeline->worker.bank_id] = pipeline->tokens_per_bank;
    pipeline->banks[pipeline->worker.bank_id].available = 0;
    return 0;
}

static int soko_find_empty_bank(const soko_falcon_refill_pipeline_t *pipeline)
{
    int bank_id;

    for (bank_id = 0; bank_id < pipeline->bank_count; bank_id++) {
        if (pipeline->bank_remaining[bank_id] == 0) {
            return bank_id;
        }
    }
    return -1;
}

int soko_falcon_refill_pipeline_init(
    soko_falcon_refill_pipeline_t **pipeline_out,
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed_base,
    int tokens_per_bank,
    int refill_threshold,
    soko_falcon_token_bank_t *banks,
    int bank_count)
{
    soko_falcon_refill_pipeline_t *pipeline;

    if (pipeline_out == NULL || h_coeffs == NULL || banks == NULL || bank_count < 2 || tokens_per_bank <= 0) {
        return -210;
    }

    pipeline = (soko_falcon_refill_pipeline_t *)calloc(1, sizeof(*pipeline));
    if (pipeline == NULL) {
        return -210;
    }

    pipeline->bank_remaining = (int *)calloc((size_t)bank_count, sizeof(*pipeline->bank_remaining));
    if (pipeline->bank_remaining == NULL) {
        free(pipeline);
        return -210;
    }

    pipeline->h_coeffs = h_coeffs;
    pipeline->logn = logn;
    pipeline->next_seed = seed_base;
    pipeline->tokens_per_bank = tokens_per_bank;
    pipeline->refill_threshold = refill_threshold;
    pipeline->banks = banks;
    pipeline->bank_count = bank_count;

    if (!soko_queue_init(&pipeline->queue, tokens_per_bank * bank_count)) {
        free(pipeline->bank_remaining);
        free(pipeline);
        return -210;
    }

    atomic_store(&pipeline->worker.in_flight, 0);
    atomic_store(&pipeline->worker.done, 0);
    atomic_store(&pipeline->worker.started, 0);

    if (!soko_enqueue_bank_tokens(&pipeline->queue, 0, tokens_per_bank)) {
        soko_queue_destroy(&pipeline->queue);
        free(pipeline->bank_remaining);
        free(pipeline);
        return -211;
    }

    pipeline->bank_remaining[0] = tokens_per_bank;
    pipeline->banks[0].available = 0;
    pipeline->banks[1].available = 1;

    *pipeline_out = pipeline;
    return 0;
}

int soko_falcon_refill_pipeline_acquire(
    soko_falcon_refill_pipeline_t *pipeline,
    soko_falcon_token_view_t *token,
    double *wait_ms)
{
    if (pipeline == NULL || token == NULL || wait_ms == NULL) {
        return -215;
    }

    *wait_ms = 0.0;

    {
        int status = soko_finish_completed_refill(pipeline, -212);
        if (status != 0) {
            return status;
        }
    }

    if (pipeline->queue.count <= pipeline->refill_threshold && !atomic_load(&pipeline->worker.in_flight)) {
        int bank_id = soko_find_empty_bank(pipeline);
        if (bank_id >= 0) {
            pipeline->banks[bank_id].available = 0;
            if (!soko_start_refill_worker(&pipeline->worker, bank_id, &pipeline->banks[bank_id],
                    pipeline->h_coeffs, pipeline->logn, pipeline->next_seed++, pipeline->tokens_per_bank)) {
                return -213;
            }
        }
    }

    while (pipeline->queue.count == 0) {
        if (atomic_load(&pipeline->worker.done)) {
            int status = soko_finish_completed_refill(pipeline, -214);
            if (status != 0) {
                return status;
            }
            break;
        }

        {
            double w0;
            double w1;

            w0 = (double)GetTickCount64();
            Sleep(1);
            w1 = (double)GetTickCount64();
            *wait_ms += w1 - w0;
        }
    }

    {
        soko_token_ref_t ref;
        size_t offset;

        if (!soko_queue_pop(&pipeline->queue, &ref)) {
            return -215;
        }
        pipeline->bank_remaining[ref.bank_id]--;
        if (pipeline->bank_remaining[ref.bank_id] == 0) {
            pipeline->banks[ref.bank_id].available = 1;
        }

        offset = (size_t)ref.token_index * ((size_t)1 << pipeline->logn);
        token->bank_id = ref.bank_id;
        token->token_index = ref.token_index;
        token->s1 = pipeline->banks[ref.bank_id].s1 + offset;
        token->s2 = pipeline->banks[ref.bank_id].s2 + offset;
        token->target = pipeline->banks[ref.bank_id].target + offset;
    }

    return 0;
}

void soko_falcon_refill_pipeline_destroy(soko_falcon_refill_pipeline_t *pipeline)
{
    if (pipeline == NULL) {
        return;
    }

    if (atomic_load(&pipeline->worker.started)) {
        soko_join_refill_worker(&pipeline->worker);
    }
    soko_queue_destroy(&pipeline->queue);
    free(pipeline->bank_remaining);
    free(pipeline);
}

#else

struct soko_falcon_refill_pipeline {
    int unused;
};

int soko_falcon_refill_pipeline_init(
    soko_falcon_refill_pipeline_t **pipeline_out,
    const uint16_t *h_coeffs,
    unsigned logn,
    uint32_t seed_base,
    int tokens_per_bank,
    int refill_threshold,
    soko_falcon_token_bank_t *banks,
    int bank_count)
{
    (void)pipeline_out;
    (void)h_coeffs;
    (void)logn;
    (void)seed_base;
    (void)tokens_per_bank;
    (void)refill_threshold;
    (void)banks;
    (void)bank_count;
    return -100;
}

int soko_falcon_refill_pipeline_acquire(
    soko_falcon_refill_pipeline_t *pipeline,
    soko_falcon_token_view_t *token,
    double *wait_ms)
{
    (void)pipeline;
    (void)token;
    (void)wait_ms;
    return -100;
}

void soko_falcon_refill_pipeline_destroy(soko_falcon_refill_pipeline_t *pipeline)
{
    (void)pipeline;
}

#endif