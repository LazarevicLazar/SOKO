#include "benchmark/benchmark.h"
#include "testlib.h"

static void falcon_dyn_lazy_offline(benchmark::State& state) {
    // Perform setup here
    const uint64_t logn = 9;
    const uint64_t n = 1 << logn;
    inner_shake256_context rng;
    inner_shake256_init(&rng);
    falcon_key_t key = keygen(logn, &rng);
    vec_modQ hq = to_vec_modQ(key.h);
    std::vector<int16_t> sig(n);
// use a random hash of message
    std::vector<uint16_t> hm(n);
    for (uint64_t i=0; i<n; ++i) {
        hm[i]=rand()%F_Q;
    }
    std::vector<int8_t> sample1(n);
    std::vector<int8_t> sample2(n);
    std::vector<uint16_t> sample_target(n);
    std::vector<fpr> f_FFT(n);
    std::vector<fpr> g_FFT(n);
    std::vector<fpr> F_FFT(n);
    std::vector<fpr> G_FFT(n);

    for (auto _ : state) {
        sign_dyn_lazy_offline(&rng, key.f.data(), key.g.data(), key.F.data(), key.G.data(), key.h.data(), logn,
                              sample1.data(), sample2.data(), sample_target.data(), f_FFT.data(), g_FFT.data(),
                              F_FFT.data(), G_FFT.data());
    }
}

static void falcon_dyn_lazy_online(benchmark::State& state) {
    // Perform setup here
    const uint64_t logn = 9;
    const uint64_t n = 1 << logn;
    inner_shake256_context rng;
    inner_shake256_init(&rng);
    falcon_key_t key = keygen(logn, &rng);
    vec_modQ hq = to_vec_modQ(key.h);
    std::vector<int16_t> sig(n);
    // use a random hash of message
    std::vector<uint16_t> hm(n);
    for (uint64_t i=0; i<n; ++i) {
        hm[i]=rand()%F_Q;
    }
    std::vector<int8_t> sample1(n);
    std::vector<int8_t> sample2(n);
    std::vector<uint16_t> sample_target(n);
    std::vector<fpr> f_FFT(n);
    std::vector<fpr> g_FFT(n);
    std::vector<fpr> F_FFT(n);
    std::vector<fpr> G_FFT(n);

    for (uint64_t i=0; i<1; ++i) {
        sign_dyn_lazy_offline(&rng, key.f.data(), key.g.data(), key.F.data(), key.G.data(), key.h.data(), logn,
                              sample1.data(), sample2.data(), sample_target.data(), f_FFT.data(), g_FFT.data(),
                              F_FFT.data(), G_FFT.data());
    }
    std::vector<uint16_t> orig_sample_target = sample_target;
    for (auto _ : state) {
        sign_dyn_lazy_online(sample1.data(), sample2.data(), sample_target.data(), sig.data(),
                             f_FFT.data(), g_FFT.data(), F_FFT.data(), G_FFT.data(), hm.data(), logn,
                             nullptr);
    }
}

static void falcon_dyn_orig(benchmark::State& state) {
    const uint64_t logn = 9;
    const uint64_t n = 1 << logn;
    inner_shake256_context rng;
    inner_shake256_init(&rng);
    falcon_key_t key = keygen(logn, &rng);
    vec_modQ hq = to_vec_modQ(key.h);
    std::vector<int16_t> sig(n);
    std::vector<uint16_t> hm(n);
// use a random hash of message
    for (uint64_t i = 0; i < n; ++i) {
        hm[i] = rand() % F_Q;
    }
    uint8_t *tmp = (uint8_t *) aligned_alloc(64, 1024 * 1024);
    for (auto _ : state) {
        falcon_inner_sign_dyn(
                sig.data(),
                &rng,
                key.f.data(), key.g.data(),
                key.F.data(), key.G.data(),
                hm.data(),
                logn, tmp);
    }
    free(tmp);
}


// Register the function as a benchmark
BENCHMARK(falcon_dyn_lazy_offline);
BENCHMARK(falcon_dyn_lazy_online);
BENCHMARK(falcon_dyn_orig);

#include "ed25519.h"

static void ed25519(benchmark::State& state) {
    static const uint64_t MSGBYTES=64;
    unsigned char public_key[32], private_key[64], seed[32];
    unsigned char signature[64];
    uint8_t message[MSGBYTES];


    /* create a random seed, and a keypair out of that seed */
    for (uint64_t i=0; i<MSGBYTES; ++i) message[i] = random_u64();
    ed25519_create_seed(seed);
    ed25519_create_keypair(public_key, private_key, seed);
    for (auto _ : state) {
        ed25519_sign(signature, (const uint8_t*) message, MSGBYTES, public_key, private_key);
    }
}

BENCHMARK(ed25519);

extern "C" {
#include "dilithium/ref/sign.h"
}

static void dilithium_ref(benchmark::State& state) {
    static const uint64_t MSGBYTES=64;
    size_t siglen;
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_BYTES];
    uint8_t message[MSGBYTES];

    // TODO (initialize / keygen / etc)
    pqcrystals_dilithium2_ref_keypair(pk, sk);
    for (uint64_t i=0; i<MSGBYTES; ++i) message[i] = random_u64();
    for (auto _ : state) {
        pqcrystals_dilithium2_ref_signature(sig, &siglen, message, MSGBYTES, sk);
    }
    //pqcrystals_dilithium2_ref_verify(sig, CRYPTO_BYTES, sig, CRHBYTES, pk);
}

BENCHMARK(dilithium_ref);

#ifdef __x86_64__
// workaround since the macros system does not allow to include dilithium avx2 after ref
extern "C" typeof(pqcrystals_dilithium2_ref_keypair) pqcrystals_dilithium2_avx2_keypair;
extern "C" typeof(pqcrystals_dilithium2_ref_signature) pqcrystals_dilithium2_avx2_signature;

static void dilithium_avx(benchmark::State& state) {
    static const uint64_t MSGBYTES=64;
    size_t siglen;
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t sig[CRYPTO_BYTES];
    uint8_t message[MSGBYTES];

    // TODO (initialize / keygen / etc)
    pqcrystals_dilithium2_avx2_keypair(pk, sk);
    for (uint64_t i=0; i<MSGBYTES; ++i) message[i] = random_u64();
    for (auto _ : state) {
        pqcrystals_dilithium2_avx2_signature(sig, &siglen, message, MSGBYTES, sk);
    }
    //pqcrystals_dilithium2_avx2_verify(sig, CRYPTO_BYTES, sig, CRHBYTES, pk);
}

BENCHMARK(dilithium_avx);
#endif
