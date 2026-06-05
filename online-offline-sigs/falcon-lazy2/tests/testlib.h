#ifndef FALCON_LAZY2_TESTLIB_H
#define FALCON_LAZY2_TESTLIB_H

#define restrict
extern "C" {
#include "../inner.h"
}

#include <iostream>
#include <vector>
#include <tuple>
#include <random>

#define EXPORT extern "C"

#define REQUIRE_DRAMATICALLY(req_contition, error_msg)                                                        \
  do {                                                                                                        \
    if (!(req_contition)) {                                                                                   \
      std::cerr << "REQUIREMENT FAILED at " << __FILE__ << ":" << __LINE__ << ": " << error_msg << std::endl; \
      abort();                                                                                                \
    }                                                                                                         \
  } while (0)

#define F_Q 12289

// from falcon.h
EXPORT void shake256_init_prng_from_seed(inner_shake256_context *sc,
                                         const void *seed, size_t seed_len);



int64_t posmod(int64_t a, int64_t q);
int64_t centermod(int64_t a, int64_t q);

struct modQ {
    int64_t v;
    modQ(int64_t x=0);
};

modQ operator+(modQ a, modQ b);
modQ operator-(modQ a, modQ b);
modQ operator*(modQ a, modQ b);
modQ& operator+=(modQ& a, modQ b);
modQ& operator-=(modQ& a, modQ b);
modQ& operator*=(modQ& a, modQ b);
bool operator==(modQ& a, modQ b);
std::ostream& operator<<(std::ostream& out, modQ x);

void convert(modQ* res, const int8_t* x, uint64_t n);
void convert(modQ* res, const int16_t* x, uint64_t n);
void convert(modQ* res, const uint16_t* x, uint64_t n);
void convert(modQ* res, const int32_t* x, uint64_t n);
void starproduct(modQ* res, const modQ* x, const modQ* y, uint64_t n);

typedef std::vector<modQ> vec_modQ;
vec_modQ to_vec_modQ(const std::vector<int8_t>& v);
vec_modQ to_vec_modQ(const std::vector<uint16_t>& v);
vec_modQ to_vec_modQ(const std::vector<int16_t>& v);
vec_modQ to_vec_modQ(const std::vector<int32_t>& v);
vec_modQ starproduct(const vec_modQ& x, const vec_modQ& y);
vec_modQ operator-(const vec_modQ& x, const vec_modQ& y);
vec_modQ operator+(const vec_modQ& x, const vec_modQ& y);
bool operator==(const vec_modQ& x, const vec_modQ& y);
std::default_random_engine& randgen();
uint64_t random_u64();
uint64_t random_i64();

template<typename T>
double print_statistics(const std::vector<T>& vec) {
    uint64_t n = vec.size();
    double minv = 1./0.;
    double maxv = -1./0.;
    double sumv = 0;
    double sumsq = 0;
    for (uint64_t i=0; i<n; ++i) {
        double v = vec[i];
        if (v > maxv) maxv = v;
        if (v < minv) minv = v;
        sumv += v;
        sumsq += v * v;
    }
    double mean = sumv/n;
    double var = sumsq/n - mean*mean;
    std::cout
            << "min: " << minv << "\n"
            << "max: " << maxv << "\n"
            << "mean: " << mean << "\n"
            << "stdv: " << sqrt(var) << "\n"
            << "norm: " << sqrt(sumsq) << std::endl;
    return sqrt(sumsq);
}

template<typename T>
std::tuple<double, double> cal_statistics(const std::vector<T>& vec) {
    uint64_t n = vec.size();
    double maxv = -1./0.;
    double sumsq = 0;
    for (uint64_t i=0; i<n; ++i) {
        double v = std::abs(vec[i]);
        if (v > maxv) maxv = v;
        sumsq += v * v;
    }
    return std::make_tuple(std::sqrt(sumsq), maxv);
}

struct falcon_key_t {
    std::vector<int8_t> f;
    std::vector<int8_t> g;
    std::vector<int8_t> F;
    std::vector<int8_t> G;
    std::vector<uint16_t> h;
};

falcon_key_t keygen(uint64_t logn, inner_shake256_context* rng);

// declaration of useful stuff in falcon.h

EXPORT int sign_dyn_lazy_online(
        int8_t* sample1, int8_t* sample2, uint16_t* sample_target,
        int16_t *s2,
        const fpr *restrict f_fft, const fpr *restrict g_fft,
        const fpr *restrict F_fft, const fpr *restrict G_fft,
        const uint16_t *hm, unsigned logn, fpr *restrict tmp __attribute((unused)));

EXPORT void sign_dyn_lazy_offline(
        // inputs
        inner_shake256_context *rng,
        const int8_t *restrict f, const int8_t *restrict g,
        const int8_t *restrict F, const int8_t *restrict G,
        const uint16_t *h,
        unsigned logn,
        //outputs
        int8_t* sample1, int8_t* sample2, uint16_t* sample_target,
        fpr *restrict f_fft, fpr *restrict g_fft,
        fpr *restrict F_fft, fpr *restrict G_fft
);


#endif //FALCON_LAZY2_TESTLIB_H
