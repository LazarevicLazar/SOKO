#include "testlib.h"

int64_t posmod(int64_t a, int64_t q) {
    int64_t t = a%q;
    return t<0?t+q:t;
}
int64_t centermod(int64_t a, int64_t q) {
    int64_t t = a%q;
    if (t>q/2) return t-q;
    if (t<-q/2) return t+q;
    return t;
}

modQ::modQ(int64_t x): v(posmod(x,F_Q)) {}

modQ operator+(modQ a, modQ b) { return posmod(a.v+b.v, F_Q); }
modQ operator-(modQ a, modQ b) { return posmod(a.v-b.v, F_Q); }
modQ operator*(modQ a, modQ b) { return posmod(a.v*b.v, F_Q); }
modQ& operator+=(modQ& a, modQ b) { a.v = posmod(a.v+b.v, F_Q); return a; }
modQ& operator-=(modQ& a, modQ b) { a.v = posmod(a.v-b.v, F_Q); return a; }
modQ& operator*=(modQ& a, modQ b) { a.v = posmod(a.v*b.v, F_Q); return a; }
bool operator==(modQ& a, modQ b) { return a.v == b.v; }
std::ostream& operator<<(std::ostream& out, modQ x) { return out << x.v; }

void convert(modQ* res, const int8_t* x, uint64_t n) {
    for (uint64_t k=0; k<n; ++k) {
        res[k]=modQ(x[k]);
    }
}
void convert(modQ* res, const int16_t* x, uint64_t n) {
    for (uint64_t k=0; k<n; ++k) {
        res[k]=modQ(x[k]);
    }
}
void convert(modQ* res, const uint16_t* x, uint64_t n) {
    for (uint64_t k=0; k<n; ++k) {
        res[k]=modQ(x[k]);
    }
}
void convert(modQ* res, const int32_t* x, uint64_t n) {
    for (uint64_t k=0; k<n; ++k) {
        res[k]=modQ(x[k]);
    }
}
void starproduct(modQ* res, const modQ* x, const modQ* y, uint64_t n) {
    for (uint64_t k=0; k<n; ++k) {
        modQ rk = 0;
        for (uint64_t i=0; i<=k; ++i) {
            rk += x[k-i] * y[i];
        }
        for (uint64_t i=k+1; i<n; ++i) {
            rk -= x[n+k-i] * y[i];
        }
        res[k]=rk;
    }
}

typedef std::vector<modQ> vec_modQ;
vec_modQ to_vec_modQ(const std::vector<int8_t>& v) {
    vec_modQ res(v.size());
    convert(res.data(), v.data(), v.size());
    return res;
}
vec_modQ to_vec_modQ(const std::vector<uint16_t>& v) {
    vec_modQ res(v.size());
    convert(res.data(), v.data(), v.size());
    return res;
}
vec_modQ to_vec_modQ(const std::vector<int16_t>& v) {
    vec_modQ res(v.size());
    convert(res.data(), v.data(), v.size());
    return res;
}
vec_modQ to_vec_modQ(const std::vector<int32_t>& v) {
    vec_modQ res(v.size());
    convert(res.data(), v.data(), v.size());
    return res;
}
vec_modQ starproduct(const vec_modQ& x, const vec_modQ& y) {
    uint64_t n = x.size();
    REQUIRE_DRAMATICALLY(y.size()==n, "wrong size");
    vec_modQ res(n);
    starproduct(res.data(), x.data(), y.data(), n);
    return res;
}
vec_modQ operator-(const vec_modQ& x, const vec_modQ& y) {
    uint64_t n = x.size();
    REQUIRE_DRAMATICALLY(y.size()==n, "wrong size");
    vec_modQ res(n);
    for (uint16_t i=0; i<n; ++i) {
        res[i] = x[i] - y[i];
    }
    return res;
}
vec_modQ operator+(const vec_modQ& x, const vec_modQ& y) {
    uint64_t n = x.size();
    REQUIRE_DRAMATICALLY(y.size()==n, "wrong size");
    vec_modQ res(n);
    for (uint16_t i=0; i<n; ++i) {
        res[i] = x[i] + y[i];
    }
    return res;
}
bool operator==(const vec_modQ& x, const vec_modQ& y) {
    uint64_t n = x.size();
    REQUIRE_DRAMATICALLY(y.size()==n, "wrong size");
    for (uint64_t i=0; i<n; ++i) {
        if (x[i].v != y[i].v) return false;
    }
    return true;
}
std::default_random_engine& randgen() {
    static std::default_random_engine eng;
    return eng;
}
uint64_t random_u64() {
    static std::uniform_int_distribution<uint64_t> dist;
    return dist(randgen());
}
uint64_t random_i64() {
    static std::uniform_int_distribution<int64_t> dist;
    return dist(randgen());
}

falcon_key_t keygen(uint64_t logn, inner_shake256_context* rng) {
    falcon_key_t res;
    const uint64_t n = 1<<logn;
    uint8_t* tmp = (uint8_t*) aligned_alloc(64, 1024*1024);
    res.f.resize(n);
    res.g.resize(n);
    res.F.resize(n);
    res.G.resize(n);
    res.h.resize(n);
    Zf(keygen)(
            rng,
            res.f.data(),res.g.data(),
            res.F.data(),res.G.data(),
            res.h.data(),logn, tmp);
    free(tmp);
    return res;
}

