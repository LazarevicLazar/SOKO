#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>

typedef std::complex<double> cplx_t;
static_assert(sizeof(cplx_t)==16, "bug!");

#define REQUIRE_DRAMATICALLY(XXXXX,MESSAGE) { \
if (!(XXXXX)) {                  \
std::cerr << "REQUIREMENT FAILED at " << __FILE__ << ":" << __LINE__ \
<< std::endl << MESSAGE << std::endl; \
abort();                         \
}}

/** @brief fft modulo X^m-exp(i.2pi.entry+pwr) -- reference code */
void cplx_fft_naive(const uint32_t m, const double entry_pwr, cplx_t* data) {
    if (m == 1) return;
    const double pom = entry_pwr / 2.;
    const uint32_t h = m / 2;
    // apply the twiddle factors
    cplx_t cpom(cos(2*M_PI*pom),sin(2*M_PI*pom));
    for (uint64_t i = 0; i < h; ++i) {
        cplx_t prod = data[i+h]*cpom;
        data[i + h] = data[i] - prod;
        data[i] += prod;
    }
    // do the recursive calls
    cplx_fft_naive(h, pom, data);
    cplx_fft_naive(h, pom + 0.5, data + h);
}

void cplx_ifft_naive(const uint32_t m, const double entry_pwr, cplx_t* data) {
    if (m == 1) return;
    const double pom = entry_pwr / 2.;
    const uint32_t h = m / 2;
    cplx_t cpom(cos(2*M_PI*pom),-sin(2*M_PI*pom));
    // do the recursive calls
    cplx_ifft_naive(h, pom, data);
    cplx_ifft_naive(h, pom + 0.5, data + h);
    // apply the inverse twiddle factors
    for (uint64_t i = 0; i < h; ++i) {
        cplx_t diff = (data[i]-data[i+h])*cpom;
        data[i]+=data[i+h];
        data[i+h]=diff;
    }
}


void negacyclic_fft(const uint64_t N, cplx_t* res, const double* x) {
    const uint64_t m = N/2;
    //let's make a copy in case we wish the transformation in place
    std::vector<double> v(x,x+N);
    double norm = 0;
    for (uint64_t i=0; i<m; ++i) {
        res[i]=cplx_t(v[i],v[i+m]);
        norm += pow(v[i],2)+pow(v[i+m],2);
    }
    cplx_fft_naive(m, 0.25, res);
    double vnorm = 0;
    for (uint64_t i=0; i<m; ++i) {
        vnorm += pow(abs(res[i]),2);
    }
    REQUIRE_DRAMATICALLY(fabs(1-m*norm/vnorm)<1e-8, "fft problem");
}

void negacyclic_ifft(const uint64_t N, double* res, const cplx_t* x) {
    const uint64_t m = N/2;
    std::vector<cplx_t> v(x, x+m);
    double vnorm = 0;
    for (uint64_t i=0; i<m; ++i) {
        vnorm += pow(abs(v[i]),2);
    }
    cplx_ifft_naive(m, 0.25, v.data());
    double norm = 0;
    for (uint64_t i=0; i<m; ++i) {
        res[i]=v[i].real()/m;
        res[i + m]=v[i].imag()/m;
        norm += pow(res[i],2)+pow(res[i+m],2);
    }
    REQUIRE_DRAMATICALLY(fabs(1.-m*norm/vnorm)<1e-8, "fft problem");
}


void addto_ubar_times_v(const uint64_t m, cplx_t* res, const cplx_t* u, const cplx_t* v) {
    for (uint64_t i=0; i<m; ++i) {
        res[i]+=conj(u[i])*v[i];
    }
}

/**
 * Computes the fft form of the input samples
 * @param res_fft result is (d1,d2,w) that represents this matrix [[d1,w][wbar,d2]]
 * @param data_fft input samples in fft form
 */
void data_fft(const uint64_t N, const uint64_t samples, cplx_t* res_fft, const double* data) {
    const uint64_t m = N/2;
    for (uint64_t i=0; i<samples; ++i) {
        const double *dptr = data + i * N * 2;
        cplx_t* rptr = res_fft + i * m * 2;
        negacyclic_fft(N, rptr, dptr);
        negacyclic_fft(N, rptr+m, dptr + N);
    }
}

/**
 * Computes the covariance matrix of data_fft
 * @param res_fft result is (d1,d2,w) that represents this matrix [[d1,w][wbar,d2]]
 * @param data_fft input samples in fft form
 */
void covariance_fft(const uint64_t m, const uint64_t samples, cplx_t* res_fft, const cplx_t* data_fft) {
    cplx_t* d1 = res_fft;
    cplx_t* d2 = res_fft+m;
    cplx_t* w = res_fft+2*m;
    memset(res_fft, 0, 3*m*sizeof(cplx_t));
    for (uint64_t i=0; i<samples; ++i) {
        const cplx_t* x1 = data_fft + i*m*2;
        const cplx_t* x2 = x1 + m;
        addto_ubar_times_v(m, d1, x1, x1);
        addto_ubar_times_v(m, d2, x2, x2);
        addto_ubar_times_v(m, w, x1, x2);
    }
    //normalize the result (note: the 1/12 is the parallepiped distribution)
    const double factor = 12./(double(samples)*2.*m);
    for (uint64_t j=0; j<m; ++j) {
        d1[j]*=factor;
        d2[j]*=factor;
        w[j]*=factor;
    }
}

void sqrt_inv_covariance(const uint64_t m, cplx_t* sqrt_fft, cplx_t* invsqrt_fft, const cplx_t* covariance_fft) {
    for (uint64_t i=0; i<m; ++i) {
        cplx_t d1 = covariance_fft[i];
        cplx_t d2 = covariance_fft[i+m];
        cplx_t w = covariance_fft[i+2*m];
        REQUIRE_DRAMATICALLY(
                d1.real()>0 && fabs(d1.imag())<1e-10,
                "d1 is not positive:" << d1);
        REQUIRE_DRAMATICALLY(
                d2.real()>0 && fabs(d2.imag())<1e-10,
                "d2 is not positive:" << d2);
        double sd1 = sqrt(abs(d1));
        double si1 = 1./sd1;
        d1=1;
        cplx_t u = w*si1;
        double v = abs(d2) - abs(conj(u)*u);
        REQUIRE_DRAMATICALLY(
                v>0,
                "v is not positive:" << v);
        double sd2 = sqrt(v);
        double si2 = 1/sd2;
        invsqrt_fft[i]=si1;
        invsqrt_fft[i+m]=si2;
        invsqrt_fft[i+2*m]=-si1*si2*u;
        sqrt_fft[i]=sd1;
        sqrt_fft[i+m]=sd2;
        sqrt_fft[i+2*m]=u;
    }
}

void make_it_square(const uint64_t m, const uint64_t nsamples, cplx_t* samples_fft, cplx_t* invsqrt_fft) {
    for (uint64_t i=0; i<nsamples; ++i) {
        cplx_t* s = samples_fft + i*2*m;
        for (uint64_t j=0; j<m; ++j) {
            cplx_t s1 = s[j] * invsqrt_fft[j];
            cplx_t s2 = s[j] * invsqrt_fft[j+2*m] + s[j+m] * invsqrt_fft[j+m];
            s[j] = s1;
            s[j+m] = s2;
        }
    }
}

/**
 * computes score and gradient
 * @param theta_fft: current unitary direction of size 2m
 * @param samples_fft: the samples in FFT form
 * @param score: sum(<samples,theta>^4) on all rotations of samples
 * @param grad_fft: gradient of the score function
 */
void score_and_gradient(const uint64_t m, const uint64_t nsamples,
                        double* score, cplx_t* grad_fft,
                        const cplx_t* theta_fft, const cplx_t* samples_fft) {
    const uint64_t N = 2*m;
    *score = 0;
    memset(grad_fft, 0, 2*m*sizeof(cplx_t));
    std::vector<cplx_t> v1(m);
    std::vector<cplx_t> v2(m);
    std::vector<double> realv1(2*m);
    std::vector<double> realv2(2*m);
    for (uint64_t i=0; i<nsamples; ++i) {
        const cplx_t *s = samples_fft + i * 2 * m;
        for (uint64_t j = 0; j < m; ++j) {
            v1[j] = s[j] * conj(theta_fft[j]);
            v2[j] = s[j + m] * conj(theta_fft[j + m]);
        }
        negacyclic_ifft(N, realv1.data(), v1.data());
        negacyclic_ifft(N, realv2.data(), v2.data());
        for (uint64_t j = 0; j < N; ++j) {
            *score += pow(realv1[j] + realv2[j], 4);
        }
        for (uint64_t j = 0; j < N; ++j) {
            realv1[j] = pow(realv1[j] + realv2[j], 3);
        }
        negacyclic_fft(N, v1.data(), realv1.data());
        for (uint64_t j = 0; j < m; ++j) {
            grad_fft[j] += s[j] * conj(v1[j]);
            grad_fft[j + m] += s[j + m] * conj(v1[j]);
        }
    }
    // normalize the result
    *score /= (nsamples * N);
    for (uint64_t j = 0; j < m; ++j) {
        grad_fft[j] *= 4./nsamples;
        grad_fft[j+m] *= 4./nsamples;
    }
}

#include <NTL/LLL.h>
#include <random>

double random_double() {
    static std::random_device rd;  // Will be used to obtain a seed for the random number engine
    static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}

void generate_fake_dataset(const uint64_t N, const uint64_t nsamples, const int32_t Bnorm,
                           std::vector<double>& basis,
                           std::vector<double>& samples) {
    const uint64_t m = N/2;
    basis.resize(4*N);
    samples.resize(2*N*nsamples);
    // generate a basis
    for (uint64_t i=0; i<4*N; ++i) {
        basis[i]=(rand()%(2*Bnorm+1))-Bnorm;
    }
    std::vector<cplx_t> basis_fft(4*m);
    cplx_t* a = basis_fft.data();
    cplx_t* b = a+m;    cplx_t* c = a+2*m;
    cplx_t* d = a+3*m;
    negacyclic_fft(N, a, basis.data());
    negacyclic_fft(N, b, basis.data()+N);
    negacyclic_fft(N, c, basis.data()+2*N);
    negacyclic_fft(N, d, basis.data()+3*N);
    // generate samples in the parallelepiped
    std::vector<cplx_t> coeffs_fft(2*m);
    cplx_t* u = coeffs_fft.data();
    cplx_t* v = u+m;
    for (uint64_t i=0; i<nsamples; ++i) {
        double* s = samples.data()+i*2*N;
        for (uint64_t j=0; j<N; ++j) {
            s[j] = random_double()-0.5;//(rand()/double(RAND_MAX))-0.5;
            s[j+N] = random_double()-0.5;//(rand()/double(RAND_MAX))-0.5;
        }
        negacyclic_fft(N, u, s);
        negacyclic_fft(N, v, s+N);
        for (uint64_t j=0; j<m; ++j) {
            cplx_t nx = u[j]*a[j] + v[j]*c[j];
            v[j] = u[j]*b[j] + v[j]*d[j];
            u[j] = nx;
        }
        negacyclic_ifft(N, s, u);
        negacyclic_ifft(N, s+N, v);
    }
}

int main(int argc, char** argv) {
    // read the samples (here, we generate a fake dataset instead)
    uint64_t N = 512;
    uint64_t m = N/2;
    uint64_t nsamples = 5000;
    std::vector<double> basis; // secret basis (just to verify)
    std::vector<double> samples;
    generate_fake_dataset(N, nsamples, 5, basis, samples);
    // put the samples in fft form
    std::vector<cplx_t> samples_fft(nsamples*2*m);
    data_fft(N, nsamples, samples_fft.data(), samples.data());
    {
        // OPTIONAL SANITY CHECK BLOCK:
        // verify that the samples
        // really have uniformly distributed coordinates over the basis
        // check the samples coordinates
        std::vector<cplx_t> basis_fft(4*m);
        std::vector<cplx_t> invbasis_fft(4*m);
        std::vector<cplx_t> coord_fft(2*m);
        std::vector<double> coord(2*N);
        cplx_t* a = basis_fft.data();
        cplx_t* b = a+m;
        cplx_t* c = a+2*m;
        cplx_t* d = a+3*m;
        cplx_t* ia = invbasis_fft.data();
        cplx_t* ib = ia+m;
        cplx_t* ic = ia+2*m;
        cplx_t* id = ia+3*m;
        negacyclic_fft(N, a, basis.data());
        negacyclic_fft(N, b, basis.data()+N);
        negacyclic_fft(N, c, basis.data()+2*N);
        negacyclic_fft(N, d, basis.data()+3*N);
        for (uint64_t j=0; j<m; ++j) {
            cplx_t det = a[j] * d[j] - b[j] * c[j];
            ia[j] = d[j] / det;
            ib[j] = -b[j] / det;
            ic[j] = -c[j] / det;
            id[j] = a[j] / det;
        }
        double cmin=1./0.;
        double cmax=-1./0.;
        double cavg=0;
        double cstd=0;
        for (uint64_t i=0; i<nsamples; ++i) {
            const cplx_t* s = samples_fft.data()+2*m*i;
            for (uint64_t j=0; j<m; ++j) {
                coord_fft[j] = s[j] * ia[j] + s[j+m] * ic[j];
                coord_fft[j+m] = s[j] * ib[j] + s[j+m] * id[j];
            }
            negacyclic_ifft(N, coord.data(), coord_fft.data());
            negacyclic_ifft(N, coord.data()+N, coord_fft.data()+m);
            for (uint64_t j=0; j<2*N; ++j) {
                double cj = coord[j];
                if (cj<cmin) cmin=cj;
                if (cj>cmax) cmax=cj;
                cavg += cj;
                cstd += cj*cj;
            }
        }
        cavg /= nsamples*2*N;
        cstd /= nsamples*2*N;
        cstd = sqrt(cstd-cavg*cavg);
        std::cout << "coords: min: " << cmin << " max: " << cmax
        << " avg: " << cavg << " std: " << cstd << std::endl;
    }
    // estimate the covariance matrix (B^t.B)
    std::vector<cplx_t> covar_fft(3*m);
    covariance_fft(m, nsamples, covar_fft.data(), samples_fft.data());
    {
        // OPTIONAL SANITY CHECK BLOCK:
        // test that the estimated covariance is close to the real basis (b^t.b)
        std::vector<cplx_t> basis_fft(4*m);
        cplx_t* a = basis_fft.data();
        cplx_t* b = a+m;
        cplx_t* c = a+2*m;
        cplx_t* d = a+3*m;
        negacyclic_fft(N, a, basis.data());
        negacyclic_fft(N, b, basis.data()+N);
        negacyclic_fft(N, c, basis.data()+2*N);
        negacyclic_fft(N, d, basis.data()+3*N);
        for (uint64_t j=0; j<m; ++j) {
            cplx_t d1 = conj(a[j])*a[j]+conj(c[j])*c[j];
            cplx_t d2 = conj(b[j])*b[j]+conj(d[j])*d[j];
            cplx_t w = conj(a[j])*b[j]+conj(c[j])*d[j];
            std::cout << "d1[" << j << "] : " << covar_fft[j] << " vs " << d1 << std::endl;
            std::cout << "d2[" << j << "] : " << covar_fft[j+m] << " vs " << d2 << std::endl;
            std::cout << "w[" << j << "] : " << covar_fft[j+2*m] << " vs " << w << std::endl;
        }
    }
    // compute R and invR s.t. (R^t.R) = (B^t.B)
    std::vector<cplx_t> r_fft(3*m);
    std::vector<cplx_t> invr_fft(3*m);
    sqrt_inv_covariance(m, r_fft.data(), invr_fft.data(), covar_fft.data());
    // rescale the samples by invR
    make_it_square(m, nsamples, samples_fft.data(), invr_fft.data());
    {
        //OPTIONAL SANITY CHECK BLOCK
        //check that the score function is maximal along the secret dirs vectors
        std::vector<cplx_t> q_fft(4*m);
        std::vector<cplx_t> basis_fft(4*m);
        cplx_t* a = basis_fft.data();
        cplx_t* b = a+m;
        cplx_t* c = a+2*m;
        cplx_t* d = a+3*m;
        negacyclic_fft(N, a, basis.data());
        negacyclic_fft(N, b, basis.data()+N);
        negacyclic_fft(N, c, basis.data()+2*N);
        negacyclic_fft(N, d, basis.data()+3*N);
        double norm0=0;
        double norm1=0;
        double norm2=0;
        double norm3=0;
        for (uint64_t j=0; j<m; ++j) {
            q_fft[j]=a[j]*invr_fft[j];
            q_fft[j+m]=a[j]*invr_fft[j+2*m]+b[j]*invr_fft[j+m];
            q_fft[j+2*m]=c[j]*invr_fft[j];
            q_fft[j+3*m]=c[j]*invr_fft[j+2*m]+d[j]*invr_fft[j+m];
            norm0 += pow(abs(q_fft[j]), 2) + pow(abs(q_fft[j+m]), 2);
            norm1 += pow(abs(q_fft[j+2*m]), 2) + pow(abs(q_fft[j+3*m]), 2);
            norm2 += pow(abs(q_fft[j]), 2) + pow(abs(q_fft[j+2*m]), 2);
            norm3 += pow(abs(q_fft[j+m]), 2) + pow(abs(q_fft[j+3*m]), 2);
        }
        std::cout << norm0 << " " << norm1 << std::endl;
        std::cout << norm2 << " " << norm3 << std::endl;
        // check the norms
        std::vector<cplx_t> check_grad_fft(2*m);
        double check_score;
        score_and_gradient(m, nsamples, &check_score, check_grad_fft.data(), q_fft.data(), samples_fft.data());
        std::cout << "score0: " << check_score << std::endl;
        score_and_gradient(m, nsamples, &check_score, check_grad_fft.data(), q_fft.data()+2*m, samples_fft.data());
        std::cout << "score1: " << check_score << std::endl;
    }
    // do a gradient descent to minimize the 4-th moment
    std::vector<cplx_t> theta_fft(2*m);
    std::vector<cplx_t> grad_fft(2*m);
    double score;
    uint64_t niters = 1000;
    double learn_rate = 0.02;
    // gradient descent init: initialize theta_fft at random (unitary)
    {
        double norm = 0;
        for (uint64_t j=0; j<2*m; ++j) {
            theta_fft[j] = cplx_t(
                    (rand() / double(RAND_MAX))-0.5,
                    (rand() / double(RAND_MAX))-0.5);
            norm += pow(abs(theta_fft[j]),2);
        }
        norm = sqrt(norm/m);
        for (uint64_t j=0; j<2*m; ++j) {
            theta_fft[j] /= norm;
        }
    }
    double last_score = 1./0.;
    clock_t last_update = 0;
    for (uint64_t i=0; i<niters; ++i) {
        // compute score and gradient
        score_and_gradient(m, nsamples, &score, grad_fft.data(), theta_fft.data(), samples_fft.data());
        if (clock()>last_update+CLOCKS_PER_SEC) {
            last_update = clock();
            std::cout << "iteration " << i << "; score " << score << std::endl;
        }
        if (score < 2./80. && score > last_score) break;
        last_score = score;
        // move in the direction of the gradient and normalize
        double norm = 0;
        for (uint64_t j=0; j<2*m; ++j) {
            theta_fft[j] -= learn_rate*grad_fft[j];
            norm += pow(abs(theta_fft[j]),2);
        }
        norm = sqrt(norm/m);
        for (uint64_t j=0; j<2*m; ++j) {
            theta_fft[j] /= norm;
        }
    }
    // once the gradient descent converges, theta * R should be one basis vector
    std::vector<cplx_t> solution_fft(2*m);
    std::vector<double> solution(2*N);
    for (uint64_t j=0; j<m; ++j) {
        solution_fft[j]=theta_fft[j]*r_fft[j];
        solution_fft[j+m]=theta_fft[j]*r_fft[j+2*m]+theta_fft[j+m]*r_fft[j+m];
    }
    negacyclic_ifft(N, solution.data(), solution_fft.data());
    negacyclic_ifft(N, solution.data()+N, solution_fft.data()+m);
    std::cout << "solution: " << std::endl;
    double dist = 0;
    for (uint64_t j=0; j<N; ++j) {
        double d = solution[j]-rint(solution[j]);
        std::cout << rint(solution[j]);
        if (fabs(d)>0.25) std::cout << "?";
        std::cout << " ";
        if (fabs(d)>dist) dist=fabs(d);
    }
    std::cout << "distance: " << dist << std::endl;
    std::cout << "basis: " << std::endl;
    for (uint64_t j=0; j<N; ++j) {
        std::cout << basis[j] << " ";
    }
    std::cout << std::endl;
    for (uint64_t j=0; j<N; ++j) {
        std::cout << basis[j+2*N] << " ";
    }
    std::cout << std::endl;

    return 0;
}
