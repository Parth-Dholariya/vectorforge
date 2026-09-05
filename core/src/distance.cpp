#include "vectorforge/distance.hpp"

#include <cmath>
#include <immintrin.h>

namespace vectorforge {
namespace {

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

float l2_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float ip_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return 1.0f - sum;
}

// ---------------------------------------------------------------------------
// AVX2 + FMA
//
// These carry a target attribute rather than being compiled with a global
// -mavx2, so the binary still starts on a pre-Haswell CPU and the dispatcher
// below picks a legal kernel. Compiling the whole TU with -mavx2 would let GCC
// emit VEX instructions into the dispatcher itself and fault before we ever
// get to check cpuid.
//
// Four independent accumulators, not one. A single accumulator serialises on
// the FMA latency chain (~4 cycles on Zen+); four in flight keep both FMA
// pipes fed and is worth roughly 3x over the naive version.
// ---------------------------------------------------------------------------

__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

__attribute__((target("avx2,fma")))
float l2_avx2(const float* a, const float* b, size_t dim) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    for (; i + 8 <= dim; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = _mm256_fmadd_ps(d, d, s0);
    }

    float sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < dim; ++i) {  // tail: dims like SIFT-128 divide evenly, GIST-960 does too,
        const float d = a[i] - b[i];  // but 768-dim BERT embeddings need this branch.
        sum += d * d;
    }
    return sum;
}

__attribute__((target("avx2,fma")))
float ip_avx2(const float* a, const float* b, size_t dim) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= dim; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    }

    float sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < dim; ++i) sum += a[i] * b[i];
    return 1.0f - sum;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

struct Kernels {
    DistanceFn l2;
    DistanceFn ip;
    const char* name;
};

Kernels resolve() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return {l2_avx2, ip_avx2, "avx2+fma"};
    }
    return {l2_scalar, ip_scalar, "scalar"};
}

const Kernels& kernels() {
    static const Kernels k = resolve();  // thread-safe init, C++11 magic statics
    return k;
}

}  // namespace

DistanceFn get_distance_fn(Metric metric) {
    switch (metric) {
        case Metric::L2:
            return kernels().l2;
        case Metric::InnerProduct:
        case Metric::Cosine:
            // Cosine collapses to inner product because HnswIndex normalises
            // every vector on the way in. See the header for the reasoning.
            return kernels().ip;
    }
    return kernels().l2;
}

const char* simd_backend() { return kernels().name; }

void normalize(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) norm += vec[i] * vec[i];
    if (norm <= 0.0f) return;  // a zero vector has no direction; leave it be
    const float inv = 1.0f / std::sqrt(norm);
    for (size_t i = 0; i < dim; ++i) vec[i] *= inv;
}

}  // namespace vectorforge
