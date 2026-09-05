#pragma once

#include <cstddef>
#include <cstdint>

namespace vectorforge {

enum class Metric {
    L2,            // squared euclidean (we never take the sqrt - monotonic, so ranking is identical)
    InnerProduct,  // 1 - dot(a,b)
    Cosine,        // 1 - cos(a,b); the index normalises on insert and then uses InnerProduct
};

// All distance functions return a value where SMALLER MEANS CLOSER. That
// invariant is what lets the graph search use one comparator everywhere.
using DistanceFn = float (*)(const float* a, const float* b, size_t dim);

// Resolved once at startup against the host's CPU features. The returned
// pointer is stable for the life of the process.
DistanceFn get_distance_fn(Metric metric);

// "avx2+fma", "sse2", or "scalar" - whichever kernel family the dispatcher
// picked. Benchmarks print this so a result is never ambiguous about which
// code path produced it.
const char* simd_backend();

// L2-normalise in place. Used when Metric::Cosine is requested: once every
// vector is unit length, cosine distance and inner-product distance are the
// same function, so we pay the normalisation once at insert instead of paying
// two extra reductions on every single distance evaluation in the search loop.
void normalize(float* vec, size_t dim);

}  // namespace vectorforge
