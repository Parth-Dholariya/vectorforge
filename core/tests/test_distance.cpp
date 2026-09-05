#include "vectorforge/distance.hpp"

#include <cstdio>
#include <random>
#include <vector>

#include "test_util.hpp"

using namespace vectorforge;

namespace {

// Independent reference. Deliberately written the dumb way so it shares no
// code with the kernels it is checking.
double ref_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        s += d * d;
    }
    return s;
}

double ref_ip(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return 1.0 - s;
}

std::vector<float> random_vec(std::mt19937& rng, size_t dim) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = g(rng);
    return v;
}

}  // namespace

int main() {
    std::printf("SIMD backend: %s\n", simd_backend());
    std::mt19937 rng(42);

    // Dimensions chosen to exercise every path through the kernels: the 32-wide
    // unrolled body, the 8-wide body, and the scalar tail. 768 is a real BERT
    // embedding width and is the one that actually hits the tail (768 = 24*32,
    // so it doesn't - but 100 and 13 do). 128 = SIFT, 960 = GIST.
    const std::vector<size_t> dims = {1, 3, 8, 13, 100, 128, 384, 768, 960};

    auto l2 = get_distance_fn(Metric::L2);
    auto ip = get_distance_fn(Metric::InnerProduct);

    for (size_t dim : dims) {
        for (int trial = 0; trial < 20; ++trial) {
            auto a = random_vec(rng, dim);
            auto b = random_vec(rng, dim);

            // Tolerance scales with dim: float32 accumulation error grows with
            // the number of terms summed, and the SIMD version sums in a
            // different order than the reference, so bit-exactness is not the
            // bar. Relative agreement is.
            const double tol = 1e-4 * double(dim);
            CHECK_NEAR(double(l2(a.data(), b.data(), dim)), ref_l2(a, b), tol);
            CHECK_NEAR(double(ip(a.data(), b.data(), dim)), ref_ip(a, b), tol);
        }
    }

    // Self-distance must be exactly zero, not merely small - the graph search
    // relies on a query finding itself at distance 0 when it is in the index.
    for (size_t dim : dims) {
        auto a = random_vec(rng, dim);
        CHECK_EQ(l2(a.data(), a.data(), dim), 0.0f);
    }

    // L2 is symmetric.
    for (size_t dim : dims) {
        auto a = random_vec(rng, dim);
        auto b = random_vec(rng, dim);
        CHECK_NEAR(l2(a.data(), b.data(), dim), l2(b.data(), a.data(), dim), 1e-3f);
    }

    // normalize() must produce unit length, and after normalising both sides
    // the inner-product kernel must equal true cosine distance. This is the
    // identity the Cosine metric is built on, so it is worth pinning down.
    for (size_t dim : dims) {
        if (dim < 2) continue;
        auto a = random_vec(rng, dim);
        auto b = random_vec(rng, dim);

        double cos_num = 0, na = 0, nb = 0;
        for (size_t i = 0; i < dim; ++i) {
            cos_num += double(a[i]) * double(b[i]);
            na += double(a[i]) * double(a[i]);
            nb += double(b[i]) * double(b[i]);
        }
        const double true_cos_dist = 1.0 - cos_num / (std::sqrt(na) * std::sqrt(nb));

        normalize(a.data(), dim);
        normalize(b.data(), dim);

        double len = 0;
        for (size_t i = 0; i < dim; ++i) len += double(a[i]) * double(a[i]);
        CHECK_NEAR(len, 1.0, 1e-5);

        CHECK_NEAR(double(ip(a.data(), b.data(), dim)), true_cos_dist, 1e-4);
    }

    // A zero vector has no direction; normalize must leave it alone rather
    // than emit NaNs that would poison every later comparison.
    {
        std::vector<float> z(64, 0.0f);
        normalize(z.data(), z.size());
        bool finite = true;
        for (float x : z) finite = finite && std::isfinite(x) && x == 0.0f;
        CHECK(finite);
    }

    return vftest::summary("test_distance");
}
