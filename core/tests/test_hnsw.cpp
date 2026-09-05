#include "vectorforge/hnsw.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "test_util.hpp"

using namespace vectorforge;

namespace {

std::vector<float> random_dataset(std::mt19937& rng, size_t n, size_t dim) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = g(rng);
    return data;
}

// Exhaustive ground truth, computed in double precision from the metric's
// definition. It deliberately does NOT call get_distance_fn: reusing the
// library's own kernel to check the library's own index would let a wrong
// kernel agree with itself and pass.
//
// The cosine case is the one that matters. get_distance_fn(Metric::Cosine)
// hands back the inner-product kernel, which equals cosine only once both
// operands are unit length - the index arranges that by normalising on insert.
// A reference that skips the normalisation is ranking by raw inner product,
// which on Gaussian data is dominated by vector magnitude and produces a
// genuinely different top-k.
std::vector<label_t> brute_force(const std::vector<float>& data, size_t n, size_t dim,
                                 const float* query, size_t k, Metric metric) {
    auto norm = [&](const float* v) {
        double s = 0;
        for (size_t i = 0; i < dim; ++i) s += double(v[i]) * double(v[i]);
        return std::sqrt(s);
    };
    const double qn = (metric == Metric::Cosine) ? norm(query) : 1.0;

    std::vector<std::pair<double, label_t>> all;
    all.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const float* v = data.data() + i * dim;
        double d = 0;
        if (metric == Metric::L2) {
            for (size_t j = 0; j < dim; ++j) {
                const double diff = double(query[j]) - double(v[j]);
                d += diff * diff;
            }
        } else {
            double dot = 0;
            for (size_t j = 0; j < dim; ++j) dot += double(query[j]) * double(v[j]);
            const double vn = (metric == Metric::Cosine) ? norm(v) : 1.0;
            const double denom = qn * vn;
            d = 1.0 - (denom > 0 ? dot / denom : 0.0);
        }
        all.emplace_back(d, label_t(i));
    }
    std::partial_sort(all.begin(), all.begin() + std::min(k, all.size()), all.end());
    std::vector<label_t> out;
    for (size_t i = 0; i < std::min(k, all.size()); ++i) out.push_back(all[i].second);
    return out;
}

double measure_recall(HnswIndex& index, const std::vector<float>& data, size_t n,
                      size_t dim, const std::vector<float>& queries, size_t nq,
                      size_t k, size_t ef, Metric metric) {
    size_t hits = 0;
    for (size_t q = 0; q < nq; ++q) {
        const float* query = queries.data() + q * dim;
        auto truth = brute_force(data, n, dim, query, k, metric);
        auto got = index.search(query, k, ef);
        for (const auto& r : got) {
            if (std::find(truth.begin(), truth.end(), r.label) != truth.end()) ++hits;
        }
    }
    return double(hits) / double(nq * k);
}

}  // namespace

int main() {
    std::mt19937 rng(1234);
    const size_t n = 5000, dim = 64, nq = 100, k = 10;

    auto data = random_dataset(rng, n, dim);
    auto queries = random_dataset(rng, nq, dim);

    // --- L2: the core recall guarantee ------------------------------------
    {
        HnswConfig cfg;
        cfg.dim = dim;
        cfg.max_elements = n;
        cfg.M = 16;
        cfg.ef_construction = 200;
        cfg.metric = Metric::L2;
        HnswIndex index(cfg);

        for (size_t i = 0; i < n; ++i) index.add(data.data() + i * dim, label_t(i));
        CHECK_EQ(index.size(), n);

        // Gaussian data in 64 dims is close to the hardest case for ANN (no
        // cluster structure to exploit), so 0.95 here is a conservative floor.
        // If a refactor drops recall below this, something is genuinely broken.
        const double recall = measure_recall(index, data, n, dim, queries, nq, k, 100,
                                             Metric::L2);
        std::printf("  L2   recall@%zu (ef=100) = %.4f  levels=%d  mem=%.1f MB\n", k,
                    recall, index.max_level(),
                    double(index.memory_usage()) / (1024.0 * 1024.0));
        CHECK_GE(recall, 0.95);

        // Raising ef must not lower recall - that monotonicity is the whole
        // contract of the parameter, and breaking it means the beam search's
        // termination condition is wrong.
        const double recall_low = measure_recall(index, data, n, dim, queries, nq, k, 16,
                                                 Metric::L2);
        const double recall_high = measure_recall(index, data, n, dim, queries, nq, k, 300,
                                                  Metric::L2);
        std::printf("  L2   recall@%zu ef=16 -> %.4f, ef=300 -> %.4f\n", k, recall_low,
                    recall_high);
        CHECK_GE(recall_high, recall_low);

        // A vector already in the index must retrieve itself first, at
        // distance 0. If this fails the graph is disconnected somewhere.
        size_t self_found = 0;
        for (size_t i = 0; i < 200; ++i) {
            auto r = index.search(data.data() + i * dim, 1, 64);
            if (!r.empty() && r[0].label == label_t(i)) ++self_found;
        }
        std::printf("  L2   self-retrieval = %zu/200\n", self_found);
        CHECK_GE(self_found, size_t(198));

        // Results must come back sorted by distance.
        auto r = index.search(queries.data(), k, 100);
        bool sorted = true;
        for (size_t i = 1; i < r.size(); ++i) sorted = sorted && r[i - 1].distance <= r[i].distance;
        CHECK(sorted);
        CHECK_EQ(r.size(), k);
    }

    // --- Cosine: exercises the normalise-on-insert path --------------------
    {
        HnswConfig cfg;
        cfg.dim = dim;
        cfg.max_elements = n;
        cfg.metric = Metric::Cosine;
        HnswIndex index(cfg);
        for (size_t i = 0; i < n; ++i) index.add(data.data() + i * dim, label_t(i));

        // Ground truth is true cosine in double precision on the raw vectors;
        // the index stores normalised copies and evaluates a dot product.
        // Agreement between the two is what proves the normalise-on-insert
        // identity holds end to end.
        const double recall = measure_recall(index, data, n, dim, queries, nq, k, 100,
                                             Metric::Cosine);
        std::printf("  COS  recall@%zu (ef=100) = %.4f\n", k, recall);
        CHECK_GE(recall, 0.90);
    }

    // --- concurrent insert --------------------------------------------------
    {
        HnswConfig cfg;
        cfg.dim = dim;
        cfg.max_elements = n;
        cfg.metric = Metric::L2;
        HnswIndex index(cfg);

        const size_t threads = std::max(2u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (size_t t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                for (size_t i = t; i < n; i += threads) {
                    index.add(data.data() + i * dim, label_t(i));
                }
            });
        }
        for (auto& th : pool) th.join();

        CHECK_EQ(index.size(), n);
        const double recall = measure_recall(index, data, n, dim, queries, nq, k, 100,
                                             Metric::L2);
        std::printf("  MT   recall@%zu after %zu-thread build = %.4f\n", k, threads, recall);
        // A data race in the link lists shows up as a shredded graph and
        // recall collapsing, so this is really a race detector.
        CHECK_GE(recall, 0.95);
    }

    // --- capacity ----------------------------------------------------------
    {
        HnswConfig cfg;
        cfg.dim = 4;
        cfg.max_elements = 2;
        HnswIndex index(cfg);
        float v[4] = {1, 2, 3, 4};
        index.add(v, 0);
        index.add(v, 1);
        bool threw = false;
        try {
            index.add(v, 2);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
        // The rejected insert must not have consumed a slot.
        CHECK_EQ(index.size(), size_t(2));
    }

    // Searching an empty index is legal and returns nothing.
    {
        HnswConfig cfg;
        cfg.dim = 4;
        cfg.max_elements = 8;
        HnswIndex index(cfg);
        float v[4] = {1, 2, 3, 4};
        CHECK(index.search(v, 5, 32).empty());
    }

    return vftest::summary("test_hnsw");
}
