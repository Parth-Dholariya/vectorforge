// Recall / QPS benchmark.
//
//   bench_recall --synthetic --n 100000 --dim 128
//   bench_recall --base data/sift/sift_base.fvecs \
//                --query data/sift/sift_query.fvecs \
//                --gt data/sift/sift_groundtruth.ivecs
//
// Reports recall@k against ground truth and single-thread QPS across a sweep of
// ef. That curve - not any single number - is the thing to compare against
// FAISS, because any ANN index can trade one for the other.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "vectorforge/hnsw.hpp"

using namespace vectorforge;
using Clock = std::chrono::steady_clock;

namespace {

// .fvecs / .ivecs: little-endian, each record is int32 dim followed by dim
// float32 (or int32) values. The format the SIFT/GIST corpora ship in.
template <typename T>
std::vector<T> read_xvecs(const std::string& path, size_t& n, size_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    int32_t d = 0;
    in.read(reinterpret_cast<char*>(&d), 4);
    if (d <= 0) {
        std::fprintf(stderr, "%s: bogus dimension %d\n", path.c_str(), d);
        std::exit(1);
    }
    dim = size_t(d);

    const size_t record_bytes = 4 + dim * sizeof(T);
    if (size_t(bytes) % record_bytes != 0) {
        std::fprintf(stderr, "%s: size %lld is not a multiple of record size %zu\n",
                     path.c_str(), (long long)bytes, record_bytes);
        std::exit(1);
    }
    n = size_t(bytes) / record_bytes;

    std::vector<T> out(n * dim);
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; ++i) {
        int32_t rd = 0;
        in.read(reinterpret_cast<char*>(&rd), 4);
        if (size_t(rd) != dim) {
            std::fprintf(stderr, "%s: ragged record at %zu\n", path.c_str(), i);
            std::exit(1);
        }
        in.read(reinterpret_cast<char*>(out.data() + i * dim), std::streamsize(dim * sizeof(T)));
    }
    return out;
}

std::vector<float> synthetic(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

std::vector<int32_t> compute_gt(const std::vector<float>& base, size_t n, size_t dim,
                                const std::vector<float>& queries, size_t nq, size_t k) {
    std::fprintf(stderr, "computing brute-force ground truth (%zu x %zu)...\n", nq, n);
    auto dist = get_distance_fn(Metric::L2);
    std::vector<int32_t> gt(nq * k);
    std::vector<std::pair<float, int32_t>> scratch(n);
    for (size_t q = 0; q < nq; ++q) {
        const float* query = queries.data() + q * dim;
        for (size_t i = 0; i < n; ++i) {
            scratch[i] = {dist(query, base.data() + i * dim, dim), int32_t(i)};
        }
        std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end());
        for (size_t j = 0; j < k; ++j) gt[q * k + j] = scratch[j].second;
    }
    return gt;
}

const char* arg(int argc, char** argv, const char* flag, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const size_t k = std::stoul(arg(argc, argv, "--k", "10"));
    const size_t M = std::stoul(arg(argc, argv, "--M", "16"));
    const size_t efc = std::stoul(arg(argc, argv, "--efc", "200"));

    std::vector<float> base, queries;
    std::vector<int32_t> gt;
    size_t n = 0, dim = 0, nq = 0, gt_stride = 0;

    if (has_flag(argc, argv, "--synthetic")) {
        n = std::stoul(arg(argc, argv, "--n", "100000"));
        dim = std::stoul(arg(argc, argv, "--dim", "128"));
        nq = std::stoul(arg(argc, argv, "--nq", "1000"));
        base = synthetic(n, dim, 7);
        queries = synthetic(nq, dim, 99);
        gt = compute_gt(base, n, dim, queries, nq, k);
        gt_stride = k;
    } else {
        size_t qdim = 0;
        base = read_xvecs<float>(arg(argc, argv, "--base", "data/sift/sift_base.fvecs"), n, dim);
        queries = read_xvecs<float>(arg(argc, argv, "--query", "data/sift/sift_query.fvecs"), nq, qdim);
        if (qdim != dim) {
            std::fprintf(stderr, "dim mismatch: base %zu, query %zu\n", dim, qdim);
            return 1;
        }
        const char* gt_path = arg(argc, argv, "--gt", "");
        if (gt_path[0]) {
            size_t gtn = 0;
            gt = read_xvecs<int32_t>(gt_path, gtn, gt_stride);
            if (gtn != nq) {
                std::fprintf(stderr, "ground truth has %zu rows, expected %zu\n", gtn, nq);
                return 1;
            }
        } else {
            gt = compute_gt(base, n, dim, queries, nq, k);
            gt_stride = k;
        }
    }

    std::printf("dataset : n=%zu dim=%zu nq=%zu k=%zu\n", n, dim, nq, k);
    std::printf("params  : M=%zu efConstruction=%zu\n", M, efc);
    std::printf("backend : %s\n", simd_backend());

    HnswConfig cfg;
    cfg.dim = dim;
    cfg.max_elements = n;
    cfg.M = M;
    cfg.ef_construction = efc;
    cfg.metric = Metric::L2;
    HnswIndex index(cfg);

    auto t0 = Clock::now();
    for (size_t i = 0; i < n; ++i) {
        index.add(base.data() + i * dim, label_t(i));
        if ((i + 1) % 100000 == 0) {
            std::fprintf(stderr, "  inserted %zu / %zu\n", i + 1, n);
        }
    }
    const double build_s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("build   : %.1f s (%.0f vec/s), %.1f MB, %d levels\n\n", build_s,
                double(n) / build_s, double(index.memory_usage()) / (1024.0 * 1024.0),
                index.max_level());

    std::printf("%6s %10s %12s %12s\n", "ef", "recall@10", "QPS", "p99 (ms)");
    std::printf("%6s %10s %12s %12s\n", "------", "----------", "------------", "------------");

    for (size_t ef : {10, 16, 32, 64, 100, 200, 400, 800}) {
        if (ef < k) continue;

        // One untimed pass so the comparison isn't measuring cold page faults
        // on the arena rather than search work.
        for (size_t q = 0; q < std::min<size_t>(nq, 100); ++q) {
            index.search(queries.data() + q * dim, k, ef);
        }

        std::vector<double> latencies;
        latencies.reserve(nq);
        size_t hits = 0;

        for (size_t q = 0; q < nq; ++q) {
            auto qt = Clock::now();
            auto res = index.search(queries.data() + q * dim, k, ef);
            latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - qt).count());

            const int32_t* truth = gt.data() + q * gt_stride;
            for (const auto& r : res) {
                for (size_t j = 0; j < k; ++j) {
                    if (truth[j] == int32_t(r.label)) { ++hits; break; }
                }
            }
        }

        double total = 0;
        for (double l : latencies) total += l;
        std::sort(latencies.begin(), latencies.end());
        const double p99 = latencies[size_t(double(latencies.size()) * 0.99)];

        std::printf("%6zu %10.4f %12.0f %12.3f\n", ef, double(hits) / double(nq * k),
                    double(nq) / (total / 1000.0), p99);
    }

    return 0;
}
