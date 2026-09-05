#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#include "vectorforge/distance.hpp"
#include "vectorforge/visited_list.hpp"

namespace vectorforge {

using idx_t = uint32_t;    // internal slot id, dense, assigned on insert
using label_t = uint64_t;  // caller-supplied id, arbitrary

struct HnswConfig {
    size_t dim = 0;
    size_t max_elements = 0;
    size_t M = 16;                 // out-degree on layers >= 1
    size_t ef_construction = 200;  // beam width while building
    Metric metric = Metric::L2;
    uint64_t seed = 100;
};

struct SearchResult {
    float distance;
    label_t label;
};

// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016/2018).
//
// MEMORY LAYOUT
// -------------
// Layer 0 holds every element and is where the overwhelming majority of
// distance evaluations happen, so its data is packed into one contiguous arena
// with the links, the vector, and the label adjacent per element:
//
//   element i lives at data_level0_ + i * stride_
//   [ 0                ) uint32 degree
//   [ 4                ) idx_t  neighbours[M0]        M0 = 2*M
//   [ offset_data_      ) float  vector[dim]
//   [ offset_label_     ) label_t label
//
// The point is that reading a node's neighbour list and then its vector is one
// cache line walk instead of two pointer chases into unrelated heap blocks.
// This layout is also the reason Phase 2 can mmap the arena and get a usable
// index with zero deserialisation - the on-disk bytes are already the in-memory
// representation.
//
// Layers >= 1 are sparse (roughly 1/M of the elements reach layer 1, 1/M^2
// reach layer 2, ...) so they get separate small allocations per element rather
// than a dense arena.
//
// CONCURRENCY
// -----------
// add() is safe to call from many threads: each element's link list is guarded
// by a striped mutex, and the global entry point is guarded separately.
// search() is lock-free and safe concurrently with other searches. Searching
// while adding is NOT supported in v1 - see docs/DESIGN.md.
class HnswIndex {
public:
    explicit HnswIndex(const HnswConfig& config);
    ~HnswIndex();

    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    void add(const float* vec, label_t label);
    std::vector<SearchResult> search(const float* query, size_t k, size_t ef) const;

    size_t size() const { return cur_element_count_.load(std::memory_order_acquire); }
    size_t dim() const { return dim_; }
    int max_level() const { return max_level_; }
    // Bytes held by the layer-0 arena plus the upper-layer link lists.
    size_t memory_usage() const;

private:
    // --- layout helpers -----------------------------------------------------
    char* element(idx_t i) const { return data_level0_ + i * stride_; }
    uint32_t* links0(idx_t i) const { return reinterpret_cast<uint32_t*>(element(i)); }
    const float* vector_at(idx_t i) const {
        return reinterpret_cast<const float*>(element(i) + offset_data_);
    }
    label_t label_at(idx_t i) const {
        return *reinterpret_cast<const label_t*>(element(i) + offset_label_);
    }
    // Link list for element i on the given level. level 0 -> the arena,
    // level > 0 -> the per-element upper-layer block.
    uint32_t* links(idx_t i, int level) const {
        if (level == 0) return links0(i);
        return reinterpret_cast<uint32_t*>(link_lists_[i] +
                                           (level - 1) * upper_link_bytes_);
    }
    size_t max_degree(int level) const { return level == 0 ? M0_ : M_; }

    // --- algorithm ----------------------------------------------------------
    int sample_level();
    // Greedy descent through a single upper layer: walk to the local minimum.
    // Locking=true is the insert path (a concurrent writer may be rewriting the
    // list); Locking=false is the query path, where taking a mutex per hop
    // would cost real QPS for no benefit since v1 forbids search-during-add.
    template <bool Locking>
    idx_t greedy_descend(const float* query, idx_t entry, float& entry_dist, int level) const;
    // Algorithm 2: beam search on one layer. Returns the ef best, ascending.
    template <bool Locking>
    std::vector<std::pair<float, idx_t>> search_layer(const float* query, idx_t entry,
                                                      size_t ef, int level,
                                                      VisitedList& visited) const;
    // Algorithm 4: the pruning heuristic. Keeps a neighbour only if it is
    // closer to the base than to every neighbour already kept, which is what
    // preserves long-range links and keeps the graph navigable.
    void select_neighbors_heuristic(std::vector<std::pair<float, idx_t>>& candidates,
                                    size_t M) const;
    void connect(idx_t node, std::vector<std::pair<float, idx_t>>& neighbors, int level);

    // --- config -------------------------------------------------------------
    size_t dim_ = 0;
    size_t max_elements_ = 0;
    size_t M_ = 16;
    size_t M0_ = 32;
    size_t ef_construction_ = 200;
    Metric metric_ = Metric::L2;
    bool normalize_input_ = false;
    DistanceFn dist_ = nullptr;

    // --- layout -------------------------------------------------------------
    size_t stride_ = 0;
    size_t offset_data_ = 0;
    size_t offset_label_ = 0;
    size_t upper_link_bytes_ = 0;

    char* data_level0_ = nullptr;
    std::vector<char*> link_lists_;
    std::vector<int> element_levels_;

    // --- state --------------------------------------------------------------
    std::atomic<size_t> cur_element_count_{0};
    idx_t entry_point_ = 0;
    int max_level_ = -1;

    mutable std::vector<std::mutex> link_locks_;
    mutable std::mutex global_mutex_;  // guards entry_point_ / max_level_
    std::mutex level_gen_mutex_;       // guards level_rng_
    std::mt19937 level_rng_;
    double level_mult_ = 0.0;  // mL = 1 / ln(M)

    std::unique_ptr<VisitedListPool> visited_pool_;
};

}  // namespace vectorforge
