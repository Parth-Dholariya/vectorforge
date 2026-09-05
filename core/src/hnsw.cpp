#include "vectorforge/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <queue>
#include <stdexcept>

namespace vectorforge {
namespace {

constexpr size_t kAlign = 32;          // AVX2 register width
constexpr size_t kLockStripes = 65536; // one mutex per element would cost more
                                       // than the graph itself at 10M vectors

size_t round_up(size_t n, size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

using Cand = std::pair<float, idx_t>;
using MaxHeap = std::priority_queue<Cand>;
using MinHeap = std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>>;

}  // namespace

HnswIndex::HnswIndex(const HnswConfig& config)
    : dim_(config.dim),
      max_elements_(config.max_elements),
      M_(config.M),
      M0_(config.M * 2),
      ef_construction_(std::max<size_t>(config.ef_construction, config.M)),
      metric_(config.metric),
      normalize_input_(config.metric == Metric::Cosine),
      dist_(get_distance_fn(config.metric)),
      link_locks_(kLockStripes),
      level_rng_(static_cast<uint32_t>(config.seed)),
      level_mult_(1.0 / std::log(static_cast<double>(config.M))) {
    if (dim_ == 0) throw std::invalid_argument("dim must be > 0");
    if (max_elements_ == 0) throw std::invalid_argument("max_elements must be > 0");
    if (M_ < 2) throw std::invalid_argument("M must be >= 2");

    const size_t links0_bytes = sizeof(uint32_t) + M0_ * sizeof(idx_t);
    // Pad so the vector lands on a 32-byte boundary. The AVX2 kernels use
    // unaligned loads so this is not a correctness requirement, but an aligned
    // load that never straddles a cache line is measurably cheaper in the
    // inner loop, and the inner loop is the whole game here.
    offset_data_ = round_up(links0_bytes, kAlign);
    offset_label_ = offset_data_ + dim_ * sizeof(float);
    stride_ = round_up(offset_label_ + sizeof(label_t), kAlign);
    upper_link_bytes_ = sizeof(uint32_t) + M_ * sizeof(idx_t);

    const size_t arena_bytes = stride_ * max_elements_;
    data_level0_ = static_cast<char*>(
        ::operator new(arena_bytes, std::align_val_t(kAlign)));
    std::memset(data_level0_, 0, arena_bytes);

    link_lists_.assign(max_elements_, nullptr);
    element_levels_.assign(max_elements_, 0);

    // Enough lists for every hardware thread to search concurrently without
    // hitting the pool's allocation path.
    visited_pool_ = std::make_unique<VisitedListPool>(16, max_elements_);
}

HnswIndex::~HnswIndex() {
    for (char* p : link_lists_) free(p);
    if (data_level0_) {
        ::operator delete(data_level0_, std::align_val_t(kAlign));
    }
}

size_t HnswIndex::memory_usage() const {
    size_t bytes = stride_ * max_elements_;
    const size_t n = size();
    for (size_t i = 0; i < n; ++i) {
        if (link_lists_[i]) bytes += element_levels_[i] * upper_link_bytes_;
    }
    return bytes;
}

int HnswIndex::sample_level() {
    std::lock_guard<std::mutex> lock(level_gen_mutex_);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // Paper eq. (1): l = floor(-ln(U(0,1)) * mL). Produces a geometric decay
    // where each layer holds ~1/M of the one below, which is what makes the
    // descent cost logarithmic.
    double r = uniform(level_rng_);
    if (r <= 0.0) r = std::numeric_limits<double>::min();
    return static_cast<int>(-std::log(r) * level_mult_);
}

// ---------------------------------------------------------------------------

template <bool Locking>
idx_t HnswIndex::greedy_descend(const float* query, idx_t entry, float& entry_dist,
                                int level) const {
    std::vector<idx_t> snapshot;
    bool moved = true;
    while (moved) {
        moved = false;

        const idx_t* nbrs;
        uint32_t degree;
        if constexpr (Locking) {
            std::lock_guard<std::mutex> lock(link_locks_[entry % kLockStripes]);
            const uint32_t* list = links(entry, level);
            snapshot.assign(list + 1, list + 1 + list[0]);
            nbrs = snapshot.data();
            degree = static_cast<uint32_t>(snapshot.size());
        } else {
            const uint32_t* list = links(entry, level);
            nbrs = list + 1;
            degree = list[0];
        }

        for (uint32_t i = 0; i < degree; ++i) {
            const idx_t cand = nbrs[i];
            const float d = dist_(query, vector_at(cand), dim_);
            if (d < entry_dist) {
                entry_dist = d;
                entry = cand;
                moved = true;
            }
        }
    }
    return entry;
}

template <bool Locking>
std::vector<Cand> HnswIndex::search_layer(const float* query, idx_t entry, size_t ef,
                                          int level, VisitedList& visited) const {
    MaxHeap top;      // the ef best so far, worst at the top so we can evict
    MinHeap frontier; // unexplored candidates, best at the top

    const float d0 = dist_(query, vector_at(entry), dim_);
    visited.test_and_set(entry);
    top.emplace(d0, entry);
    frontier.emplace(d0, entry);
    float worst = d0;

    std::vector<idx_t> snapshot;  // reused across hops, see the Locking branch
    while (!frontier.empty()) {
        const Cand cur = frontier.top();
        // Every remaining candidate is further than our current worst keeper
        // and we already have ef of them, so nothing left can improve the
        // result. This early exit is what bounds the search.
        if (cur.first > worst && top.size() >= ef) break;
        frontier.pop();

        const idx_t* nbrs;
        uint32_t degree;
        if constexpr (Locking) {
            // Under concurrent insert the neighbour list can be rewritten
            // while we walk it, so copy it out and drop the lock before doing
            // any distance work. snapshot is hoisted out of the loop so this is
            // a reused buffer, not an allocation per node visited.
            std::lock_guard<std::mutex> lock(link_locks_[cur.second % kLockStripes]);
            const uint32_t* l = links(cur.second, level);
            snapshot.assign(l + 1, l + 1 + l[0]);
            nbrs = snapshot.data();
            degree = static_cast<uint32_t>(snapshot.size());
        } else {
            const uint32_t* l = links(cur.second, level);
            nbrs = l + 1;
            degree = l[0];
        }

        for (uint32_t i = 0; i < degree; ++i) {
            const idx_t cand = nbrs[i];
            if (visited.test_and_set(cand)) continue;

            const float d = dist_(query, vector_at(cand), dim_);
            if (top.size() < ef || d < worst) {
                frontier.emplace(d, cand);
                top.emplace(d, cand);
                if (top.size() > ef) top.pop();
                worst = top.top().first;
            }
        }
    }

    // Drain the max-heap (worst first) then flip, so callers always get
    // ascending distance.
    std::vector<Cand> out;
    out.reserve(top.size());
    while (!top.empty()) {
        out.push_back(top.top());
        top.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

void HnswIndex::select_neighbors_heuristic(std::vector<Cand>& candidates, size_t M) const {
    if (candidates.size() <= M) return;

    // Algorithm 4. Take a candidate only if it is closer to the base element
    // than to any neighbour already selected. Plain "keep the M nearest" builds
    // a graph where all of one node's links point into the same tight cluster;
    // this rule forces diversity of direction and is what preserves the
    // long-range edges the greedy descent needs to escape local minima.
    std::vector<Cand> kept;
    kept.reserve(M);
    for (const Cand& c : candidates) {  // already ascending
        if (kept.size() >= M) break;
        bool good = true;
        for (const Cand& k : kept) {
            const float d_between =
                dist_(vector_at(c.second), vector_at(k.second), dim_);
            if (d_between < c.first) {
                good = false;
                break;
            }
        }
        if (good) kept.push_back(c);
    }
    candidates.swap(kept);
}

void HnswIndex::connect(idx_t node, std::vector<Cand>& neighbors, int level) {
    const size_t maxd = max_degree(level);

    {
        std::lock_guard<std::mutex> lock(link_locks_[node % kLockStripes]);
        uint32_t* list = links(node, level);
        list[0] = static_cast<uint32_t>(neighbors.size());
        for (size_t i = 0; i < neighbors.size(); ++i) {
            list[1 + i] = neighbors[i].second;
        }
    }

    // Links are bidirectional. Adding the back-edge can overflow the
    // neighbour's budget, in which case it re-runs the same heuristic over its
    // existing links plus the new one and keeps the best spread.
    for (const Cand& nb : neighbors) {
        std::lock_guard<std::mutex> lock(link_locks_[nb.second % kLockStripes]);
        uint32_t* list = links(nb.second, level);
        const uint32_t degree = list[0];

        if (degree < maxd) {
            list[1 + degree] = node;
            list[0] = degree + 1;
            continue;
        }

        std::vector<Cand> merged;
        merged.reserve(degree + 1);
        const float* base = vector_at(nb.second);
        for (uint32_t i = 0; i < degree; ++i) {
            const idx_t other = list[1 + i];
            merged.emplace_back(dist_(base, vector_at(other), dim_), other);
        }
        merged.emplace_back(nb.first, node);  // dist(nb, node) == dist(node, nb)
        std::sort(merged.begin(), merged.end());

        select_neighbors_heuristic(merged, maxd);
        list[0] = static_cast<uint32_t>(merged.size());
        for (size_t i = 0; i < merged.size(); ++i) {
            list[1 + i] = merged[i].second;
        }
    }
}

// ---------------------------------------------------------------------------

void HnswIndex::add(const float* vec, label_t label) {
    const size_t slot = cur_element_count_.fetch_add(1, std::memory_order_acq_rel);
    if (slot >= max_elements_) {
        cur_element_count_.fetch_sub(1, std::memory_order_acq_rel);
        throw std::runtime_error("index is full: max_elements reached");
    }
    const idx_t cur = static_cast<idx_t>(slot);
    const int level = sample_level();

    // Copy the vector into the arena, normalising first if the metric is
    // cosine so that every later distance evaluation is a plain dot product.
    float* dst = reinterpret_cast<float*>(element(cur) + offset_data_);
    std::memcpy(dst, vec, dim_ * sizeof(float));
    if (normalize_input_) normalize(dst, dim_);
    *reinterpret_cast<label_t*>(element(cur) + offset_label_) = label;

    element_levels_[cur] = level;
    if (level > 0) {
        const size_t bytes = level * upper_link_bytes_;
        link_lists_[cur] = static_cast<char*>(calloc(1, bytes));
        if (!link_lists_[cur]) throw std::bad_alloc();
    }

    // Serialise only the inserts that reach above the current top layer; those
    // are rare (probability ~1/M per level) so this lock is nearly uncontended.
    std::unique_lock<std::mutex> global(global_mutex_, std::defer_lock);
    if (level > max_level_) global.lock();

    if (cur == 0) {
        entry_point_ = cur;
        max_level_ = level;
        return;
    }

    idx_t entry = entry_point_;
    const int top_level = max_level_;
    float entry_dist = dist_(dst, vector_at(entry), dim_);

    // Phase 1: zoom in through the layers this element does not belong to,
    // greedily and with a beam of one. Cheap, and it lands us near the right
    // neighbourhood before the expensive part starts.
    for (int lvl = top_level; lvl > level; --lvl) {
        entry = greedy_descend<true>(dst, entry, entry_dist, lvl);
    }

    // Phase 2: from this element's own top layer down to 0, run the real beam
    // search and wire up links.
    ScopedVisitedList visited(*visited_pool_);
    for (int lvl = std::min(level, top_level); lvl >= 0; --lvl) {
        visited->reset();
        std::vector<Cand> candidates =
            search_layer<true>(dst, entry, ef_construction_, lvl, *visited);
        if (candidates.empty()) continue;

        entry = candidates.front().second;  // best on this layer seeds the next
        // A new element gets at most M outgoing links even on layer 0. M0 is
        // only the ceiling for back-edges accumulated by existing nodes, which
        // connect() enforces separately.
        select_neighbors_heuristic(candidates, M_);
        connect(cur, candidates, lvl);
    }

    if (level > max_level_) {
        entry_point_ = cur;
        max_level_ = level;
    }
}

std::vector<SearchResult> HnswIndex::search(const float* query, size_t k, size_t ef) const {
    if (size() == 0) return {};

    // Cosine queries must live in the same normalised space as the stored
    // vectors, so normalise a scratch copy rather than the caller's buffer.
    std::vector<float> scratch;
    const float* q = query;
    if (normalize_input_) {
        scratch.assign(query, query + dim_);
        normalize(scratch.data(), dim_);
        q = scratch.data();
    }

    idx_t entry = entry_point_;
    float entry_dist = dist_(q, vector_at(entry), dim_);
    for (int lvl = max_level_; lvl > 0; --lvl) {
        entry = greedy_descend<false>(q, entry, entry_dist, lvl);
    }

    ef = std::max(ef, k);
    ScopedVisitedList visited(*visited_pool_);
    std::vector<Cand> found = search_layer<false>(q, entry, ef, 0, *visited);

    if (found.size() > k) found.resize(k);
    std::vector<SearchResult> out;
    out.reserve(found.size());
    for (const Cand& c : found) {
        out.push_back({c.first, label_at(c.second)});
    }
    return out;
}

}  // namespace vectorforge
