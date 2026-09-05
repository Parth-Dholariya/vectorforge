#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace vectorforge {

// A visited-set for one graph traversal.
//
// The obvious implementation is std::vector<bool> cleared per query, but that
// clear is O(max_elements) while a single search only touches O(ef * M) nodes.
// At 1M elements the memset dominates the query. Instead every slot holds a
// generation stamp: bumping the generation invalidates the whole array in O(1),
// and we only pay a real memset on the rare wrap of the 16-bit counter.
class VisitedList {
public:
    explicit VisitedList(size_t capacity)
        : capacity_(capacity), generation_(1), marks_(new uint16_t[capacity]()) {}

    void reset() {
        if (++generation_ == 0) {                                  // wrapped
            std::memset(marks_.get(), 0, capacity_ * sizeof(uint16_t));
            generation_ = 1;
        }
    }

    bool test_and_set(size_t i) {
        if (marks_[i] == generation_) return true;
        marks_[i] = generation_;
        return false;
    }

    size_t capacity() const { return capacity_; }

private:
    size_t capacity_;
    uint16_t generation_;
    std::unique_ptr<uint16_t[]> marks_;
};

// Concurrent searches each need their own visited list. Allocating one per
// query would mean a fresh 2 MB zeroed allocation per query at 1M elements, so
// we hand them out from a pool and recycle.
class VisitedListPool {
public:
    VisitedListPool(size_t initial, size_t capacity) : capacity_(capacity) {
        for (size_t i = 0; i < initial; ++i) {
            pool_.push_back(std::make_unique<VisitedList>(capacity));
        }
    }

    std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            auto v = std::move(pool_.back());
            pool_.pop_back();
            v->reset();
            return v;
        }
        return std::make_unique<VisitedList>(capacity_);
    }

    void release(std::unique_ptr<VisitedList> v) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(v));
    }

private:
    size_t capacity_;
    std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> pool_;
};

// RAII wrapper so an early return or a throw inside search() cannot leak a
// list out of the pool.
class ScopedVisitedList {
public:
    explicit ScopedVisitedList(VisitedListPool& pool)
        : pool_(pool), list_(pool.acquire()) {}
    ~ScopedVisitedList() { pool_.release(std::move(list_)); }

    ScopedVisitedList(const ScopedVisitedList&) = delete;
    ScopedVisitedList& operator=(const ScopedVisitedList&) = delete;

    VisitedList* operator->() const { return list_.get(); }
    VisitedList& operator*() const { return *list_; }

private:
    VisitedListPool& pool_;
    std::unique_ptr<VisitedList> list_;
};

}  // namespace vectorforge
