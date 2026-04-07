#pragma once

#include "ring/typed_ring_set.hpp"
#include <cstddef>
#include <memory>
#include <vector>

// Owns all TypedRingSets for every worker thread on one rank.
// Constructed once during NodePlan; read-only (by structure) after that.
//
// unique_ptr is used because TypedRingSet is non-movable (it contains
// std::atomic members via SpscRing). std::vector requires MoveInsertable,
// which unique_ptr satisfies while TypedRingSet itself does not.
template<std::size_t RingSize, typename... Types>
class RingPool {
    std::vector<std::unique_ptr<TypedRingSet<RingSize, Types...>>> rings_;

public:
    // Constructs num_workers TypedRingSets, one per worker thread.
    explicit RingPool(std::size_t num_workers) {
        rings_.reserve(num_workers);
        for (std::size_t i = 0; i < num_workers; ++i) {
            rings_.emplace_back(
                std::make_unique<TypedRingSet<RingSize, Types...>>());
        }
    }

    RingPool(const RingPool&)            = delete;
    RingPool& operator=(const RingPool&) = delete;
    RingPool(RingPool&&)                 = delete;
    RingPool& operator=(RingPool&&)      = delete;

    // Returns the TypedRingSet for the given worker thread.
    [[nodiscard]] TypedRingSet<RingSize, Types...>& get(std::size_t worker_id) noexcept {
        return *rings_[worker_id];
    }

    // True only when every ring for every worker has zero items.
    [[nodiscard]] bool all_empty() const noexcept {
        for (const auto& rs : rings_) {
            if (!rs->all_empty()) {
                return false;
            }
        }
        return true;
    }

    // Returns the number of worker threads this pool was constructed for.
    [[nodiscard]] std::size_t num_workers() const noexcept {
        return rings_.size();
    }
};
