#pragma once

#include "ring/spsc_ring.hpp"
#include "type_utils.hpp"
#include <cstddef>
#include <tuple>
#include <utility>

// One SpscRing per output type for a single worker thread.
// ring<T>() is a compile error if T is not in Types... (via TypeIndex).
// Types... must be unique (AllUnique static_assert).
template<std::size_t RingSize, typename... Types>
class TypedRingSet {
    static_assert(AllUnique<Types...>::value,
                  "TypedRingSet: duplicate types in pack are not allowed");

    std::tuple<SpscRing<Types, RingSize>...> rings_;

    // Helper: sum available() across all rings via index sequence
    template<std::size_t... Is>
    std::size_t total_available_impl(std::index_sequence<Is...>) const noexcept {
        // fold expression: adds available() for each ring in the tuple
        return (std::get<Is>(rings_).available() + ...);
    }

    // Helper: check all_empty() across all rings via index sequence
    template<std::size_t... Is>
    bool all_empty_impl(std::index_sequence<Is...>) const noexcept {
        // fold expression: true only if every ring reports zero items
        return (... && (std::get<Is>(rings_).available() == 0));
    }

public:
    TypedRingSet() = default;
    TypedRingSet(const TypedRingSet&)            = delete;
    TypedRingSet& operator=(const TypedRingSet&) = delete;
    TypedRingSet(TypedRingSet&&)                 = delete;
    TypedRingSet& operator=(TypedRingSet&&)      = delete;

    // Returns a reference to the SpscRing for type T.
    // Compile error if T is not in Types...
    template<typename T>
    [[nodiscard]] SpscRing<T, RingSize>& ring() noexcept {
        // TypeIndex<T, Types...>::value resolves to the tuple index for T
        return std::get<TypeIndex<T, Types...>::value>(rings_);
    }

    template<typename T>
    [[nodiscard]] const SpscRing<T, RingSize>& ring() const noexcept {
        return std::get<TypeIndex<T, Types...>::value>(rings_);
    }

    // Sum of available() across all type rings. Used for test observability.
    // NOTE: not called by CommThread in production — remove if it stays unused
    // outside tests.
    [[nodiscard]] std::size_t total_available() const noexcept {
        return total_available_impl(std::index_sequence_for<Types...>{});
    }

    // True only when every ring in this set has zero items ready to read.
    [[nodiscard]] bool all_empty() const noexcept {
        return all_empty_impl(std::index_sequence_for<Types...>{});
    }
};
