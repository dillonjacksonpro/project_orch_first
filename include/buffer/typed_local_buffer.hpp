#pragma once

#include "ring/typed_ring_set.hpp"
#include "type_utils.hpp"
#include <array>
#include <cstddef>
#include <tuple>

// Per-worker write-combining buffer. Accumulates items locally until the buffer
// for a given type reaches LocalBufSize, then bulk-flushes to the corresponding
// SpscRing via write_bulk(). This amortizes the atomic write_idx_ update cost
// over LocalBufSize writes rather than paying it per item.
//
// Template parameters:
//   LocalBufSize — items to buffer per type before auto-flushing; must be > 0
//   RingSize     — forwarded to TypedRingSet; must be a power of 2
//   Types...     — output types; must be trivially copyable and unique
template<std::size_t LocalBufSize, std::size_t RingSize, typename... Types>
class TypedLocalBuffer {
    static_assert(AllUnique<Types...>::value,
                  "TypedLocalBuffer: duplicate types in pack are not allowed");
    static_assert(LocalBufSize > 0, "TypedLocalBuffer: LocalBufSize must be > 0");

    std::tuple<std::array<Types, LocalBufSize>...> bufs_;
    std::array<std::size_t, sizeof...(Types)>       counts_{};

public:
    // Write one item of type T to the local buffer. Triggers a bulk flush to
    // ring_set when the buffer for T reaches LocalBufSize items.
    // Compile error if T is not in Types...
    template<typename T>
    void write(const T& val, TypedRingSet<RingSize, Types...>& ring_set) noexcept;

    // Flush partial buffers for all types to their rings. Call at end of a
    // work unit to ensure no items remain in the local buffer.
    void flush_all(TypedRingSet<RingSize, Types...>& ring_set) noexcept;

private:
    // Bulk-flush the buffer for type T to its ring, then reset the count.
    // No-op if the buffer for T is empty.
    template<typename T>
    void flush_type(TypedRingSet<RingSize, Types...>& ring_set) noexcept;
};

template<std::size_t LocalBufSize, std::size_t RingSize, typename... Types>
template<typename T>
void TypedLocalBuffer<LocalBufSize, RingSize, Types...>::write(
    const T& val, TypedRingSet<RingSize, Types...>& ring_set) noexcept {
    // TypeIndex<T, Types...>::value is a compile error if T is not in the pack
    constexpr std::size_t idx = TypeIndex<T, Types...>::value;
    std::get<idx>(bufs_)[counts_[idx]++] = val;
    if (counts_[idx] == LocalBufSize) {
        flush_type<T>(ring_set);
    }
}

template<std::size_t LocalBufSize, std::size_t RingSize, typename... Types>
template<typename T>
void TypedLocalBuffer<LocalBufSize, RingSize, Types...>::flush_type(
    TypedRingSet<RingSize, Types...>& ring_set) noexcept {
    constexpr std::size_t idx = TypeIndex<T, Types...>::value;
    auto& count = counts_[idx];
    if (count == 0) return;
    ring_set.template ring<T>().write_bulk(std::get<idx>(bufs_).data(), count);
    count = 0;
}

template<std::size_t LocalBufSize, std::size_t RingSize, typename... Types>
void TypedLocalBuffer<LocalBufSize, RingSize, Types...>::flush_all(
    TypedRingSet<RingSize, Types...>& ring_set) noexcept {
    // fold expression: flush partial buffer for each type in the pack
    (flush_type<Types>(ring_set), ...);
}
