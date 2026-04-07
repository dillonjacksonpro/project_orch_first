#pragma once
#include <type_traits>
#include <cstddef>

// Compile-time index of T in the pack Types...
// Produces a compile error if T is not found (instantiation of incomplete specialisation).
//
// Uses two specialisations rather than a ternary expression so that the recursive
// TypeIndex<T, Rest...> is never instantiated when T == First. A ternary would force
// both branches to be instantiated as class-template members, hitting the incomplete
// primary template even when T is at the front of the pack.
template<typename T, typename... Types>
struct TypeIndex;

// Found: T matches the head of the pack — index is 0.
template<typename T, typename... Rest>
struct TypeIndex<T, T, Rest...> {
    static constexpr std::size_t value = 0;
};

// Not yet found: skip First and increment the index.
template<typename T, typename First, typename... Rest>
struct TypeIndex<T, First, Rest...> {
    static constexpr std::size_t value = 1 + TypeIndex<T, Rest...>::value;
};

// True if all types in the pack are distinct.
// Used by JobInterface to reject duplicate OutputTypes at compile time.
template<typename... Types>
struct AllUnique;

template<>
struct AllUnique<> : std::true_type {};          // zero types: trivially unique

template<typename T>
struct AllUnique<T> : std::true_type {};         // one type: trivially unique

template<typename T, typename... Rest>
struct AllUnique<T, Rest...> {
    // fold expression: checks T != each Rest, then recurses
    static constexpr bool value =
        (!std::is_same_v<T, Rest> && ...) && AllUnique<Rest...>::value;
};
