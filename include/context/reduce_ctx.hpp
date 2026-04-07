#pragma once

#include "type_utils.hpp"
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

// Trivially-copyable read-only range over a contiguous block of T.
// Defined here as a C++17 substitute for std::span<const T> (which is C++20).
// Trivial copyability is required so lambdas can safely capture a Span by value.
template<typename T>
struct Span {
    const T*    data;
    std::size_t size;
    const T* begin() const noexcept { return data; }
    const T* end()   const noexcept { return data + size; }
};

// Passed to on_reduce (rank 0 only). Provides read access to collected output
// data for each type, and a task registration interface for OMP-parallel reduction.
//
// data_vecs_ stores const-references to orchestrator-owned vectors that outlive
// this context object — Span values captured by lambda are valid after ReduceCtx
// goes out of scope.
template<typename... Types>
class ReduceCtx {
public:
    explicit ReduceCtx(const std::vector<Types>&... vecs) : data_vecs_(vecs...) {}

    // Returns a read-only span over the orchestrator-owned vector for type T.
    // Compile error if T is not in Types...
    template<typename T>
    [[nodiscard]] Span<const T> data() const noexcept {
        // TypeIndex<T, Types...>::value is a compile error if T is not in the pack
        constexpr std::size_t idx = TypeIndex<T, Types...>::value;
        const auto& vec = std::get<idx>(data_vecs_);
        return {vec.data(), vec.size()};
    }

    // Register an OMP task for the reduce phase. Each task should write to a
    // separate field on the job object to avoid data races — tasks run in parallel.
    void add_task(std::string label, std::function<void()> fn) {
        labels_.push_back(std::move(label));
        fns_.push_back(std::move(fn));
    }

    // Orchestrator-facing accessors
    [[nodiscard]] const std::vector<std::string>&              labels() const noexcept { return labels_; }
    [[nodiscard]] const std::vector<std::function<void()>>&    fns()    const noexcept { return fns_; }

private:
    // Const-references to orchestrator-owned vectors — do not outlive them.
    std::tuple<const std::vector<Types>&...> data_vecs_;

    std::vector<std::string>           labels_;
    std::vector<std::function<void()>> fns_;
};
