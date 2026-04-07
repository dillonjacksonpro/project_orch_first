#pragma once

#include "buffer/typed_local_buffer.hpp"
#include "orchestrator_config.hpp"
#include "ring/typed_ring_set.hpp"

// Returns the global worker ID of thread 0 on the given rank.
//   Root (rank 0)   has cpus_per_task - 1 workers (one slot reserved for comm thread).
//   Non-root ranks  have cpus_per_task workers each.
//
//   first = 0                                            for rank == 0
//   first = (cpus_per_task - 1) + (rank - 1) * cpus_per_task  for rank > 0
[[nodiscard]] constexpr int first_global_worker_id(int rank, int cpus_per_task) noexcept {
    return (rank == 0) ? 0 : (cpus_per_task - 1) + (rank - 1) * cpus_per_task;
}

// Returns the global worker ID for thread tid on the given rank.
[[nodiscard]] constexpr int compute_global_id(int rank, int tid, int cpus_per_task) noexcept {
    return first_global_worker_id(rank, cpus_per_task) + tid;
}

// Passed to on_map for each worker thread. Wraps a TypedLocalBuffer that
// bulk-flushes to the worker's ring set. The destructor flushes any remaining
// items so the caller does not need to call flush_all explicitly.
//
// Copy and move are deleted — WorkCtx holds a reference to thread-local storage
// and must fire its destructor exactly once in the owning OMP thread.
template<typename... Types>
class WorkCtx {
public:
    const int id;          // global worker ID (unique across all ranks)
    const int num_workers; // total workers across all ranks

    WorkCtx(int worker_id, int num_workers_total,
            TypedRingSet<RING_SIZE, Types...>& ring_set)
        : id(worker_id), num_workers(num_workers_total), ring_set_(ring_set) {}

    // Flush remaining buffered items to the ring on destruction.
    ~WorkCtx() { buffer_.flush_all(ring_set_); }

    WorkCtx(const WorkCtx&)            = delete;
    WorkCtx& operator=(const WorkCtx&) = delete;
    WorkCtx(WorkCtx&&)                 = delete;
    WorkCtx& operator=(WorkCtx&&)      = delete;

    // Emit one item of type T. Triggers a bulk flush when the local buffer for T
    // reaches LOCAL_BUF_SIZE items. Compile error if T is not in Types...
    template<typename T>
    void emit(const T& val) { buffer_.write(val, ring_set_); }

private:
    TypedRingSet<RING_SIZE, Types...>&                ring_set_;
    TypedLocalBuffer<LOCAL_BUF_SIZE, RING_SIZE, Types...> buffer_;
};
