#pragma once

// mpi.h must precede fatal.hpp: FATAL_MPI and MPI_CHECK use MPI symbols
// that fatal.hpp does not itself include. See handoff §Phase 0 item 2.
#include <mpi.h>
#include "fatal.hpp"
#include "orchestrator_config.hpp"
#include "ring/ring_pool.hpp"
#include "type_utils.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <thread>
#include <tuple>
#include <vector>

// Sole caller of all MPI functions on a rank. One instance per rank.
// Drains worker rings round-robin, batches items by type, and sends each
// batch to rank 0 via MPI_Isend. Root accumulates all received data.
//
// Shutdown is coordinated via a sentinel (zero-byte MPI_BYTE message with
// SENTINEL_TAG). MPI per-sender ordering ensures root receives all data
// from rank R before R's sentinel arrives, so root can safely drain
// remaining receives and enter MPI_Barrier only after all data is in.
//
// MPI threading level: MPI_THREAD_SERIALIZED. Only CommThread calls MPI,
// so no mutex is needed in production. Test binaries that call MPI from
// the test thread (e.g., in-test barriers) must init with MPI_THREAD_MULTIPLE.
template<std::size_t RingSize, typename... Types>
class CommThread {
    static_assert(sizeof...(Types) > 0,
                  "CommThread: at least one output type required");
    static_assert(AllUnique<Types...>::value,
                  "CommThread: duplicate types in pack are not allowed");

    // SENTINEL_TAG is one past the last data tag (0..sizeof...(Types)-1).
    // No collision with data tags is possible by construction.
    static constexpr int SENTINEL_TAG = static_cast<int>(sizeof...(Types));

    // ── Private nested types ─────────────────────────────────────────────────

    struct PendingSend {
        MPI_Request req;
        // -1 = sentinel; 0..sizeof...(Types)-1 = data type index
        int type_idx;
    };

    struct Backoff {
        long wait_ns{1};
        void reset() noexcept { wait_ns = 1; }
        // Busy-spin for wait_ns nanoseconds, then double (capped at MAX_BACKOFF_NS).
        void spin() noexcept {
            auto start = std::chrono::high_resolution_clock::now();
            while (
                (std::chrono::high_resolution_clock::now() - start).count()
                < wait_ns) {}
            wait_ns = std::min(wait_ns * 2L, MAX_BACKOFF_NS);
        }
    };

    // ── Private data members ─────────────────────────────────────────────────

    RingPool<RingSize, Types...>&     pool_;
    int                               rank_;
    int                               world_size_;
    int                               num_workers_;
    std::tuple<std::vector<Types>...> recv_bufs_;         // root only
    std::thread                       thread_;
    std::atomic<bool>                 flush_requested_{false};
    int                               sentinel_count_{0}; // root only

    // Shared 1-byte buffer for all sentinel MPI_Irecv calls.
    // Zero-byte sends never write to any buffer — sharing a single byte is safe.
    char sentinel_recv_buf_{0};

public:
    CommThread(RingPool<RingSize, Types...>& pool,
               int rank, int world_size, int num_workers)
        : pool_(pool), rank_(rank), world_size_(world_size),
          num_workers_(num_workers) {}

    CommThread(const CommThread&)            = delete;
    CommThread& operator=(const CommThread&) = delete;
    CommThread(CommThread&&)                 = delete;
    CommThread& operator=(CommThread&&)      = delete;

    // Spawns the background thread. Must be called exactly once before any
    // other method.
    void start() { thread_ = std::thread(&CommThread::run, this); }

    // Signals the comm thread to drain remaining data and shut down.
    // Called by the main thread after the OMP parallel region exits.
    void request_flush() noexcept {
        flush_requested_.store(true, std::memory_order_release);
    }

    // Blocks until the internal MPI_Barrier completes (all ranks flushed).
    void join() { thread_.join(); }

    // Root only: returns a reference to all received data after join().
    // The tuple is stable and never modified after join() returns.
    [[nodiscard]] std::tuple<std::vector<Types>...>& received() {
        assert(rank_ == 0 && "received() is only valid on rank 0");
        return recv_bufs_;
    }

private:

    // ── Compile-time type iteration ──────────────────────────────────────────

    // Invokes fn(integral_constant<std::size_t, I>{}) for each I in 0..N-1.
    // C++17 fold expression over an index sequence; zero runtime overhead.
    template<typename Fn, std::size_t... Is>
    static void for_each_type(Fn&& fn, std::index_sequence<Is...>) {
        // fold expression: calls fn once per index, left-to-right
        (fn(std::integral_constant<std::size_t, Is>{}), ...);
    }

    // ── Ring drain ───────────────────────────────────────────────────────────

    // Collects up to n items of type T from all worker rings (round-robin
    // across workers 0..num_workers_-1). Returns the actual count collected,
    // which may be less than n when total available across all rings is < n.
    template<typename T>
    std::size_t collect_from_rings(T* dst, std::size_t n) noexcept {
        std::size_t collected = 0;
        for (int w = 0; w < num_workers_ && collected < n; ++w) {
            auto& ring = pool_.get(static_cast<std::size_t>(w)).template ring<T>();
            const std::size_t avail = ring.available();
            const std::size_t take  = std::min(avail, n - collected);
            if (take == 0) continue;
            // read_ptr(k) handles modular wrap internally (no contiguous copy needed).
            // Copy individually, then consume as a unit to preserve SPSC ordering.
            for (std::size_t k = 0; k < take; ++k) {
                dst[collected++] = *ring.read_ptr(k);
            }
            ring.consume(take);
        }
        return collected;
    }

    // ── Thread body ──────────────────────────────────────────────────────────

    void run() {
        // One send staging buffer per type (BATCH_SIZE items each).
        // A tuple is used because types have different sizes — a 2D array is
        // not possible.
        std::tuple<std::array<Types, BATCH_SIZE>...> send_bufs;

        // True while the send buffer for type i is held by a live MPI_Isend.
        // The buffer must not be reused until the send completes.
        std::array<bool, sizeof...(Types)> send_in_flight{};

        std::vector<PendingSend> pending_sends;
        // +1 for the sentinel send on non-root ranks
        pending_sends.reserve(sizeof...(Types) + 1);

        Backoff backoff;

        // ── Root-only state ────────────────────────────────────────────────
        // data_recv_reqs[type_idx][slot] with matching per-type recv buffers.
        std::array<std::array<MPI_Request, PREPOST_RECVS>, sizeof...(Types)>
            data_recv_reqs{};
        // Tuple of per-type recv slot buffers: std::array<std::array<T, BATCH_SIZE>,
        // PREPOST_RECVS> per type. Achieves correct per-type sizing without a
        // fixed-size 2D byte array.
        std::tuple<std::array<std::array<Types, BATCH_SIZE>, PREPOST_RECVS>...>
            data_recv_bufs;
        std::vector<MPI_Request> sentinel_reqs; // world_size_ - 1 entries
        bool all_data_received = false;

        // Non-root-only state
        bool sentinel_sent = false;

        // ── Pre-post receives on root ──────────────────────────────────────
        if (rank_ == 0) {
            // For each type i, pre-post PREPOST_RECVS data receives so the
            // comm thread always has a buffer ready for incoming messages.
            for_each_type([&](auto idx_c) {
                constexpr std::size_t i = idx_c.value;
                // T is the i-th type in the pack
                using T = std::tuple_element_t<i, std::tuple<Types...>>;
                auto& bufs_i = std::get<i>(data_recv_bufs);
                for (int s = 0; s < PREPOST_RECVS; ++s) {
                    MPI_CHECK(MPI_Irecv(
                        bufs_i[s].data(),
                        static_cast<int>(BATCH_SIZE * sizeof(T)),
                        MPI_BYTE, MPI_ANY_SOURCE,
                        static_cast<int>(i),
                        MPI_COMM_WORLD, &data_recv_reqs[i][s]));
                }
            }, std::index_sequence_for<Types...>{});

            // Pre-post one sentinel receive per non-root rank.
            // Zero-byte sends match these; the shared 1-byte buffer is never written.
            sentinel_reqs.resize(world_size_ - 1);
            for (int r = 0; r < world_size_ - 1; ++r) {
                MPI_CHECK(MPI_Irecv(
                    &sentinel_recv_buf_, 1, MPI_BYTE,
                    MPI_ANY_SOURCE, SENTINEL_TAG,
                    MPI_COMM_WORLD, &sentinel_reqs[r]));
            }
            // world_size == 1: no sentinels expected; exit is gated only on
            // flush_requested + pool_empty + pending_sends_empty.
            if (world_size_ == 1) all_data_received = true;
        }

        // ── Main loop ──────────────────────────────────────────────────────
        for (;;) {
            bool did_work = false;

            // Drain rings → MPI_Isend to rank 0.
            // Runs on all ranks (root sends to itself via MPI loopback).
            for_each_type([&](auto idx_c) {
                constexpr std::size_t i = idx_c.value;
                using T = std::tuple_element_t<i, std::tuple<Types...>>;

                // Buffer still held by an in-flight send; skip until it completes.
                if (send_in_flight[i]) return;

                auto& buf = std::get<i>(send_bufs);
                const std::size_t n = collect_from_rings<T>(buf.data(), BATCH_SIZE);
                if (n == 0) return;

                MPI_Request req;
                MPI_CHECK(MPI_Isend(
                    buf.data(),
                    static_cast<int>(n * sizeof(T)),
                    MPI_BYTE, 0,
                    static_cast<int>(i),
                    MPI_COMM_WORLD, &req));
                pending_sends.push_back({req, static_cast<int>(i)});
                send_in_flight[i] = true;
                did_work = true;
            }, std::index_sequence_for<Types...>{});

            // Drive async progress on pending sends via MPI_Testsome.
            if (!pending_sends.empty()) {
                // MPI_Testsome requires a contiguous MPI_Request array.
                std::vector<MPI_Request> reqs;
                reqs.reserve(pending_sends.size());
                for (const auto& ps : pending_sends) reqs.push_back(ps.req);

                int outcount = 0;
                std::vector<int>        indices(pending_sends.size());
                std::vector<MPI_Status> statuses(pending_sends.size());
                MPI_CHECK(MPI_Testsome(
                    static_cast<int>(reqs.size()), reqs.data(),
                    &outcount, indices.data(), statuses.data()));

                if (outcount != MPI_UNDEFINED && outcount > 0) {
                    // Sort descending so we can erase by index without shifting
                    // earlier entries and invalidating remaining indices.
                    std::sort(indices.begin(), indices.begin() + outcount,
                              [](int a, int b) { return a > b; });
                    for (int k = 0; k < outcount; ++k) {
                        const int idx  = indices[k];
                        const int tidx =
                            pending_sends[static_cast<std::size_t>(idx)].type_idx;
                        if (tidx >= 0) {
                            // Data send completed: release buffer for reuse.
                            send_in_flight[static_cast<std::size_t>(tidx)] = false;
                        }
                        pending_sends.erase(pending_sends.begin() + idx);
                    }
                }
            }

            // Root: drain completed data receives, then poll sentinel receives.
            if (rank_ == 0) {
                for_each_type([&](auto idx_c) {
                    constexpr std::size_t i = idx_c.value;
                    using T = std::tuple_element_t<i, std::tuple<Types...>>;

                    auto& reqs_i = data_recv_reqs[i];
                    auto& bufs_i = std::get<i>(data_recv_bufs);
                    auto& dst    = std::get<i>(recv_bufs_);

                    int outcount = 0;
                    // Stack-allocated: PREPOST_RECVS is a small compile-time constant
                    std::array<int,        PREPOST_RECVS> completed_slots;
                    std::array<MPI_Status, PREPOST_RECVS> statuses;
                    MPI_CHECK(MPI_Testsome(
                        PREPOST_RECVS, reqs_i.data(),
                        &outcount, completed_slots.data(), statuses.data()));

                    if (outcount == MPI_UNDEFINED || outcount == 0) return;

                    for (int k = 0; k < outcount; ++k) {
                        const int s = completed_slots[k];
                        int n_bytes = 0;
                        // statuses[k] corresponds to reqs_i[completed_slots[k]]
                        MPI_CHECK(MPI_Get_count(&statuses[k], MPI_BYTE, &n_bytes));
                        if (n_bytes > 0) {
                            const std::size_t n_items =
                                static_cast<std::size_t>(n_bytes) / sizeof(T);
                            dst.insert(dst.end(),
                                       bufs_i[s].data(),
                                       bufs_i[s].data() + n_items);
                            did_work = true;
                        }
                        // Re-post this slot to maintain PREPOST_RECVS active receives.
                        MPI_CHECK(MPI_Irecv(
                            bufs_i[s].data(),
                            static_cast<int>(BATCH_SIZE * sizeof(T)),
                            MPI_BYTE, MPI_ANY_SOURCE,
                            static_cast<int>(i),
                            MPI_COMM_WORLD, &reqs_i[s]));
                    }
                }, std::index_sequence_for<Types...>{});

                // Poll sentinel receives (only until all world_size-1 arrive).
                if (!sentinel_reqs.empty() && !all_data_received) {
                    int outcount = 0;
                    std::vector<int>        sentinel_indices(sentinel_reqs.size());
                    std::vector<MPI_Status> sentinel_statuses(sentinel_reqs.size());
                    MPI_CHECK(MPI_Testsome(
                        static_cast<int>(sentinel_reqs.size()),
                        sentinel_reqs.data(),
                        &outcount,
                        sentinel_indices.data(),
                        sentinel_statuses.data()));
                    if (outcount != MPI_UNDEFINED && outcount > 0) {
                        sentinel_count_ += outcount;
                        if (sentinel_count_ == world_size_ - 1) {
                            all_data_received = true;
                        }
                    }
                }
            }

            // Backoff: reset on any progress, else exponential sleep.
            if (did_work) backoff.reset();
            else          backoff.spin();

            // ── Non-root shutdown ─────────────────────────────────────────
            if (rank_ != 0) {
                // Once all local rings are drained and all pending sends have
                // completed, send a zero-byte sentinel to root.
                if (!sentinel_sent
                    && flush_requested_.load(std::memory_order_acquire)
                    && pool_.all_empty()
                    && pending_sends.empty()) {
                    MPI_Request req;
                    MPI_CHECK(MPI_Isend(
                        nullptr, 0, MPI_BYTE, 0,
                        SENTINEL_TAG, MPI_COMM_WORLD, &req));
                    pending_sends.push_back({req, -1});
                    sentinel_sent = true;
                }
                // Exit only after sentinel send is also confirmed complete.
                if (sentinel_sent && pending_sends.empty()) {
                    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                    return;
                }
                continue; // skip root shutdown check below
            }

            // ── Root shutdown ─────────────────────────────────────────────
            // Exit when: flush requested, all local rings drained, all sentinels
            // received (=> all non-root data is in-flight or already received),
            // and all pending sends are complete.
            if (flush_requested_.load(std::memory_order_acquire)
                && pool_.all_empty()
                && all_data_received
                && pending_sends.empty()) {

                // Drain remaining in-flight data before cancelling.
                //
                // When sentinel_count_ == world_size_-1 all non-root sends have
                // been submitted (sentinels arrive after all data per MPI ordering),
                // but some sends may still be in the MPI transport layer — not yet
                // matched to a pre-posted recv buffer. Cancelling immediately would
                // race with those arriving messages.
                //
                // Drain via repeated MPI_Testsome until a full pass with no
                // completions (quiet). After that, no more sends are outstanding
                // and any remaining active recvs are truly unmatched; they can be
                // safely cancelled without losing data.
                {
                    bool had_completions = true;
                    while (had_completions) {
                        had_completions = false;
                        for_each_type([&](auto idx_c) {
                            constexpr std::size_t i = idx_c.value;
                            using T = std::tuple_element_t<i, std::tuple<Types...>>;

                            auto& reqs_i = data_recv_reqs[i];
                            auto& bufs_i = std::get<i>(data_recv_bufs);
                            auto& dst    = std::get<i>(recv_bufs_);

                            int outcount = 0;
                            std::array<int,        PREPOST_RECVS> completed_slots;
                            std::array<MPI_Status, PREPOST_RECVS> statuses;
                            MPI_CHECK(MPI_Testsome(
                                PREPOST_RECVS, reqs_i.data(),
                                &outcount, completed_slots.data(), statuses.data()));

                            if (outcount == MPI_UNDEFINED || outcount == 0) return;

                            had_completions = true;
                            for (int k = 0; k < outcount; ++k) {
                                const int s = completed_slots[k];
                                int n_bytes = 0;
                                MPI_CHECK(MPI_Get_count(&statuses[k], MPI_BYTE, &n_bytes));
                                if (n_bytes > 0) {
                                    const std::size_t n_items =
                                        static_cast<std::size_t>(n_bytes) / sizeof(T);
                                    dst.insert(dst.end(),
                                               bufs_i[s].data(),
                                               bufs_i[s].data() + n_items);
                                }
                                // Re-post to keep matching any late-arriving messages.
                                MPI_CHECK(MPI_Irecv(
                                    bufs_i[s].data(),
                                    static_cast<int>(BATCH_SIZE * sizeof(T)),
                                    MPI_BYTE, MPI_ANY_SOURCE,
                                    static_cast<int>(i),
                                    MPI_COMM_WORLD, &reqs_i[s]));
                            }
                        }, std::index_sequence_for<Types...>{});
                    }
                }

                // All in-flight data is now in recv_bufs_. Cancel the remaining
                // pre-posted recvs (those that never matched a send). MPI_Cancel
                // is safe: if a send somehow arrives after the drain loop (should
                // not happen given the exit condition), the cancel will fail and
                // MPI_Wait will deliver the data.
                for_each_type([&](auto idx_c) {
                    constexpr std::size_t i = idx_c.value;
                    using T = std::tuple_element_t<i, std::tuple<Types...>>;

                    auto& reqs_i = data_recv_reqs[i];
                    auto& bufs_i = std::get<i>(data_recv_bufs);
                    auto& dst    = std::get<i>(recv_bufs_);

                    // Request cancellation of all remaining pre-posted recvs.
                    for (int s = 0; s < PREPOST_RECVS; ++s) {
                        MPI_CHECK(MPI_Cancel(&reqs_i[s]));
                    }

                    // Wait for all to complete: each either delivers late data or
                    // completes as cancelled.
                    std::array<MPI_Status, PREPOST_RECVS> statuses;
                    MPI_CHECK(MPI_Waitall(
                        PREPOST_RECVS, reqs_i.data(), statuses.data()));

                    for (int s = 0; s < PREPOST_RECVS; ++s) {
                        int was_cancelled = 0;
                        // MPI_Test_cancelled does not return an error code per spec
                        MPI_Test_cancelled(&statuses[s], &was_cancelled);
                        if (!was_cancelled) {
                            // Late arrival (cancel lost the race): collect data.
                            int n_bytes = 0;
                            MPI_CHECK(MPI_Get_count(&statuses[s], MPI_BYTE, &n_bytes));
                            if (n_bytes > 0) {
                                const std::size_t n_items =
                                    static_cast<std::size_t>(n_bytes) / sizeof(T);
                                dst.insert(dst.end(),
                                           bufs_i[s].data(),
                                           bufs_i[s].data() + n_items);
                            }
                        }
                        // Cancelled: request freed by MPI_Waitall; no data.
                    }
                }, std::index_sequence_for<Types...>{});

                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                return;
            }
        }
    }
};
