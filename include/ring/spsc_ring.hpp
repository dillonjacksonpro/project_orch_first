#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#ifdef SPSC_DEBUG
#include <thread>
#endif

// Lock-free single-producer single-consumer ring buffer.
// Zero heap allocation. Power-of-2 capacity only.
// T must be trivially copyable (memcpy-safe).
// write_idx_ and read_idx_ on separate cache lines to prevent false sharing.
template<typename T, std::size_t N>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing<T,N>: T must be trivially copyable");
    static_assert((N & (N - 1)) == 0,
                  "SpscRing<T,N>: N must be a power of 2");

    alignas(64) std::atomic<std::size_t> write_idx_{0};
    alignas(64) std::atomic<std::size_t> read_idx_{0};
    T slots_[N];

#ifdef SPSC_DEBUG
    // Thread ownership tracking: detect misuse from two writer or two reader threads.
    std::atomic<std::thread::id> writer_owner_{};
    std::atomic<std::thread::id> reader_owner_{};
#endif

public:
    SpscRing() = default;
    SpscRing(const SpscRing&)            = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&)                 = delete;
    SpscRing& operator=(SpscRing&&)      = delete;

    // ── Writer thread only ────────────────────────────────────────────────────

    // Spin-waits until space is available; returns a reference to the next
    // write slot. Caller must call commit() after filling the slot.
    [[nodiscard]] T& acquire() noexcept {
#ifdef SPSC_DEBUG
        check_writer_owner_();
#endif
        const std::size_t w = write_idx_.load(std::memory_order_relaxed);
        // acquire: synchronize with consumer's release in consume()
        while (w - read_idx_.load(std::memory_order_acquire) >= N) { /* spin */ }
        return slots_[w & (N - 1)];
    }

    // Publishes the slot filled after acquire(). Must be called exactly once
    // per acquire().
    void commit() noexcept {
#ifdef SPSC_DEBUG
        check_writer_owner_();
#endif
        // release: makes the written slot visible to the reader
        write_idx_.fetch_add(1, std::memory_order_release);
    }

    // Writes count items from src in a single atomic batch.
    // Spin-waits until count slots are free.
    // PRECONDITION: count <= N — if count > N the spin is unsatisfiable.
    void write_bulk(const T* src, std::size_t count) noexcept {
#ifdef SPSC_DEBUG
        check_writer_owner_();
#endif
        // count > N would deadlock — fast-fail in debug builds
        assert(count <= N && "write_bulk: count exceeds ring capacity");

        const std::size_t w = write_idx_.load(std::memory_order_relaxed);
        // acquire: synchronize with consumer's release in consume()
        while (N - (w - read_idx_.load(std::memory_order_acquire)) < count) { /* spin */ }

        const std::size_t start = w & (N - 1);
        const std::size_t tail  = N - start;  // slots remaining from start to end of array

        if (count <= tail) {
            std::memcpy(&slots_[start], src, count * sizeof(T));
        } else {
            // batch wraps around the end of the array — two copies
            std::memcpy(&slots_[start], src,        tail          * sizeof(T));
            std::memcpy(&slots_[0],     src + tail, (count - tail) * sizeof(T));
        }

        // Single release store publishes the entire batch atomically
        write_idx_.store(w + count, std::memory_order_release);
    }

    // ── Reader thread only ────────────────────────────────────────────────────

    // Returns the number of items currently readable.
    [[nodiscard]] std::size_t available() const noexcept {
#ifdef SPSC_DEBUG
        check_reader_owner_();
#endif
        // acquire: synchronize with producer's release in commit()/write_bulk()
        return write_idx_.load(std::memory_order_acquire)
             - read_idx_.load(std::memory_order_relaxed);
    }

    // Returns a pointer to the item at read_idx_ + offset.
    // Does NOT consume the item — call consume() when done.
    // NOTE: if offset causes a wrap, the returned pointer may not be the start
    // of a contiguous run. Caller is responsible for wrap-around logic.
    [[nodiscard]] const T* read_ptr(std::size_t offset = 0) const noexcept {
#ifdef SPSC_DEBUG
        check_reader_owner_();
#endif
        const std::size_t r = read_idx_.load(std::memory_order_relaxed);
        return &slots_[(r + offset) & (N - 1)];
    }

    // Advances the read index by count, making those slots available to writer.
    void consume(std::size_t count) noexcept {
#ifdef SPSC_DEBUG
        check_reader_owner_();
#endif
        // release: makes freed slots visible to the writer
        read_idx_.fetch_add(count, std::memory_order_release);
    }

private:
#ifdef SPSC_DEBUG
    void check_writer_owner_() noexcept {
        auto expected      = std::thread::id{};
        const auto current = std::this_thread::get_id();
        // compare_exchange sets expected to the stored value on failure
        if (!writer_owner_.compare_exchange_strong(
                expected, current,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            assert(expected == current
                   && "SpscRing: writer method called from two different threads");
        }
    }

    void check_reader_owner_() const noexcept {
        auto expected      = std::thread::id{};
        const auto current = std::this_thread::get_id();
        if (!const_cast<std::atomic<std::thread::id>&>(reader_owner_)
                .compare_exchange_strong(
                    expected, current,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
            assert(expected == current
                   && "SpscRing: reader method called from two different threads");
        }
    }
#endif
};
