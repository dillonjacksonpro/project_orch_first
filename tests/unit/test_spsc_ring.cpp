#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ring/spsc_ring.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <thread>

#ifndef NDEBUG
#include <sys/wait.h>
#include <unistd.h>
#endif

// RING_SIZE is injected by the Makefile (-DRING_SIZE=N).
// In the test build it is set to 8 to force many wrap-arounds.
using Ring = SpscRing<int, RING_SIZE>;

// ─────────────────────────────────────────────────────────────────────────────
// Sequential correctness
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SpscRing sequential correctness") {
    Ring ring;

    SUBCASE("write 1 item, read 1 item — value matches") {
        ring.acquire() = 42;
        ring.commit();

        CHECK(ring.available() == 1);
        CHECK(*ring.read_ptr() == 42);
        ring.consume(1);
        CHECK(ring.available() == 0);
    }

    SUBCASE("write N-1 items, read all back in order") {
        constexpr std::size_t count = RING_SIZE - 1;
        for (std::size_t i = 0; i < count; ++i) {
            ring.acquire() = static_cast<int>(i);
            ring.commit();
        }
        CHECK(ring.available() == count);
        for (std::size_t i = 0; i < count; ++i) {
            CHECK(*ring.read_ptr(i) == static_cast<int>(i));
        }
        ring.consume(count);
        CHECK(ring.available() == 0);
    }

    SUBCASE("write_bulk K items: available() == K before consume, 0 after") {
        constexpr std::size_t k = RING_SIZE / 2;
        int src[k];
        for (std::size_t i = 0; i < k; ++i) {
            src[i] = static_cast<int>(i * 10);
        }
        ring.write_bulk(src, k);
        CHECK(ring.available() == k);
        ring.consume(k);
        CHECK(ring.available() == 0);
    }

    SUBCASE("consume(k) reduces available by exactly k") {
        constexpr std::size_t k = RING_SIZE / 2;
        int src[k];
        for (std::size_t i = 0; i < k; ++i) {
            src[i] = static_cast<int>(i);
        }
        ring.write_bulk(src, k);
        const std::size_t pre = ring.available();
        constexpr std::size_t drain = k / 2;
        ring.consume(drain);
        CHECK(ring.available() == pre - drain);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// write_bulk atomicity
// Consumer must never observe an intermediate count between pre and pre+K.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SpscRing write_bulk atomicity — no intermediate observations") {
    // Use a separate ring so the test is self-contained
    SpscRing<int, RING_SIZE> ring;
    constexpr std::size_t k = RING_SIZE / 2;

    const std::size_t pre = ring.available();  // 0 before write

    std::atomic<bool> violation{false};
    std::atomic<bool> done{false};

    // Consumer: spin-reads available() and validates every observation
    std::thread consumer([&] {
        while (!done.load(std::memory_order_acquire)) {
            const std::size_t n = ring.available();
            if (n != pre && n != pre + k) {
                violation.store(true, std::memory_order_relaxed);
            }
        }
    });

    // Producer: write one bulk batch
    int src[k];
    for (std::size_t i = 0; i < k; ++i) {
        src[i] = static_cast<int>(i);
    }
    ring.write_bulk(src, k);
    done.store(true, std::memory_order_release);

    // 5-second wall-clock timeout — a hang here means deadlock
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    consumer.join();
    CHECK(std::chrono::steady_clock::now() < deadline);

    CHECK_FALSE(violation.load());
}

// ─────────────────────────────────────────────────────────────────────────────
// write_bulk count > N must trigger assert() (debug builds only)
// ─────────────────────────────────────────────────────────────────────────────

#ifndef NDEBUG
TEST_CASE("SpscRing write_bulk count > N asserts") {
    // Fork a child process to perform the bad call; assert() fires → SIGABRT.
    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // Child: deliberately pass count > N — should abort
        SpscRing<int, RING_SIZE> ring;
        int src[RING_SIZE + 1] = {};
        ring.write_bulk(src, RING_SIZE + 1);
        _exit(0);  // should never reach here
    }

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status));  // killed by SIGABRT from assert()
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Concurrent stress — 10,000 sequential integers
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SpscRing concurrent stress — 10000 items sum correct") {
    SpscRing<int, RING_SIZE> ring;
    constexpr int item_count = 10'000;
    constexpr long expected_sum = static_cast<long>(item_count - 1) * item_count / 2;

    std::atomic<bool> timed_out{false};

    std::thread producer([&] {
        for (int i = 0; i < item_count; ++i) {
            ring.acquire() = i;
            ring.commit();
        }
    });

    long sum = 0;
    std::thread consumer([&] {
        int received = 0;
        while (received < item_count) {
            const std::size_t avail = ring.available();
            if (avail == 0) {
                continue;
            }
            for (std::size_t j = 0; j < avail; ++j) {
                sum += *ring.read_ptr(j);
            }
            ring.consume(avail);
            received += static_cast<int>(avail);
        }
    });

    // 5-second wall-clock timeout — fail if threads hang
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    producer.join();
    consumer.join();
    CHECK(std::chrono::steady_clock::now() < deadline);

    CHECK(sum == expected_sum);
}

// ─────────────────────────────────────────────────────────────────────────────
// Backpressure — producer fills ring, consumer drains after delay
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SpscRing backpressure — no items lost") {
    SpscRing<int, RING_SIZE> ring;
    // Write twice the ring capacity to guarantee backpressure
    constexpr int item_count = RING_SIZE * 2;

    std::thread producer([&] {
        for (int i = 0; i < item_count; ++i) {
            ring.acquire() = i;
            ring.commit();
        }
    });

    // Consumer wakes after 50 ms, then drains
    std::thread consumer([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        int received = 0;
        while (received < item_count) {
            const std::size_t avail = ring.available();
            if (avail == 0) {
                continue;
            }
            ring.consume(avail);
            received += static_cast<int>(avail);
        }
    });

    // 2-second wall-clock timeout
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    producer.join();
    consumer.join();
    CHECK(std::chrono::steady_clock::now() < deadline);
}

// ─────────────────────────────────────────────────────────────────────────────
// SPSC_DEBUG: two writers must trigger abort
// ─────────────────────────────────────────────────────────────────────────────

#ifdef SPSC_DEBUG

TEST_CASE("SpscRing SPSC_DEBUG two writers abort") {
    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        // Child: first thread acquires, second thread also calls acquire() —
        // SPSC_DEBUG assert fires.
        SpscRing<int, RING_SIZE> ring;

        // Fill all but one slot so acquire() won't spin waiting for space
        // (we want to hit the thread-check, not the full-ring spin)
        std::atomic<bool> first_in{false};
        std::atomic<bool> second_done{false};

        std::thread t1([&] {
            // Acquire a slot and hold it while t2 tries to enter
            [[maybe_unused]] volatile int& slot = ring.acquire();
            first_in.store(true, std::memory_order_release);
            // Wait until t2 has had a chance to fire the assert
            while (!second_done.load(std::memory_order_acquire)) {}
        });

        std::thread t2([&] {
            while (!first_in.load(std::memory_order_acquire)) {}
            // This should trigger the SPSC_DEBUG assert
            [[maybe_unused]] volatile int& slot = ring.acquire();
            second_done.store(true, std::memory_order_release);
        });

        t1.join();
        t2.join();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status));
}

TEST_CASE("SpscRing SPSC_DEBUG two readers abort") {
    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0) {
        SpscRing<int, RING_SIZE> ring;
        // Put an item in so available() returns immediately
        ring.acquire() = 1;
        ring.commit();

        std::atomic<bool> first_in{false};
        std::atomic<bool> second_done{false};

        std::thread t1([&] {
            [[maybe_unused]] std::size_t avail = ring.available();
            first_in.store(true, std::memory_order_release);
            while (!second_done.load(std::memory_order_acquire)) {}
        });

        std::thread t2([&] {
            while (!first_in.load(std::memory_order_acquire)) {}
            // This should trigger the SPSC_DEBUG assert
            [[maybe_unused]] std::size_t avail = ring.available();
            second_done.store(true, std::memory_order_release);
        });

        t1.join();
        t2.join();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status));
}

#endif  // SPSC_DEBUG
