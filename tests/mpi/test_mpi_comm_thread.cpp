// CommThread MPI integration tests.
// Compiled with TEST_RING_SIZE=8, TEST_BATCH_SIZE=4 (injected by Makefile).
//
// Tests write directly to RingPool rings, bypassing TypedLocalBuffer.
// This is intentional — CommThread is tested in isolation of the buffer layer.
// See plan/phase-4.md stack coverage note.
//
// Uses DOCTEST_CONFIG_IMPLEMENT (not _WITH_MAIN) so we can provide a custom
// main() that wraps MPI_Init_thread / MPI_Finalize.
// Only one translation unit in the test binary may define either form.
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

// mpi.h before fatal.hpp (which comm_thread.hpp transitively includes).
#include <mpi.h>
#include "comm/comm_thread.hpp"
#include "ring/ring_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// ── Test output types ──────────────────────────────────────────────────────

// Both must be trivially copyable (SpscRing<T,N> static_assert).
struct TypeA { int   val; };
struct TypeB { float val; };

static_assert(std::is_trivially_copyable_v<TypeA>, "TypeA must be trivially copyable");
static_assert(std::is_trivially_copyable_v<TypeB>, "TypeB must be trivially copyable");

// Pool and comm thread parameterised with test-scale constants.
// RING_SIZE=8, BATCH_SIZE=4 are injected by Makefile via TEST_CXXFLAGS.
using TestPool = RingPool<RING_SIZE, TypeA, TypeB>;
using TestComm = CommThread<RING_SIZE, TypeA, TypeB>;

// ── Global MPI state ───────────────────────────────────────────────────────

static int  g_rank               = 0;
static int  g_world_size         = 1;
// True when MPI_Init_thread provided MPI_THREAD_MULTIPLE.
// The "root does not exit early" test calls MPI_Barrier from the test thread
// concurrently with CommThread's MPI calls; this requires THREAD_MULTIPLE.
static bool g_thread_multiple    = false;

// ── Helpers ────────────────────────────────────────────────────────────────

// Blocks on comm.join() from a helper thread, with a wall-clock timeout.
// Calls FAIL and detaches the helper thread if the timeout is exceeded,
// preventing an infinite hang in the test process.
template<std::size_t RS, typename... Ts>
static void join_with_timeout(CommThread<RS, Ts...>& comm, int timeout_ms = 2000) {
    std::atomic<bool> done{false};
    std::thread t([&] {
        comm.join();
        done.store(true, std::memory_order_release);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!done.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() > deadline) {
            t.detach();
            FAIL("comm.join() did not complete within " << timeout_ms << " ms");
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    t.join();
}

// Writes n items to ring in chunks of at most RING_SIZE.
// write_bulk asserts count <= N; chunking ensures we never exceed that limit
// regardless of n, while the comm thread drains between chunks.
template<typename T>
static void write_chunked(SpscRing<T, RING_SIZE>& ring, const T* items, int n) {
    for (int base = 0; base < n; ) {
        const int chunk = std::min(static_cast<int>(RING_SIZE), n - base);
        ring.write_bulk(items + base, static_cast<std::size_t>(chunk));
        base += chunk;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

TEST_CASE("empty map phase") {
    TestPool pool(1);
    TestComm comm(pool, g_rank, g_world_size, 1);
    comm.start();
    comm.request_flush();
    join_with_timeout(comm, 500);
    CHECK(pool.all_empty());
}

TEST_CASE("single type single worker to root") {
    // Works with any rank count: root writes to its own ring and receives via
    // MPI loopback (rank 0 → rank 0). Non-root ranks write nothing.
    const int num_workers = 1;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 0) {
        TypeA items[20];
        for (int i = 0; i < 20; ++i) items[i] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), items, 20);
    }

    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        REQUIRE(ra.size() == 20);
        // Order of delivery is not guaranteed (MPI_ANY_SOURCE receives); sort
        // before comparing values.
        std::vector<int> vals;
        vals.reserve(ra.size());
        for (const auto& a : ra) vals.push_back(a.val);
        std::sort(vals.begin(), vals.end());
        for (int i = 0; i < 20; ++i) CHECK(vals[i] == i);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("multi-type multi-worker to root") {
    // Annotated (run with -n 2): skipped when world_size < 2.
    if (g_world_size < 2) return;

    const int num_workers = 4;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    // Only rank 1's workers write: 4 workers × 3 TypeA + 5 TypeB each.
    if (g_rank == 1) {
        for (int w = 0; w < num_workers; ++w) {
            TypeA ta[3]; for (int i = 0; i < 3; ++i) ta[i] = {w * 10 + i};
            TypeB tb[5]; for (int i = 0; i < 5; ++i)
                tb[i] = {static_cast<float>(w * 10 + i)};
            write_chunked(pool.get(w).ring<TypeA>(), ta, 3);
            write_chunked(pool.get(w).ring<TypeB>(), tb, 5);
        }
    }

    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        // 4 workers × 3 TypeA = 12; 4 workers × 5 TypeB = 20
        CHECK(ra.size() == 12);
        CHECK(rb.size() == 20);
        // Verify all expected TypeA values are present (sorted comparison)
        std::vector<int> avals;
        avals.reserve(ra.size());
        for (const auto& a : ra) avals.push_back(a.val);
        std::sort(avals.begin(), avals.end());
        const std::vector<int> expected = {0,1,2,10,11,12,20,21,22,30,31,32};
        CHECK(avals == expected);
    }
    CHECK(pool.all_empty());
}

TEST_CASE("round-robin drain fairness") {
    // Worker 0 writes more than others; verify no worker ring is starved.
    const int num_workers = 4;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 0) {
        // Worker 0 writes 8 items first (fills ring), then workers 1-3 write
        // 4 each, then worker 0 writes 2 more (total 10 for worker 0).
        TypeA a[8]; for (int i = 0; i < 8; ++i) a[i] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), a, 8);

        for (int w = 1; w < num_workers; ++w) {
            TypeA aw[4]; for (int i = 0; i < 4; ++i) aw[i] = {w * 100 + i};
            write_chunked(pool.get(w).ring<TypeA>(), aw, 4);
        }

        TypeA a2[2] = {{8}, {9}};
        write_chunked(pool.get(0).ring<TypeA>(), a2, 2);
    }

    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        // Worker 0: 10, workers 1-3: 4 each = 22 total
        CHECK(ra.size() == 22);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("send buffer reuse under load") {
    // 3 × BATCH_SIZE items forces the same send buffer to be reused 3 times.
    // BATCH_SIZE=4 in tests → 12 items total.
    const int num_workers = 1;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 0) {
        const int total = 3 * static_cast<int>(BATCH_SIZE);
        std::vector<TypeA> items(static_cast<std::size_t>(total));
        for (int i = 0; i < total; ++i) items[static_cast<std::size_t>(i)] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), items.data(), total);
    }

    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        CHECK(ra.size() == 3 * BATCH_SIZE);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("backoff convergence") {
    // No items written; comm thread enters exponential backoff.
    // Verifies backoff does not prevent clean shutdown within 500 ms.
    TestPool pool(1);
    TestComm comm(pool, g_rank, g_world_size, 1);
    comm.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    comm.request_flush();
    join_with_timeout(comm, 500);
    CHECK(pool.all_empty());
}

TEST_CASE("large volume") {
    // Annotated (run with -n 2): skipped when world_size < 2.
    // Stress test: 4 workers × 1000 items each = 4000 TypeA, far exceeding
    // ring capacity (8) and batch size (4).
    if (g_world_size < 2) return;

    const int num_workers      = 4;
    const int items_per_worker = 1000;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 1) {
        for (int w = 0; w < num_workers; ++w) {
            std::vector<TypeA> items(static_cast<std::size_t>(items_per_worker));
            for (int i = 0; i < items_per_worker; ++i)
                items[static_cast<std::size_t>(i)] = {w * items_per_worker + i};
            write_chunked(pool.get(w).ring<TypeA>(),
                          items.data(), items_per_worker);
        }
    }

    comm.request_flush();
    join_with_timeout(comm, 10000);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        CHECK(ra.size() ==
              static_cast<std::size_t>(num_workers * items_per_worker));
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("request flush while send in flight") {
    // Writes exactly BATCH_SIZE items, then calls request_flush() immediately
    // (before the comm thread may have processed the Isend).
    const int num_workers = 1;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 0) {
        // write_bulk (not chunked): BATCH_SIZE=4 <= RING_SIZE=8, so single call is safe
        TypeA items[BATCH_SIZE];
        for (std::size_t i = 0; i < BATCH_SIZE; ++i) items[i] = {static_cast<int>(i)};
        pool.get(0).ring<TypeA>().write_bulk(items, BATCH_SIZE);
    }

    // Immediate flush — races with the comm thread's first collect_from_rings
    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        CHECK(ra.size() == BATCH_SIZE);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("sentinel ordering guarantee") {
    // Annotated (run with -n 2): skipped when world_size < 2.
    // MPI per-sender ordering guarantees all data from rank R arrives before
    // R's sentinel. Root must have all 10 items after join().
    if (g_world_size < 2) return;

    const int num_workers = 1;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    if (g_rank == 1) {
        // Write 10 items in two chunks (RING_SIZE=8, so split 8+2)
        TypeA a[8]; for (int i = 0; i < 8; ++i) a[i] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), a, 8);
        TypeA b[2] = {{8}, {9}};
        write_chunked(pool.get(0).ring<TypeA>(), b, 2);
    }

    comm.request_flush();
    join_with_timeout(comm);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        CHECK(ra.size() == 10);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

TEST_CASE("root does not exit early") {
    // Annotated (run with -n 3): skipped when world_size < 3.
    // Also requires MPI_THREAD_MULTIPLE: the test thread calls MPI_Barrier
    // while the comm thread may be calling MPI_Isend / MPI_Testsome.
    if (g_world_size < 3 || !g_thread_multiple) return;

    const int num_workers = 1;
    TestPool  pool(num_workers);
    TestComm  comm(pool, g_rank, g_world_size, num_workers);
    comm.start();

    // Rank 1 writes its data BEFORE the barrier.
    if (g_rank == 1) {
        TypeA items[8]; for (int i = 0; i < 8; ++i) items[i] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), items, 8);
    }

    // All 3 ranks synchronize here. Rank 2 cannot proceed past this barrier
    // until rank 1 has already written all its data above.
    // REQUIRES MPI_THREAD_MULTIPLE because CommThread may be calling MPI
    // concurrently on the same process.
    REQUIRE(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);

    // Rank 2 writes its data AFTER the barrier, overlapping with root
    // potentially draining rank 1's data. Root must not exit early.
    if (g_rank == 2) {
        TypeA items[8]; for (int i = 100; i < 108; ++i) items[i - 100] = {i};
        write_chunked(pool.get(0).ring<TypeA>(), items, 8);
    }

    comm.request_flush();
    join_with_timeout(comm, 5000);

    if (g_rank == 0) {
        const auto& [ra, rb] = comm.received();
        // Rank 1 wrote 8 TypeA, rank 2 wrote 8 TypeA → 16 total at root.
        CHECK(ra.size() == 16);
        CHECK(rb.empty());
    }
    CHECK(pool.all_empty());
}

// ── MPI-aware entry point ──────────────────────────────────────────────────

int main(int argc, char** argv) {
    int provided = 0;
    // Request MPI_THREAD_MULTIPLE so the test thread can call MPI_Barrier in
    // "root does not exit early" while CommThread's MPI calls run concurrently.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world_size);
    g_thread_multiple = (provided >= MPI_THREAD_MULTIPLE);

    if (!g_thread_multiple && g_rank == 0) {
        std::fprintf(stderr,
            "[rank 0] WARNING: MPI_THREAD_MULTIPLE not available (provided=%d). "
            "\"root does not exit early\" test will be skipped.\n", provided);
    }

    doctest::Context ctx;
    ctx.applyCommandLine(argc, argv);
    const int result = ctx.run();

    MPI_Finalize();
    return result;
}
