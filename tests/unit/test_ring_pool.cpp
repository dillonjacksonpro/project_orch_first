#include "doctest.h"

#include "ring/ring_pool.hpp"

struct TypeA { int x; };
struct TypeB { float y; };

using Pool = RingPool<RING_SIZE, TypeA, TypeB>;

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("RingPool get(i) returns distinct TypedRingSet objects") {
    constexpr std::size_t nworkers = 4;
    Pool pool{nworkers};

    // All four pointers must be distinct
    for (std::size_t i = 0; i < nworkers; ++i) {
        for (std::size_t j = i + 1; j < nworkers; ++j) {
            CHECK(&pool.get(i) != &pool.get(j));
        }
    }
}

TEST_CASE("RingPool isolation — write to get(0) does not affect get(1)") {
    Pool pool{2};

    pool.get(0).ring<TypeA>().acquire() = TypeA{42};
    pool.get(0).ring<TypeA>().commit();

    CHECK(pool.get(0).ring<TypeA>().available() == 1);
    CHECK(pool.get(1).ring<TypeA>().available() == 0);
}

TEST_CASE("RingPool all_empty — false if any ring has data, true only when all empty") {
    Pool pool{3};

    CHECK(pool.all_empty());

    // Write to one worker's ring
    pool.get(1).ring<TypeB>().acquire() = TypeB{3.14f};
    pool.get(1).ring<TypeB>().commit();

    CHECK_FALSE(pool.all_empty());

    // Drain it
    pool.get(1).ring<TypeB>().consume(1);
    CHECK(pool.all_empty());
}

TEST_CASE("RingPool num_workers returns value passed to constructor") {
    CHECK(Pool{1}.num_workers() == 1);
    CHECK(Pool{4}.num_workers() == 4);
    CHECK(Pool{16}.num_workers() == 16);
}
