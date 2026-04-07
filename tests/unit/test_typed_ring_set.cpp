#include "doctest.h"

#include "ring/typed_ring_set.hpp"

// Two trivially-copyable output types for testing
struct TypeA { int x; };
struct TypeB { float y; };

using RingSet = TypedRingSet<RING_SIZE, TypeA, TypeB>;

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TypedRingSet ring<A> and ring<B> return distinct objects") {
    RingSet rs;
    CHECK(&rs.ring<TypeA>() != static_cast<void*>(&rs.ring<TypeB>()));
}

TEST_CASE("TypedRingSet write to ring A, ring B unaffected; all_empty transitions") {
    RingSet rs;
    CHECK(rs.all_empty());

    rs.ring<TypeA>().acquire() = TypeA{7};
    rs.ring<TypeA>().commit();

    CHECK_FALSE(rs.all_empty());
    CHECK(rs.ring<TypeA>().available() == 1);
    CHECK(rs.ring<TypeB>().available() == 0);

    rs.ring<TypeA>().consume(1);
    CHECK(rs.all_empty());
}

TEST_CASE("TypedRingSet mixed state — all_empty false when only one ring has data") {
    RingSet rs;

    // Write 3 items to ring A, nothing to ring B
    for (int i = 0; i < 3; ++i) {
        rs.ring<TypeA>().acquire() = TypeA{i};
        rs.ring<TypeA>().commit();
    }

    CHECK(rs.ring<TypeA>().available() == 3);
    CHECK(rs.ring<TypeB>().available() == 0);
    CHECK_FALSE(rs.all_empty());  // partial state — A has data, B does not
}

TEST_CASE("TypedRingSet all_empty true on construction") {
    RingSet rs;
    CHECK(rs.all_empty());
    CHECK(rs.total_available() == 0);
}

TEST_CASE("TypedRingSet all_empty true after all items consumed") {
    RingSet rs;

    rs.ring<TypeA>().acquire() = TypeA{1};
    rs.ring<TypeA>().commit();
    rs.ring<TypeB>().acquire() = TypeB{2.0f};
    rs.ring<TypeB>().commit();

    CHECK_FALSE(rs.all_empty());
    CHECK(rs.total_available() == 2);

    rs.ring<TypeA>().consume(1);
    rs.ring<TypeB>().consume(1);

    CHECK(rs.all_empty());
    CHECK(rs.total_available() == 0);
}

TEST_CASE("TypedRingSet single-type pack works") {
    TypedRingSet<RING_SIZE, TypeA> single;
    CHECK(single.all_empty());
    CHECK(single.total_available() == 0);

    single.ring<TypeA>().acquire() = TypeA{99};
    single.ring<TypeA>().commit();

    CHECK(single.total_available() == 1);
    CHECK(single.ring<TypeA>().available() == 1);
    CHECK_FALSE(single.all_empty());

    single.ring<TypeA>().consume(1);
    CHECK(single.all_empty());
}
