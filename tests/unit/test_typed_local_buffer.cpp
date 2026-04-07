#include "doctest.h"

#include "buffer/typed_local_buffer.hpp"
#include "ring/typed_ring_set.hpp"

struct TypeA {
    int x;
};
struct TypeB {
    float y;
};

// Convenience alias using Makefile-injected constants
using Buf  = TypedLocalBuffer<LOCAL_BUF_SIZE, RING_SIZE, TypeA, TypeB>;
using Ring = TypedRingSet<RING_SIZE, TypeA, TypeB>;

TEST_CASE("write below threshold — no flush") {
    Ring ring_set;
    Buf  buf;
    for (std::size_t i = 0; i < LOCAL_BUF_SIZE - 1; ++i) {
        buf.write(TypeA{static_cast<int>(i)}, ring_set);
    }
    CHECK(ring_set.ring<TypeA>().available() == 0);
}

TEST_CASE("write at threshold — auto-flush triggered") {
    Ring ring_set;
    Buf  buf;
    for (std::size_t i = 0; i < LOCAL_BUF_SIZE; ++i) {
        buf.write(TypeA{static_cast<int>(i)}, ring_set);
    }
    CHECK(ring_set.ring<TypeA>().available() == LOCAL_BUF_SIZE);
}

TEST_CASE("write double threshold — two bulk transfers") {
    Ring ring_set;
    Buf  buf;
    for (std::size_t i = 0; i < 2 * LOCAL_BUF_SIZE; ++i) {
        buf.write(TypeA{static_cast<int>(i)}, ring_set);
    }
    CHECK(ring_set.ring<TypeA>().available() == 2 * LOCAL_BUF_SIZE);
}

TEST_CASE("flush_all with partial buffer") {
    Ring ring_set;
    Buf  buf;
    buf.write(TypeA{1}, ring_set);
    buf.write(TypeA{2}, ring_set);
    buf.flush_all(ring_set);
    CHECK(ring_set.ring<TypeA>().available() == 2);
}

TEST_CASE("flush_all on empty buffer — ring unaffected") {
    Ring ring_set;
    Buf  buf;
    buf.flush_all(ring_set);
    CHECK(ring_set.ring<TypeA>().available() == 0);
    CHECK(ring_set.ring<TypeB>().available() == 0);
}

TEST_CASE("multi-type: independent flush per type") {
    Ring ring_set;
    Buf  buf;

    // Fill TypeA to threshold; leave TypeB one short
    for (std::size_t i = 0; i < LOCAL_BUF_SIZE; ++i) {
        buf.write(TypeA{static_cast<int>(i)}, ring_set);
    }
    for (std::size_t i = 0; i < LOCAL_BUF_SIZE - 1; ++i) {
        buf.write(TypeB{static_cast<float>(i)}, ring_set);
    }

    // TypeA flushed automatically; TypeB not yet
    CHECK(ring_set.ring<TypeA>().available() == LOCAL_BUF_SIZE);
    CHECK(ring_set.ring<TypeB>().available() == 0);

    buf.flush_all(ring_set);

    CHECK(ring_set.ring<TypeB>().available() == LOCAL_BUF_SIZE - 1);
}

TEST_CASE("values preserved byte-for-byte") {
    Ring ring_set;
    Buf  buf;

    for (std::size_t i = 0; i < LOCAL_BUF_SIZE; ++i) {
        buf.write(TypeA{static_cast<int>(i) * 10}, ring_set);
    }

    auto& ring = ring_set.ring<TypeA>();
    CHECK(ring.available() == LOCAL_BUF_SIZE);

    for (std::size_t i = 0; i < LOCAL_BUF_SIZE; ++i) {
        CHECK(ring.read_ptr(i)->x == static_cast<int>(i) * 10);
    }
    ring.consume(LOCAL_BUF_SIZE);
    CHECK(ring.available() == 0);
}
