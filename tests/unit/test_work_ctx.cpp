#include "doctest.h"

#include "context/work_ctx.hpp"
#include "ring/typed_ring_set.hpp"

struct IntVal {
    int v;
};

using Ring = TypedRingSet<RING_SIZE, IntVal>;
using Ctx  = WorkCtx<IntVal>;

TEST_CASE("emit at threshold — auto-flush triggers ring write") {
    Ring ring;
    {
        Ctx ctx{0, 1, ring};
        // LOCAL_BUF_SIZE writes triggers exactly one bulk flush
        for (std::size_t i = 0; i < LOCAL_BUF_SIZE; ++i) {
            ctx.emit(IntVal{static_cast<int>(i)});
        }
        CHECK(ring.ring<IntVal>().available() == LOCAL_BUF_SIZE);
    } // destructor flush — nothing left to flush
    CHECK(ring.ring<IntVal>().available() == LOCAL_BUF_SIZE);
}

TEST_CASE("destructor flush — partial buffer drained on scope exit") {
    Ring ring;
    {
        Ctx ctx{0, 1, ring};
        // Write half a buffer — no auto-flush should occur
        for (std::size_t i = 0; i < LOCAL_BUF_SIZE / 2; ++i) {
            ctx.emit(IntVal{static_cast<int>(i)});
        }
        CHECK(ring.ring<IntVal>().available() == 0); // still buffered
    } // destructor flushes the partial buffer
    CHECK(ring.ring<IntVal>().available() == LOCAL_BUF_SIZE / 2);
}

TEST_CASE("no items emitted — destructor is a no-op") {
    Ring ring;
    {
        Ctx ctx{0, 1, ring};
    }
    CHECK(ring.ring<IntVal>().available() == 0);
}

TEST_CASE("id and num_workers populated from constructor args") {
    Ring ring;
    Ctx  ctx{42, 7, ring};
    CHECK(ctx.id          == 42);
    CHECK(ctx.num_workers == 7);
}
