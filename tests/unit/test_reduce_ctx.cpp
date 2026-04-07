#include "doctest.h"

#include "context/output_ctx.hpp"
#include "context/reduce_ctx.hpp"

struct TypeA { int   x; };
struct TypeB { float y; };

// Span<T> must be trivially copyable for safe lambda capture by value
static_assert(std::is_trivially_copyable_v<Span<int>>,
              "Span<int> must be trivially copyable");
static_assert(std::is_trivially_copyable_v<Span<TypeA>>,
              "Span<TypeA> must be trivially copyable");

TEST_CASE("data<T>() returns span with correct pointer and size") {
    std::vector<TypeA> va = {{1}, {2}, {3}};
    std::vector<TypeB> vb = {{1.0f}, {2.0f}};

    ReduceCtx<TypeA, TypeB> ctx{va, vb};

    auto sa = ctx.data<TypeA>();
    CHECK(sa.data == va.data());
    CHECK(sa.size == 3);

    auto sb = ctx.data<TypeB>();
    CHECK(sb.data == vb.data());
    CHECK(sb.size == 2);
}

TEST_CASE("add_task — labels and fns both grow") {
    std::vector<TypeA> va;
    std::vector<TypeB> vb;

    ReduceCtx<TypeA, TypeB> ctx{va, vb};
    ctx.add_task("task_a", [] {});
    ctx.add_task("task_b", [] {});

    CHECK(ctx.labels().size() == 2);
    CHECK(ctx.fns().size()    == 2);
}

TEST_CASE("span lifetime — valid after ReduceCtx goes out of scope") {
    std::vector<TypeA> va = {{10}, {20}, {30}};
    std::vector<TypeB> vb;

    Span<const TypeA> captured_span{};

    {
        ReduceCtx<TypeA, TypeB> ctx{va, vb};
        // Capture span by value — Span is trivially copyable so this is safe
        captured_span = ctx.data<TypeA>();
    } // ctx destroyed here; va still alive

    // The span points into va (orchestrator-owned), not into ctx
    CHECK(captured_span.size == 3);
    CHECK(captured_span.data[0].x == 10);
    CHECK(captured_span.data[1].x == 20);
    CHECK(captured_span.data[2].x == 30);
}

TEST_CASE("labels from add_task match OutputCtx task_labels") {
    std::vector<TypeA> va;
    std::vector<TypeB> vb;

    ReduceCtx<TypeA, TypeB> ctx{va, vb};
    ctx.add_task("alpha", [] {});
    ctx.add_task("beta",  [] {});

    // OutputCtx takes a const-ref to the same labels vector
    OutputCtx out_ctx{0, ctx.labels()};
    CHECK(out_ctx.task_labels.size() == 2);
    CHECK(out_ctx.task_labels[0] == "alpha");
    CHECK(out_ctx.task_labels[1] == "beta");
}

// NOTE: Antipattern — do NOT write to a shared field from two tasks.
// Tasks registered via add_task run in OMP parallel; writes to the same
// memory location are a data race with no compile-time detection.
//
// CORRECT pattern: each task writes to a distinct field on the job object.
//   ctx.add_task("count_a", [span_a]{ job.count_a = compute(span_a); });
//   ctx.add_task("count_b", [span_b]{ job.count_b = compute(span_b); });
//
// WRONG pattern (data race, undefined behaviour):
//   ctx.add_task("t1", [&result]{ result += compute_part1(); });
//   ctx.add_task("t2", [&result]{ result += compute_part2(); });
