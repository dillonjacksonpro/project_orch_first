#include "doctest.h"

#include "job_interface.hpp"
#include "type_utils.hpp"

// -----------------------------------------------------------------------
// Dummy output types used across all tests
// -----------------------------------------------------------------------

struct TypeA { int   x; };
struct TypeB { float y; };
struct TypeC { char  z; };

// -----------------------------------------------------------------------
// Concrete job stubs — only exist to let JobOutputTypes be deduced
// -----------------------------------------------------------------------

class MyJob1 : public JobInterface<TypeA> {
public:
    void on_map(WorkCtx<TypeA>&) override {}
};

class MyJob2 : public JobInterface<TypeA, TypeB> {
public:
    void on_map(WorkCtx<TypeA, TypeB>&) override {}
};

class MyJob3 : public JobInterface<TypeA, TypeB, TypeC> {
public:
    void on_map(WorkCtx<TypeA, TypeB, TypeC>&) override {}
};

// -----------------------------------------------------------------------
// JobOutputTypes deduction (compile-time static_assert)
// -----------------------------------------------------------------------

static_assert(std::is_same_v<JobOutputTypes<MyJob1>, std::tuple<TypeA>>,
              "JobOutputTypes<MyJob1> should be tuple<TypeA>");

static_assert(std::is_same_v<JobOutputTypes<MyJob2>, std::tuple<TypeA, TypeB>>,
              "JobOutputTypes<MyJob2> should be tuple<TypeA, TypeB>");

static_assert(std::is_same_v<JobOutputTypes<MyJob3>, std::tuple<TypeA, TypeB, TypeC>>,
              "JobOutputTypes<MyJob3> should be tuple<TypeA, TypeB, TypeC>");

// -----------------------------------------------------------------------
// TypeIndex spot checks (compile-time static_assert)
// -----------------------------------------------------------------------

static_assert(TypeIndex<TypeA, TypeA, TypeB, TypeC>::value == 0, "TypeA is at index 0");
static_assert(TypeIndex<TypeB, TypeA, TypeB, TypeC>::value == 1, "TypeB is at index 1");
static_assert(TypeIndex<TypeC, TypeA, TypeB, TypeC>::value == 2, "TypeC is at index 2");

// -----------------------------------------------------------------------
// AllUnique edge cases (compile-time static_assert)
// -----------------------------------------------------------------------

static_assert(AllUnique<>::value,           "zero types must be unique");
static_assert(AllUnique<TypeA>::value,      "one type must be unique");
static_assert(AllUnique<TypeA, TypeB>::value, "two distinct types must be unique");
static_assert(!AllUnique<TypeA, TypeA>::value, "duplicate types must not be unique");

// -----------------------------------------------------------------------
// first_global_worker_id / compute_global_id — table-driven
// -----------------------------------------------------------------------

// | world_size | cpus_per_task | rank | tid | expected global_id | expected first |
// |------------|---------------|------|-----|--------------------|-|
// |          1 |             4 |    0 |   0 |                  0 | 0 |
// |          1 |             4 |    0 |   2 |                  2 | 0 |
// |          3 |             4 |    0 |   2 |                  2 | 0 |
// |          3 |             4 |    1 |   0 |                  3 | 3 |
// |          3 |             4 |    1 |   3 |                  6 | 3 |
// |          3 |             4 |    2 |   0 |                  7 | 7 |
// |          3 |             4 |    2 |   3 |                 10 | 7 |
// |          2 |             8 |    1 |   7 |                 14 | 7 |

TEST_CASE("first_global_worker_id and compute_global_id — table driven") {
    struct Row {
        int cpus_per_task;
        int rank;
        int tid;
        int expected_global_id;
        int expected_first;
    };

    constexpr Row table[] = {
        {4, 0, 0,  0,  0},
        {4, 0, 2,  2,  0},
        {4, 0, 2,  2,  0},
        {4, 1, 0,  3,  3},
        {4, 1, 3,  6,  3},
        {4, 2, 0,  7,  7},
        {4, 2, 3, 10,  7},
        {8, 1, 7, 14,  7},
    };

    for (const auto& row : table) {
        CHECK(first_global_worker_id(row.rank, row.cpus_per_task) == row.expected_first);
        CHECK(compute_global_id(row.rank, row.tid, row.cpus_per_task) == row.expected_global_id);
    }
}
