// Orchestrator MPI integration tests.
//
// No doctest framework is used here. orch.run() calls MPI_Init_thread and
// MPI_Finalize internally, which means no doctest TEST_CASE can wrap it (calling
// MPI_Init twice is undefined behaviour). Instead, this binary uses a standalone
// main() with assertions. The binary exits 0 on success, non-zero on failure.
//
// Run with: mpirun -n 2 build/tests/test_mpi
//           mpirun -n 3 build/tests/test_mpi
// Both invocations must exit 0.
//
// SLURM env validation paths (not automated — FATAL_MPI calls MPI_Abort which
// terminates the test runner; subprocess isolation would be needed):
//   SLURM_CPUS_PER_TASK <= 0 || > 65536:
//     "[rank N] [FATAL] SLURM_CPUS_PER_TASK out of range: <value>"
//
// The rank-0 >= 2 CPUs requirement is enforced in run.sh before srun is invoked.
// node_count and node_id are derived from MPI (world_size and rank) under the
// --ntasks-per-node=1 invariant — no SLURM_NNODES or SLURM_NODEID are read.
//
// MPI threading level: the fatal-exit path for provided < MPI_THREAD_SERIALIZED
// is untestable without a crippled MPI installation; documented for code review.

#include <mpi.h>

#include "orchestrator.hpp"

#include <atomic>
#include <cassert>
#include <cstdio>

// ---------------------------------------------------------------------------
// Output type for the mock job
// ---------------------------------------------------------------------------

struct TestRecord {
    int val; // set to the emitting worker's global ID
};
static_assert(std::is_trivially_copyable_v<TestRecord>, "MPI requires trivially copyable");

// ---------------------------------------------------------------------------
// Mock job — records hook invocations and data flow for post-run assertions
// ---------------------------------------------------------------------------

struct MockJob : public JobInterface<TestRecord> {
    // Incremented by multiple OMP threads — must be atomic.
    std::atomic<int> cmdline_calls{0};
    std::atomic<int> dist_plan_calls{0};
    std::atomic<int> node_plan_calls{0};
    std::atomic<int> map_calls{0};

    // Single-threaded (called only from root's main thread after map phase).
    int reduce_calls{0};
    int output_calls{0};

    // Captured in on_dist_plan so post-run code knows which rank this process was.
    int captured_rank{-1};
    int captured_num_workers{0};
    int captured_total_workers{0};

    // Set in on_reduce (root only).
    std::size_t total_received{0};

    void on_cmdline(CmdlineCtx&) override {
        cmdline_calls.fetch_add(1, std::memory_order_relaxed);
    }

    void on_dist_plan(DistPlanCtx& ctx) override {
        dist_plan_calls.fetch_add(1, std::memory_order_relaxed);
        captured_rank         = ctx.rank;
        captured_num_workers  = ctx.num_workers;
        captured_total_workers = ctx.total_workers;
    }

    void on_node_plan(NodePlanCtx&) override {
        node_plan_calls.fetch_add(1, std::memory_order_relaxed);
    }

    void on_map(WorkCtx<TestRecord>& ctx) override {
        map_calls.fetch_add(1, std::memory_order_relaxed);
        // Emit one record per worker thread; val carries the global worker ID.
        ctx.emit(TestRecord{ctx.id});
    }

    void on_reduce(ReduceCtx<TestRecord>& ctx) override {
        reduce_calls++;
        total_received = ctx.data<TestRecord>().size;
        // Register a no-op reduce task to exercise the OMP task dispatch path.
        ctx.add_task("count", []{ /* nothing to do in this mock */ });
    }

    void on_output(OutputCtx&) override {
        output_calls++;
    }
};

// ---------------------------------------------------------------------------
// SLURM env fallback verification (single-rank, no SLURM vars set)
// ---------------------------------------------------------------------------
// This is not a separate test case because it would require another MPI_Init.
// The fallback logic is exercised implicitly when mpirun runs outside SLURM
// (SLURM_CPUS_PER_TASK is absent). Verified by inspection that slurm_read_env
// falls back to hardware_concurrency() for cpus_per_task. node_count and
// node_id are derived from MPI world_size and rank, not env vars.

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MockJob job;
    Orchestrator<MockJob> orch(job, argc, argv);
    orch.run(); // MPI_Init_thread → all phases → MPI_Finalize

    // ── Hook call counts — every rank ──────────────────────────────────────
    // Each hook (except on_map) is called exactly once on every rank.
    assert(job.cmdline_calls   == 1);
    assert(job.dist_plan_calls == 1);
    assert(job.node_plan_calls == 1);
    // on_map is called once per worker thread on this rank.
    assert(job.map_calls >= 1);
    assert(job.map_calls == job.captured_num_workers);

    // ── Rank-conditional assertions ────────────────────────────────────────
    if (job.captured_rank == 0) {
        // on_reduce and on_output called exactly once on root.
        assert(job.reduce_calls == 1);
        assert(job.output_calls == 1);

        // Data flow: every worker across all ranks emitted exactly 1 record.
        // total_received should equal total_workers (one record per global worker).
        // We cannot call MPI_Allreduce here (MPI is finalized), so verify against
        // captured_total_workers which was set during on_dist_plan.
        assert(job.total_received > 0);
        assert(job.total_received ==
               static_cast<std::size_t>(job.captured_total_workers));
    } else {
        // on_reduce and on_output must NOT be called on non-root ranks.
        assert(job.reduce_calls == 0);
        assert(job.output_calls == 0);
    }

    // If we reach here, all assertions passed.
    return 0;
}
