#pragma once

#include <mpi.h>

#include "comm/comm_thread.hpp"
#include "context/dist_plan_ctx.hpp"
#include "context/node_plan_ctx.hpp"
#include "context/output_ctx.hpp"
#include "context/reduce_ctx.hpp"
#include "context/work_ctx.hpp"
#include "fatal.hpp"
#include "job_interface.hpp"
#include "orchestrator_config.hpp"
#include "ring/ring_pool.hpp"

#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// Non-template helper: reads SLURM_CPUS_PER_TASK, validates the value, populates cpus_per_task.
// node_count and node_id are derived from MPI (invariant: --ntasks-per-node=1).
// Calls FATAL_MPI(rank, ...) on invalid values.
void slurm_read_env(int rank, int& cpus_per_task);

// Primary template — never instantiated directly; serves as the declaration hook
// that the partial specialization below specializes.
template<typename Job, typename Tuple = JobOutputTypes<Job>>
class OrchestratorImpl;

// Partial specialization: unpacks std::tuple<Types...> from JobOutputTypes<Job>
// so all downstream types (Pool, Comm, contexts) are parameterized on the same pack.
template<typename Job, typename... Types>
class OrchestratorImpl<Job, std::tuple<Types...>> {
    using Pool = RingPool<RING_SIZE, Types...>;
    using Comm = CommThread<RING_SIZE, Types...>;

    Job& job_;
    int rank_{0};
    int world_size_{1};
    int cpus_per_task_{1};
    int num_workers_{1};
    int total_workers_{1};
    int node_count_{1};
    int node_id_{0};
    int argc_;
    char** argv_;

    std::unique_ptr<Pool> pool_;
    std::unique_ptr<Comm> comm_;
    std::tuple<std::vector<Types>...> recv_bufs_; // root only, populated after join()
    std::vector<std::string> reduce_labels_;
    std::vector<std::function<void()>> reduce_fns_;

public:
    OrchestratorImpl(Job& job, int argc, char** argv)
        : job_(job), argc_(argc), argv_(argv) {}

    void run() {
        phase_cmdline(argc_, argv_);

        int provided = 0;
        MPI_Init_thread(&argc_, &argv_, MPI_THREAD_SERIALIZED, &provided);
        if (provided < MPI_THREAD_SERIALIZED) {
            // NOTE: MPI_THREAD_SERIALIZED unavailability is not automatable in tests
            // because it requires a crippled MPI installation. Documented here for
            // code review. Expected error: "[orchestrator] MPI_THREAD_SERIALIZED not available"
            std::cerr << "[orchestrator] MPI_THREAD_SERIALIZED not available\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        read_slurm_env();

        pool_ = std::make_unique<Pool>(static_cast<std::size_t>(num_workers_));
        comm_ = std::make_unique<Comm>(*pool_, rank_, world_size_, num_workers_);

        phase_dist_plan();
        phase_node_plan();

        comm_->start();

        if (rank_ == 0) {
            std::fprintf(stderr, "[orchestrator] phase: map\n");
        }
        phase_map();

        comm_->request_flush();
        comm_->join(); // CommThread handles the cross-rank MPI_Barrier internally

        if (rank_ == 0) {
            recv_bufs_ = std::move(comm_->received());
            phase_reduce();
            phase_output();
        }

        MPI_Finalize();
    }

private:
    void read_slurm_env() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Delegate env parsing and FATAL_MPI validation to the non-template free function.
        slurm_read_env(rank_, cpus_per_task_);

        // Invariant: --ntasks-per-node=1 means exactly one MPI rank per node.
        // Under this invariant MPI rank == SLURM node index, world_size == node count.
        node_count_ = world_size_;
        node_id_    = rank_;

        // Root loses one CPU slot to the comm thread; non-root does not.
        num_workers_   = (rank_ == 0) ? cpus_per_task_ - 1 : cpus_per_task_;
        // Total workers = root's workers + all non-root ranks at full cpus_per_task.
        total_workers_ = (cpus_per_task_ - 1) + (world_size_ - 1) * cpus_per_task_;

        if (rank_ == 0) {
            std::fprintf(stderr, "[orchestrator] %d ranks, %d cpus/rank, %d total workers\n",
                         world_size_, cpus_per_task_, total_workers_);
        }
    }

    void phase_cmdline(int argc, char** argv) {
        CmdlineCtx ctx{argc, argv};
        job_.on_cmdline(ctx);
    }

    void phase_dist_plan() {
        // first_global_worker_id() from work_ctx.hpp accounts for root's -1 offset.
        DistPlanCtx ctx{rank_, world_size_, node_count_, node_id_,
                        num_workers_, total_workers_,
                        first_global_worker_id(rank_, cpus_per_task_)};
        job_.on_dist_plan(ctx);
    }

    void phase_node_plan() {
        NodePlanCtx ctx{rank_, node_id_, num_workers_};
        job_.on_node_plan(ctx);
    }

    void phase_map() {
        const int n   = num_workers_;
        const int cpt = cpus_per_task_;
        const int tw  = total_workers_;
        // One WorkCtx per worker thread. WorkCtx destructor flushes remaining local
        // buffer items to the ring, so no explicit flush is needed after on_map.
        #pragma omp parallel num_threads(n)
        {
            const int local_tid = omp_get_thread_num();
            // compute_global_id() from work_ctx.hpp correctly accounts for root
            // having cpus_per_task-1 workers rather than cpus_per_task.
            const int worker_id = compute_global_id(rank_, local_tid, cpt);
            WorkCtx<Types...> ctx(worker_id, tw,
                                  pool_->get(static_cast<std::size_t>(local_tid)));
            job_.on_map(ctx);
        }
    }

    void phase_reduce() {
        // Count total received records across all output types before building ReduceCtx.
        std::size_t total = 0;
        std::apply([&](const auto&... vecs) {
            // fold expression: accumulate size of each type's received vector
            ((total += vecs.size()), ...);
        }, recv_bufs_);
        std::fprintf(stderr, "[orchestrator] phase: reduce (%zu records)\n", total);

        // Build ReduceCtx holding const-refs into orchestrator-owned recv_bufs_.
        // std::as_const ensures the lambda receives const auto& for each vector.
        auto ctx = std::apply(
            [](const auto&... vecs) { return ReduceCtx<Types...>(vecs...); },
            std::as_const(recv_bufs_));
        job_.on_reduce(ctx);

        // Copy labels and fns before ctx is destroyed at scope end.
        reduce_labels_ = ctx.labels();
        reduce_fns_    = ctx.fns();

        const int n_tasks = static_cast<int>(reduce_fns_.size());
        // Each task writes to a separate output field — no synchronization needed.
        #pragma omp parallel for num_threads(num_workers_) schedule(dynamic)
        for (int i = 0; i < n_tasks; ++i) {
            reduce_fns_[i]();
        }
    }

    void phase_output() {
        std::fprintf(stderr, "[orchestrator] phase: output\n");
        // OutputCtx holds a const-ref to reduce_labels_ (a member) — lifetime is safe.
        OutputCtx ctx{rank_, reduce_labels_};
        job_.on_output(ctx);
    }
};

// Public alias — users write Orchestrator<MyJob>, not OrchestratorImpl<MyJob>.
template<typename Job>
using Orchestrator = OrchestratorImpl<Job>;
