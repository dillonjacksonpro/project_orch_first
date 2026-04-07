#pragma once

#include "context/cmdline_ctx.hpp"
#include "context/dist_plan_ctx.hpp"
#include "context/node_plan_ctx.hpp"
#include "context/output_ctx.hpp"
#include "context/reduce_ctx.hpp"
#include "context/work_ctx.hpp"
#include "type_utils.hpp"
#include <tuple>

// Abstract base class for user-defined jobs.
//
// OutputTypes... are the types the job emits during on_map. Each type must be
// trivially copyable (enforced downstream by SpscRing). Duplicate types are
// rejected at compile time because they would produce ambiguous ring/buffer
// lookups in WorkCtx and ReduceCtx.
template<typename... OutputTypes>
class JobInterface {
    // Duplicate types in the pack produce ambiguous ring/buffer lookups at compile time.
    // Detect and reject them early with a clear message.
    static_assert(AllUnique<OutputTypes...>::value,
                  "OutputTypes must be unique — duplicate type in JobInterface<...>");

public:
    virtual ~JobInterface() = default;

    // ALL RANKS — called before MPI_Init. Parse argc/argv; store into job fields.
    virtual void on_cmdline(CmdlineCtx&) {}

    // ALL RANKS — post-MPI. Partition work across ranks; store into job fields.
    virtual void on_dist_plan(DistPlanCtx&) {}

    // ALL RANKS — post-MPI. Sub-partition this rank's work across threads; store into job fields.
    virtual void on_node_plan(NodePlanCtx&) {}

    // ALL RANKS — OMP parallel, one call per worker thread. Emit results via ctx.emit<T>().
    // WARNING: do NOT call any MPI functions from this hook. The comm thread is the
    // sole MPI caller; calling MPI from a worker thread violates MPI_THREAD_SERIALIZED.
    virtual void on_map(WorkCtx<OutputTypes...>&) = 0;

    // RANK 0 ONLY — register OMP tasks via ctx.add_task(). Each task should write
    // to a separate field to avoid data races (no compile-time enforcement).
    virtual void on_reduce(ReduceCtx<OutputTypes...>&) {}

    // RANK 0 ONLY — write final results; ctx.task_labels holds registered task names.
    virtual void on_output(OutputCtx&) {}
};

// Deduces OutputTypes from a concrete Job class that inherits JobInterface<Types...>.
// Used by Orchestrator<Job> to unpack the type pack without requiring the user to
// re-declare it.
template<typename... Types>
auto job_output_types_helper(const JobInterface<Types...>*) -> std::tuple<Types...>;

// JobOutputTypes<MyJob> == std::tuple<T1, T2, ...> matching the JobInterface base
template<typename Job>
using JobOutputTypes = decltype(job_output_types_helper(std::declval<const Job*>()));
