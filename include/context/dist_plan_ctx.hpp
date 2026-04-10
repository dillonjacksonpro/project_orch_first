#pragma once

struct DistPlanCtx {
    int rank;
    int world_size;
    int node_count;             // == MPI world_size (invariant: --ntasks-per-node=1)
    int node_id;                // == MPI rank       (invariant: --ntasks-per-node=1)
    int num_workers;            // workers on THIS rank (root: cpus-1, non-root: cpus)
    int total_workers;          // sum of workers across all ranks
    int first_global_worker_id; // global_id of thread 0 on this rank
                                // = 0 for root; = workers_root + (rank-1)*workers_nonroot for others
};
