#pragma once

struct DistPlanCtx {
    int rank;
    int world_size;
    int node_count;             // SLURM_NNODES
    int node_id;                // SLURM_NODEID
    int num_workers;            // workers on THIS rank (root: cpus-1, non-root: cpus)
    int total_workers;          // sum of workers across all ranks
    int first_global_worker_id; // global_id of thread 0 on this rank
                                // = 0 for root; = workers_root + (rank-1)*workers_nonroot for others
};
