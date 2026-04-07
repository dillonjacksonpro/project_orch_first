#pragma once

struct NodePlanCtx {
    int rank;
    int node_id;
    int num_workers; // workers on this rank
};
