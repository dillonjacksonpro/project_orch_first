#pragma once

#include <string>
#include <vector>

struct OutputCtx {
    int rank;                                      // always 0
    const std::vector<std::string>& task_labels;   // labels in add_task registration order
                                                   // const-ref to orchestrator-owned vector,
                                                   // valid for the duration of on_output
};
