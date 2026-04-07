#include "orchestrator.hpp"
#include USER_JOB_HEADER

int main(int argc, char** argv) {
    USER_JOB_CLASS job;
    Orchestrator<USER_JOB_CLASS> orch(job, argc, argv);
    orch.run();
    return 0;
}
