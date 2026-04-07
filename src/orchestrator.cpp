#include <mpi.h>

#include "fatal.hpp"
#include "orchestrator.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

// SLURM env validation error paths — all three call FATAL_MPI and cannot be
// tested automatically because MPI_Abort terminates the test runner. They are
// verified by code review. Expected error strings:
//   cpus_per_task <= 0 || cpus_per_task > 65536:
//     "[rank N] [FATAL] SLURM_CPUS_PER_TASK out of range: <value>"
//   node_count <= 0:
//     "[rank N] [FATAL] SLURM_NNODES out of range: <value>"
//   rank == 0 && cpus_per_task < 2:
//     "[rank N] [FATAL] rank 0 requires SLURM_CPUS_PER_TASK >= 2 ..."
void slurm_read_env(int rank, int world_size,
                    int& cpus_per_task, int& node_count, int& node_id) {
    // Parse an env var as int. Returns fallback if the variable is absent.
    // Calls FATAL_MPI if the variable is present but not a valid integer.
    auto getenv_int = [&](const char* name, int fallback) -> int {
        const char* v = std::getenv(name);
        if (!v) {
            return fallback;
        }
        try {
            return std::stoi(v);
        } catch (...) {
            FATAL_MPI(rank, std::string(name) + " is not a valid integer: " + v);
        }
        return fallback; // unreachable; silences -Wreturn-type
    };

    const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    cpus_per_task = getenv_int("SLURM_CPUS_PER_TASK", hw_threads > 0 ? hw_threads : 1);
    if (cpus_per_task <= 0 || cpus_per_task > 65536) {
        FATAL_MPI(rank, "SLURM_CPUS_PER_TASK out of range: " +
                            std::to_string(cpus_per_task));
    }

    node_count = getenv_int("SLURM_NNODES", 1);
    if (node_count <= 0) {
        FATAL_MPI(rank, "SLURM_NNODES out of range: " + std::to_string(node_count));
    }

    node_id = getenv_int("SLURM_NODEID", rank);

    // Root rank must have at least 2 CPUs: one for the comm thread, one for work.
    // num_workers_ = cpus_per_task - 1 for root, so cpus_per_task == 1 → num_workers_ == 0.
    if (rank == 0 && cpus_per_task < 2) {
        FATAL_MPI(rank, "rank 0 requires SLURM_CPUS_PER_TASK >= 2 "
                        "(one slot reserved for comm thread)");
    }

    (void)world_size; // reserved for future SLURM_NTASKS cross-checking
}
