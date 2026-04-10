#include <mpi.h>

#include "fatal.hpp"
#include "orchestrator.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

// SLURM env error path — not automated because FATAL_MPI calls MPI_Abort which
// terminates the test runner. Verified by code review. Expected error string:
//   cpus_per_task <= 0 || cpus_per_task > 65536:
//     "[rank N] [FATAL] SLURM_CPUS_PER_TASK out of range: <value>"
void slurm_read_env(int rank, int& cpus_per_task) {
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
}
