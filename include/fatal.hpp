#pragma once
#include <cstdlib>
#include <iostream>

// Pre-MPI fatal error (startup, env reading before MPI_Init).
// Uses std::exit — MPI is not yet initialised.
#define FATAL(msg) do {                                    \
    std::cerr << "[FATAL] " << (msg) << "\n";              \
    std::exit(1);                                          \
} while(0)

// Post-MPI fatal error on the main thread (bad env values, assertion failures).
// Calls MPI_Abort so all ranks terminate atomically.
#define FATAL_MPI(rank, msg) do {                          \
    std::cerr << "[rank " << (rank) << "] [FATAL] "        \
              << (msg) << "\n";                             \
    MPI_Abort(MPI_COMM_WORLD, 1);                          \
} while(0)

// Comm thread only — wraps every MPI call return code.
// Calls MPI_Abort with the MPI error string on failure.
// PRECONDITION: rank_ must be in scope (CommThread member variable).
#define MPI_CHECK(call) do {                               \
    int _rc = (call);                                      \
    if (_rc != MPI_SUCCESS) {                              \
        char _s[MPI_MAX_ERROR_STRING]; int _len;           \
        MPI_Error_string(_rc, _s, &_len);                  \
        std::cerr << "[rank " << rank_ << "] MPI error in "\
                  << #call << ": " << _s << "\n";          \
        MPI_Abort(MPI_COMM_WORLD, _rc);                    \
    }                                                      \
} while(0)
