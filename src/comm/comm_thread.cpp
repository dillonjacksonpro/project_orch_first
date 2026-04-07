// Minimal stub — CommThread is fully header-only (template class).
// This file exists to satisfy the Makefile prerequisite and force a
// standalone compilation check of the header. See handoff §Phase 1 item 4.
//
// mpi.h must precede fatal.hpp (which comm_thread.hpp includes).
// See handoff §Phase 0 item 2.
#include <mpi.h>
#include "comm/comm_thread.hpp"
