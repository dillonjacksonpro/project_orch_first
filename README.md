# HPC Orchestrator

A C++17 framework for embarrassingly parallel workloads on SLURM clusters. You implement
six lifecycle hooks; the framework handles MPI initialization, OpenMP thread management,
lock-free data transfer from workers to the root rank, and reduction. The orchestrator is
agnostic to what work is being done — it never calls into user code except through the
hooks you override.

Communication uses a star topology: all output records flow from worker threads on every
rank to rank 0, through lock-free SPSC ring buffers and async MPI sends. Only one thread
per rank ever calls MPI (`MPI_THREAD_SERIALIZED`), eliminating lock contention on the
critical path.

A reference job (`CsvStatsJob`) is included. It reads a directory of CSV files and
computes totals, averages, exact medians, and top/bottom-10 rankings across three
numeric columns.


## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| C++ compiler | C++17 | Must be exposed via the `mpicxx` wrapper |
| MPI | Any | OpenMPI, MPICH, Cray, Intel MPI — any that provides `mpicxx` |
| OpenMP | 2.0+ | Included with most MPI toolchains |
| GNU Make | 3.8+ | |
| Python 3 | 3.6+ | E2E test data generation only (stdlib, no extra packages) |
| SLURM | Optional | Required only for `run.sh`; `mpirun` works without it |

On an HPC cluster, load the appropriate MPI module before building:

```bash
module load openmpi   # or mpich, intel-mpi, etc.
```

On a workstation (Ubuntu/Debian):

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```


## Quick Start

Build the included example job and validate it end-to-end:

```bash
make                          # builds build/orchestrator using CsvStatsJob
make test_e2e                 # generates test data, runs three configs, diffs output
```

Run it directly against the test data:

```bash
SLURM_CPUS_PER_TASK=4 mpirun -n 2 ./build/orchestrator --dir tests/e2e/data
```


## Building

### Default build (example job)

```bash
make
# or equivalently
make orchestrator
```

Output binary: `build/orchestrator`

### Custom job

Pass three variables to `make`:

```bash
make USER_JOB_HEADER=my_job/my_job.hpp \
     USER_JOB_CLASS=MyJob \
     USER_SRC=my_job/my_job.cpp
```

| Variable | Default | Description |
|----------|---------|-------------|
| `USER_JOB_HEADER` | `user/csv_stats_job.hpp` | Header file containing your job class |
| `USER_JOB_CLASS` | `CsvStatsJob` | Name of the class |
| `USER_SRC` | `user/csv_stats_job.cpp` | Implementation file |

### Performance tuning overrides

```bash
make RING_SIZE=8192 LOCAL_BUF_SIZE=128 BATCH_SIZE=512
```

See [Performance Tuning](#performance-tuning) for guidance on what to change.

### Clean

```bash
make clean    # removes build/
```


## Running

### Local (mpirun)

Set `SLURM_CPUS_PER_TASK` in the environment before launching. Most MPI launchers
forward the current environment to all ranks by default.

```bash
# 2 ranks, 4 threads per rank (rank 0 uses 3 workers + 1 comm thread)
SLURM_CPUS_PER_TASK=4 mpirun -n 2 ./build/orchestrator --dir /path/to/data
```

If `SLURM_CPUS_PER_TASK` is unset, the orchestrator falls back to
`std::thread::hardware_concurrency()`. Rank 0 always requires at least 2 CPUs
(1 worker + 1 comm thread); launching with fewer causes a fatal error.

### SLURM cluster (run.sh)

Copy the example config and fill in your values:

```bash
cp run.cfg.example run.cfg
$EDITOR run.cfg
bash run.sh run.cfg
```

**Config file format** (`run.cfg`):

```bash
# Required
DATA_DIR=/scratch/myproject/data
SLURM_NTASKS=8
SLURM_CPUS_PER_TASK=4

# Optional (default: runs/)
OUTPUT_ROOT=runs
```

Each invocation creates a timestamped subdirectory:

```
runs/
  20260407_143022/
    config.cfg      # copy of the config used for this run
    stdout.txt      # orchestrator output (on_output results)
    stderr.txt      # orchestrator lifecycle logs
    exit_code.txt   # numeric exit code of srun
```

Both `stdout.txt` and `stderr.txt` are also echoed to the terminal live.


## Implementing a Custom Job

### 1. Define output types

Output types are the records your workers emit during the map phase. Each type must be
**trivially copyable** (no `std::string`, `std::vector`, or other heap-owning members).
The compiler will reject non-trivially-copyable types with a static assert.

```cpp
// Good: plain data, no heap members
struct WordCount {
    uint64_t word_count;
    uint64_t doc_id;
};

// Bad: std::string is not trivially copyable — compile error
struct Bad {
    std::string word;
    int count;
};
```

If you need to store a variable-length string, use a fixed-size char array:

```cpp
struct Hit {
    char filename[256];
    uint64_t offset;
};
```

### 2. Implement the job class

Inherit from `JobInterface<Types...>` and override the hooks you need.
Only `on_map` is required.

```cpp
#pragma once
#include "job_interface.hpp"
#include <cstdint>

struct WordCount { uint64_t count; uint32_t doc_id; };

class MyJob : public JobInterface<WordCount> {
public:
    // Pre-MPI: parse your own CLI arguments from argc/argv
    void on_cmdline(CmdlineCtx& ctx) override;

    // All ranks: partition work across ranks; store in job fields
    void on_dist_plan(DistPlanCtx& ctx) override;

    // All ranks: sub-partition this rank's work across threads
    void on_node_plan(NodePlanCtx& ctx) override;

    // All ranks, OMP parallel — REQUIRED. Emit results via ctx.emit<T>()
    // Do NOT call MPI from this hook.
    void on_map(WorkCtx<WordCount>& ctx) override;

    // Rank 0 only: register OMP reduction tasks via ctx.add_task()
    // Each task must write to separate fields (no synchronization)
    void on_reduce(ReduceCtx<WordCount>& ctx) override;

    // Rank 0 only: write final results to stdout
    void on_output(OutputCtx& ctx) override;

private:
    std::vector<std::string> files_;   // populated in dist/node_plan
    uint64_t total_words_ = 0;         // written by reduce task
};
```

### 3. Hook reference

| Hook | Context | Scope | OMP | Required | Purpose |
|------|---------|-------|-----|----------|---------|
| `on_cmdline` | `CmdlineCtx` | All ranks | No | No | Parse `argc`/`argv` (pre-MPI) |
| `on_dist_plan` | `DistPlanCtx` | All ranks | No | No | Partition work across ranks |
| `on_node_plan` | `NodePlanCtx` | All ranks | No | No | Sub-partition across threads |
| `on_map` | `WorkCtx<T...>` | All ranks | Yes | **Yes** | Emit records via `ctx.emit<T>()` |
| `on_reduce` | `ReduceCtx<T...>` | Rank 0 | No | No | Register OMP reduction tasks |
| `on_output` | `OutputCtx` | Rank 0 | No | No | Write final results to stdout |

**Context fields available in each hook:**

`DistPlanCtx`: `rank`, `world_size`, `node_count`, `node_id`, `num_workers`,
`total_workers`, `first_global_worker_id`

`NodePlanCtx`: `rank`, `node_id`, `num_workers`

`WorkCtx<T...>`: `id` (global worker ID, 0-indexed), `num_workers`

`ReduceCtx<T...>`: access records via `ctx.data<T>()`, register tasks via `ctx.add_task()`

`OutputCtx`: `rank` (always 0), `task_labels` (in registration order)

### 4. Key API methods

```cpp
// Map phase: emit a record (copied into lock-free ring buffer → comm thread → rank 0)
ctx.emit<T>(value);

// Reduce phase: read all received records of type T from all workers
Span<const T> records = ctx.data<T>();

// Reduce phase: register an OMP task (runs in parallel after on_reduce returns)
ctx.add_task("label", [this, records]() {
    // compute and write into job member fields
    // each task must write to separate fields — no shared mutable state
});

// Output phase: access task labels in registration order
for (const auto& label : ctx.task_labels) { ... }
```

### 5. Build

```bash
make USER_JOB_HEADER=my_job/my_job.hpp \
     USER_JOB_CLASS=MyJob \
     USER_SRC=my_job/my_job.cpp
```

The entry point (`main.cpp`) is never modified. It instantiates your class and runs the
orchestrator lifecycle automatically.

### Common pitfalls

- **Non-trivially-copyable output type** — compile error in `SpscRing`. Use fixed-size
  char arrays instead of `std::string`.
- **Duplicate types in `JobInterface<A, A>`** — compile error. Each output type must
  be unique.
- **MPI calls inside `on_map`** — undefined behaviour. Only the comm thread calls MPI
  (`MPI_THREAD_SERIALIZED`). Never call MPI from worker threads.
- **Data races in reduce tasks** — each `add_task` lambda runs as an OMP task in
  parallel. Write to separate job member fields per task; do not share mutable state.
- **`SLURM_CPUS_PER_TASK` < 2 on rank 0** — rank 0 reserves one thread for the comm
  thread and needs at least one worker. Setting this to 1 causes a fatal error.
- **`on_output` writing to stderr** — the E2E script captures stdout for diffing.
  All user output must go to stdout (`printf` / `std::cout`); use stderr for debug
  messages only.


## Performance Tuning

Three constants can be overridden at build time:

| Constant | Default | Build override | Effect |
|----------|---------|----------------|--------|
| `RING_SIZE` | 4096 | `make RING_SIZE=N` | Slots per worker's SPSC ring buffer. Larger reduces MPI send frequency but increases memory per worker. Must be a power of 2. |
| `LOCAL_BUF_SIZE` | 64 | `make LOCAL_BUF_SIZE=N` | Records accumulated in each worker's local buffer before flushing to the ring. Larger improves cache efficiency; smaller lowers per-record latency. |
| `BATCH_SIZE` | 256 | `make BATCH_SIZE=N` | Records batched into a single `MPI_Isend`. Larger improves MPI throughput; smaller reduces end-of-map latency. |

Two additional constants in `include/orchestrator_config.hpp` are not Makefile-overridable:

| Constant | Value | Effect |
|----------|-------|--------|
| `MAX_BACKOFF_NS` | 100,000 ns | Exponential backoff cap for the comm thread on non-root ranks when the ring is empty. |
| `PREPOST_RECVS` | 8 | Number of receives root pre-posts. Increase if in-flight messages exceed this count at peak. |

**Tuning heuristics:**

- High MPI call overhead (profiled): increase `BATCH_SIZE` or `RING_SIZE`.
- High per-record latency (first result arrives late): decrease `LOCAL_BUF_SIZE`.
- Memory constrained: decrease `RING_SIZE`.
- Start with defaults; profile with `mpiP` or `Tau` before tuning.


## Testing

| Target | Description |
|--------|-------------|
| `make test` | Unit tests (no MPI) — ring buffers, local buffers, type utilities |
| `make test_mpi` | MPI integration tests at 2 and 3 ranks |
| `make test_e2e` | End-to-end: generates CSV data, runs three rank/thread configs, diffs output |
| `make test_all` | All of the above plus compile-fail and link-fail negative tests |
| `make test_debug` | Unit tests with `SPSC_DEBUG` thread-ownership tracking enabled |

On environments without a full toolchain (e.g., missing the right compiler flags for
negative tests):

```bash
SKIP_NEGATIVE_TESTS=1 make test_all
```

To use a different MPI launcher for tests:

```bash
MPIRUN=srun make test_mpi
```


## Project Layout

```
.
├── main.cpp                    # Entry point — never modified by users
├── Makefile                    # Build system
├── run.sh                      # SLURM runner script
├── run.cfg.example             # Template config for run.sh
├── include/
│   ├── job_interface.hpp       # JobInterface<Types...> base class
│   ├── orchestrator.hpp        # Orchestrator<Job> — 6-phase lifecycle
│   ├── orchestrator_config.hpp # Tuning constants (RING_SIZE etc.)
│   ├── fatal.hpp               # FATAL / MPI_CHECK error macros
│   ├── type_utils.hpp          # TypeIndex, AllUnique — compile-time utilities
│   ├── context/                # CmdlineCtx, DistPlanCtx, WorkCtx, ReduceCtx, OutputCtx
│   ├── ring/                   # SpscRing, TypedRingSet, RingPool
│   ├── buffer/                 # TypedLocalBuffer
│   └── comm/                   # CommThread
├── src/                        # Orchestrator implementation
├── user/                       # Reference job: CsvStatsJob
│   ├── csv_stats_job.hpp
│   └── csv_stats_job.cpp
└── tests/
    ├── unit/                   # Doctest unit tests (no MPI)
    ├── mpi/                    # MPI integration tests
    ├── static_assert/          # Negative compile/link-fail tests
    └── e2e/                    # End-to-end script + test data generator
```

For a detailed description of the internal architecture, threading model, and data flow,
see `design.md`.
