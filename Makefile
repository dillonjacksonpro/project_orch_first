# ── User configures these three variables ─────────────────────────────────────
USER_JOB_HEADER ?= user/csv_stats_job.hpp
USER_JOB_CLASS  ?= CsvStatsJob
USER_SRC        ?= user/csv_stats_job.cpp

# ── Toolchain ─────────────────────────────────────────────────────────────────
CXX     = mpicxx
MPIRUN ?= mpirun

# ── Production tuning (override on command line) ──────────────────────────────
RING_SIZE      ?= 4096
LOCAL_BUF_SIZE ?= 64
BATCH_SIZE     ?= 256

# ── Test tuning: forces every boundary condition with minimal data ─────────────
TEST_RING_SIZE      = 8
TEST_LOCAL_BUF_SIZE = 4
TEST_BATCH_SIZE     = 4

SKIP_NEGATIVE_TESTS ?= 0

# ── Common flags (production) ─────────────────────────────────────────────────
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic -Werror \
           -Iinclude \
           -DRING_SIZE=$(RING_SIZE) \
           -DLOCAL_BUF_SIZE=$(LOCAL_BUF_SIZE) \
           -DBATCH_SIZE=$(BATCH_SIZE) \
           -DUSER_JOB_HEADER='"$(USER_JOB_HEADER)"' \
           -DUSER_JOB_CLASS=$(USER_JOB_CLASS) \
           -DOMPI_SKIP_MPICXX \
           -Wno-stringop-overflow

# ── Test flags (override tuning constants) ─────────────────────────────────────
TEST_CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic -Werror \
                -Iinclude -Itests/third_party \
                -DRING_SIZE=$(TEST_RING_SIZE) \
                -DLOCAL_BUF_SIZE=$(TEST_LOCAL_BUF_SIZE) \
                -DBATCH_SIZE=$(TEST_BATCH_SIZE) \
                -DOMPI_SKIP_MPICXX \
                -Wno-stringop-overflow

# ── Orchestrator sources (never modified by user) ──────────────────────────────
ORCH_SRC = src/orchestrator.cpp \
           src/ring/ring_pool.cpp \
           src/comm/comm_thread.cpp

TARGET = build/orchestrator

.PHONY: all orchestrator test test_static_assert test_link_fail \
        test_mpi test_e2e test_all test_debug clean

# ── Main binary ───────────────────────────────────────────────────────────────
all: orchestrator

orchestrator: $(TARGET)

$(TARGET): $(ORCH_SRC) $(USER_SRC) main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Iuser $^ -o $@

# ── Unit tests (no MPI) ───────────────────────────────────────────────────────
build/tests/test_unit: tests/unit/test_*.cpp src/ring/ring_pool.cpp
	@mkdir -p build/tests
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

test: build/tests/test_unit
	build/tests/test_unit

# ── Negative compile-fail tests ───────────────────────────────────────────────
test_static_assert: orchestrator
ifeq ($(SKIP_NEGATIVE_TESTS),1)
	@echo "Skipping negative compile tests (SKIP_NEGATIVE_TESTS=1)"
else
	@fail=0; \
	for f in tests/static_assert/sa_*.cpp; do \
	    case $$f in *sa_job_no_on_map*) continue ;; esac; \
	    if $(CXX) $(CXXFLAGS) -Iinclude -c $$f -o /dev/null 2>/dev/null; then \
	        echo "FAIL (compiled when it should not): $$f"; fail=1; \
	    else \
	        echo "PASS (correctly rejected): $$f"; \
	    fi; \
	done; \
	exit $$fail
endif

# ── Negative link-fail tests ──────────────────────────────────────────────────
test_link_fail: orchestrator
ifeq ($(SKIP_NEGATIVE_TESTS),1)
	@echo "Skipping negative link tests (SKIP_NEGATIVE_TESTS=1)"
else
	@fail=0; \
	for f in tests/static_assert/sa_job_no_on_map.cpp; do \
	    if $(CXX) $(CXXFLAGS) -Iinclude $$f -o /dev/null 2>/dev/null; then \
	        echo "FAIL (linked when it should not): $$f"; fail=1; \
	    else \
	        echo "PASS (correctly rejected at link): $$f"; \
	    fi; \
	done; \
	exit $$fail
endif

# ── MPI integration tests ─────────────────────────────────────────────────────
build/tests/test_mpi: tests/mpi/test_mpi_*.cpp src/ring/ring_pool.cpp \
                      src/comm/comm_thread.cpp
	@mkdir -p build/tests
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

# Orchestrator integration test — separate binary because orch.run() owns
# MPI_Init_thread/MPI_Finalize, preventing use of the doctest main() above.
build/tests/test_orch: tests/mpi/test_orchestrator.cpp src/ring/ring_pool.cpp \
                       src/comm/comm_thread.cpp src/orchestrator.cpp
	@mkdir -p build/tests
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

test_mpi: build/tests/test_mpi build/tests/test_orch
	$(MPIRUN) -n 2 build/tests/test_mpi
	$(MPIRUN) -n 3 build/tests/test_mpi
	$(MPIRUN) -n 2 build/tests/test_orch
	$(MPIRUN) -n 3 build/tests/test_orch

# ── End-to-end tests ──────────────────────────────────────────────────────────
test_e2e: orchestrator
	tests/e2e/run_e2e.sh

# ── Debug build (SPSC_DEBUG enables thread-ownership tracking in SpscRing) ────
build/tests/test_debug: tests/unit/test_spsc_ring.cpp src/ring/ring_pool.cpp
	@mkdir -p build/tests
	$(CXX) $(TEST_CXXFLAGS) -DSPSC_DEBUG $^ -o $@

test_debug: build/tests/test_debug
	build/tests/test_debug

# ── Full test suite ───────────────────────────────────────────────────────────
test_all: orchestrator test test_static_assert test_link_fail test_mpi test_e2e

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf build/
