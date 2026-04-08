#!/usr/bin/env bash
# Runner script for the HPC orchestrator.
# Usage: ./run.sh <config-file>
#
# From a login node: submits itself to SLURM via sbatch with the correct
# resource flags, then exits. The job re-runs this script and goes to the
# srun branch directly.
#
# From a compute node (inside an existing allocation): skips sbatch and runs
# srun immediately against the current allocation.
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument validation and config parsing (always runs first)
# ---------------------------------------------------------------------------
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config-file>" >&2
    exit 1
fi

config_file="$1"

if [ ! -f "$config_file" ]; then
    echo "error: config file not found: $config_file" >&2
    exit 1
fi

DATA_DIR=""
SLURM_NTASKS=""
SLURM_CPUS_PER_TASK=""
SLURM_PARTITION=""
OUTPUT_ROOT="runs"

while IFS= read -r line; do
    # Strip leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [ -z "$line" ] && continue
    [ "${line#\#}" != "$line" ] && continue

    key="${line%%=*}"
    value="${line#*=}"

    case "$key" in
        DATA_DIR)             DATA_DIR="$value" ;;
        SLURM_NTASKS)         SLURM_NTASKS="$value" ;;
        SLURM_CPUS_PER_TASK)  SLURM_CPUS_PER_TASK="$value" ;;
        SLURM_PARTITION)      SLURM_PARTITION="$value" ;;
        OUTPUT_ROOT)          OUTPUT_ROOT="$value" ;;
        *) echo "warning: unrecognised config key: $key" >&2 ;;
    esac
done < "$config_file"

missing=0
for var in DATA_DIR SLURM_NTASKS SLURM_CPUS_PER_TASK; do
    if [ -z "${!var}" ]; then
        echo "error: required config key missing: $var" >&2
        missing=1
    fi
done
[ "$missing" -ne 0 ] && exit 1

# Make DATA_DIR absolute relative to CWD.
if [ "${DATA_DIR#/}" = "$DATA_DIR" ]; then
    DATA_DIR="$(pwd)/$DATA_DIR"
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist or is not a directory: $DATA_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: create a timestamped run directory
# ---------------------------------------------------------------------------
make_run_dir() {
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local base
    if [ "${OUTPUT_ROOT#/}" = "$OUTPUT_ROOT" ]; then
        base="$(pwd)/${OUTPUT_ROOT}"
    else
        base="${OUTPUT_ROOT}"
    fi
    local dir="${base}/${timestamp}"
    mkdir -p "$dir"
    cp "$config_file" "$dir/config.cfg"
    echo "$dir"
}

# ---------------------------------------------------------------------------
# In-job branch: inside a SLURM allocation — run srun directly.
# Reached either via sbatch self-submission (RUN_DIR already exported)
# or when the user runs the script directly on a compute node.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # If submitted via sbatch, RUN_DIR is already set in the environment.
    # If run directly on a compute node, create it now.
    if [ -z "${RUN_DIR:-}" ]; then
        RUN_DIR="$(make_run_dir)"
    fi

    # SLURM_MEM_PER_CPU (from partition DefMemPerCPU) and SLURM_MEM_PER_NODE
    # are mutually exclusive from srun's perspective — drop the node-level one.
    unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU

    echo "Job $SLURM_JOB_ID running on $(hostname)"
    echo "Run directory: $RUN_DIR"

    exit_code=0
    srun --export=ALL \
        ./build/orchestrator --dir "$DATA_DIR" \
        > "$RUN_DIR/stdout.txt" \
        2> "$RUN_DIR/stderr.txt" \
        || exit_code=$?

    echo "$exit_code" > "$RUN_DIR/exit_code.txt"
    echo "Exit code: $exit_code"
    exit "$exit_code"
fi

# ---------------------------------------------------------------------------
# Login-node branch: submit this script to sbatch with correct resources.
# The job re-runs this script; SLURM_JOB_ID triggers the in-job branch above.
# ---------------------------------------------------------------------------
RUN_DIR="$(make_run_dir)"

partition_flag=""
if [ -n "$SLURM_PARTITION" ]; then
    partition_flag="--partition=$SLURM_PARTITION"
fi

# --parsable outputs "jobid" or "jobid;cluster" on federated setups.
job_id=$(sbatch --parsable \
    --nodes=1 \
    -n "$SLURM_NTASKS" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    ${partition_flag:+$partition_flag} \
    --chdir="$(pwd)" \
    --export="ALL,RUN_DIR=$RUN_DIR,DATA_DIR=$DATA_DIR" \
    --output="$RUN_DIR/job.log" \
    --error="$RUN_DIR/job.log" \
    "$(realpath "$0")" "$config_file" | cut -d';' -f1)

echo "Job ID:       $job_id"
echo "Run dir:      $RUN_DIR"
echo "stdout:       tail -f $RUN_DIR/stdout.txt"
echo "stderr/logs:  tail -f $RUN_DIR/job.log"
