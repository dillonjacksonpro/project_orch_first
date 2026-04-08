#!/usr/bin/env bash
# Runner script for the HPC orchestrator.
# Usage: ./run.sh <config-file>
#
# Run from the repo root on a login node. Reads a key=value config file,
# then submits itself to SLURM via sbatch with the correct resource flags.
# When the job runs, srun launches the orchestrator with a properly-sized
# allocation. Output goes to a timestamped subdirectory under OUTPUT_ROOT.
set -euo pipefail

# ---------------------------------------------------------------------------
# In-job phase
# Reached when sbatch re-runs this script inside the allocation.
# RUN_DIR and DATA_DIR are injected via --export at submission time;
# SLURM_CPUS_PER_TASK is set automatically by SLURM from --cpus-per-task.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # RUN_DIR and DATA_DIR must be injected via --export at submission time.
    # If they're missing, the script was run directly on a compute node instead
    # of being submitted through the login-node path.
    if [ -z "${RUN_DIR:-}" ] || [ -z "${DATA_DIR:-}" ]; then
        echo "error: run.sh must be run from a login node, not directly on a compute node." >&2
        echo "       Log out of this session and run: ./run.sh <config-file>" >&2
        exit 1
    fi

    # SLURM_MEM_PER_CPU (from partition DefMemPerCPU) and SLURM_MEM_PER_NODE
    # are mutually exclusive from srun's perspective — drop the node-level one.
    unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU

    echo "Job $SLURM_JOB_ID running on $(hostname)"

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
# Login-node phase
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

# ---------------------------------------------------------------------------
# Parse config: KEY=VALUE pairs, skip comments and blank lines
# ---------------------------------------------------------------------------
DATA_DIR=""
SLURM_NTASKS=""
SLURM_CPUS_PER_TASK=""
SLURM_PARTITION=""
OUTPUT_ROOT="runs"

while IFS= read -r line; do
    # Strip leading/trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    # Skip blank lines and comments
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

# ---------------------------------------------------------------------------
# Validate required keys
# ---------------------------------------------------------------------------
missing=0
for var in DATA_DIR SLURM_NTASKS SLURM_CPUS_PER_TASK; do
    if [ -z "${!var}" ]; then
        echo "error: required config key missing: $var" >&2
        missing=1
    fi
done
[ "$missing" -ne 0 ] && exit 1

# ---------------------------------------------------------------------------
# Resolve and validate DATA_DIR
# ---------------------------------------------------------------------------
# Make absolute relative to CWD (not the script's location).
if [ "${DATA_DIR#/}" = "$DATA_DIR" ]; then
    DATA_DIR="$(pwd)/$DATA_DIR"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist or is not a directory: $DATA_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create timestamped run directory
# ---------------------------------------------------------------------------
timestamp="$(date +%Y%m%d_%H%M%S)"
# Support absolute OUTPUT_ROOT as well as relative.
if [ "${OUTPUT_ROOT#/}" = "$OUTPUT_ROOT" ]; then
    run_dir="$(pwd)/${OUTPUT_ROOT}/${timestamp}"
else
    run_dir="${OUTPUT_ROOT}/${timestamp}"
fi
mkdir -p "$run_dir"
cp "$config_file" "$run_dir/config.cfg"

# ---------------------------------------------------------------------------
# Submit this script to sbatch with the correct resource allocation.
# sbatch re-runs the script inside the job; SLURM_JOB_ID triggers the
# in-job phase above. RUN_DIR and DATA_DIR are passed via --export=ALL,...
# --nodes=1 keeps all ranks on one node, avoiding whole-node allocation
# under OverSubscribe=NO when tasks would otherwise spread across nodes.
# ---------------------------------------------------------------------------
partition_flag=""
if [ -n "$SLURM_PARTITION" ]; then
    partition_flag="--partition=$SLURM_PARTITION"
fi

# --parsable outputs "jobid" or "jobid;cluster" on federated setups; keep only the ID.
job_id=$(sbatch --parsable \
    --nodes=1 \
    -n "$SLURM_NTASKS" \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    ${partition_flag:+$partition_flag} \
    --chdir="$(pwd)" \
    --export="ALL,RUN_DIR=$run_dir,DATA_DIR=$DATA_DIR" \
    --output="$run_dir/job.log" \
    --error="$run_dir/job.log" \
    "$(realpath "$0")" "$config_file" | cut -d';' -f1)

echo "Job ID:       $job_id"
echo "Run dir:      $run_dir"
echo "stdout:       tail -f $run_dir/stdout.txt"
echo "stderr/logs:  tail -f $run_dir/job.log"
