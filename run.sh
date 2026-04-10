#!/usr/bin/env bash
#SBATCH --job-name=hpc_orch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=runs/slurm-%j.log
#SBATCH --error=runs/slurm-%j.log

# Runner script for the HPC orchestrator.
# Usage: sbatch [--ntasks=N] [--cpus-per-task=C]  [--partition=P] run.sh <config-file>
#
# SLURM resources are set via #SBATCH directives above (overridable on the
# sbatch command line). The config file controls DATA_DIR and OUTPUT_ROOT only.
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument validation and config parsing
# ---------------------------------------------------------------------------
if [ $# -ne 1 ]; then
    echo "Usage: sbatch run.sh <config-file>" >&2
    exit 1
fi

config_file="$1"

if [ ! -f "$config_file" ]; then
    echo "error: config file not found: $config_file" >&2
    exit 1
fi

DATA_DIR=""
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
        DATA_DIR)    DATA_DIR="$value" ;;
        OUTPUT_ROOT) OUTPUT_ROOT="$value" ;;
        *) ;;
    esac
done < "$config_file"

if [ -z "$DATA_DIR" ]; then
    echo "error: required config key missing: DATA_DIR" >&2
    exit 1
fi

# Make DATA_DIR absolute relative to CWD.
if [ "${DATA_DIR#/}" = "$DATA_DIR" ]; then
    DATA_DIR="$(pwd)/$DATA_DIR"
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "error: DATA_DIR does not exist or is not a directory: $DATA_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create a timestamped run directory
# ---------------------------------------------------------------------------
timestamp="$(date +%Y%m%d_%H%M%S)"
if [ "${OUTPUT_ROOT#/}" = "$OUTPUT_ROOT" ]; then
    run_dir="$(pwd)/${OUTPUT_ROOT}/${timestamp}"
else
    run_dir="${OUTPUT_ROOT}/${timestamp}"
fi
mkdir -p "$run_dir"
cp "$config_file" "$run_dir/config.cfg"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# SLURM_MEM_PER_CPU and SLURM_MEM_PER_NODE are mutually exclusive for srun.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU

echo "Job $SLURM_JOB_ID running on $(hostname)"
echo "Run directory: $run_dir"

exit_code=0
srun --export=ALL \
    --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    env SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
        SLURM_NNODES="$SLURM_NNODES" \
    ./build/orchestrator --dir "$DATA_DIR" \
    > "$run_dir/stdout.txt" \
    2> "$run_dir/stderr.txt" \
    || exit_code=$?

echo "$exit_code" > "$run_dir/exit_code.txt"
echo "Exit code: $exit_code"
exit "$exit_code"
