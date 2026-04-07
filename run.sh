#!/usr/bin/env bash
# Runner script for the HPC orchestrator.
# Usage: bash run.sh <config-file>
#
# Reads a key=value config file, launches the orchestrator via srun, and saves
# stdout, stderr, and the exit code to a timestamped directory under OUTPUT_ROOT.
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument validation
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
run_dir="${OUTPUT_ROOT}/${timestamp}"
mkdir -p "$run_dir"

# Resolve to absolute path for cleaner output.
run_dir="$(cd "$run_dir" && pwd)"

cp "$config_file" "$run_dir/config.cfg"

echo "Run directory: $run_dir"
echo "Launching: srun -n $SLURM_NTASKS --export=SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK ./build/orchestrator --dir $DATA_DIR"
echo "---"

# ---------------------------------------------------------------------------
# Run srun — tee stdout and stderr to screen and separate files
# ---------------------------------------------------------------------------
exit_code=0
srun -n "$SLURM_NTASKS" \
    --export=SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
    ./build/orchestrator --dir "$DATA_DIR" \
    > >(tee "$run_dir/stdout.txt") \
    2> >(tee "$run_dir/stderr.txt" >&2) \
    || exit_code=$?   # capture non-zero exit without triggering set -e
wait   # flush tee subprocesses before writing exit_code.txt

echo "$exit_code" > "$run_dir/exit_code.txt"

echo "---"
echo "Exit code: $exit_code"
echo "Output saved to: $run_dir"

exit "$exit_code"
