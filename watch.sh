#!/usr/bin/env bash
# Usage: ./watch.sh <run-directory>
# Tails stdout.txt and job.log side by side for a run directory.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run-directory>" >&2
    exit 1
fi

run_dir="$1"

if [ ! -d "$run_dir" ]; then
    echo "error: directory not found: $run_dir" >&2
    exit 1
fi

tail -f "$run_dir/stdout.txt" "$run_dir/job.log"
