#!/usr/bin/env bash
# Run smoke configs for exp3 / exp4 / expA / expB sequentially and analyze each.
# Usage: bash scripts/run_remaining_smokes.sh
set -e

for cfg in configs/exp3_smoke.yaml configs/exp4_smoke.yaml configs/expA_smoke.yaml configs/expB_smoke.yaml; do
    echo "=== Running $cfg ==="
    rundir=$(uv run python scripts/run_experiment.py --config "$cfg" 2>&1 | tail -1)
    echo "Run dir: $rundir"
    echo "--- Analyzing ---"
    uv run python scripts/analyze_run.py "$rundir" 2>&1 | tail -2
    echo
done
echo "DONE all remaining smokes"
