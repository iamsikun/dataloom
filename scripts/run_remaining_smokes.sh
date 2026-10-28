#!/usr/bin/env bash
# Run and analyze all ten smoke profiles, or the supplied smoke configs.
# Run from the repository root. Prepare Adult data before the complete suite:
#   uv run --frozen python scripts/prepare_adult.py
#   bash scripts/run_remaining_smokes.sh
set -euo pipefail

if [ "$#" -eq 0 ]; then
    set -- configs/exp1_smoke.yaml configs/exp2_smoke.yaml \
        configs/exp3_smoke.yaml configs/exp4_smoke.yaml \
        configs/expA_smoke.yaml configs/expB_smoke.yaml \
        configs/expR1_boundary_smoke.yaml configs/expR2_remainder_smoke.yaml \
        configs/expR3_hierarchical_smoke.yaml configs/expR4_adult_auxiliary_smoke.yaml
fi

# Check every requested config before starting an expensive sequence.
uv run --frozen python - "$@" <<'PY'
import sys
from pathlib import Path
import yaml

needs_adult = False
for filename in sys.argv[1:]:
    config = yaml.safe_load(Path(filename).read_text())
    if config.get("profile") != "smoke":
        raise SystemExit(f"Smoke helper rejects non-smoke config: {filename}")
    needs_adult |= config["experiment_id"] in {"expA_tabular", "expR4_adult_auxiliary"}
if needs_adult:
    from scripts.prepare_adult import prepare_adult
    prepare_adult(check_only=True)
PY

export MPLBACKEND=Agg
for cfg in "$@"; do
    echo "Running $cfg"
    rundir=$(uv run --frozen python scripts/run_experiment.py --config "$cfg")
    test -d "$rundir"
    uv run --frozen python scripts/analyze_run.py "$rundir"
    echo "Completed $cfg: $rundir"
done
echo "All requested smoke profiles completed."
