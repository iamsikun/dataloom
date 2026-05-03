"""Generic experiment runner.

Usage:
    uv run python scripts/run_experiment.py --config configs/exp1_smoke.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path


EXP_DISPATCH = {
    "exp1_phase_diagram": "dataloom.experiments.exp1_phase_diagram",
    "exp2_adaptive": "dataloom.experiments.exp2_adaptive",
    "exp3_multichannel": "dataloom.experiments.exp3_multichannel",
    "exp4_inference": "dataloom.experiments.exp4_inference",
    "expA_tabular": "dataloom.experiments.expA_tabular",
    "expB_causal": "dataloom.experiments.expB_causal",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a dataloom experiment.")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to YAML config file.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    exp_id = raw["experiment_id"]

    if exp_id not in EXP_DISPATCH:
        print(f"Unknown experiment_id: {exp_id}", file=sys.stderr)
        print(f"Known: {sorted(EXP_DISPATCH)}", file=sys.stderr)
        return 2

    mod = importlib.import_module(EXP_DISPATCH[exp_id])
    run_dir = mod.run(args.config)
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
