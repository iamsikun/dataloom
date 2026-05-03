"""Aggregate raw parquet, build tables, and render figures for one run.

Usage:
    uv run python scripts/analyze_run.py results/exp1_phase_diagram__abc__...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one experiment run.")
    parser.add_argument("run_dir", type=Path,
                        help="Path to results/{run_id}/ directory.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.run_dir.exists():
        print(f"run_dir not found: {args.run_dir}", file=sys.stderr)
        return 1

    name = args.run_dir.name
    if "exp1_phase_diagram" in name:
        from dataloom.analysis.aggregate import aggregate_run
        from dataloom.analysis.figures import render_exp1_figures
        out = aggregate_run(args.run_dir)
        render_exp1_figures(args.run_dir, out["raw"], out["table_s1"], out["table_s2"])
        print(f"raw rows={len(out['raw'])}, "
              f"table_s1={out['table_s1'].shape}, "
              f"table_s2={out['table_s2'].shape}")
    elif "exp2_adaptive" in name:
        from dataloom.analysis.exp2 import aggregate_run_exp2
        out = aggregate_run_exp2(args.run_dir)
        print(f"raw rows={len(out['raw'])}, "
              f"table_a1={getattr(out.get('table_a1'), 'shape', None)}, "
              f"table_a2={out['table_a2'].shape}")
    elif "exp3_multichannel" in name:
        from dataloom.analysis.exp3 import aggregate_run_exp3
        out = aggregate_run_exp3(args.run_dir)
        print(f"raw rows={len(out['raw'])}, table_m1={out['table_m1'].shape}")
    elif "exp4_inference" in name:
        from dataloom.analysis.exp4 import aggregate_run_exp4
        out = aggregate_run_exp4(args.run_dir)
        print(f"raw rows={len(out['raw'])}, table_c1={out['table_c1'].shape}")
    elif "expA_tabular" in name:
        from dataloom.analysis.exp_semisynth import aggregate_expA
        out = aggregate_expA(args.run_dir)
        print(f"raw rows={len(out['raw'])}, table_t2={out['table_t2'].shape}")
    elif "expB_causal" in name:
        from dataloom.analysis.exp_semisynth import aggregate_expB
        out = aggregate_expB(args.run_dir)
        print(f"raw rows={len(out['raw'])}, table_d1={out['table_d1'].shape}")
    else:
        print(f"unknown experiment kind for run dir: {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
