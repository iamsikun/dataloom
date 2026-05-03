"""Experiment 1: phase diagram validation (docs/experiments.md §4).

Estimators tested (§4.5):
    real_only_all, synthetic_only_full_calibration, naive_pooling,
    fixed_half_split_oracle_alpha, old_fixed_share_oracle_alpha,
    corrected_oracle_gn, safe_corrected_oracle_gn.

Note: §2.3 #1 `synthetic_only_oracle_x` is intentionally omitted — under
the §1 risk model its definition collapses to `synthetic_only_full_calibration`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..io.config import RunConfig, load_run_config, prepare_run_dir
from ..runner import run_experiment

log = logging.getLogger(__name__)


def run(config_path: str | Path) -> Path:
    """Load config, prepare run directory, execute Monte Carlo."""
    config: RunConfig = load_run_config(config_path)
    if config.experiment_id != "exp1_phase_diagram":
        log.warning(
            "config experiment_id=%s (expected exp1_phase_diagram)",
            config.experiment_id,
        )
    run_dir = prepare_run_dir(config, config_path)
    log.info("run_dir=%s", run_dir)
    n_rows = run_experiment(config, run_dir)
    log.info("wrote %d rows to %s/raw", n_rows, run_dir)
    return run_dir
