"""Experiment 4: inference and coverage (§7)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..io.config import RunConfig, load_run_config, prepare_run_dir
from ..runner import run_experiment

log = logging.getLogger(__name__)


def run(config_path: str | Path) -> Path:
    config: RunConfig = load_run_config(config_path)
    if config.experiment_id != "exp4_inference":
        log.warning("config experiment_id=%s (expected exp4_inference)",
                    config.experiment_id)
    run_dir = prepare_run_dir(config, config_path)
    log.info("run_dir=%s", run_dir)
    n_rows = run_experiment(config, run_dir)
    log.info("wrote %d rows to %s/raw", n_rows, run_dir)
    return run_dir
