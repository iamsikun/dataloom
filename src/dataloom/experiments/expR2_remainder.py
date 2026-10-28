"""Rebuttal Experiment R2: remainder and non-power-law sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..oracle import oracle_grid_curve


def _curve(family: str, *, n: int, q: float):
    if family == "exact":
        return lambda x: n**-1 + np.asarray(x, dtype=float) ** -2
    if family == "positive_remainder":
        return lambda x: (
            n**-1
            + np.asarray(x, dtype=float) ** -2
            + q * np.asarray(x, dtype=float) ** -2.5
        )
    if family == "negative_remainder":
        return lambda x: (
            n**-1
            + np.asarray(x, dtype=float) ** -2
            + q * np.asarray(x, dtype=float) ** -2.5
        )
    if family == "oscillatory_remainder":
        return lambda x: (
            n**-1
            + np.asarray(x, dtype=float) ** -2
            + q
            * np.sin(3.0 * np.log(np.asarray(x, dtype=float)))
            * np.asarray(x, dtype=float) ** -2.5
        )
    if family == "bounded_remainder":
        return lambda x: (
            n**-1
            + 0.1 * np.asarray(x, dtype=float) ** -0.5
            + q * np.asarray(x, dtype=float) ** -1
        )
    if family == "plateau":
        return lambda x: (
            n**-1 + 1e-3 + np.asarray(x, dtype=float) ** -2
        )
    raise KeyError(family)


def run(config_path: str | Path) -> Path:
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    config = RunConfig.from_dict(raw)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")
    rows: list[dict] = []
    cases = [
        ("exact", 0.0, 1.0, 1.0),
        ("positive_remainder", 0.5, 1.0, 1.0),
        ("negative_remainder", -0.5, 1.0, 1.0),
        ("oscillatory_remainder", 0.5, 1.0, 1.0),
    ]
    for family, q, beta, rho in cases:
        for n in map(int, raw["diverging_n_grid"]):
            curve = _curve(family, n=n, q=q)
            result = oracle_grid_curve(n=n, a=1.0, B_curve=curve)
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "replication": 0,
                    "n": n,
                    "m": n,
                    "rho": rho,
                    "beta": beta,
                    "a": 1.0,
                    "c": 1.0,
                    "dataset": "deterministic_working_risk",
                    "generator": "analytic_curve",
                    "estimand": "mean",
                    "method": "curve_oracle",
                    "x_selected": result.x_star,
                    "lambda_selected": result.x_star / n,
                    "alpha_selected": result.alpha_star,
                    "true_risk_selected": result.risk_star,
                    "B_true": result.B_eff_star,
                    "curve_family": family,
                    "remainder_scale": q,
                    "failure_flag": False,
                }
            )

    for q in map(float, raw["bounded_q_grid"]):
        for n in map(int, raw["bounded_n_grid"]):
            curve = _curve("bounded_remainder", n=n, q=q)
            result = oracle_grid_curve(n=n, a=1.0, B_curve=curve)
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "replication": 0,
                    "n": n,
                    "m": n,
                    "rho": 1.0,
                    "beta": 0.25,
                    "a": 1.0,
                    "c": 0.1,
                    "dataset": "deterministic_working_risk",
                    "generator": "analytic_curve",
                    "estimand": "mean",
                    "method": "curve_oracle",
                    "x_selected": result.x_star,
                    "lambda_selected": result.x_star / n,
                    "alpha_selected": result.alpha_star,
                    "true_risk_selected": result.risk_star,
                    "B_true": result.B_eff_star,
                    "curve_family": "bounded_remainder",
                    "remainder_scale": q,
                    "failure_flag": False,
                }
            )

    for n in map(int, raw["diverging_n_grid"]):
        curve = _curve("plateau", n=n, q=0.0)
        result = oracle_grid_curve(n=n, a=1.0, B_curve=curve)
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "replication": 0,
                "n": n,
                "m": n,
                "rho": 1.0,
                "beta": 1.0,
                "a": 1.0,
                "c": 1.0,
                "dataset": "deterministic_working_risk",
                "generator": "analytic_curve",
                "estimand": "mean",
                "method": "curve_oracle",
                "x_selected": result.x_star,
                "lambda_selected": result.x_star / n,
                "alpha_selected": result.alpha_star,
                "true_risk_selected": result.risk_star,
                "B_true": result.B_eff_star,
                "curve_family": "plateau",
                "remainder_scale": 1e-3,
                "failure_flag": False,
            }
        )
    writer.write_cell({"kind": "all"}, rows)
    return run_dir
