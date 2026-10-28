"""Rebuttal Experiment R1: equality boundary and comparator provenance."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..oracle import R_profile, oracle_grid


METHODS = (
    "real_only_all",
    "synthetic_only_full_calibration",
    "product_surrogate_fixed_share_oracle_alpha",
    "corrected_oracle_gn",
)


def _allocation(
    method: str, *, n: int, a: float, v_n: float, c: float, beta: float
) -> tuple[int, float]:
    if method == "real_only_all":
        return 0, 1.0
    if method == "synthetic_only_full_calibration":
        return n, 0.0
    if method == "corrected_oracle_gn":
        result = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
        return result.x_star, result.alpha_star
    if method == "product_surrogate_fixed_share_oracle_alpha":
        share = 2.0 * beta / (1.0 + 2.0 * beta)
        x = max(1, min(n - 1, int(np.floor(share * n))))
        b = v_n + c * x ** (-2.0 * beta)
        v = a / (n - x)
        return x, float(b / (b + v))
    raise KeyError(method)


def _draw_estimate(
    rng: np.random.Generator,
    *,
    x: int,
    alpha: float,
    n: int,
    a: float,
    v_n: float,
    c: float,
    beta: float,
) -> float:
    if x == 0:
        return float(rng.normal(0.0, math.sqrt(a / n)))
    bias = math.sqrt(c) * x ** (-beta)
    theta_s = float(rng.normal(bias, math.sqrt(v_n)))
    if x == n:
        return theta_s
    theta_r = float(rng.normal(0.0, math.sqrt(a / (n - x))))
    return float(alpha * theta_r + (1.0 - alpha) * theta_s)


def run(config_path: str | Path) -> Path:
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    config = RunConfig.from_dict(raw)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")
    beta = float(raw.get("beta", 1.0))
    rho = float(raw.get("rho", beta + 0.5))
    a = float(raw.get("a", 1.0))
    c = float(raw.get("c", 1.0))
    replications = int(raw["replications"]["default"])
    methods = tuple(raw.get("estimators", METHODS))

    worker = 0
    for v0 in map(float, raw["v0_grid"]):
        for n in map(int, raw["n_grid"]):
            v_n = v0 * n ** (-rho)
            oracle = oracle_grid(n=n, a=a, v_n=v_n, c=c, beta=beta)
            rows: list[dict] = []
            for method in methods:
                x, alpha = _allocation(
                    method, n=n, a=a, v_n=v_n, c=c, beta=beta
                )
                risk = float(
                    R_profile(x, n=n, a=a, v_n=v_n, c=c, beta=beta)
                )
                for rep in range(replications):
                    seed = replication_seed(
                        config.seed_root,
                        f"{config.experiment_id}:{method}",
                        (n, int(round(v0 * 1_000_000))),
                        rep,
                    )
                    estimate = _draw_estimate(
                        make_rng(seed),
                        x=x,
                        alpha=alpha,
                        n=n,
                        a=a,
                        v_n=v_n,
                        c=c,
                        beta=beta,
                    )
                    rows.append(
                        {
                            "experiment_id": config.experiment_id,
                            "replication": rep,
                            "seed": seed,
                            "n": n,
                            "m": int(np.ceil(n**rho)),
                            "rho": rho,
                            "beta": beta,
                            "a": a,
                            "sigma_s2": v0,
                            "B0": math.sqrt(c),
                            "c": c,
                            "dataset": "analytic_gaussian",
                            "generator": "analytic_mean",
                            "estimand": "mean",
                            "method": method,
                            "theta_true": 0.0,
                            "theta_hat": estimate,
                            "x_selected": x,
                            "alpha_selected": alpha,
                            "oracle_x": oracle.x_star,
                            "oracle_lambda": oracle.x_star / n,
                            "oracle_alpha": oracle.alpha_star,
                            "oracle_risk": oracle.risk_star,
                            "true_risk_selected": risk,
                            "v0": v0,
                            "failure_flag": False,
                        }
                    )
            writer.write_cell(
                {"n": n, "v0": f"{v0:.12g}"}, rows, worker_id=worker
            )
            worker += 1
    return run_dir
