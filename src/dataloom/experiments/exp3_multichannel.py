"""Experiment 3: multichannel allocation (§6).

Has its own runner because the allocation is a 2-tuple, not a scalar x.
The output schema still follows §11; (x_1, x_2) are stored in the row's
extras-derived columns x1, x2 (added to the row dict alongside x_selected
which holds x_1 + x_2).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from joblib import Parallel, delayed

from ..dgp.multichannel import (
    GaussianTwoChannelParams,
    make_synth_fn_2,
    sample_real,
)
from ..dgp.gaussian import estimand_mean
from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..multichannel.estimators import METHODS_2
from ..multichannel.oracle import oracle_grid_2

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCellMC:
    n: int
    beta1: float
    beta2: float
    rho: float

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.n,
            int(round(self.beta1 * 1000)),
            int(round(self.beta2 * 1000)),
            int(round(self.rho * 1000)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "beta1": self.beta1,
            "beta2": self.beta2, "rho": self.rho,
        }


def load_run_config_mc(path: str | Path) -> tuple[RunConfig, dict[str, Any]]:
    """Multichannel needs slightly different schema (beta1/beta2 grids)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Reuse the standard RunConfig for the shared fields, set beta_grid to a
    # placeholder for type compat.
    raw_for_run = dict(raw)
    # The standard RunConfig expects beta_grid; for multichannel we use beta_pairs.
    if "beta_grid" not in raw_for_run:
        first_pair = raw["beta_pairs"][0]
        raw_for_run["beta_grid"] = [float(first_pair[0])]
    raw_for_run.setdefault("rho_grid", raw.get("rho_grid", [1.0]))
    raw_for_run.setdefault("n_grid", raw["n_grid"])
    cfg = RunConfig.from_dict(raw_for_run)
    return cfg, raw


def run(config_path: str | Path) -> Path:
    config, raw = load_run_config_mc(config_path)
    if config.experiment_id != "exp3_multichannel":
        log.warning("config experiment_id=%s (expected exp3_multichannel)",
                    config.experiment_id)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")

    cells: list[MCellMC] = []
    for n in raw["n_grid"]:
        for pair in raw["beta_pairs"]:
            for rho in raw["rho_grid"]:
                cells.append(MCellMC(n=int(n), beta1=float(pair[0]),
                                     beta2=float(pair[1]), rho=float(rho)))

    methods = raw["estimators"]

    log.info("running exp3_multichannel: %d cells, methods=%s, profile=%s",
             len(cells), methods, raw.get("profile", "smoke"))

    def run_cell(cell: MCellMC, worker_id: int) -> int:
        params = GaussianTwoChannelParams(
            n=cell.n, beta1=cell.beta1, beta2=cell.beta2, rho=cell.rho,
            a=config.constants.get("a", 1.0),
            B0_1=config.constants.get("B0_1", 1.0),
            B0_2=config.constants.get("B0_2", 1.0),
            sigma_s2=config.constants.get("sigma_s2", 1.0),
            kappa=config.constants.get("kappa", 1.0),
            theta_star=config.constants.get("theta_star", 0.0),
        )
        truth = {
            "a": params.a, "v_n": params.v_n,
            "c1": params.c1, "beta1": params.beta1,
            "c2": params.c2, "beta2": params.beta2,
        }
        oracle_res = oracle_grid_2(
            n=cell.n, a=params.a, v_n=params.v_n,
            c1=params.c1, beta1=params.beta1,
            c2=params.c2, beta2=params.beta2,
            coarse_step=max(1, cell.n // 200),
        )
        R = config.replications.for_n(cell.n)
        rows: list[dict[str, Any]] = []
        for rep in range(R):
            seed = replication_seed(
                config.seed_root, config.experiment_id, cell.key, rep,
            )
            rng = make_rng(seed)
            X = sample_real(params, rng)
            synth_fn_2 = make_synth_fn_2(params)
            for name in methods:
                fn = METHODS_2[name]
                t0 = time.perf_counter()
                try:
                    res = fn(X, synth_fn_2,
                             n=cell.n, rng=rng,
                             params=truth, estimand=estimand_mean)
                    fail = False
                    reason = None
                except Exception as e:  # noqa: BLE001
                    log.exception("estimator %s failed", name)
                    res = None
                    fail = True
                    reason = f"{type(e).__name__}: {e}"
                runtime = time.perf_counter() - t0
                if res is None:
                    rows.append({
                        "experiment_id": config.experiment_id,
                        "replication": rep, "seed": seed,
                        "n": cell.n, "m": params.m, "rho": cell.rho,
                        "beta": cell.beta1, "method": name,
                        "estimand": "mean", "theta_true": params.theta_star,
                        "theta_hat": float("nan"),
                        "failure_flag": True, "failure_reason": reason,
                    })
                    continue
                ex = res.extras or {}
                rows.append({
                    "experiment_id": config.experiment_id,
                    "replication": rep, "seed": seed,
                    "n": cell.n, "m": params.m, "rho": cell.rho,
                    "beta": cell.beta1,  # store beta1 in beta column for partition
                    "a": params.a, "sigma_s2": params.sigma_s2,
                    "B0": params.B0_1, "c": params.c1,
                    "dataset": "gaussian_two_channel",
                    "generator": "analytic_gaussian",
                    "estimand": "mean", "method": name,
                    "theta_true": params.theta_star,
                    "theta_hat": res.theta_hat,
                    "x_selected": res.x_selected,
                    "alpha_selected": res.alpha_selected,
                    "B_eff_selected": res.B_eff_selected,
                    "V_R_selected": res.V_R_selected,
                    "oracle_x": oracle_res.x1_star + oracle_res.x2_star,
                    "oracle_alpha": oracle_res.alpha_star,
                    "oracle_risk": oracle_res.risk_star,
                    "estimated_risk_selected": res.estimated_risk_selected,
                    "safe_pass": res.safe_pass,
                    "safe_margin": res.safe_margin,
                    "fallback_used": res.fallback_used,
                    "runtime_seconds": runtime,
                    "failure_flag": fail, "failure_reason": reason,
                    # MC-specific extras get stored as JSON in failure_reason for now
                    # (can be extended; outside §11 master schema).
                })
        # Write per-cell.
        cell_dict = {**cell.to_dict(),
                     "x1_star": oracle_res.x1_star,
                     "x2_star": oracle_res.x2_star}
        # Store oracle (x1*, x2*) alongside the partition dir for later analysis.
        writer.write_cell({"n": cell.n, "beta1": cell.beta1,
                           "beta2": cell.beta2, "rho": cell.rho},
                          rows, worker_id=worker_id)
        # Side-car JSON with the full oracle decomposition (x1*, x2*).
        cell_meta_dir = writer.raw_dir / _cell_dir_name_mc(cell)
        with open(cell_meta_dir / "oracle.json", "w") as f:
            json.dump({
                "x1_star": oracle_res.x1_star,
                "x2_star": oracle_res.x2_star,
                "risk_star": oracle_res.risk_star,
                "mv1": oracle_res.mv1,
                "mv2": oracle_res.mv2,
            }, f, indent=2)
        return len(rows)

    if config.parallel.unit == "cell" and config.parallel.n_jobs not in (0, 1):
        totals = Parallel(
            n_jobs=config.parallel.n_jobs,
            backend=config.parallel.backend,
        )(delayed(run_cell)(c, i) for i, c in enumerate(cells))
        n_rows = int(sum(totals))
    else:
        n_rows = sum(run_cell(c, i) for i, c in enumerate(cells))

    log.info("wrote %d rows to %s/raw", n_rows, run_dir)
    return run_dir


def _cell_dir_name_mc(cell: MCellMC) -> str:
    d = {"n": cell.n, "beta1": cell.beta1, "beta2": cell.beta2, "rho": cell.rho}
    parts = [f"{k}={d[k]}" for k in sorted(d)]
    return "cell=" + "_".join(parts)
