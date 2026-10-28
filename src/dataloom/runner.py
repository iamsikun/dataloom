"""Generic Monte Carlo runner.

Iterates over (n, beta, rho) cells, runs R replications per cell, calls
each registered estimator, and writes one Parquet part per worker per cell.

Parallelization is over cells by default (joblib loky); replications run
sequentially inside each cell so cell-level intermediates (oracle x*,
B_eff arrays) are computed once.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from joblib import Parallel, delayed

from .dgp.gaussian import (
    GaussianParams,
    estimand_mean,
    make_synth_fn,
    sample_real,
)
from .estimators.api import EstimatorResult, get
from .io.config import RunConfig
from .io.results import ResultsWriter
from .io.seeds import make_rng, replication_seed
from .notation import classify_regime
from .oracle import oracle_grid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cell:
    n: int
    beta: float
    rho: float

    @property
    def key(self) -> tuple[int, int, int]:
        # Stable integer representation for the seed spawn_key.
        return (self.n, int(round(self.beta * 1000)), int(round(self.rho * 1000)))

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "beta": self.beta, "rho": self.rho}


def iter_cells(config: RunConfig) -> Iterable[Cell]:
    for n in config.n_grid:
        for beta in config.beta_grid:
            for rho in config.rho_grid:
                yield Cell(n=n, beta=beta, rho=rho)


def _run_one_replication(
    cell: Cell,
    rep: int,
    config: RunConfig,
    params: GaussianParams,
    truth_params: dict[str, Any],
    oracle_x: int,
    oracle_lambda: float,
    oracle_alpha: float | None,
    oracle_risk: float,
) -> list[dict[str, Any]]:
    """Run all configured estimators on a single replication and return rows."""
    seed = replication_seed(
        config.seed_root, config.experiment_id, cell.key, rep
    )
    rng = make_rng(seed)
    X = sample_real(params, rng)
    synth_fn = make_synth_fn(params)

    rows: list[dict[str, Any]] = []
    for name in config.estimators:
        est = get(name)
        est_cfg = config.estimator_configs.get(name, {})
        t0 = time.perf_counter()
        try:
            res: EstimatorResult = est(
                X, synth_fn,
                n=cell.n, rng=rng,
                truth_params=truth_params,
                estimand=estimand_mean,
                config=est_cfg,
            )
            failure = False
            reason = None
        except Exception as e:  # noqa: BLE001  - record, don't drop (§16)
            log.exception("estimator %s failed on cell %s rep %d", name, cell, rep)
            res = EstimatorResult(
                theta_hat=float("nan"),
                failure_flag=True,
                failure_reason=f"{type(e).__name__}: {e}",
            )
            failure = True
            reason = res.failure_reason
        runtime = time.perf_counter() - t0

        rows.append({
            "experiment_id": config.experiment_id,
            "replication": rep,
            "seed": seed,
            "n": cell.n,
            "m": params.m,
            "rho": cell.rho,
            "beta": cell.beta,
            "a": params.a,
            "sigma_s2": params.sigma_s2,
            "B0": params.B0,
            "c": params.c,
            "dataset": "gaussian_synthetic",
            "generator": "analytic_gaussian",
            "estimand": "mean",
            "method": name,
            "theta_true": params.theta_star,
            "theta_hat": res.theta_hat,
            "x_selected": res.x_selected,
            "alpha_selected": res.alpha_selected,
            "B_eff_selected": res.B_eff_selected,
            "V_R_selected": res.V_R_selected,
            "beta_hat": res.beta_hat,
            "c_hat": res.c_hat,
            "a_hat": res.a_hat,
            "v_hat": res.v_hat,
            "oracle_x": oracle_x,
            "oracle_lambda": oracle_lambda,
            "oracle_alpha": oracle_alpha,
            "oracle_risk": oracle_risk,
            "estimated_risk_selected": res.estimated_risk_selected,
            "true_risk_selected": _true_risk_at(res.x_selected, params),
            "safe_pass": res.safe_pass,
            "safe_margin": res.safe_margin,
            "fallback_used": res.fallback_used,
            "ci_lower": res.ci_lower,
            "ci_upper": res.ci_upper,
            "runtime_seconds": runtime,
            "failure_flag": failure,
            "failure_reason": reason,
        })
    return rows


def _true_risk_at(x: int | None, params: GaussianParams) -> float | None:
    if x is None:
        return None
    if x == 0:
        return params.a / params.n
    if x >= params.n:
        return params.v_n + params.c * (params.n ** (-2.0 * params.beta))
    b = params.v_n + params.c * (x ** (-2.0 * params.beta))
    v = params.a / (params.n - x)
    return float(v * b / (v + b))


def run_cell(
    cell: Cell,
    config: RunConfig,
    writer: ResultsWriter,
    worker_id: int = 0,
) -> int:
    """Execute all replications for one cell and write a parquet part. Returns
    the number of rows written."""
    constants = config.constants
    params = GaussianParams(
        n=cell.n, beta=cell.beta, rho=cell.rho,
        a=constants.get("a", 1.0),
        B0=constants.get("B0", 1.0),
        sigma_s2=constants.get("sigma_s2", 1.0),
        kappa=constants.get("kappa", 1.0),
        theta_star=constants.get("theta_star", 0.0),
        fast_mean=bool(constants.get("fast_mean", False)),
    )
    truth_params = {
        "a": params.a,
        "v_n": params.v_n,
        "c": params.c,
        "beta": params.beta,
        "sigma_s2": params.sigma_s2,
        "B0": params.B0,
        "m": params.m,
    }

    oracle_res = oracle_grid(
        n=cell.n, a=params.a, v_n=params.v_n, c=params.c, beta=params.beta,
    )
    oracle_x = oracle_res.x_star
    oracle_lambda = oracle_x / cell.n
    oracle_alpha = oracle_res.alpha_star
    oracle_risk = oracle_res.risk_star

    R = config.replications.for_n(cell.n)
    rows: list[dict[str, Any]] = []
    for rep in range(R):
        rows.extend(_run_one_replication(
            cell, rep, config, params, truth_params,
            oracle_x, oracle_lambda, oracle_alpha, oracle_risk,
        ))

    writer.write_cell(cell.to_dict(), rows, worker_id=worker_id)
    return len(rows)


def run_experiment(config: RunConfig, run_dir, worker_id_offset: int = 0) -> int:
    """Execute every cell in `config` and return the total row count."""
    writer = ResultsWriter(run_dir / "raw")
    cells = list(iter_cells(config))

    log.info(
        "running %s: %d cells, estimators=%s, profile=%s",
        config.experiment_id, len(cells), config.estimators, config.profile,
    )

    if config.parallel.unit == "cell" and config.parallel.n_jobs not in (0, 1):
        results = Parallel(
            n_jobs=config.parallel.n_jobs,
            backend=config.parallel.backend,
        )(
            delayed(run_cell)(cell, config, writer, worker_id=i + worker_id_offset)
            for i, cell in enumerate(cells)
        )
        return int(sum(results))

    # Serial fallback.
    total = 0
    for i, cell in enumerate(cells):
        total += run_cell(cell, config, writer, worker_id=i + worker_id_offset)
    return total
