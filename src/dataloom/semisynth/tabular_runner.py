"""Experiment A tabular semi-synthetic runner (§8).

Per-replication flow:
1. Sample n rows from the pseudo-population without replacement.
2. Reserve a validation subset of size n_v.
3. For each estimator, run a method-specific protocol (with adaptive bias
   estimation when the estimator is in the adaptive family).
4. Compute the estimand on the truth (whole pseudo-population), real subset,
   and synthetic samples; combine with the corrected MSE-optimal alpha.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from ..adaptive.bias_curve import default_pilot_grid
from ..datasets.adult import ESTIMANDS as ADULT_ESTIMANDS, load_adult
from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..oracle import oracle_grid
from .generators import get_generator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TCell:
    n: int
    m_factor: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.n, self.m_factor)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "m_factor": self.m_factor}


def _adaptive_pipeline(
    *,
    df_real: pd.DataFrame,
    estimand: Callable[[pd.DataFrame], float],
    generator_name: str,
    m: int,
    pilot_grid: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Estimate the bias curve and pick x̂ via grid minimization of R̂_n(x)."""
    n = len(df_real)
    n_v = max(1, min(n // 5, n // 4))
    val = df_real.iloc[n - n_v :]
    rest = df_real.iloc[: n - n_v]
    theta_R_V = estimand(val)
    a_hat = max(float(val.var(numeric_only=True).mean()) if len(val) >= 2 else 1.0, 1e-12)
    var_R_V = a_hat / max(n_v, 1)

    bias2 = np.empty(len(pilot_grid))
    sigma_s2_estimates = []
    for i, x_j in enumerate(pilot_grid):
        if x_j > len(rest):
            bias2[i] = 0.0
            sigma_s2_estimates.append(a_hat)
            continue
        cal = rest.iloc[: int(x_j)]
        gen = get_generator(generator_name).fit(cal)
        Z = gen.sample(m, rng)
        try:
            theta_S = float(estimand(Z))
        except Exception:  # noqa: BLE001
            theta_S = float("nan")
        if not np.isfinite(theta_S):
            bias2[i] = 0.0
            sigma_s2_estimates.append(a_hat)
            continue
        s2 = max(float(Z.var(numeric_only=True).mean()) if len(Z) >= 2 else a_hat, 1e-12)
        sigma_s2_estimates.append(s2)
        var_S_hat = s2 / m
        diff2 = (theta_S - theta_R_V) ** 2
        bias2[i] = max(diff2 - var_S_hat - var_R_V, 0.0)

    sigma_s2_hat = float(np.median(sigma_s2_estimates))
    v_hat = sigma_s2_hat / m

    pos = bias2 > 0
    fit_reliable = True
    if pos.sum() >= 3:
        log_x = np.log(pilot_grid[pos].astype(float))
        log_b2 = np.log(bias2[pos])
        slope, intercept = np.polyfit(log_x, log_b2, 1)
        beta_hat = float(-slope / 2.0)
        c_hat = float(np.exp(intercept))
        if not np.isfinite(beta_hat) or not np.isfinite(c_hat) or beta_hat <= 0.60:
            fit_reliable = False
    else:
        beta_hat = 0.0
        c_hat = max(a_hat, float(bias2.max()), 1e-12)
        fit_reliable = False

    n_eff = n - n_v
    grid = np.unique(np.concatenate([
        [0],
        np.arange(int(max(1, pilot_grid.min())), n_eff, dtype=int),
        [n_eff],
    ]))
    res = oracle_grid(n=n_eff, a=a_hat, v_n=v_hat,
                      c=c_hat, beta=max(beta_hat, 1e-3),
                      grid=grid, include_boundaries=False)
    return {
        "n_eff": n_eff, "n_v": n_v,
        "x_hat": res.x_star, "alpha_hat": res.alpha_star,
        "B_eff_hat": res.B_eff_star or float("inf"),
        "V_R_hat": res.V_R_star or (a_hat / n_eff),
        "beta_hat": beta_hat, "c_hat": c_hat,
        "a_hat": a_hat, "v_hat": v_hat,
        "estimated_risk_selected": res.risk_star,
        "fit_reliable": fit_reliable,
    }


def _combine_real_synth(
    *,
    df_real: pd.DataFrame,
    n_eff: int,
    x_hat: int,
    alpha: float,
    estimand: Callable[[pd.DataFrame], float],
    generator_name: str,
    m: int,
    rng: np.random.Generator,
) -> float:
    """Combine real-estimation and synthetic estimators with the given alpha."""
    if x_hat <= 0:
        return float(estimand(df_real))
    rest = df_real.iloc[:n_eff]
    perm = rng.permutation(n_eff)
    cal_idx = perm[:x_hat]
    est_idx = perm[x_hat:]
    cal = rest.iloc[cal_idx]
    est = rest.iloc[est_idx]
    if len(est) == 0:
        # Synthetic-only.
        gen = get_generator(generator_name).fit(cal)
        Z = gen.sample(m, rng)
        return float(estimand(Z))
    theta_R = float(estimand(est))
    gen = get_generator(generator_name).fit(cal)
    Z = gen.sample(m, rng)
    theta_S = float(estimand(Z))
    return alpha * theta_R + (1.0 - alpha) * theta_S


def _run_one_estimator(
    *,
    name: str,
    df_real: pd.DataFrame,
    truth: float,
    estimand: Callable[[pd.DataFrame], float],
    generator_name: str,
    m: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one tabular estimator. Returns a dict slotting into the master schema."""
    n = len(df_real)
    if name == "real_only_all":
        theta_hat = float(estimand(df_real))
        return {
            "method": name, "theta_hat": theta_hat, "x_selected": 0,
            "alpha_selected": 1.0, "fallback_used": False,
        }
    if name == "synthetic_only_full_calibration":
        gen = get_generator(generator_name).fit(df_real)
        Z = gen.sample(m, rng)
        theta_hat = float(estimand(Z))
        return {"method": name, "theta_hat": theta_hat,
                "x_selected": n, "alpha_selected": 0.0, "fallback_used": False}
    if name == "naive_pooling":
        x = n // 2
        cal = df_real.iloc[:x]
        est = df_real.iloc[x:]
        gen = get_generator(generator_name).fit(cal)
        Z = gen.sample(m, rng)
        # Pool real-est + synth, both as sample units. For non-mean estimands
        # this is the locked-in stacked-refit convention.
        # We must make Z's columns subset of real columns.
        common = [c for c in df_real.columns if c in Z.columns]
        pooled = pd.concat([est[common], Z[common]], ignore_index=True)
        theta_hat = float(estimand(pooled))
        return {"method": name, "theta_hat": theta_hat,
                "x_selected": x, "alpha_selected": len(est) / (len(est) + len(Z)),
                "fallback_used": False}

    if name in ("fixed_half_split_plugin_alpha", "old_fixed_share_plugin_alpha",
                "corrected_adaptive_gn", "safe_corrected_adaptive_gn",
                "validation_debiased_gn", "empirical_oracle_grid"):
        pilot_grid = default_pilot_grid(n)
        if pilot_grid.size == 0 or len(pilot_grid) < 2:
            return {"method": name, "theta_hat": float("nan"),
                    "failure_flag": True, "failure_reason": "pilot grid empty"}

        sel = _adaptive_pipeline(
            df_real=df_real, estimand=estimand, generator_name=generator_name,
            m=m, pilot_grid=pilot_grid, rng=rng,
        )
        x_hat = sel["x_hat"]
        alpha = sel["alpha_hat"]
        n_eff = sel["n_eff"]

        if name in ("corrected_adaptive_gn", "safe_corrected_adaptive_gn",
                    "validation_debiased_gn") and not sel["fit_reliable"]:
            return {
                "method": name, "theta_hat": float(estimand(df_real)),
                "x_selected": 0, "alpha_selected": 1.0,
                "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
                "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
                "fallback_used": True,
            }

        if name == "fixed_half_split_plugin_alpha":
            x_hat = max(1, n // 2)
            x_hat = min(x_hat, n_eff - 1)
            b = sel["v_hat"] + sel["c_hat"] * (x_hat ** (-2.0 * sel["beta_hat"]))
            v = sel["a_hat"] / max(n_eff - x_hat, 1)
            alpha = b / (v + b)
        elif name == "old_fixed_share_plugin_alpha":
            beta_hat = max(sel["beta_hat"], 1e-3)
            lam = (2.0 * beta_hat) / (1.0 + 2.0 * beta_hat)
            x_hat = max(1, min(n_eff - 1, int(np.floor(n_eff * lam))))
            b = sel["v_hat"] + sel["c_hat"] * (x_hat ** (-2.0 * beta_hat))
            v = sel["a_hat"] / max(n_eff - x_hat, 1)
            alpha = b / (v + b)
        elif name == "safe_corrected_adaptive_gn":
            b_at_x = (sel["v_hat"] +
                      sel["c_hat"] * (x_hat ** (-2.0 * max(sel["beta_hat"], 1e-3)))
                      if x_hat > 0 else float("inf"))
            if x_hat <= 0 or x_hat * b_at_x >= sel["a_hat"]:
                return {
                    "method": name, "theta_hat": float(estimand(df_real)),
                    "x_selected": 0, "alpha_selected": 1.0,
                    "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
                    "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
                    "fallback_used": True,
                }
        elif name == "validation_debiased_gn":
            n_v = sel["n_v"]
            val = df_real.iloc[n_eff:]
            rest = df_real.iloc[:n_eff]
            cal = rest.iloc[:x_hat] if x_hat > 0 else rest
            est = rest.iloc[x_hat:] if x_hat > 0 else pd.DataFrame()
            gen = get_generator(generator_name).fit(cal)
            Z = gen.sample(m, rng)
            Z_v = gen.sample(m, rng)
            theta_R = float(estimand(est)) if len(est) else float(estimand(rest))
            theta_R_V = float(estimand(val))
            theta_S = float(estimand(Z))
            theta_S_V = float(estimand(Z_v))
            bias_hat = theta_S_V - theta_R_V
            theta_S_tilde = theta_S - bias_hat
            theta_hat = alpha * theta_R + (1.0 - alpha) * theta_S_tilde
            return {
                "method": name, "theta_hat": theta_hat,
                "x_selected": x_hat, "alpha_selected": alpha,
                "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
                "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
                "fallback_used": False,
            }
        elif name == "empirical_oracle_grid":
            # Skip in first pass; would require expensive sweeping.
            return {"method": name, "theta_hat": float("nan"),
                    "failure_flag": True, "failure_reason": "empirical_oracle_grid not implemented in first pass"}

        theta_hat = _combine_real_synth(
            df_real=df_real, n_eff=n_eff, x_hat=x_hat, alpha=alpha,
            estimand=estimand, generator_name=generator_name,
            m=m, rng=rng,
        )
        return {
            "method": name, "theta_hat": theta_hat,
            "x_selected": x_hat, "alpha_selected": alpha,
            "B_eff_selected": sel["B_eff_hat"], "V_R_selected": sel["V_R_hat"],
            "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
            "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
            "estimated_risk_selected": sel["estimated_risk_selected"],
            "fallback_used": False,
        }

    return {"method": name, "theta_hat": float("nan"),
            "failure_flag": True,
            "failure_reason": f"unknown estimator '{name}' for tabular runner"}


def run(config_path: str | Path) -> Path:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw_for_run = dict(raw)
    raw_for_run.setdefault("beta_grid", [1.0])
    raw_for_run.setdefault("rho_grid", [1.0])
    config = RunConfig.from_dict(raw_for_run)
    if config.experiment_id != "expA_tabular":
        log.warning("config experiment_id=%s (expected expA_tabular)",
                    config.experiment_id)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")

    # Load pseudo-population.
    df_pop = load_adult()
    if "income_binary" not in df_pop.columns:
        raise RuntimeError("loaded adult dataset missing 'income_binary'")
    estimands_to_run = raw.get("estimands", ["income_mean"])
    truth_by_estimand = {
        e: float(ADULT_ESTIMANDS[e](df_pop)) for e in estimands_to_run
    }
    log.info("pseudo-population N=%d, truths=%s", len(df_pop), truth_by_estimand)

    cells = [TCell(n=int(n), m_factor=int(mf))
             for n in raw["n_grid"] for mf in raw.get("m_factor_grid", [1])]
    methods = raw["estimators"]
    generator_name = raw.get("generator", "bootstrap_smoothed")

    def run_cell(cell: TCell, worker_id: int) -> int:
        m = cell.n * cell.m_factor
        R = config.replications.for_n(cell.n)
        rows: list[dict[str, Any]] = []
        for rep in range(R):
            seed = replication_seed(
                config.seed_root, config.experiment_id, cell.key, rep,
            )
            rng = make_rng(seed)
            idx = rng.choice(len(df_pop), size=cell.n, replace=False)
            df_real = df_pop.iloc[idx].reset_index(drop=True)

            for estimand_name in estimands_to_run:
                estimand = ADULT_ESTIMANDS[estimand_name]
                truth = truth_by_estimand[estimand_name]
                for name in methods:
                    t0 = time.perf_counter()
                    try:
                        out = _run_one_estimator(
                            name=name, df_real=df_real, truth=truth,
                            estimand=estimand, generator_name=generator_name,
                            m=m, rng=rng,
                        )
                        fail = bool(out.get("failure_flag", False))
                        reason = out.get("failure_reason")
                    except Exception as e:  # noqa: BLE001
                        log.exception("estimator %s failed", name)
                        out = {"method": name, "theta_hat": float("nan")}
                        fail = True
                        reason = f"{type(e).__name__}: {e}"
                    runtime = time.perf_counter() - t0
                    base = {
                        "experiment_id": config.experiment_id,
                        "replication": rep, "seed": seed,
                        "n": cell.n, "m": m,
                        "dataset": "adult", "generator": generator_name,
                        "estimand": estimand_name, "theta_true": truth,
                        "runtime_seconds": runtime,
                        "failure_flag": fail, "failure_reason": reason,
                    }
                    base.update(out)
                    rows.append(base)
        writer.write_cell(cell.to_dict(), rows, worker_id=worker_id)
        return len(rows)

    log.info("running expA_tabular: %d cells, methods=%s, generator=%s",
             len(cells), methods, generator_name)

    if config.parallel.unit in ("cell",) and config.parallel.n_jobs not in (0, 1):
        totals = Parallel(
            n_jobs=config.parallel.n_jobs,
            backend=config.parallel.backend,
        )(delayed(run_cell)(c, i) for i, c in enumerate(cells))
        total = int(sum(totals))
    else:
        total = sum(run_cell(c, i) for i, c in enumerate(cells))

    log.info("wrote %d rows to %s/raw", total, run_dir)
    return run_dir
