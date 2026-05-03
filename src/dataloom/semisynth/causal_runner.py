"""Experiment B causal semi-synthetic runner (§9).

Uses the IHDP-style dataset from datasets/ihdp.py. Real ATE estimator:
difference-in-means with covariate adjustment via plug-in linear outcome
model on each treatment arm (a simplified AIPW that avoids cross-fitting
overhead in the first pass).

The synthetic generator replays observed (X, A) pairs and *imputes* a
counterfactual outcome from a calibrated outcome model. Calibration size x
controls the size of the training set used to fit the outcome model.

Estimators implemented for first pass:
    real_only_diff_in_means, real_only_aipw, synthetic_only_full_calibration,
    naive_pooling, fixed_half_split_plugin_alpha, old_fixed_share_plugin_alpha,
    corrected_adaptive_gn, safe_corrected_adaptive_gn.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from ..adaptive.bias_curve import default_pilot_grid
from ..datasets.ihdp import IHDPSpec, make_ihdp, true_ate
from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..oracle import oracle_grid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CCell:
    n: int
    m_factor: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.n, self.m_factor)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "m_factor": self.m_factor}


COVS = ["x1", "x2", "x3", "x4", "x5", "x6", "z1", "z2", "z3"]


def _ate_diff_in_means(df: pd.DataFrame) -> float:
    treated = df[df["A"] == 1]["Y"]
    control = df[df["A"] == 0]["Y"]
    if len(treated) == 0 or len(control) == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def _fit_outcome_model(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit separate OLS models for treated and control. Returns (beta1, beta0)."""
    cols = ["intercept"] + COVS
    df_aug = df.copy()
    df_aug["intercept"] = 1.0
    treated = df_aug[df_aug["A"] == 1]
    control = df_aug[df_aug["A"] == 0]
    if len(treated) < len(cols) + 1 or len(control) < len(cols) + 1:
        return None, None
    X1 = treated[cols].to_numpy(dtype=float)
    y1 = treated["Y"].to_numpy(dtype=float)
    X0 = control[cols].to_numpy(dtype=float)
    y0 = control["Y"].to_numpy(dtype=float)
    beta1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    beta0, *_ = np.linalg.lstsq(X0, y0, rcond=None)
    return beta1, beta0


def _ate_aipw(df: pd.DataFrame) -> float:
    """Plug-in AIPW with linear outcome models (no cross-fitting)."""
    beta1, beta0 = _fit_outcome_model(df)
    if beta1 is None:
        return _ate_diff_in_means(df)
    cols = ["intercept"] + COVS
    df_aug = df.copy()
    df_aug["intercept"] = 1.0
    Xall = df_aug[cols].to_numpy(dtype=float)
    mu1 = Xall @ beta1
    mu0 = Xall @ beta0
    # Logistic propensity via simple plug-in.
    A = df_aug["A"].to_numpy(dtype=float)
    Y = df_aug["Y"].to_numpy(dtype=float)
    p_hat = float(A.mean())
    e = np.full_like(A, np.clip(p_hat, 0.05, 0.95))
    aug = mu1 - mu0 + A * (Y - mu1) / e - (1 - A) * (Y - mu0) / (1 - e)
    return float(aug.mean())


def _synthetic_ate(
    cal: pd.DataFrame, m: int, rng: np.random.Generator
) -> float:
    """Use calibration data to fit the outcome model, then bootstrap m rows
    from the calibration covariate distribution and impute counterfactuals."""
    beta1, beta0 = _fit_outcome_model(cal)
    if beta1 is None:
        return _ate_diff_in_means(cal)
    cols = ["intercept"] + COVS
    bs_idx = rng.integers(0, len(cal), size=m)
    df_bs = cal.iloc[bs_idx].copy().reset_index(drop=True)
    df_bs["intercept"] = 1.0
    Xbs = df_bs[cols].to_numpy(dtype=float)
    Y1 = Xbs @ beta1 + rng.normal(0, 1, size=m)
    Y0 = Xbs @ beta0 + rng.normal(0, 1, size=m)
    return float((Y1 - Y0).mean())


def _adaptive_pipeline_causal(
    *, df_real: pd.DataFrame,
    m: int,
    pilot_grid: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(df_real)
    n_v = max(2, n // 5)
    val = df_real.iloc[n - n_v :]
    rest = df_real.iloc[: n - n_v]
    theta_R_V = _ate_aipw(val)
    a_hat = max(float(val["Y"].var(ddof=1)), 1e-6)
    var_R_V = a_hat / max(n_v, 1)

    bias2 = np.empty(len(pilot_grid))
    sigma_estimates = []
    for i, x_j in enumerate(pilot_grid):
        if x_j > len(rest):
            bias2[i] = 0.0
            sigma_estimates.append(a_hat)
            continue
        cal = rest.iloc[: int(x_j)]
        try:
            theta_S = _synthetic_ate(cal, m, rng)
        except Exception:  # noqa: BLE001
            bias2[i] = 0.0
            sigma_estimates.append(a_hat)
            continue
        # Approximate var(theta_S) from variance of imputed differences.
        sigma_estimates.append(a_hat)
        var_S_hat = a_hat / m
        diff2 = (theta_S - theta_R_V) ** 2
        bias2[i] = max(diff2 - var_S_hat - var_R_V, 0.0)

    sigma_s2_hat = float(np.median(sigma_estimates))
    v_hat = sigma_s2_hat / m
    pos = bias2 > 0
    if pos.sum() >= 2:
        log_x = np.log(pilot_grid[pos].astype(float))
        log_b2 = np.log(bias2[pos])
        slope, intercept = np.polyfit(log_x, log_b2, 1)
        beta_hat = float(-slope / 2.0)
        c_hat = float(np.exp(intercept))
    else:
        beta_hat = 5.0
        c_hat = max(float(bias2.max()), 1e-12)

    n_eff = n - n_v
    res = oracle_grid(n=n_eff, a=a_hat, v_n=v_hat,
                      c=c_hat, beta=max(beta_hat, 1e-3))
    return {
        "n_eff": n_eff, "n_v": n_v, "x_hat": res.x_star,
        "alpha_hat": res.alpha_star,
        "beta_hat": beta_hat, "c_hat": c_hat,
        "a_hat": a_hat, "v_hat": v_hat,
        "estimated_risk_selected": res.risk_star,
        "B_eff_hat": res.B_eff_star or float("inf"),
        "V_R_hat": res.V_R_star or (a_hat / n_eff),
    }


def _run_one(
    *, name: str, df_real: pd.DataFrame, truth: float,
    m: int, rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(df_real)
    if name == "real_only_diff_in_means":
        return {"method": name, "theta_hat": _ate_diff_in_means(df_real),
                "x_selected": 0, "alpha_selected": 1.0, "fallback_used": False}
    if name == "real_only_aipw":
        return {"method": name, "theta_hat": _ate_aipw(df_real),
                "x_selected": 0, "alpha_selected": 1.0, "fallback_used": False}
    if name == "synthetic_only_full_calibration":
        return {"method": name, "theta_hat": _synthetic_ate(df_real, m, rng),
                "x_selected": n, "alpha_selected": 0.0, "fallback_used": False}
    if name == "naive_pooling":
        x = n // 2
        cal = df_real.iloc[:x]
        est = df_real.iloc[x:]
        try:
            beta1, beta0 = _fit_outcome_model(cal)
            cols = ["intercept"] + COVS
            est_aug = est.copy()
            est_aug["intercept"] = 1.0
            X_e = est_aug[cols].to_numpy(dtype=float)
            mu1 = X_e @ beta1 if beta1 is not None else None
            mu0 = X_e @ beta0 if beta0 is not None else None
            if mu1 is None:
                return {"method": name, "theta_hat": _ate_diff_in_means(est),
                        "x_selected": x, "alpha_selected": None, "fallback_used": False}
            # Stack imputed potential outcomes from synthetic + real est.
            theta_S = _synthetic_ate(cal, m, rng)
            theta_R = _ate_aipw(est)
            n_e = len(est)
            theta_hat = (n_e * theta_R + m * theta_S) / (n_e + m)
            return {"method": name, "theta_hat": theta_hat,
                    "x_selected": x, "alpha_selected": n_e / (n_e + m),
                    "fallback_used": False}
        except Exception as e:  # noqa: BLE001
            return {"method": name, "theta_hat": float("nan"),
                    "failure_flag": True,
                    "failure_reason": f"{type(e).__name__}: {e}"}

    pilot_grid = default_pilot_grid(n)
    if len(pilot_grid) < 2:
        return {"method": name, "theta_hat": float("nan"),
                "failure_flag": True, "failure_reason": "pilot grid too small"}
    sel = _adaptive_pipeline_causal(
        df_real=df_real, m=m, pilot_grid=pilot_grid, rng=rng,
    )
    x_hat = sel["x_hat"]
    alpha = sel["alpha_hat"]
    n_eff = sel["n_eff"]

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
        beta_hat = max(sel["beta_hat"], 1e-3)
        b_at_x = sel["v_hat"] + sel["c_hat"] * (x_hat ** (-2.0 * beta_hat)) \
                 if x_hat > 0 else float("inf")
        if x_hat <= 0 or x_hat * b_at_x >= sel["a_hat"]:
            return {"method": name, "theta_hat": _ate_aipw(df_real),
                    "x_selected": 0, "alpha_selected": 1.0,
                    "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
                    "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
                    "fallback_used": True}
    elif name == "validation_debiased_gn":
        n_v = sel["n_v"]
        val = df_real.iloc[n_eff:]
        rest = df_real.iloc[:n_eff]
        cal = rest.iloc[:x_hat] if x_hat > 0 else rest
        est = rest.iloc[x_hat:] if x_hat > 0 else pd.DataFrame()
        theta_R = _ate_aipw(est) if len(est) >= 10 else _ate_aipw(rest)
        theta_R_V = _ate_aipw(val)
        theta_S = _synthetic_ate(cal, m, rng)
        # Use a fresh synthetic batch to estimate bias.
        theta_S_V = _synthetic_ate(cal, m, rng)
        bias_hat = theta_S_V - theta_R_V
        theta_S_tilde = theta_S - bias_hat
        theta_hat = alpha * theta_R + (1 - alpha) * theta_S_tilde
        return {"method": name, "theta_hat": theta_hat,
                "x_selected": x_hat, "alpha_selected": alpha,
                "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
                "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
                "fallback_used": False}

    rest = df_real.iloc[:n_eff]
    cal = rest.iloc[:x_hat] if x_hat > 0 else pd.DataFrame()
    est = rest.iloc[x_hat:] if x_hat > 0 else rest
    if len(est) >= 10:
        theta_R = _ate_aipw(est)
    else:
        theta_R = _ate_aipw(rest)
    if len(cal) > 0:
        theta_S = _synthetic_ate(cal, m, rng)
    else:
        theta_S = theta_R
    theta_hat = alpha * theta_R + (1 - alpha) * theta_S
    return {"method": name, "theta_hat": theta_hat,
            "x_selected": x_hat, "alpha_selected": alpha,
            "beta_hat": sel["beta_hat"], "c_hat": sel["c_hat"],
            "a_hat": sel["a_hat"], "v_hat": sel["v_hat"],
            "fallback_used": False,
            "B_eff_selected": sel["B_eff_hat"],
            "V_R_selected": sel["V_R_hat"],
            "estimated_risk_selected": sel["estimated_risk_selected"]}


def run(config_path: str | Path) -> Path:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw_for_run = dict(raw)
    raw_for_run.setdefault("beta_grid", [1.0])
    raw_for_run.setdefault("rho_grid", [1.0])
    config = RunConfig.from_dict(raw_for_run)
    if config.experiment_id != "expB_causal":
        log.warning("config experiment_id=%s (expected expB_causal)",
                    config.experiment_id)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")

    df_pop = make_ihdp(IHDPSpec(
        n=int(raw.get("population_size", 5000)),
        true_ate=float(raw.get("true_ate", 4.0)),
        seed=int(raw.get("population_seed", 0)),
    ))
    truth = true_ate(df_pop)
    log.info("IHDP-style pop: N=%d, true ATE=%.4f", len(df_pop), truth)

    cells = [CCell(n=int(n), m_factor=int(mf))
             for n in raw["n_grid"]
             for mf in raw.get("m_factor_grid", [1])]
    methods = raw["estimators"]

    def run_cell(cell: CCell, worker_id: int) -> int:
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
            for name in methods:
                t0 = time.perf_counter()
                try:
                    out = _run_one(name=name, df_real=df_real, truth=truth,
                                   m=m, rng=rng)
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
                    "dataset": "ihdp_synthetic",
                    "generator": "linear_outcome_model",
                    "estimand": "ate", "theta_true": truth,
                    "runtime_seconds": runtime,
                    "failure_flag": fail, "failure_reason": reason,
                }
                base.update(out)
                rows.append(base)
        writer.write_cell(cell.to_dict(), rows, worker_id=worker_id)
        return len(rows)

    log.info("running expB_causal: %d cells, methods=%s", len(cells), methods)

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
