"""Aggregation: long-format raw results -> Tables S1, S2 (and analogues).

Tables defined in docs/experiments.md §4.8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.results import read_run
from ..notation import classify_regime, theory_slope
from .scaling import fit_scaling_slopes


def estimator_perf_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table S2: per-estimator bias / variance / MSE / ratios / regret.

    Columns: beta, rho, n, estimator, bias, variance, mse, mse_ratio_to_real,
             oracle_regret, x_selected, lambda_selected, alpha_selected.
    """
    work = df.dropna(subset=["theta_hat"]).copy()
    work["error"] = work["theta_hat"] - work["theta_true"]
    work["squared_error"] = work["error"] ** 2

    grouped = work.groupby(
        ["beta", "rho", "n", "method"], dropna=False, observed=True
    )
    summary = grouped.agg(
        bias=("error", "mean"),
        variance=("error", lambda s: float(np.var(s, ddof=1))),
        mse=("squared_error", "mean"),
        x_selected=("x_selected", "mean"),
        lambda_selected=("lambda_selected", "mean"),
        alpha_selected=("alpha_selected", "mean"),
        n_replications=("replication", "count"),
    ).reset_index()

    # MSE ratios
    real_mse = (
        summary[summary["method"] == "real_only_all"]
        .set_index(["beta", "rho", "n"])["mse"]
        .rename("real_mse")
    )
    summary = summary.join(real_mse, on=["beta", "rho", "n"])
    summary["mse_ratio_to_real"] = summary["mse"] / summary["real_mse"]

    oracle_mse = (
        summary[summary["method"] == "corrected_oracle_gn"]
        .set_index(["beta", "rho", "n"])["mse"]
        .rename("oracle_mse")
    )
    summary = summary.join(oracle_mse, on=["beta", "rho", "n"])
    summary["oracle_regret"] = summary["mse"] / summary["oracle_mse"] - 1.0

    summary = summary.drop(columns=["real_mse", "oracle_mse"])
    summary["regime"] = [
        classify_regime(b, r) for b, r in zip(summary["beta"], summary["rho"])
    ]
    return summary.rename(columns={"method": "estimator"})


def scaling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table S1: predicted vs empirical allocation scaling per (beta, rho).

    Empirical slope is the OLS slope of log(oracle_x) vs log(n) using
    *cell-level* oracle_x (not per-replication, since oracle_x is deterministic
    per cell).
    """
    cell_oracle = (
        df.groupby(["beta", "rho", "n"], observed=True)["oracle_x"]
        .first()
        .reset_index()
    )
    out = fit_scaling_slopes(cell_oracle)
    out["theory_regime"] = [
        classify_regime(b, r) for b, r in zip(out["beta"], out["rho"])
    ]
    out["theory_slope"] = [
        theory_slope(b, r) for b, r in zip(out["beta"], out["rho"])
    ]

    largest_n = cell_oracle["n"].max()
    last = cell_oracle[cell_oracle["n"] == largest_n].copy()
    last["mean_oracle_lambda_at_largest_n"] = last["oracle_x"] / last["n"]
    out = out.merge(
        last[["beta", "rho", "mean_oracle_lambda_at_largest_n"]],
        on=["beta", "rho"], how="left",
    )

    # Boundary-selected rate: fraction of n values where oracle picked x=0 or x=n.
    cell_oracle["boundary"] = (cell_oracle["oracle_x"] == 0) | (
        cell_oracle["oracle_x"] == cell_oracle["n"]
    )
    boundary_rate = (
        cell_oracle.groupby(["beta", "rho"], observed=True)["boundary"]
        .mean()
        .rename("boundary_selected_rate")
        .reset_index()
    )
    out = out.merge(boundary_rate, on=["beta", "rho"], how="left")
    return out


def aggregate_run(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read raw parquet, build the standard tables, and write them to
    {run_dir}/aggregated/. Returns a dict of dataframes."""
    run_dir = Path(run_dir)
    df = read_run(run_dir / "raw")

    s2 = estimator_perf_table(df)
    s1 = scaling_table(df)

    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1.to_parquet(out_dir / "table_s1_scaling.parquet")
    s2.to_parquet(out_dir / "table_s2_estimator_perf.parquet")
    return {"raw": df, "table_s1": s1, "table_s2": s2}
