"""Aggregation + figures for Experiment 2 (adaptive)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.results import read_run
from ..notation import classify_regime
from .aggregate import estimator_perf_table


def learning_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table A1: per-cell summary of β̂, ĉ across replications."""
    work = df.dropna(subset=["beta_hat"]).copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(["beta", "rho", "n", "method"], observed=True)
    out = grouped.agg(
        mean_beta_hat=("beta_hat", "mean"),
        sd_beta_hat=("beta_hat", "std"),
        mean_c_hat=("c_hat", "mean"),
        sd_c_hat=("c_hat", "std"),
        n_replications=("replication", "count"),
    ).reset_index()
    out["bias_beta_hat"] = out["mean_beta_hat"] - out["beta"]
    out["regime"] = [
        classify_regime(b, r) for b, r in zip(out["beta"], out["rho"])
    ]
    return out


def adaptive_perf_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table A2: adaptive performance summary.

    Columns: beta, rho, n, method, mse_ratio_to_real, oracle_regret,
             allocation_relative_error, safe_fallback_rate, harm_rate.
    """
    work = df.dropna(subset=["theta_hat"]).copy()
    work["sq_error"] = (work["theta_hat"] - work["theta_true"]) ** 2

    grouped = work.groupby(["beta", "rho", "n", "method"], observed=True)
    summary = grouped.agg(
        mse=("sq_error", "mean"),
        x_selected=("x_selected", "mean"),
        oracle_x=("oracle_x", "first"),
        safe_fallback_rate=("fallback_used", "mean"),
        n_replications=("replication", "count"),
    ).reset_index()

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

    # Allocation relative error: |x_selected - oracle_x| / max(oracle_x, 1)
    summary["allocation_relative_error"] = np.where(
        summary["oracle_x"] > 0,
        np.abs(summary["x_selected"] - summary["oracle_x"]) / summary["oracle_x"],
        np.abs(summary["x_selected"]),
    )

    # Harm rate: fraction of replications where method's MSE > real_only's MSE in this cell.
    # We approximate at the cell level via mse_ratio > 1.
    summary["harm_rate"] = (summary["mse_ratio_to_real"] > 1.0).astype(float)

    summary = summary.drop(columns=["real_mse", "oracle_mse"])
    summary["regime"] = [
        classify_regime(b, r) for b, r in zip(summary["beta"], summary["rho"])
    ]
    return summary


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_a2_adaptive_vs_oracle(df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure A2: scatter of x̂ (corrected_adaptive_gn) vs oracle_x."""
    sub = df[df["method"] == "corrected_adaptive_gn"].dropna(subset=["x_selected"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        sub["oracle_x"], sub["x_selected"],
        c=sub["n"], cmap="viridis", alpha=0.4, s=10,
    )
    lim = max(sub["oracle_x"].max(), sub["x_selected"].max())
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
    ax.set_xlabel("oracle x*")
    ax.set_ylabel(r"adaptive $\hat x$")
    ax.set_title(r"Figure A2: adaptive $\hat x$ vs oracle $x^*$ (color = n)")
    plt.colorbar(sc, ax=ax, label="n")
    _save(fig, fig_dir, "figure_a2_adaptive_vs_oracle")


def figure_a3_adaptive_regret(adaptive_table: pd.DataFrame, fig_dir: Path) -> None:
    """Figure A3: adaptive regret vs n, faceted by (beta, rho)."""
    methods = ["corrected_adaptive_gn", "adaptive_parametric_foc",
               "adaptive_nonparametric_grid", "safe_corrected_adaptive_gn"]
    sub = adaptive_table[adaptive_table["method"].isin(methods)]
    if sub.empty:
        return
    pairs = sorted(sub[["beta", "rho"]].drop_duplicates().itertuples(index=False))
    cols = 3
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for ax, (b, r) in zip(axes.flat, pairs):
        cell = sub[(sub["beta"] == b) & (sub["rho"] == r)]
        for m, m_grp in cell.groupby("method", observed=True):
            m_grp = m_grp.sort_values("n")
            ax.semilogx(m_grp["n"], m_grp["oracle_regret"], "o-",
                        label=m, alpha=0.8)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_title(fr"$\beta={b}, \rho={r}$  ({classify_regime(b, r)})", fontsize=9)
        ax.set_xlabel("n")
        ax.set_ylabel("oracle regret")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Figure A3: adaptive regret relative to corrected_oracle_gn")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, fig_dir, "figure_a3_adaptive_regret")


def aggregate_run_exp2(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df = read_run(run_dir / "raw")
    a1 = learning_curve_table(df)
    a2 = adaptive_perf_table(df)
    s2 = estimator_perf_table(df)  # also useful for Exp 2 estimators

    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a1.empty:
        a1.to_parquet(out_dir / "table_a1_learning_curve.parquet")
    a2.to_parquet(out_dir / "table_a2_adaptive_perf.parquet")
    s2.to_parquet(out_dir / "table_s2_estimator_perf.parquet")

    fig_dir = run_dir / "figures"
    figure_a2_adaptive_vs_oracle(df, fig_dir)
    figure_a3_adaptive_regret(a2, fig_dir)

    return {"raw": df, "table_a1": a1, "table_a2": a2, "table_s2": s2}
