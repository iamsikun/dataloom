"""Aggregation + figures for Experiment 4 (inference)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.results import read_run


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table C1: coverage / length / bias / SE summary per (β, n, method)."""
    work = df.dropna(subset=["theta_hat", "ci_lower", "ci_upper"]).copy()
    work["bias"] = work["theta_hat"] - work["theta_true"]
    work["covered"] = (work["ci_lower"] <= work["theta_true"]) & (
        work["theta_true"] <= work["ci_upper"]
    )
    work["length"] = work["ci_upper"] - work["ci_lower"]

    grouped = work.groupby(["beta", "rho", "n", "method"], observed=True)
    out = grouped.agg(
        coverage=("covered", "mean"),
        avg_length=("length", "mean"),
        median_length=("length", "median"),
        bias=("bias", "mean"),
        empirical_sd=("theta_hat", lambda s: float(np.std(s, ddof=1))),
        n_replications=("replication", "count"),
    ).reset_index()
    out["bias_over_se"] = (
        out["bias"].abs() / (out["empirical_sd"] / np.sqrt(out["n_replications"]))
    )
    out = out.rename(columns={"method": "interval_method"})
    return out


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_c1_coverage_vs_n(table_c1: pd.DataFrame, fig_dir: Path) -> None:
    """Figure C1: coverage for each interval method across n, faceted by β."""
    methods = ["ci_real_only", "ci_gn_naive", "ci_gn_bias_aware",
               "ci_gn_undersmoothed", "ci_validation_debiased"]
    sub = table_c1[table_c1["interval_method"].isin(methods)]
    if sub.empty:
        return
    pairs = sorted(sub[["beta", "rho"]].drop_duplicates().itertuples(index=False))
    cols = 2
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    for ax, (b, r) in zip(axes.flat, pairs):
        cell = sub[(sub["beta"] == b) & (sub["rho"] == r)]
        for method, mg in cell.groupby("interval_method", observed=True):
            mg = mg.sort_values("n")
            ax.plot(mg["n"], mg["coverage"], "o-", label=method, alpha=0.85)
        ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("n")
        ax.set_ylabel("coverage")
        ax.set_title(fr"$\beta={b}, \rho={r}$", fontsize=10)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Figure C1: empirical coverage by interval method")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    _save(fig, fig_dir, "figure_c1_coverage")


def figure_c2_coverage_vs_length(table_c1: pd.DataFrame, fig_dir: Path) -> None:
    """Figure C2: scatter of coverage vs interval length."""
    if table_c1.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, mg in table_c1.groupby("interval_method", observed=True):
        ax.scatter(mg["avg_length"], mg["coverage"],
                   label=method, alpha=0.7, s=40)
    ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("average interval length")
    ax.set_ylabel("coverage")
    ax.set_title("Figure C2: coverage vs interval length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, fig_dir, "figure_c2_coverage_vs_length")


def aggregate_run_exp4(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df = read_run(run_dir / "raw")
    table_c1 = coverage_table(df)
    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_c1.to_parquet(out_dir / "table_c1_coverage.parquet")
    fig_dir = run_dir / "figures"
    figure_c1_coverage_vs_n(table_c1, fig_dir)
    figure_c2_coverage_vs_length(table_c1, fig_dir)
    return {"raw": df, "table_c1": table_c1}
