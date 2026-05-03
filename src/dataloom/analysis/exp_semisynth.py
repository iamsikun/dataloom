"""Aggregation + figures for semi-synthetic Experiments A (tabular) and B (causal)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.results import read_run


def _perf_table(df: pd.DataFrame, group_extra: list[str]) -> pd.DataFrame:
    work = df.dropna(subset=["theta_hat"]).copy()
    work["error"] = work["theta_hat"] - work["theta_true"]
    work["sq_error"] = work["error"] ** 2
    grp_cols = ["dataset", "generator", "estimand", "n", "m"] + group_extra + ["method"]
    grouped = work.groupby(grp_cols, observed=True)
    out = grouped.agg(
        bias=("error", "mean"),
        variance=("error", lambda s: float(np.var(s, ddof=1))),
        mse=("sq_error", "mean"),
        x_selected=("x_selected", "mean"),
        alpha_selected=("alpha_selected", "mean"),
        fallback_rate=("fallback_used", "mean"),
        n_replications=("replication", "count"),
    ).reset_index()
    # Pick a baseline: prefer `real_only_all`; in the causal experiment use
    # `real_only_aipw` (since that's the natural baseline there).
    baseline_candidates = ["real_only_all", "real_only_aipw"]
    baseline = next(
        (c for c in baseline_candidates if (out["method"] == c).any()),
        None,
    )
    if baseline is not None:
        real_mse = (
            out[out["method"] == baseline]
            .set_index([c for c in grp_cols if c != "method"])["mse"]
            .rename("real_mse")
        )
        out = out.join(real_mse, on=[c for c in grp_cols if c != "method"])
        out["mse_ratio_to_real"] = out["mse"] / out["real_mse"]
        out = out.drop(columns=["real_mse"])
    else:
        out["mse_ratio_to_real"] = float("nan")
    return out


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_expA(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df = read_run(run_dir / "raw")
    out = _perf_table(df, group_extra=[])
    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "table_t2_tabular_perf.parquet")

    # Figure T3: MSE ratio bar plot per estimand.
    fig, axes = plt.subplots(
        1, max(1, df["estimand"].nunique()),
        figsize=(5 * max(1, df["estimand"].nunique()), 4),
        squeeze=False,
    )
    for ax, (estimand, eg) in zip(axes.flat, out.groupby("estimand", observed=True)):
        for m, mg in eg.groupby("method", observed=True):
            mg = mg.sort_values("n")
            ax.semilogx(mg["n"], mg["mse_ratio_to_real"], "o-", label=m, alpha=0.8)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_title(f"estimand={estimand}", fontsize=9)
        ax.set_xlabel("n")
        ax.set_ylabel("MSE / MSE(real_only_all)")
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Figure T3: tabular MSE ratio to real-only")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, run_dir / "figures", "figure_t3_mse_ratio")
    return {"raw": df, "table_t2": out}


def aggregate_expB(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df = read_run(run_dir / "raw")
    out = _perf_table(df, group_extra=[])
    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "table_d1_causal_perf.parquet")

    # Figure D2: ATE MSE ratio
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, mg in out.groupby("method", observed=True):
        mg = mg.sort_values("n")
        ax.semilogx(mg["n"], mg["mse_ratio_to_real"], "o-", label=m, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
    ax.set_xlabel("n")
    ax.set_ylabel("MSE / MSE(real_only_all)")
    ax.set_title("Figure D2: ATE MSE ratio (causal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, run_dir / "figures", "figure_d2_ate_mse_ratio")
    return {"raw": df, "table_d1": out}
