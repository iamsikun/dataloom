"""Aggregation + figures for Experiment 3 (multichannel)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..multichannel.oracle import R_profile_2


def read_run_mc(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all multichannel parts plus per-cell oracle.json."""
    raw_dir = Path(raw_dir)
    parts = list(raw_dir.glob("cell=*/part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parts under {raw_dir}")
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)

    oracle_rows = []
    for cell_dir in raw_dir.glob("cell=*"):
        meta = cell_dir / "oracle.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            o = json.load(f)
        # Parse the cell directory name back into a dict.
        params = dict(part.split("=") for part in cell_dir.name[len("cell="):].split("_"))
        oracle_rows.append({
            "n": int(params["n"]), "beta1": float(params["beta1"]),
            "beta2": float(params["beta2"]), "rho": float(params["rho"]),
            "x1_star": int(o["x1_star"]), "x2_star": int(o["x2_star"]),
            "risk_star": float(o["risk_star"]),
            "mv1": float(o["mv1"]), "mv2": float(o["mv2"]),
        })
    return df, pd.DataFrame(oracle_rows)


def perf_table_mc(df: pd.DataFrame) -> pd.DataFrame:
    """Table M1: per-method bias / variance / MSE / MSE-ratio / regret."""
    work = df.dropna(subset=["theta_hat"]).copy()
    work["sq_error"] = (work["theta_hat"] - work["theta_true"]) ** 2
    grouped = work.groupby(["beta", "rho", "n", "method"], observed=True)
    out = grouped.agg(
        mse=("sq_error", "mean"),
        x_total_selected=("x_selected", "mean"),
        oracle_total=("oracle_x", "first"),
        n_replications=("replication", "count"),
    ).reset_index()

    # MSE ratios vs corrected_multichannel_oracle (the benchmark for Exp 3).
    oracle_mse = (
        out[out["method"] == "corrected_multichannel_oracle"]
        .set_index(["beta", "rho", "n"])["mse"]
        .rename("oracle_mse")
    )
    out = out.join(oracle_mse, on=["beta", "rho", "n"])
    out["mse_ratio_to_corrected"] = out["mse"] / out["oracle_mse"]
    out = out.drop(columns=["oracle_mse"])
    return out


def _save(fig, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_m1_risk_surface(oracle_df: pd.DataFrame, fig_dir: Path,
                           a: float = 1.0, sigma_s2: float = 1.0,
                           c1: float = 1.0, c2: float = 1.0) -> None:
    """Figure M1: risk surface for a representative cell, with method markers."""
    if oracle_df.empty:
        return
    # Pick the largest n in the data.
    cell = oracle_df.sort_values("n", ascending=False).iloc[0]
    n = int(cell["n"])
    beta1, beta2 = float(cell["beta1"]), float(cell["beta2"])
    v_n = sigma_s2 / n  # rho=1 assumed for Exp 3 default
    step = max(1, n // 80)
    xs = np.arange(1, n, step, dtype=int)
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    feas = (X1 + X2) < n
    R = np.full(X1.shape, np.nan)
    R[feas] = R_profile_2(X1[feas], X2[feas], n=n, a=a, v_n=v_n,
                          c1=c1, beta1=beta1, c2=c2, beta2=beta2)

    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(X1, X2, np.log10(R), shading="auto", cmap="viridis")
    plt.colorbar(pcm, ax=ax, label=r"$\log_{10} R_n$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(fr"Figure M1: risk surface (n={n}, $\beta_1$={beta1}, $\beta_2$={beta2})")
    # Mark corrected oracle.
    ax.plot(cell["x1_star"], cell["x2_star"], "*", markersize=18,
            color="red", label="corrected oracle")
    # Mark equal split.
    es = max(1, n // 4)
    ax.plot(es, es, "s", markersize=10, color="white", label="equal split")
    # Mark old fixed-share.
    denom = 1.0 + 2.0 * (beta1 + beta2)
    x1_old = max(1, int(np.floor(n * 2.0 * beta1 / denom)))
    x2_old = max(1, int(np.floor(n * 2.0 * beta2 / denom)))
    ax.plot(x1_old, x2_old, "^", markersize=10, color="orange", label="old fixed-share")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, fig_dir, "figure_m1_risk_surface")


def figure_m2_marginal_values(oracle_df: pd.DataFrame, fig_dir: Path) -> None:
    """Figure M2: MV1 vs MV2 at the corrected oracle for each (β1, β2, n)."""
    if oracle_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for (b1, b2), g in oracle_df.groupby(["beta1", "beta2"], observed=True):
        ax.scatter(g["mv1"], g["mv2"], label=fr"$\beta=({b1}, {b2})$", alpha=0.7, s=50)
    lim = max(oracle_df["mv1"].max(), oracle_df["mv2"].max())
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="MV1 = MV2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$MV_1$ at oracle")
    ax.set_ylabel(r"$MV_2$ at oracle")
    ax.set_title("Figure M2: KKT marginal values at corrected oracle")
    ax.legend(fontsize=8)
    _save(fig, fig_dir, "figure_m2_marginal_values")


def aggregate_run_exp3(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    df, oracle_df = read_run_mc(run_dir / "raw")
    table_m1 = perf_table_mc(df)
    out_dir = run_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_m1.to_parquet(out_dir / "table_m1_multichannel_perf.parquet")
    if not oracle_df.empty:
        oracle_df.to_parquet(out_dir / "table_m1_oracle.parquet")

    fig_dir = run_dir / "figures"
    figure_m1_risk_surface(oracle_df, fig_dir)
    figure_m2_marginal_values(oracle_df, fig_dir)
    return {"raw": df, "oracle": oracle_df, "table_m1": table_m1}
