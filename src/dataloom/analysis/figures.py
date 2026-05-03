"""Figure generation for Experiment 1 (Figures S1-S4 from §4.7).

Each figure is saved as both .pdf and .png in {run_dir}/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..notation import classify_regime, theory_slope


REGIME_COLORS = {
    "persistent_variance": "#999999",
    "slow_learning": "#1f77b4",
    "parametric_knife_edge": "#ff7f0e",
    "sublinear_fast_learning": "#2ca02c",
    "boundary_full_calibration": "#d62728",
    "boundary_knife_edge": "#9467bd",
}


def _save(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_s1_phase_diagram(
    beta_grid: Iterable[float],
    rho_grid: Iterable[float],
    fig_dir: Path,
) -> None:
    """Heatmap of theoretical regimes over (beta, rho), with simulated grid markers."""
    betas = list(beta_grid)
    rhos = list(rho_grid)
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in betas:
        for r in rhos:
            regime = classify_regime(b, r)
            ax.scatter(
                b, r, s=200, c=REGIME_COLORS[regime],
                marker="s", edgecolor="black", linewidth=0.5,
            )
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=c, markersize=12, label=k)
        for k, c in REGIME_COLORS.items()
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    ax.set_xlabel(r"$\beta$ (synthetic learning rate)")
    ax.set_ylabel(r"$\rho$ (synthetic-variance decay)")
    ax.set_title("Figure S1: phase diagram (regime per simulated cell)")
    ax.grid(True, alpha=0.3)
    _save(fig, fig_dir, "figure_s1_phase_diagram")


def figure_s2_calibration_scaling(
    table_s1: pd.DataFrame,
    cell_oracle: pd.DataFrame,
    fig_dir: Path,
    panels: list[tuple[float, float, str]] | None = None,
) -> None:
    """Figure S2: log x_n* vs log n for representative regimes, with theory slope."""
    if panels is None:
        panels = [
            (1.0, 0.0, "fixed m (rho=0)"),
            (0.4, 1.0, "slow learner (beta=0.4, rho=1)"),
            (1.0, 1.0, "common fast learner (beta=1, rho=1)"),
            (1.0, 2.0, "boundary regime (beta=1, rho=2)"),
        ]
    panels_present = [
        p for p in panels
        if not cell_oracle[
            (cell_oracle["beta"] == p[0]) & (cell_oracle["rho"] == p[1])
        ].empty
    ]
    if not panels_present:
        return
    n_panels = len(panels_present)
    cols = 2
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    for ax, (b, r, title) in zip(axes.flat, panels_present):
        sub = cell_oracle[(cell_oracle["beta"] == b) & (cell_oracle["rho"] == r)]
        sub = sub[sub["oracle_x"] > 0]
        if sub.empty:
            ax.set_title(f"{title}\n(no interior points)")
            continue
        ax.loglog(sub["n"], sub["oracle_x"], "o-", label=r"empirical $x_n^*$")
        slope_theory = theory_slope(b, r)
        if slope_theory is not None:
            n_arr = np.array(sub["n"])
            # Anchor to first point for a visual reference.
            ref = sub["oracle_x"].iloc[0] / (n_arr[0] ** slope_theory)
            ax.loglog(n_arr, ref * n_arr ** slope_theory, "--",
                      label=f"theory slope = {slope_theory:.3f}")
        ax.set_xlabel("n")
        ax.set_ylabel(r"$x_n^*$")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(panels_present):]:
        ax.axis("off")
    fig.suptitle("Figure S2: oracle calibration size scaling")
    _save(fig, fig_dir, "figure_s2_calibration_scaling")


def figure_s3_calibration_share(cell_oracle: pd.DataFrame, fig_dir: Path) -> None:
    """Figure S3: lambda_n* = x_n* / n vs n on log scale.

    Highlight the §4.7 visual: in (beta=1, rho=1), x* increases but lambda* decreases.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for (b, r), sub in cell_oracle.groupby(["beta", "rho"], observed=True):
        sub = sub.sort_values("n")
        if sub["oracle_x"].max() == 0:
            continue
        lam = sub["oracle_x"] / sub["n"]
        ax.semilogx(sub["n"], lam, "o-", alpha=0.7, label=fr"$\beta={b}, \rho={r}$")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\lambda_n^* = x_n^* / n$")
    ax.set_title("Figure S3: oracle calibration share")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, fig_dir, "figure_s3_calibration_share")


def figure_s4_mse_ratio(table_s2: pd.DataFrame, fig_dir: Path) -> None:
    """Figure S4: MSE / MSE(real_only) for all estimators across n, faceted by (beta, rho)."""
    pairs = sorted(table_s2[["beta", "rho"]].drop_duplicates().itertuples(index=False))
    cols = 3
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for ax, (b, r) in zip(axes.flat, pairs):
        sub = table_s2[(table_s2["beta"] == b) & (table_s2["rho"] == r)]
        for est, est_grp in sub.groupby("estimator", observed=True):
            est_grp = est_grp.sort_values("n")
            ax.semilogx(
                est_grp["n"], est_grp["mse_ratio_to_real"],
                "o-", label=est, alpha=0.8,
            )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_title(fr"$\beta={b}, \rho={r}$  ({classify_regime(b, r)})", fontsize=9)
        ax.set_xlabel("n")
        ax.set_ylabel("MSE / MSE(real_only_all)")
        ax.set_ylim(0, 2.5)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Figure S4: MSE ratio to real-only across estimators")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, fig_dir, "figure_s4_mse_ratio")


def render_exp1_figures(run_dir: str | Path, raw: pd.DataFrame,
                        table_s1: pd.DataFrame, table_s2: pd.DataFrame) -> None:
    """Render all four Exp 1 figures into {run_dir}/figures/."""
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    cell_oracle = (
        raw.groupby(["beta", "rho", "n"], observed=True)["oracle_x"]
        .first()
        .reset_index()
    )
    figure_s1_phase_diagram(
        beta_grid=sorted(raw["beta"].unique()),
        rho_grid=sorted(raw["rho"].unique()),
        fig_dir=fig_dir,
    )
    figure_s2_calibration_scaling(table_s1, cell_oracle, fig_dir)
    figure_s3_calibration_share(cell_oracle, fig_dir)
    figure_s4_mse_ratio(table_s2, fig_dir)
