"""Aggregation and figures for the targeted rebuttal experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.results import read_run
from ..oracle import oracle_grid_curve


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def aggregate_boundary(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    raw = read_run(run_dir / "raw")
    work = raw.copy()
    work["squared_error"] = (work["theta_hat"] - work["theta_true"]) ** 2
    performance = (
        work.groupby(["v0", "n", "method"], observed=True)
        .agg(
            mse=("squared_error", "mean"),
            mse_se=("squared_error", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
            bias=("error", "mean"),
            variance=("error", lambda x: float(np.var(x, ddof=1))),
            x_selected=("x_selected", "first"),
            alpha_selected=("alpha_selected", "first"),
            analytic_risk=("true_risk_selected", "first"),
            replications=("replication", "count"),
        )
        .reset_index()
    )
    real = (
        performance[performance["method"] == "real_only_all"]
        .set_index(["v0", "n"])["mse"]
        .rename("real_mse")
    )
    performance = performance.join(real, on=["v0", "n"])
    performance["mse_ratio_to_real"] = performance["mse"] / performance["real_mse"]
    performance["analytic_standardized_error"] = (
        performance["mse"] - performance["analytic_risk"]
    ) / performance["mse_se"]
    performance = performance.drop(columns="real_mse")

    boundary = (
        performance[performance["method"] == "corrected_oracle_gn"][
            ["v0", "n", "x_selected", "mse", "analytic_risk"]
        ]
        .copy()
        .sort_values(["v0", "n"])
    )
    boundary["lambda_selected"] = boundary["x_selected"] / boundary["n"]
    boundary["lambda_limit"] = np.minimum(
        1.0, (2.0 / boundary["v0"] ** 2) ** (1.0 / 3.0)
    )
    boundary["absolute_share_error"] = np.abs(
        boundary["lambda_selected"] - boundary["lambda_limit"]
    )

    zero_rows = []
    for beta in (0.25, 0.4, 0.5, 0.75, 1.0, 1.5):
        n = 10_000
        curve = lambda x, beta=beta: np.asarray(x, dtype=float) ** (
            -2.0 * beta
        )
        result = oracle_grid_curve(n=n, a=1.0, B_curve=curve)
        if beta < 0.5:
            expected = "real_only_tiebreak"
        elif beta > 0.5:
            expected = "synthetic_only"
        else:
            expected = "all_allocations_tied"
        zero_rows.append(
            {
                "beta": beta,
                "n": n,
                "x_star_tiebreak": result.x_star,
                "n_minimizers": len(result.minimizers),
                "min_x": min(result.minimizers),
                "max_x": max(result.minimizers),
                "minimizer_class": expected,
                "product_surrogate_share": 2 * beta / (1 + 2 * beta),
            }
        )
    zero_variance = pd.DataFrame(zero_rows)

    out = run_dir / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    performance.to_parquet(out / "table_r1_performance.parquet")
    boundary.to_parquet(out / "table_r1_boundary.parquet")
    zero_variance.to_parquet(out / "table_r1_zero_variance_oracle.parquet")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for v0, group in boundary.groupby("v0", observed=True):
        ax.semilogx(group["n"], group["lambda_selected"], "o-", label=f"v0={v0:.3g}")
        ax.axhline(group["lambda_limit"].iloc[0], linestyle=":", linewidth=1)
    ax.set_xlabel("n")
    ax.set_ylabel("corrected oracle calibration share")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, run_dir / "figures", "figure_r1_boundary_share")
    return {
        "raw": raw,
        "performance": performance,
        "boundary": boundary,
        "zero_variance": zero_variance,
    }


def aggregate_remainder(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    raw = read_run(run_dir / "raw").sort_values(["curve_family", "n"])
    slope_rows = []
    divergent = raw[
        raw["curve_family"].isin(
            ["exact", "positive_remainder", "negative_remainder", "oscillatory_remainder"]
        )
    ]
    exact_risk = (
        divergent[divergent["curve_family"] == "exact"]
        .set_index("n")["true_risk_selected"]
        .astype(float)
    )
    divergent = divergent.copy()
    divergent["risk_ratio_to_exact"] = [
        float(risk) / float(exact_risk.loc[int(n)])
        for n, risk in zip(divergent["n"], divergent["true_risk_selected"])
    ]
    for family, group in divergent.groupby("curve_family", observed=True):
        slope = float(
            np.polyfit(
                np.log(group["n"].astype(float)),
                np.log(group["x_selected"].astype(float)),
                1,
            )[0]
        )
        slope_rows.append(
            {
                "curve_family": family,
                "empirical_slope": slope,
                "theory_slope": 2.0 / 3.0,
                "absolute_slope_error": abs(slope - 2.0 / 3.0),
                "largest_n": int(group["n"].max()),
                "x_at_largest_n": int(
                    group.loc[group["n"].idxmax(), "x_selected"]
                ),
                "share_at_largest_n": float(
                    group.loc[group["n"].idxmax(), "lambda_selected"]
                ),
                "risk_ratio_at_largest_n": float(
                    group.loc[group["n"].idxmax(), "risk_ratio_to_exact"]
                ),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    bounded = raw[raw["curve_family"] == "bounded_remainder"].copy()
    bounded_final = bounded[bounded["n"] == bounded["n"].max()][
        ["remainder_scale", "n", "x_selected", "true_risk_selected"]
    ].sort_values("remainder_scale")
    plateau = raw[raw["curve_family"] == "plateau"].copy()

    out = run_dir / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    divergent.to_parquet(out / "table_r2_diverging.parquet")
    slopes.to_parquet(out / "table_r2_slopes.parquet")
    bounded_final.to_parquet(out / "table_r2_bounded.parquet")
    plateau.to_parquet(out / "table_r2_plateau.parquet")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for family, group in divergent.groupby("curve_family", observed=True):
        ax.loglog(group["n"], group["x_selected"], "o-", label=family)
    ax.set_xlabel("n")
    ax.set_ylabel("oracle calibration x")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, run_dir / "figures", "figure_r2_remainder_scaling")
    return {
        "raw": raw,
        "divergent": divergent,
        "slopes": slopes,
        "bounded": bounded_final,
        "plateau": plateau,
    }


def aggregate_hierarchical(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    raw = read_run(run_dir / "raw")
    diagnostics = raw[raw["method"] == "score_diagnostic"].copy()
    methods = raw[raw["method"] != "score_diagnostic"].copy()
    diagnostics["truncated_bias2"] = np.maximum(
        diagnostics["score_untruncated"].astype(float), 0.0
    )
    score_table = (
        diagnostics.groupby(
            [
                "beta",
                "rho",
                "n",
                "validation_size",
                "pilot_fits",
                "redraws_per_fit",
                "x_selected",
            ],
            observed=True,
        )
        .agg(
            bias2_true=("bias2_true", "first"),
            score_mean=("score_untruncated", "mean"),
            score_sd=("score_untruncated", "std"),
            truncated_mean=("truncated_bias2", "mean"),
            B_true=("B_true", "first"),
            B_hat_mean=("B_hat", "mean"),
            upper_coverage=("U_B", lambda x: float(np.mean(np.asarray(x, float) >= diagnostics.loc[x.index, "B_true"].astype(float)))),
            replications=("replication", "count"),
        )
        .reset_index()
    )
    score_table["relative_centering_error"] = (
        score_table["score_mean"] - score_table["bias2_true"]
    ) / score_table["bias2_true"]
    score_table["relative_truncation_bias"] = (
        score_table["truncated_mean"] - score_table["bias2_true"]
    ) / score_table["bias2_true"]

    methods["squared_error"] = (
        methods["theta_hat"].astype(float) - methods["theta_true"].astype(float)
    ) ** 2
    methods["false_safe"] = (
        methods["safe_pass"].fillna(False).astype(bool)
        & methods["x_selected"].fillna(0).astype(float).gt(0)
        & (
            methods["x_selected"].astype(float)
            * methods["B_true"].fillna(0).astype(float)
            >= methods["a"].astype(float)
        )
    )
    keys = [
        "beta",
        "rho",
        "n",
        "validation_size",
        "pilot_fits",
        "redraws_per_fit",
        "method",
    ]
    performance = (
        methods.groupby(keys, observed=True)
        .agg(
            mse=("squared_error", "mean"),
            mse_se=("squared_error", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
            x_median=("x_selected", "median"),
            x_q25=("x_selected", lambda x: float(np.quantile(x, 0.25))),
            x_q75=("x_selected", lambda x: float(np.quantile(x, 0.75))),
            fallback_rate=("fallback_used", "mean"),
            false_safe_rate=("false_safe", "mean"),
            median_uniform_curve_error=("uniform_curve_error", "median"),
            median_working_risk_ratio=("working_risk_ratio", "median"),
            replications=("replication", "count"),
        )
        .reset_index()
    )
    join_keys = keys[:-1]
    real = (
        performance[performance["method"] == "real_only_all"]
        .set_index(join_keys)["mse"]
        .rename("real_mse")
    )
    performance = performance.join(real, on=join_keys)
    performance["mse_ratio_to_real"] = performance["mse"] / performance["real_mse"]
    performance = performance.drop(columns="real_mse")

    out = run_dir / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    score_table.to_parquet(out / "table_r3_score.parquet")
    performance.to_parquet(out / "table_r3_performance.parquet")

    target_redraws = int(performance["redraws_per_fit"].max())
    plot = performance[
        (performance["beta"] == 1.0)
        & (performance["validation_size"] == performance["n"])
        & (performance["redraws_per_fit"] == target_redraws)
        & performance["method"].isin(
            ["direct_grid_plugin", "uncertainty_buffered_direct_grid"]
        )
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, group in plot.groupby("method", observed=True):
        for n, cell in group.groupby("n", observed=True):
            ax.plot(
                cell["pilot_fits"],
                cell["mse_ratio_to_real"],
                "o-",
                label=f"{method}, n={n}",
            )
    ax.axhline(1.0, color="black", linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("independent calibration fits")
    ax.set_ylabel("MSE ratio to real-only")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    _save(fig, run_dir / "figures", "figure_r3_adaptive_concentration")
    return {"raw": raw, "score": score_table, "performance": performance}


def _paired_ratio_ci(
    method_error: np.ndarray,
    real_error: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(method_error)
    indices = rng.integers(0, count, size=(draws, count))
    numerator = np.mean(method_error[indices], axis=1)
    denominator = np.mean(real_error[indices], axis=1)
    ratios = numerator / denominator
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def aggregate_adult_auxiliary(run_dir: str | Path) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    raw = read_run(run_dir / "raw")
    pilot = raw[raw["method"] == "external_pilot_curve"].copy()
    evaluation = raw[raw["method"] != "external_pilot_curve"].copy()
    evaluation["squared_error"] = (
        evaluation["theta_hat"].astype(float)
        - evaluation["theta_true"].astype(float)
    ) ** 2
    summary_rows = []
    for (n, estimand_name), cell in evaluation.groupby(
        ["n", "estimand"], observed=True
    ):
        real = (
            cell[cell["method"] == "real_only_all"]
            .sort_values("replication")["squared_error"]
            .to_numpy(float)
        )
        for method, group in cell.groupby("method", observed=True):
            ordered = group.sort_values("replication")
            error = ordered["squared_error"].to_numpy(float)
            ratio = float(error.mean() / real.mean())
            low, high = _paired_ratio_ci(
                error,
                real,
                seed=20260727 + int(n) + sum(map(ord, str(method))),
            )
            summary_rows.append(
                {
                    "n": int(n),
                    "estimand": estimand_name,
                    "method": method,
                    "mse": float(error.mean()),
                    "mse_ratio_to_real": ratio,
                    "ratio_ci_low": low,
                    "ratio_ci_high": high,
                    "x_selected": int(ordered["x_selected"].iloc[0]),
                    "lambda_selected": float(
                        ordered["x_selected"].iloc[0] / int(n)
                    ),
                    "alpha_selected": float(
                        ordered["alpha_selected"].iloc[0]
                    ),
                    "fallback_rate": float(
                        ordered["fallback_used"].fillna(False).mean()
                    ),
                    "beta_hat": float(ordered["beta_hat"].iloc[0]),
                    "powerlaw_r2": float(ordered["powerlaw_r2"].iloc[0]),
                    "replications": len(ordered),
                }
            )
    performance = pd.DataFrame(summary_rows)
    curve = (
        pilot.groupby(["n", "estimand", "x_selected"], observed=True)
        .agg(
            B_empirical=("B_true", "mean"),
            score_mean=("score_untruncated", "mean"),
            v_hat=("v_hat", "mean"),
            a_hat=("a_hat", "mean"),
            replications=("replication", "count"),
        )
        .reset_index()
    )
    curve["B_score"] = np.maximum(curve["score_mean"], 0.0) + curve["v_hat"]
    out = run_dir / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    performance.to_parquet(out / "table_r4_performance.parquet")
    curve.to_parquet(out / "table_r4_curve.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for estimand_name, group in performance.groupby("estimand", observed=True):
        direct = group[
            group["method"].isin(
                [
                    "empirical_grid_oracle",
                    "externally_piloted_direct_grid",
                    "uncertainty_buffered_external_grid",
                ]
            )
        ]
        for method, method_group in direct.groupby("method", observed=True):
            axes[0].plot(
                method_group["n"],
                method_group["mse_ratio_to_real"],
                "o-",
                label=f"{estimand_name}: {method}",
            )
    axes[0].axhline(1.0, color="black", linestyle=":")
    axes[0].set_xlabel("labeled budget n")
    axes[0].set_ylabel("held-out MSE ratio to real-only")
    axes[0].legend(fontsize=6)
    for (n, estimand_name), group in curve.groupby(
        ["n", "estimand"], observed=True
    ):
        axes[1].loglog(
            group["x_selected"],
            group["B_score"],
            "o-",
            label=f"{estimand_name}, n={n}",
        )
    axes[1].set_xlabel("calibration size")
    axes[1].set_ylabel("estimated synthetic error")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, run_dir / "figures", "figure_r4_adult_auxiliary")
    return {"raw": raw, "performance": performance, "curve": curve}
