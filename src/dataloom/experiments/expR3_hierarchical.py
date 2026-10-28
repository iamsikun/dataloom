"""Rebuttal Experiment R3: adaptive score concentration and buffered safety."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import chi2, t

from ..adaptive.bias_curve import default_pilot_grid
from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..oracle import oracle_grid_curve
from ..rebuttal.bounds import buffered_safe_condition


METHODS = (
    "real_only_all",
    "grid_oracle",
    "direct_grid_plugin",
    "uncertainty_buffered_direct_grid",
)


def _risk(n: int, a: float, x: int, b: float) -> float:
    if x == 0:
        return a / n
    if x == n:
        return b
    v = a / (n - x)
    return float(v * b / (v + b))


def _select(grid: np.ndarray, risks: np.ndarray) -> int:
    return int(grid[int(np.argmin(risks))])


def _final_estimate(
    rng: np.random.Generator,
    *,
    n: int,
    x: int,
    alpha: float,
    a: float,
    c: float,
    beta: float,
    v_n: float,
) -> float:
    if x == 0:
        return float(rng.normal(0.0, math.sqrt(a / n)))
    scale = math.sqrt(c / 2.0) * x ** (-beta)
    delta = float(rng.normal(scale, scale))
    theta_s = float(rng.normal(delta, math.sqrt(v_n)))
    if x == n:
        return theta_s
    theta_r = float(rng.normal(0.0, math.sqrt(a / (n - x))))
    return float(alpha * theta_r + (1.0 - alpha) * theta_s)


def _one_replication(
    *,
    rng: np.random.Generator,
    seed: int,
    n: int,
    beta: float,
    rho: float,
    validation_size: int,
    pilot_fits: int,
    redraws: int,
    alpha_level: float,
) -> tuple[list[dict], list[dict]]:
    a = c = 1.0
    v_n = n ** (-rho)
    pilot_x = default_pilot_grid(n)
    true_bias2 = c * pilot_x.astype(float) ** (-2.0 * beta)
    true_b = v_n + true_bias2
    grid = np.concatenate(([0], pilot_x))
    true_curve = lambda x: v_n + c * np.asarray(x, dtype=float) ** (
        -2.0 * beta
    )
    oracle = oracle_grid_curve(
        n=n,
        a=a,
        B_curve=true_curve,
        grid=np.concatenate(([0], pilot_x)),
        include_boundaries=False,
    )

    df_a = validation_size - 1
    a_hat = float(a * rng.chisquare(df_a) / df_a)
    validation_mean = float(rng.normal(0.0, math.sqrt(a / validation_size)))
    l_a = float(df_a * a_hat / chi2.ppf(1.0 - alpha_level / 2.0, df_a))
    b_hat = np.empty(len(pilot_x))
    u_b = np.empty(len(pilot_x))
    scores = np.empty(len(pilot_x))
    diagnostics: list[dict] = []
    per_curve_alpha = alpha_level / (2.0 * len(pilot_x))

    for j, x in enumerate(pilot_x):
        scale = math.sqrt(c / 2.0) * float(x) ** (-beta)
        delta = rng.normal(scale, scale, size=pilot_fits)
        synth = rng.normal(
            delta[:, None], math.sqrt(v_n), size=(pilot_fits, redraws)
        )
        means = synth.mean(axis=1)
        within = synth.var(axis=1, ddof=1)
        score_values = (
            (means - validation_mean) ** 2
            - within / redraws
            - a_hat / validation_size
        )
        score_mean = float(score_values.mean())
        pooled_v = float(within.mean())
        pooled_df = pilot_fits * (redraws - 1)
        v_upper = float(
            pooled_df * pooled_v / chi2.ppf(per_curve_alpha, pooled_df)
        )
        if pilot_fits < 2:
            bias_upper = float("inf")
        else:
            critical = float(t.ppf(1.0 - per_curve_alpha, pilot_fits - 1))
            bias_upper = float(
                score_mean
                + critical
                * np.std(score_values, ddof=1)
                / math.sqrt(pilot_fits)
            )
        scores[j] = score_mean
        b_hat[j] = pooled_v + max(score_mean, 0.0)
        u_b[j] = (
            float("inf")
            if not np.isfinite(bias_upper)
            else v_upper + max(bias_upper, 0.0)
        )
        diagnostics.append(
            {
                "x_selected": int(x),
                "bias2_true": true_bias2[j],
                "score_untruncated": score_mean,
                "B_true": true_b[j],
                "B_hat": b_hat[j],
                "U_B": u_b[j],
                "L_a": l_a,
            }
        )

    finite_positive = np.maximum(b_hat, np.finfo(float).tiny)
    curve_error = float(
        np.exp(np.max(np.abs(np.log(finite_positive / true_b))))
    )
    plugin_risks = np.concatenate(
        ([a_hat / n], [_risk(n, a_hat, int(x), float(b)) for x, b in zip(pilot_x, b_hat)])
    )
    plugin_x = _select(grid, plugin_risks)
    plugin_index = None if plugin_x == 0 else int(np.where(pilot_x == plugin_x)[0][0])
    if plugin_x == 0:
        plugin_b = None
        plugin_alpha = 1.0
    else:
        plugin_b = float(b_hat[plugin_index])
        plugin_v = a_hat / (n - plugin_x)
        plugin_alpha = float(plugin_b / (plugin_b + plugin_v))

    buffered_pass = bool(
        plugin_x > 0
        and buffered_safe_condition(
            plugin_x, float(u_b[plugin_index]), l_a
        )
    )
    buffered_x = plugin_x if buffered_pass else 0
    buffered_alpha = plugin_alpha if buffered_pass else 1.0

    method_specs = {
        "real_only_all": (0, 1.0, False, False),
        "grid_oracle": (oracle.x_star, oracle.alpha_star, False, True),
        "direct_grid_plugin": (plugin_x, plugin_alpha, plugin_x == 0, plugin_x > 0),
        "uncertainty_buffered_direct_grid": (
            buffered_x,
            buffered_alpha,
            not buffered_pass,
            buffered_pass,
        ),
    }
    method_rows: list[dict] = []
    oracle_risk = oracle.risk_star
    for method, (x, weight, fallback, safe_pass) in method_specs.items():
        final_seed = replication_seed(
            seed,
            f"expR3_hierarchical:final:{method}",
            (n, int(round(beta * 1000)), int(x)),
            0,
        )
        final = _final_estimate(
            make_rng(final_seed),
            n=n,
            x=int(x),
            alpha=float(weight),
            a=a,
            c=c,
            beta=beta,
            v_n=v_n,
        )
        b_true_selected = None if x == 0 else float(true_curve(x))
        true_risk = (
            a / n
            if x == 0
            else _risk(n, a, int(x), float(b_true_selected))
        )
        method_rows.append(
            {
                "method": method,
                "theta_hat": final,
                "x_selected": int(x),
                "alpha_selected": float(weight),
                "B_true": b_true_selected,
                "B_hat": None if x == 0 else plugin_b,
                "L_a": l_a,
                "uniform_curve_error": curve_error,
                "working_risk_ratio": true_risk / oracle_risk,
                "true_risk_selected": true_risk,
                "oracle_x": oracle.x_star,
                "oracle_lambda": oracle.x_star / n,
                "oracle_alpha": oracle.alpha_star,
                "oracle_risk": oracle_risk,
                "fallback_used": bool(fallback),
                "safe_pass": bool(safe_pass),
            }
        )
    return method_rows, diagnostics


def run(config_path: str | Path) -> Path:
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    config = RunConfig.from_dict(raw)
    run_dir = prepare_run_dir(config, config_path)
    writer = ResultsWriter(run_dir / "raw")
    replications = int(raw["replications"]["default"])
    alpha_level = float(raw.get("alpha_level", 0.05))
    worker = 0
    for beta, rho in raw["regimes"]:
        beta = float(beta)
        rho = float(rho)
        for n in map(int, raw["n_grid"]):
            for validation_factor in map(float, raw["validation_factor_grid"]):
                validation_size = max(2, int(round(validation_factor * n)))
                for pilot_fits in map(int, raw["pilot_fits_grid"]):
                    for redraws in map(int, raw["redraws_grid"]):
                        rows: list[dict] = []
                        cell_key = (
                            n,
                            int(round(beta * 1000)),
                            int(round(rho * 1000)),
                            validation_size,
                            pilot_fits,
                            redraws,
                        )
                        for rep in range(replications):
                            seed = replication_seed(
                                config.seed_root,
                                config.experiment_id,
                                cell_key,
                                rep,
                            )
                            method_rows, diagnostics = _one_replication(
                                rng=make_rng(seed),
                                seed=seed,
                                n=n,
                                beta=beta,
                                rho=rho,
                                validation_size=validation_size,
                                pilot_fits=pilot_fits,
                                redraws=redraws,
                                alpha_level=alpha_level,
                            )
                            common = {
                                "experiment_id": config.experiment_id,
                                "replication": rep,
                                "seed": seed,
                                "n": n,
                                "m": int(np.ceil(n**rho)),
                                "rho": rho,
                                "beta": beta,
                                "a": 1.0,
                                "sigma_s2": 1.0,
                                "B0": 1.0,
                                "c": 1.0,
                                "dataset": "hierarchical_gaussian",
                                "generator": "random_calibrated_mean",
                                "estimand": "mean",
                                "theta_true": 0.0,
                                "validation_size": validation_size,
                                "pilot_fits": pilot_fits,
                                "redraws_per_fit": redraws,
                                "failure_flag": False,
                            }
                            rows.extend([{**common, **row} for row in method_rows])
                            rows.extend(
                                [
                                    {
                                        **common,
                                        **row,
                                        "method": "score_diagnostic",
                                    }
                                    for row in diagnostics
                                ]
                            )
                        writer.write_cell(
                            {
                                "beta": beta,
                                "n": n,
                                "nv": validation_size,
                                "k": pilot_fits,
                                "l": redraws,
                            },
                            rows,
                            worker_id=worker,
                        )
                        worker += 1
    return run_dir
