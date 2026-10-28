"""Rebuttal Experiment R4: Adult pseudo-labeling on auxiliary covariates."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

from ..io.config import RunConfig, prepare_run_dir
from ..io.results import ResultsWriter
from ..io.seeds import make_rng, replication_seed
from ..rebuttal.adult_auxiliary import (
    ESTIMANDS,
    design_matrix,
    estimand,
    fit_plateau,
    fit_power_law,
    fit_ridge_logit,
    load_required_adult,
    predict_probability,
    variance_constant,
)
from ..rebuttal.bounds import buffered_safe_condition


POLICY_METHODS = (
    "real_only_all",
    "synthetic_only_full_calibration",
    "product_surrogate_fixed_share_oracle_alpha",
    "empirical_grid_oracle",
    "externally_piloted_direct_grid",
    "uncertainty_buffered_external_grid",
)


def _profile_risk(n: int, a: float, x: int, b: float) -> float:
    if x == 0:
        return a / n
    if x == n:
        return b
    v = a / (n - x)
    return float(v * b / (v + b))


def _alpha(n: int, a: float, x: int, b: float) -> float:
    if x == 0:
        return 1.0
    if x == n:
        return 0.0
    v = a / (n - x)
    return float(b / (b + v))


def _candidate_grid(n: int, base: list[int]) -> np.ndarray:
    interior = [value for value in base if 0 < value < n]
    return np.asarray(sorted(set([*interior, n])), dtype=int)


def _pilot_rows(
    *,
    config: RunConfig,
    n: int,
    grid: np.ndarray,
    replications: int,
    validation_size: int,
    auxiliary_size: int,
    x_population: np.ndarray,
    y_population: np.ndarray,
    sex_population: np.ndarray,
    truths: dict[str, float],
    ridge: float,
) -> list[dict]:
    rows: list[dict] = []
    population_size = len(y_population)
    max_x = int(grid.max())
    for rep in range(replications):
        seed = replication_seed(
            config.seed_root, f"{config.experiment_id}:pilot", (n,), rep
        )
        rng = make_rng(seed)
        calibration_indices = rng.integers(0, population_size, size=max_x)
        validation_indices = rng.integers(
            0, population_size, size=validation_size
        )
        auxiliary_indices = rng.integers(
            0, population_size, size=auxiliary_size
        )
        validation_y = y_population[validation_indices]
        validation_sex = sex_population[validation_indices]
        auxiliary_sex = sex_population[auxiliary_indices]
        for x in grid:
            calibration = calibration_indices[: int(x)]
            weights = fit_ridge_logit(
                x_population[calibration],
                y_population[calibration],
                ridge=ridge,
            )
            predicted = predict_probability(
                x_population[auxiliary_indices], weights
            )
            for name in ESTIMANDS:
                theta_s = estimand(predicted, auxiliary_sex, name)
                theta_v = estimand(validation_y, validation_sex, name)
                a_hat = variance_constant(validation_y, validation_sex, name)
                v_hat = (
                    variance_constant(predicted, auxiliary_sex, name)
                    / auxiliary_size
                )
                score = (theta_s - theta_v) ** 2 - v_hat - a_hat / validation_size
                rows.append(
                    {
                        "experiment_id": config.experiment_id,
                        "replication": rep,
                        "seed": seed,
                        "n": n,
                        "m": auxiliary_size,
                        "dataset": "uci_adult",
                        "generator": "ridge_logistic_pseudo_label",
                        "estimand": name,
                        "method": "external_pilot_curve",
                        "theta_true": truths[name],
                        "theta_hat": theta_s,
                        "x_selected": int(x),
                        "a_hat": a_hat,
                        "v_hat": v_hat,
                        "bias2_true": (theta_s - truths[name]) ** 2 - v_hat,
                        "score_untruncated": score,
                        "B_true": (theta_s - truths[name]) ** 2,
                        "validation_size": validation_size,
                        "pilot_fits": replications,
                        "auxiliary_size": auxiliary_size,
                        "failure_flag": False,
                    }
                )
    return rows


def _build_policies(
    pilot: pd.DataFrame,
    *,
    n: int,
    alpha_level: float,
) -> list[dict]:
    policies: list[dict] = []
    for name, group in pilot.groupby("estimand", observed=True):
        curve = (
            group.groupby("x_selected", observed=True)
            .agg(
                B_true=("B_true", "mean"),
                score=("score_untruncated", "mean"),
                score_sd=("score_untruncated", "std"),
                v_hat=("v_hat", "mean"),
                v_sd=("v_hat", "std"),
                a_hat=("a_hat", "mean"),
                a_sd=("a_hat", "std"),
                count=("replication", "count"),
            )
            .reset_index()
            .sort_values("x_selected")
        )
        count = int(curve["count"].iloc[0])
        grid_size = len(curve)
        critical = float(t.ppf(1.0 - alpha_level / (2 * grid_size), count - 1))
        curve["B_hat"] = np.maximum(curve["score"], 0.0) + curve["v_hat"]
        curve["U_B"] = (
            np.maximum(
                curve["score"] + critical * curve["score_sd"] / math.sqrt(count),
                0.0,
            )
            + curve["v_hat"]
            + critical * curve["v_sd"] / math.sqrt(count)
        )
        a_hat = float(curve["a_hat"].mean())
        l_a = max(
            0.0,
            float(
                curve["a_hat"].mean()
                - critical * curve["a_hat"].std(ddof=1) / math.sqrt(grid_size)
            ),
        )
        grid = curve["x_selected"].to_numpy(dtype=int)
        bias2 = np.maximum(
            curve["B_hat"].to_numpy() - curve["v_hat"].to_numpy(), 0.0
        )
        beta_hat, c_hat, powerlaw_r2 = fit_power_law(grid, bias2)
        floor_hat, plateau_c, plateau_beta = fit_plateau(grid, bias2)
        monotonicity_violations = int(
            np.sum(np.diff(curve["B_hat"].to_numpy()) > 0)
        )

        def add_policy(method: str, x: int, b: float, fallback: bool = False):
            policies.append(
                {
                    "n": n,
                    "estimand": name,
                    "method": method,
                    "x_selected": int(x),
                    "alpha_selected": _alpha(n, a_hat, int(x), float(b))
                    if x > 0
                    else 1.0,
                    "B_hat": None if x == 0 else float(b),
                    "U_B": None
                    if x == 0
                    else float(
                        np.interp(
                            x,
                            grid,
                            curve["U_B"].to_numpy(),
                        )
                    ),
                    "L_a": l_a,
                    "a_hat": a_hat,
                    "beta_hat": beta_hat,
                    "c_hat": c_hat,
                    "powerlaw_r2": powerlaw_r2,
                    "fallback_used": fallback,
                    "plateau_floor": floor_hat,
                    "plateau_c": plateau_c,
                    "plateau_beta": plateau_beta,
                    "monotonicity_violations": monotonicity_violations,
                }
            )

        add_policy("real_only_all", 0, 0.0)
        synth = curve[curve["x_selected"] == n].iloc[0]
        add_policy(
            "synthetic_only_full_calibration", n, float(synth["B_hat"])
        )

        true_risks = np.asarray(
            [
                _profile_risk(n, a_hat, int(x), float(b))
                for x, b in zip(grid, curve["B_true"])
            ]
        )
        oracle_index = int(np.argmin(np.concatenate(([a_hat / n], true_risks))))
        if oracle_index == 0:
            add_policy("empirical_grid_oracle", 0, 0.0)
        else:
            row = curve.iloc[oracle_index - 1]
            add_policy(
                "empirical_grid_oracle",
                int(row["x_selected"]),
                float(row["B_true"]),
            )

        estimated_risks = np.asarray(
            [
                _profile_risk(n, a_hat, int(x), float(b))
                for x, b in zip(grid, curve["B_hat"])
            ]
        )
        direct_index = int(
            np.argmin(np.concatenate(([a_hat / n], estimated_risks)))
        )
        if direct_index == 0:
            direct_x, direct_b, direct_u = 0, 0.0, float("inf")
        else:
            row = curve.iloc[direct_index - 1]
            direct_x = int(row["x_selected"])
            direct_b = float(row["B_hat"])
            direct_u = float(row["U_B"])
        add_policy("externally_piloted_direct_grid", direct_x, direct_b)
        if buffered_safe_condition(direct_x, direct_u, l_a):
            add_policy(
                "uncertainty_buffered_external_grid", direct_x, direct_b
            )
        else:
            add_policy(
                "uncertainty_buffered_external_grid",
                0,
                0.0,
                fallback=True,
            )

        beta_for_share = max(beta_hat, 1e-3)
        product_x = max(
            1,
            min(
                n - 1,
                int(
                    np.floor(
                        n
                        * (2.0 * beta_for_share)
                        / (1.0 + 2.0 * beta_for_share)
                    )
                ),
            ),
        )
        v_product = float(np.interp(product_x, grid, curve["v_hat"]))
        b_product = v_product + c_hat * product_x ** (-2.0 * beta_for_share)
        add_policy(
            "product_surrogate_fixed_share_oracle_alpha",
            product_x,
            b_product,
        )
    return policies


def _evaluation_rows(
    *,
    config: RunConfig,
    n: int,
    policies: pd.DataFrame,
    replications: int,
    auxiliary_size: int,
    x_population: np.ndarray,
    y_population: np.ndarray,
    sex_population: np.ndarray,
    truths: dict[str, float],
    ridge: float,
) -> list[dict]:
    rows: list[dict] = []
    population_size = len(y_population)
    unique_x = sorted(
        set(int(value) for value in policies["x_selected"] if int(value) > 0)
    )
    for rep in range(replications):
        seed = replication_seed(
            config.seed_root, f"{config.experiment_id}:evaluation", (n,), rep
        )
        rng = make_rng(seed)
        labeled = rng.integers(0, population_size, size=n)
        auxiliary = rng.integers(0, population_size, size=auxiliary_size)
        permutation = rng.permutation(n)
        auxiliary_sex = sex_population[auxiliary]
        estimates: dict[tuple[int, str], tuple[float, float | None]] = {}
        for name in ESTIMANDS:
            estimates[(0, name)] = (
                estimand(y_population[labeled], sex_population[labeled], name),
                None,
            )
        for x in unique_x:
            calibration = labeled[permutation[:x]]
            weights = fit_ridge_logit(
                x_population[calibration],
                y_population[calibration],
                ridge=ridge,
            )
            predicted = predict_probability(x_population[auxiliary], weights)
            estimation_indices = labeled[permutation[x:]]
            for name in ESTIMANDS:
                theta_s = estimand(predicted, auxiliary_sex, name)
                theta_r = (
                    None
                    if x == n
                    else estimand(
                        y_population[estimation_indices],
                        sex_population[estimation_indices],
                        name,
                    )
                )
                estimates[(x, name)] = (theta_s, theta_r)

        for policy in policies.to_dict("records"):
            name = str(policy["estimand"])
            method = str(policy["method"])
            x = int(policy["x_selected"])
            weight = float(policy["alpha_selected"])
            if x == 0:
                theta_hat = estimates[(0, name)][0]
            else:
                theta_s, theta_r = estimates[(x, name)]
                theta_hat = (
                    theta_s
                    if theta_r is None
                    else weight * theta_r + (1.0 - weight) * theta_s
                )
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "replication": rep,
                    "seed": seed,
                    "n": n,
                    "m": auxiliary_size,
                    "dataset": "uci_adult",
                    "generator": "ridge_logistic_pseudo_label",
                    "estimand": name,
                    "method": method,
                    "theta_true": truths[name],
                    "theta_hat": theta_hat,
                    "x_selected": x,
                    "alpha_selected": weight,
                    "B_hat": policy.get("B_hat"),
                    "U_B": policy.get("U_B"),
                    "L_a": policy.get("L_a"),
                    "a_hat": policy.get("a_hat"),
                    "beta_hat": policy.get("beta_hat"),
                    "c_hat": policy.get("c_hat"),
                    "powerlaw_r2": policy.get("powerlaw_r2"),
                    "fallback_used": policy.get("fallback_used"),
                    "pilot_fits": int(policy.get("pilot_fits", 0)),
                    "auxiliary_size": auxiliary_size,
                    "failure_flag": False,
                }
            )
    return rows


def run(config_path: str | Path) -> Path:
    with open(config_path) as handle:
        raw = yaml.safe_load(handle)
    config = RunConfig.from_dict(raw)
    run_dir = prepare_run_dir(config, config_path)
    frame, digest = load_required_adult(raw.get("adult_cache"))
    (run_dir / "adult_source.json").write_text(
        json.dumps(
            {
                "path": str(raw.get("adult_cache") or "default_cache"),
                "sha256": digest,
                "rows": len(frame),
                "columns": list(frame.columns),
            },
            indent=2,
        )
    )
    x_population = design_matrix(frame)
    y_population = frame["income_binary"].to_numpy(dtype=float)
    sex_population = frame["sex_binary"].to_numpy(dtype=int)
    truths = {
        name: estimand(y_population, sex_population, name) for name in ESTIMANDS
    }
    pilot_replications = int(raw["pilot_replications"])
    evaluation_replications = int(raw["evaluation_replications"])
    validation_size = int(raw["validation_size"])
    auxiliary_size = int(raw["auxiliary_size"])
    ridge = float(raw.get("ridge", 2.0))
    alpha_level = float(raw.get("alpha_level", 0.05))
    base_grid = [int(value) for value in raw["candidate_grid"]]
    writer = ResultsWriter(run_dir / "raw")
    all_policies: list[dict] = []
    worker = 0
    for n in map(int, raw["n_grid"]):
        grid = _candidate_grid(n, base_grid)
        pilot_rows = _pilot_rows(
            config=config,
            n=n,
            grid=grid,
            replications=pilot_replications,
            validation_size=validation_size,
            auxiliary_size=auxiliary_size,
            x_population=x_population,
            y_population=y_population,
            sex_population=sex_population,
            truths=truths,
            ridge=ridge,
        )
        pilot_frame = pd.DataFrame(pilot_rows)
        policies = _build_policies(
            pilot_frame, n=n, alpha_level=alpha_level
        )
        for policy in policies:
            policy["pilot_fits"] = pilot_replications
        policy_frame = pd.DataFrame(policies)
        all_policies.extend(policies)
        evaluation_rows = _evaluation_rows(
            config=config,
            n=n,
            policies=policy_frame,
            replications=evaluation_replications,
            auxiliary_size=auxiliary_size,
            x_population=x_population,
            y_population=y_population,
            sex_population=sex_population,
            truths=truths,
            ridge=ridge,
        )
        writer.write_cell(
            {"n": n, "stage": "pilot"}, pilot_rows, worker_id=worker
        )
        worker += 1
        writer.write_cell(
            {"n": n, "stage": "evaluation"},
            evaluation_rows,
            worker_id=worker,
        )
        worker += 1
    policy_dir = run_dir / "aggregated"
    policy_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_policies).to_parquet(policy_dir / "adult_policy.parquet")
    return run_dir
