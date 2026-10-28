"""Adult pseudo-label helpers for the auxiliary-covariate experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..datasets.adult import CACHE_FILE


FEATURES = (
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain",
    "capital_loss",
    "sex_binary",
)
REQUIRED_REAL_COLUMNS = {
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "native_country",
    "income_binary",
    "sex_binary",
    *FEATURES,
}
ESTIMANDS = ("income_mean", "subgroup_gap")


def load_required_adult(path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load the cached genuine UCI table and reject the synthetic stand-in."""
    cache_path = Path(path) if path is not None else CACHE_FILE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"genuine cached Adult data required at {cache_path}; "
            "the rebuttal experiment will not use the synthetic fallback"
        )
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    frame = pd.read_parquet(cache_path)
    missing = REQUIRED_REAL_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(
            "cached Adult table is not the genuine UCI schema; "
            f"missing columns: {sorted(missing)}"
        )
    return frame.reset_index(drop=True), digest


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Fixed low-dimensional outcome-model design using unlabeled covariates."""
    x = frame[list(FEATURES)].astype(float).to_numpy(copy=True)
    x[:, 3:5] = np.log1p(np.maximum(x[:, 3:5], 0.0))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    x = (x - mean) / np.where(scale > 0, scale, 1.0)
    return np.column_stack((np.ones(len(x)), x))


def fit_ridge_logit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 2.0,
    max_iter: int = 40,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Fit a ridge-logistic model by stable Newton iterations."""
    weights = np.zeros(x.shape[1], dtype=float)
    penalty = np.ones(x.shape[1], dtype=float)
    penalty[0] = 0.0
    for _ in range(max_iter):
        linear = np.clip(x @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        working = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = x.T @ (probability - y) + ridge * penalty * weights
        hessian = (x.T * working) @ x + ridge * np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < tolerance:
            break
    return weights


def predict_probability(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    linear = np.clip(x @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def estimand(values: np.ndarray, sex: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=float)
    sex = np.asarray(sex, dtype=int)
    if name == "income_mean":
        return float(values.mean())
    if name == "subgroup_gap":
        return float(values[sex == 1].mean() - values[sex == 0].mean())
    raise KeyError(name)


def influence_values(values: np.ndarray, sex: np.ndarray, name: str) -> np.ndarray:
    """Influence values whose variance is the real-estimator constant a."""
    values = np.asarray(values, dtype=float)
    sex = np.asarray(sex, dtype=int)
    if name == "income_mean":
        return values - values.mean()
    if name == "subgroup_gap":
        p1 = float(np.mean(sex == 1))
        p0 = 1.0 - p1
        if p1 <= 0 or p0 <= 0:
            return np.full(len(values), np.nan)
        mean1 = float(values[sex == 1].mean())
        mean0 = float(values[sex == 0].mean())
        return (
            (sex == 1) * (values - mean1) / p1
            - (sex == 0) * (values - mean0) / p0
        )
    raise KeyError(name)


def variance_constant(values: np.ndarray, sex: np.ndarray, name: str) -> float:
    influence = influence_values(values, sex, name)
    return float(np.var(influence, ddof=1))


def fit_power_law(
    x: np.ndarray, bias2: np.ndarray
) -> tuple[float, float, float]:
    positive = np.asarray(bias2) > 0
    if int(positive.sum()) < 3:
        return 0.0, float(np.max(bias2)), float("nan")
    log_x = np.log(np.asarray(x, dtype=float)[positive])
    log_y = np.log(np.asarray(bias2, dtype=float)[positive])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    prediction = slope * log_x + intercept
    residual = float(np.sum((log_y - prediction) ** 2))
    total = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return max(float(-slope / 2.0), 0.0), float(np.exp(intercept)), r2


def fit_plateau(
    x: np.ndarray, bias2: np.ndarray
) -> tuple[float, float, float]:
    """Fit floor + c*x^(-2 beta) with nonnegative log-parameters."""
    x = np.asarray(x, dtype=float)
    y = np.maximum(np.asarray(bias2, dtype=float), 1e-14)

    def residual(parameters: np.ndarray) -> np.ndarray:
        floor, c, beta = np.exp(parameters)
        prediction = floor + c * x ** (-2.0 * beta)
        return np.log(prediction) - np.log(y)

    initial = np.log([max(float(y.min()) / 2.0, 1e-12), float(y.max()), 0.5])
    result = least_squares(residual, initial, max_nfev=500)
    floor, c, beta = np.exp(result.x)
    return float(floor), float(c), float(beta)
