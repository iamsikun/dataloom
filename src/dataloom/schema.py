"""Master output schema (§11 of docs/experiments.md).

Every experiment writes a long-format Parquet file with one row per
(replication, method, estimand) using exactly these columns. Fields that
do not apply to a given (experiment, method) combination are stored as
NA, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd


# Order matters: this is the on-disk column order too.
MASTER_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "replication",
    "seed",
    "n",
    "m",
    "rho",
    "beta",
    "a",
    "sigma_s2",
    "B0",
    "c",
    "dataset",
    "generator",
    "estimand",
    "method",
    "theta_true",
    "theta_hat",
    "error",
    "squared_error",
    "x_selected",
    "lambda_selected",
    "n_e",
    "alpha_selected",
    "B_eff_selected",
    "V_R_selected",
    "beta_hat",
    "c_hat",
    "a_hat",
    "v_hat",
    "oracle_x",
    "oracle_lambda",
    "oracle_alpha",
    "oracle_risk",
    "estimated_risk_selected",
    "true_risk_selected",
    "safe_pass",
    "safe_margin",
    "fallback_used",
    "ci_lower",
    "ci_upper",
    "ci_length",
    "covered",
    "runtime_seconds",
    "failure_flag",
    "failure_reason",
)


# Nullable pandas dtypes for the columns where dtype matters for parquet.
COLUMN_DTYPES: dict[str, str] = {
    "experiment_id": "string",
    "replication": "Int64",
    "seed": "Int64",
    "n": "Int64",
    "m": "Int64",
    "rho": "Float64",
    "beta": "Float64",
    "a": "Float64",
    "sigma_s2": "Float64",
    "B0": "Float64",
    "c": "Float64",
    "dataset": "string",
    "generator": "string",
    "estimand": "string",
    "method": "string",
    "theta_true": "Float64",
    "theta_hat": "Float64",
    "error": "Float64",
    "squared_error": "Float64",
    "x_selected": "Int64",
    "lambda_selected": "Float64",
    "n_e": "Int64",
    "alpha_selected": "Float64",
    "B_eff_selected": "Float64",
    "V_R_selected": "Float64",
    "beta_hat": "Float64",
    "c_hat": "Float64",
    "a_hat": "Float64",
    "v_hat": "Float64",
    "oracle_x": "Int64",
    "oracle_lambda": "Float64",
    "oracle_alpha": "Float64",
    "oracle_risk": "Float64",
    "estimated_risk_selected": "Float64",
    "true_risk_selected": "Float64",
    "safe_pass": "boolean",
    "safe_margin": "Float64",
    "fallback_used": "boolean",
    "ci_lower": "Float64",
    "ci_upper": "Float64",
    "ci_length": "Float64",
    "covered": "boolean",
    "runtime_seconds": "Float64",
    "failure_flag": "boolean",
    "failure_reason": "string",
}


@dataclass
class Row:
    """One row in the master output schema. All fields default to None
    so callers only set what's applicable. The writer fills NA for the rest.
    """

    experiment_id: str | None = None
    replication: int | None = None
    seed: int | None = None
    n: int | None = None
    m: int | None = None
    rho: float | None = None
    beta: float | None = None
    a: float | None = None
    sigma_s2: float | None = None
    B0: float | None = None
    c: float | None = None
    dataset: str | None = None
    generator: str | None = None
    estimand: str | None = None
    method: str | None = None
    theta_true: float | None = None
    theta_hat: float | None = None
    error: float | None = None
    squared_error: float | None = None
    x_selected: int | None = None
    lambda_selected: float | None = None
    n_e: int | None = None
    alpha_selected: float | None = None
    B_eff_selected: float | None = None
    V_R_selected: float | None = None
    beta_hat: float | None = None
    c_hat: float | None = None
    a_hat: float | None = None
    v_hat: float | None = None
    oracle_x: int | None = None
    oracle_lambda: float | None = None
    oracle_alpha: float | None = None
    oracle_risk: float | None = None
    estimated_risk_selected: float | None = None
    true_risk_selected: float | None = None
    safe_pass: bool | None = None
    safe_margin: float | None = None
    fallback_used: bool | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_length: float | None = None
    covered: bool | None = None
    runtime_seconds: float | None = None
    failure_flag: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def rows_to_dataframe(rows: list[dict[str, Any]] | list[Row]) -> pd.DataFrame:
    """Build a DataFrame in MASTER_COLUMNS order with nullable dtypes,
    deriving error / squared_error / ci_length / covered when missing.
    """
    if rows and isinstance(rows[0], Row):
        rows = [r.to_dict() for r in rows]
    df = pd.DataFrame(rows)
    # Add any missing columns as NA so the on-disk schema is stable.
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    # Drop unexpected columns (forces callers to extend MASTER_COLUMNS, not bypass it).
    df = df[list(MASTER_COLUMNS)]

    # Derived columns.
    if df["error"].isna().all():
        df["error"] = df["theta_hat"] - df["theta_true"]
    if df["squared_error"].isna().all():
        df["squared_error"] = (df["theta_hat"] - df["theta_true"]) ** 2
    if df["ci_length"].isna().all():
        df["ci_length"] = df["ci_upper"] - df["ci_lower"]
    if df["covered"].isna().all():
        # covered := ci_lower <= theta_true <= ci_upper, NA where bounds missing
        lo = df["ci_lower"]
        hi = df["ci_upper"]
        truth = df["theta_true"]
        mask = lo.notna() & hi.notna() & truth.notna()
        cov = pd.Series(pd.NA, index=df.index, dtype="boolean")
        cov.loc[mask] = (lo.loc[mask] <= truth.loc[mask]) & (
            truth.loc[mask] <= hi.loc[mask]
        )
        df["covered"] = cov
    if df["lambda_selected"].isna().all():
        df["lambda_selected"] = df["x_selected"] / df["n"]
    if df["n_e"].isna().all():
        df["n_e"] = df["n"] - df["x_selected"].fillna(0).astype("Int64")

    # Cast to the declared dtypes.
    for col, dtype in COLUMN_DTYPES.items():
        try:
            df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            # Float coercion can fail when a column is all-None; just leave as object.
            pass
    return df
