"""log x_n* vs log n regression (§3.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_scaling_slopes(cell_oracle: pd.DataFrame) -> pd.DataFrame:
    """For each (beta, rho), regress log(oracle_x) on log(n).

    Drops rows where oracle_x is 0 (boundary regime) before regressing.
    Returns a DataFrame with columns: beta, rho, empirical_slope, slope_se,
    n_points, n_dropped_boundary.
    """
    rows = []
    for (beta, rho), grp in cell_oracle.groupby(["beta", "rho"], observed=True):
        keep = grp[(grp["oracle_x"] > 0) & (grp["oracle_x"] < grp["n"])].copy()
        n_dropped = len(grp) - len(keep)
        if len(keep) < 2:
            rows.append({
                "beta": beta, "rho": rho,
                "empirical_slope": np.nan, "slope_se": np.nan,
                "n_points": len(keep), "n_dropped_boundary": n_dropped,
            })
            continue
        log_n = np.log(keep["n"].to_numpy(dtype=float))
        log_x = np.log(keep["oracle_x"].to_numpy(dtype=float))
        # OLS via polyfit; SE via residual variance.
        slope, intercept = np.polyfit(log_n, log_x, 1)
        resid = log_x - (slope * log_n + intercept)
        if len(log_n) > 2:
            sigma2 = float(np.sum(resid ** 2) / (len(log_n) - 2))
            ss_x = float(np.sum((log_n - log_n.mean()) ** 2))
            slope_se = float(np.sqrt(sigma2 / ss_x)) if ss_x > 0 else np.nan
        else:
            slope_se = np.nan
        rows.append({
            "beta": beta, "rho": rho,
            "empirical_slope": float(slope), "slope_se": slope_se,
            "n_points": len(keep), "n_dropped_boundary": n_dropped,
        })
    return pd.DataFrame(rows)
