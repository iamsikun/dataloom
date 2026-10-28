"""Master schema writer tests."""

from __future__ import annotations

import pandas as pd

from dataloom.schema import MASTER_COLUMNS, Row, rows_to_dataframe


def test_rows_to_dataframe_preserves_column_order():
    df = rows_to_dataframe([Row(experiment_id="x", n=10, theta_true=0.0,
                                theta_hat=1.0, method="real_only_all")])
    assert list(df.columns) == list(MASTER_COLUMNS)


def test_partial_row_fills_na_for_unset_columns():
    df = rows_to_dataframe([Row(method="real_only_all", theta_hat=0.5,
                                theta_true=0.0, n=100)])
    # x_selected, alpha_selected etc. should be NA
    assert df["x_selected"].isna().all()
    assert df["alpha_selected"].isna().all()
    assert df["B_eff_selected"].isna().all()


def test_derived_columns_computed():
    df = rows_to_dataframe([Row(theta_true=0.0, theta_hat=2.0, n=100,
                                method="m", ci_lower=1.0, ci_upper=3.0,
                                x_selected=10)])
    assert df["error"].iloc[0] == 2.0
    assert df["squared_error"].iloc[0] == 4.0
    assert df["ci_length"].iloc[0] == 2.0
    # truth 0.0 not in [1, 3] -> covered = False
    assert df["covered"].iloc[0] == False  # noqa: E712
    assert df["lambda_selected"].iloc[0] == 0.1
    assert df["n_e"].iloc[0] == 90


def test_dtypes_are_nullable():
    df = rows_to_dataframe([Row(n=100, theta_hat=0.5, theta_true=0.0,
                                method="m")])
    assert str(df["x_selected"].dtype) == "Int64"
    assert str(df["safe_pass"].dtype) == "boolean"
    assert str(df["B_eff_selected"].dtype) == "Float64"
