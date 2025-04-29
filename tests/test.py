#!/usr/bin/env python3

"""
Pytest suite for predict_energy_curve using XGBoost models for both load and renewable percentage.
"""
import pytest
import pandas as pd
from greencurve.predictor import predict_energy_curve

# Common test parameters
COUNTRY = "NL"
HORIZON_DAYS = 3
FORECAST_STEPS = HORIZON_DAYS * 24
HISTORY_DAYS = 50
START_DATE = "2024-01-01 00:00:00"
EXTRA_DATA = {}


def test_predict_energy_curve_output_structure():
    """
    Test that the DataFrame returned has the correct columns and length.
    """
    df = predict_energy_curve(
        country=COUNTRY,
        extra_data=EXTRA_DATA,
        current_date=START_DATE,
        forecast_load="XGBoost",
        forecast_percentage="XGBoost",
        history_days=HISTORY_DAYS,
        days=HORIZON_DAYS,
        plot=False
    )
    # Check type
    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame"
    # Expected columns
    expected_cols = ['ds', 'total_load', 'renewable_percentage', 'renewable_mw']
    assert list(df.columns) == expected_cols, f"DataFrame columns {df.columns} do not match expected {expected_cols}"
    # Length should equal forecast steps
    assert len(df) == FORECAST_STEPS, f"Expected {FORECAST_STEPS} rows, got {len(df)}"


def test_no_missing_values():
    """
    Test that there are no NaN values in the forecast.
    """
    df = predict_energy_curve(
        country=COUNTRY,
        extra_data=EXTRA_DATA,
        current_date=START_DATE,
        forecast_load="XGBoost",
        forecast_percentage="XGBoost",
        history_days=HISTORY_DAYS,
        days=HORIZON_DAYS,
        plot=False
    )
    assert not df.isna().any().any(), "Forecast DataFrame contains NaN values"


def test_renewable_mw_calculation():
    """
    Test that renewable_mw = total_load * renewable_percentage / 100 for all rows.
    """
    df = predict_energy_curve(
        country=COUNTRY,
        extra_data=EXTRA_DATA,
        current_date=START_DATE,
        forecast_load="XGBoost",
        forecast_percentage="XGBoost",
        history_days=HISTORY_DAYS,
        days=HORIZON_DAYS,
        plot=False
    )
    calculated = df['total_load'] * df['renewable_percentage'] / 100
    # Use nearly equal for float comparison
    pd.testing.assert_series_equal(df['renewable_mw'], calculated, check_names=False)
