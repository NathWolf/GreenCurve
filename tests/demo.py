#!/usr/bin/env python3

from greencurve.predictor import predict_energy_curve
import pandas as pd

if __name__ == "__main__":
    # --- Override for the previous day (2024-01-01) ---
    # Here we force 30% renewable every hour on Jan 1, 2024.
    override_percentages = [30.0] * 24
    extra_data = {
        "2024-01-01": override_percentages
    }

    # --- Forecast starting one day later ---
    df = predict_energy_curve(
        country="NL",
        extra_data=extra_data,
        current_date="2024-01-02 00:00:00",
        forecast_load="XGBoost",
        forecast_percentage="XGBoost",
        history_days=50,
        days=3,
        plot=False
    )

    # Show all 72 forecast rows
    pd.set_option("display.max_rows", None)
    print(df)
