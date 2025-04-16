#!/usr/bin/env python3
"""
tests/test.py

Este script importa las funciones de pronóstico del módulo y ejecuta la comparación de pronósticos.
Se usa pytest para ejecutar el test.
"""

import pytest
from greencurve.predictor import compare_forecasts, load_renewable_data, load_total_load_data
import pandas as pd

def test_forecast_comparison():
    country = "NL"
    horizon_days = 3
    forecast_steps = horizon_days * 24  # hours
    history_days = 50
    start_date = "2024-01-01 00:00:00"
    
    # Load extra data from training (raw) data; only include data before the test start.
    renewable_extra = load_renewable_data(country)
    total_load_extra = load_total_load_data(country)
    renewable_extra = renewable_extra[pd.to_datetime(renewable_extra["Datetime (UTC)"]) < pd.to_datetime(start_date)]
    total_load_extra = total_load_extra[pd.to_datetime(total_load_extra["Datetime (UTC)"]) < pd.to_datetime(start_date)]
    
    compare_forecasts(country, forecast_steps, start_date=start_date,
                      extra_total_load=total_load_extra, extra_renewable=renewable_extra,
                      history_days=history_days)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
