import os
import pandas as pd
import pytest
from greencurve.predictor import predict_energy_curve

@pytest.fixture
def data_dir():
    """
    Returns the absolute path to the 'data/raw/US' directory
    inside the tests folder.
    """
    return os.path.join(os.path.dirname(__file__), "data", "raw", "US")

def test_prediction(data_dir):
    file_paths = [
        os.path.join(data_dir, "US_2021_hourly.csv"),
        os.path.join(data_dir, "US_2022_hourly.csv"),
        os.path.join(data_dir, "US_2023_hourly.csv"),
        os.path.join(data_dir, "ercot_load_act_hr_2021.csv"),
        os.path.join(data_dir, "ercot_load_act_hr_2022.csv"),
    ]
    forecast = predict_energy_curve(file_paths, current_date="2024-01-12", plot=False)
    
    # Print the forecasted values for review
    print("Predicted Energy Production Forecast:")
    print(forecast[['ds', 'yhat']])
    
    # Verify the forecast DataFrame contains the expected columns and types
    assert not forecast.empty, "Forecast DataFrame is empty"
    assert set(forecast.columns) == {"ds", "yhat"}, "Forecast columns do not match expected {'ds', 'yhat'}"
    assert pd.api.types.is_datetime64_any_dtype(forecast['ds']), "Column 'ds' is not datetime type"
    assert pd.api.types.is_float_dtype(forecast['yhat']), "Column 'yhat' is not float type"
