import pandas as pd
import pytest
from greencurve.predictor import predict_energy_curve

def test_prediction():
    # Extra data: keys are dates and values are lists of 24 renewable percentages (one per hour)
    extra_data = {
        "2023-12-25": [50.0 + i for i in range(24)],  # 
        "2024-01-01": [100.0 + i for i in range(24)],  # 
    }
    
    extra_data = {}

    # Forecast **48 hours** from January 11, 2024, and plot only from December 20, 2023
    forecast = predict_energy_curve("FR", extra_data, current_date="2024-01-14", days=2, plot=True, plot_date="2023-12-20")
        
    # Print the forecast for review (use pytest -s to see the output)
    print("Predicted Forecast:")
    print(forecast[['ds', 'yhat']])
    
    # Verify the forecast DataFrame is not empty and has the expected columns.
    assert not forecast.empty, "Forecast DataFrame is empty"
    assert set(forecast.columns) == {"ds", "yhat"}, "Forecast columns do not match expected {'ds', 'yhat'}"
    assert pd.api.types.is_datetime64_any_dtype(forecast['ds']), "Column 'ds' is not datetime type"
    assert pd.api.types.is_float_dtype(forecast['yhat']), "Column 'yhat' is not float type"
    
    # Add a helper column with the date string for grouping.
    forecast["date_str"] = forecast["ds"].dt.strftime("%Y-%m-%d")

    #TODO: make sure extra_data dictionary is used in the forecasted data
    #TODO: add other countries to the test ('FR','PL','NL')