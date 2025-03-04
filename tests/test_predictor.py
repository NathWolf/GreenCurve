import pandas as pd
import pytest
from greencurve.predictor import predict_energy_curve

def test_prediction():
    # Extra data: keys are dates and values are lists of 24 renewable percentages (one per hour)
    # extra_data = {
    #     "2023-12-25": [50.0 + i for i in range(24)],  # 
    #     "2024-01-01": [50.0 + i for i in range(24)],  # 
    # }
    
    extra_data = {}

    # Forecast **48 hours** from January 11, 2024, and plot only from December 20, 2023
    forecast = predict_energy_curve("NL", extra_data, current_date="2024-01-01", days=7, plot=False, plot_date="2023-12-20")
        
    # Print the forecast for review (use pytest -s to see the output)
    print("Predicted Forecast:")
    print(forecast[['ds', 'renewable_mw', 'renewable_percentage', 'total_load']])
    print(forecast['renewable_mw'])
