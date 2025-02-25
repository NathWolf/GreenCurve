GreenCurve
==========

GreenCurve is a machine learning tool designed to predict the 24-hour renewable energy
production curve for the upcoming day using historical data from US and ERCOT sources.
It leverages Facebook's Prophet to forecast renewable energy production and provides insights
into future energy trends.

Features:
- Load and preprocess historical US and ERCOT energy data.
- Format data for time series forecasting using Prophet.
- Train a Prophet model to forecast renewable energy production.
- Visualize the forecasted 24-hour energy curve.

Installation:
    pip install .

Usage:
    from greencurve import predict_energy_curve
    forecast = predict_energy_curve(
        ["path/to/US_data.csv", "path/to/ercot_data.csv"],
        "2024-01-12",
        plot=True
    )
