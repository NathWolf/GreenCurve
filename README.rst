# GreenCurve

GreenCurve is a machine learning tool designed to predict the 24-hour renewable energy production curve for the upcoming day using historical data from US and ERCOT sources. Leveraging multiple forecasting models—including Facebook's Prophet, SARIMAX, XGBoost, and a moving average approach—GreenCurve provides robust predictions and insights into future energy trends. It also includes visualization tools to compare model forecasts against real test data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Overview](#model-overview)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Multi-Model Forecasting:**  
  Predict renewable energy production using several methods (Prophet, SARIMAX, XGBoost, and moving average) for robust and comparative performance.

- **Data Loading & Preprocessing:**  
  Built-in functions to load, clean, and preprocess historical energy data from US and ERCOT sources.

- **Time Series Formatting:**  
  Automatically format your data for time series analysis, handling resampling, imputation, and feature engineering.

- **Visualization:**  
  Generate detailed plots comparing forecasts from different models against real energy data, including error metrics such as RMSE and MAE.

- **Easy Integration:**  
  Use GreenCurve as a standalone tool or as a module within your Python projects.

## Clone and Install Dependencies

Ensure you have Python 3.7 or later installed, then clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/GreenCurve.git
cd GreenCurve
pip install -r requirements.txt
```

Run pytest -s tests/test.py to ensure all tests pass.

## Usage

GreenCurve can be used both as a standalone script and as an importable module. Below is an example usage in a Python script:
```bash
from greencurve import predict_energy_curve

# Predict the renewable energy production curve for a given day
forecast = predict_energy_curve(
    country="US", 
    extra_data={},         # Optionally provide override data
    current_date="2024-01-12",
    days=1,                # Forecast for 24 hours (1 day)
    plot=True              # Display forecast plots
)

print(forecast.head())
```

## Configuration

GreenCurve offers flexibility in configuring forecasting parameters:

- **Training Period:**  
  Set the number of days used for training (`history_days`). This parameter determines how many past days of data are used to train the forecasting models.

- **Forecast Horizon:**  
  Define the number of forecast steps (typically set as the number of hours in the forecast period). For example, for a 24-hour forecast, `forecast_steps` is set to 24.

- **Model Tuning:**  
  You can adjust parameters for each forecasting method directly in the code:
  
  - **Prophet Parameters:**  
    Configure the number of changepoints, `changepoint_prior_scale`, `seasonality_prior_scale`, and the Fourier orders for daily and weekly seasonalities.
  
  - **SARIMAX Parameters:**  
    Set the autoregressive, differencing, and moving average orders (`order`) as well as the seasonal order (`seasonal_order`).
  
  - **XGBoost Parameters:**  
    Customize the number of estimators, learning rate, and other hyperparameters as needed.

Additional configuration can be managed through external configuration files if required.

## Model Overview

GreenCurve integrates multiple forecasting approaches to improve prediction robustness:

- **Prophet:**  
  A model specifically designed for forecasting time series data with strong seasonal effects and multiple seasonalities. It detects changepoints and captures yearly seasonality.

- **SARIMAX:**  
  A traditional autoregressive model with seasonal support (Seasonal ARIMA with exogenous variables). It is well-suited for time series data exhibiting trends and seasonality.

- **XGBoost:**  
  A high-performance gradient boosting algorithm enhanced with engineered time-based features. XGBoost is capable of capturing complex nonlinear relationships in the data.

- **Moving Average:**  
  A simple baseline method that calculates the hourly average of historical data. This approach provides a dynamic forecast that adapts to hourly patterns rather than a constant value.

Each model's predictions are evaluated using standard error metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error). This multi-model strategy allows you to compare the performance of different approaches and select the best-suited model for your forecasting needs.

