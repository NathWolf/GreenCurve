GreenCurve
==========

**GreenCurve** is a machine learning tool designed to predict the 24-hour renewable energy production curve for the upcoming day using historical data from US and ERCOT sources. Leveraging multiple forecasting models—including Facebook's Prophet, SARIMAX, XGBoost, and a moving average approach—GreenCurve provides robust predictions and insights into future energy trends. It also includes visualization tools to compare model forecasts against real test data.

Table of Contents
-----------------

- `Features <#features>`_
- `Installation <#installation>`_
- `Usage <#usage>`_
- `Configuration <#configuration>`_
- `Model Overview <#model-overview>`_
- `Examples <#examples>`_
- `Contributing <#contributing>`_
- `License <#license>`_
- `Contact <#contact>`_

Features
--------

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

Installation
------------

Ensure you have Python 3.7 or later installed. Clone the repository and install the required dependencies:

.. code-block:: bash

    git clone https://github.com/yourusername/GreenCurve.git
    cd GreenCurve
    pip install -r requirements.txt

Run the test suite to ensure everything works:

.. code-block:: bash

    pytest -s tests/test.py

Usage
-----

GreenCurve can be used both as a standalone script and as an importable module. Example usage:

.. code-block:: python

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

Configuration
-------------

GreenCurve offers flexibility in configuring forecasting parameters:

- **Training Period:**  
  Set the number of days used for training (``history_days``).

- **Forecast Horizon:**  
  Define the number of forecast steps (e.g., ``forecast_steps = 24`` for a 24-hour forecast).

- **Model Tuning:**  
  You can adjust parameters for each forecasting method directly in the code:

  - **Prophet Parameters:**  
    Configure `changepoint_prior_scale`, `seasonality_prior_scale`, and Fourier orders.

  - **SARIMAX Parameters:**  
    Set `order` (p, d, q) and `seasonal_order` (P, D, Q, s).

  - **XGBoost Parameters:**  
    Customize `n_estimators`, `learning_rate`, etc.

You may also manage configuration through external config files if needed.

Model Overview
--------------

GreenCurve integrates multiple forecasting approaches to improve prediction robustness:

- **Prophet:**  
  Designed for time series data with strong seasonal effects. Detects changepoints and captures seasonality.

- **SARIMAX:**  
  Seasonal ARIMA with support for exogenous variables. Suitable for trending and seasonal time series.

- **XGBoost:**  
  Gradient boosting with engineered time-based features. Captures nonlinear relationships.

- **Moving Average:**  
  Baseline method using historical hourly averages.

Each model's predictions are evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

This multi-model strategy enables performance comparison and model selection.

Examples
--------

Coming soon.

License
-------

This project is licensed under the MIT License.

Contact
-------

For inquiries, please contact: nathalia.wolf@inria.fr
