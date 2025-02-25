import logging
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

_logger = logging.getLogger(__name__)


def load_data(file_paths):
    """Load and concatenate datasets from multiple CSV files."""
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, delimiter=',')
        except Exception as e:
            print(f"Error reading {path}: {e}")
            raise
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def prepare_features(data: pd.DataFrame, total_load_col: str = None) -> pd.DataFrame:
    """Prepare US data for time series forecasting using Prophet.

    Args:
        data (pd.DataFrame): DataFrame containing historical data.
        total_load_col (str, optional): Column name for total load in MW. Defaults to None.

    Returns:
        pd.DataFrame: Time series formatted DataFrame for Prophet.
    """
    _logger.info("Preparing US features")
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    if total_load_col:
        # Convert renewable percentage to MW using total load
        data['Renewable MW'] = (data['Renewable Percentage'] / 100) * data[total_load_col]
    else:
        data['Renewable MW'] = data['Renewable Percentage']
    ts_data = data[['Datetime (UTC)', 'Renewable MW']].rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable MW': 'y'}
    )
    _logger.debug("US time series data prepared with shape: %s", ts_data.shape)
    return ts_data


def prepare_ercot_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare ERCOT data for time series forecasting using Prophet.

    Args:
        data (pd.DataFrame): DataFrame containing ERCOT historical data.

    Returns:
        pd.DataFrame: Time series formatted DataFrame for Prophet.
    """
    _logger.info("Preparing ERCOT features")
    data['UTC Timestamp (Interval Ending)'] = pd.to_datetime(data['UTC Timestamp (Interval Ending)'])
    data['Renewable MW'] = (data['Coast Actual Load (MW)'] +
                            data['East Actual Load (MW)'] +
                            data['Far West Actual Load (MW)'])
    ts_data = data[['UTC Timestamp (Interval Ending)', 'Renewable MW']].rename(
        columns={'UTC Timestamp (Interval Ending)': 'ds', 'Renewable MW': 'y'}
    )
    _logger.debug("ERCOT time series data prepared with shape: %s", ts_data.shape)
    return ts_data


def train_prophet_model(data: pd.DataFrame) -> Prophet:
    """Train a Prophet model on the provided time series data.

    Args:
        data (pd.DataFrame): Time series data formatted for Prophet.

    Returns:
        Prophet: Trained Prophet model.
    """
    _logger.info("Training Prophet model")
    model = Prophet()
    model.add_seasonality(name='daily', period=1, fourier_order=10)
    model.fit(data)
    _logger.info("Prophet model trained successfully")
    return model


def predict_energy_curve(file_paths: List[str], current_date: str, plot: bool = False) -> pd.DataFrame:
    """Predict the renewable energy production curve (in MW) for the next 24 hours.

    This function loads historical US and ERCOT data, prepares the data, trains a Prophet model,
    and forecasts renewable energy production starting from the given date. If the requested forecast
    start date is beyond the training data, it creates a future dataframe manually.

    Args:
        file_paths (List[str]): List of file paths for historical CSV data.
        current_date (str): The desired forecast start date in 'YYYY-MM-DD' format.
        plot (bool, optional): Whether to display a plot of the forecast. Defaults to False.

    Returns:
        pd.DataFrame: Forecast DataFrame with columns 'ds' (datetime) and 'yhat' (predicted MW).
    """
    _logger.info("Starting energy prediction for date: %s", current_date)

    # Separate US and ERCOT files based on filename conventions
    us_files = [path for path in file_paths if 'US' in path]
    ercot_files = [path for path in file_paths if 'ercot' in path.lower()]

    _logger.debug("US files: %s", us_files)
    _logger.debug("ERCOT files: %s", ercot_files)

    # Load and prepare US data
    us_data = load_data(us_files)
    us_ts = prepare_features(us_data, total_load_col='TOTAL Actual Load (MW)')

    # Load and prepare ERCOT data
    ercot_data = load_data(ercot_files)
    ercot_ts = prepare_ercot_features(ercot_data)

    # Combine datasets
    combined_ts = pd.concat([us_ts, ercot_ts], ignore_index=True)
    _logger.debug("Combined time series data shape: %s", combined_ts.shape)

    # Train Prophet model
    model = train_prophet_model(combined_ts)

    # Determine the training data's maximum date and the requested forecast start date
    training_max = combined_ts['ds'].max()
    requested_date = pd.to_datetime(current_date)

    if requested_date > training_max:
        # Create a manual future dataframe starting from the requested date
        future = pd.DataFrame({"ds": pd.date_range(start=requested_date, periods=24, freq="H")})
    else:
        # Use Prophet's built-in future dataframe and filter it by the requested date
        future = model.make_future_dataframe(periods=24, freq='H')
        future = future[future['ds'] >= requested_date]
    
    _logger.info("Forecasting for %d future time points", len(future))
    forecast = model.predict(future)

    if plot:
        plt.figure(figsize=(14, 7))
        model.plot(forecast)
        plt.title(f'Renewable Energy Prediction for {current_date} - Next 24 Hours')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Renewable Energy Production (MW)')
        plt.grid(True)
        plt.show()

    _logger.info("Prediction completed")
    return forecast[['ds', 'yhat']]
