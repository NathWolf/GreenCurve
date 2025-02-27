import logging
from typing import List, Dict, Optional
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


_logger = logging.getLogger(__name__)

def get_data_path(filename: str, folder: str = "US") -> str:
    """
    Return the absolute path to a data file located in greencurve/data/raw/<folder>.
    
    Args:
        filename (str): Name of the file.
        folder (str): Folder name (e.g., "US", "FR"). Defaults to "US".
    """
    package_dir = os.path.dirname(__file__)
    return os.path.join(package_dir, "data", "raw", folder, filename)

def load_data_from_file(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame and print its shape and file size."""
    try:
        df = pd.read_csv(file_path, delimiter=',')
        df.columns = df.columns.str.strip()  # clean up header spaces
        file_size = os.path.getsize(file_path)
        print(f"Loaded file: {file_path}")
        print(f"  Shape: {df.shape} (rows, columns)")
        print(f"  File size: {file_size} bytes")
    except Exception as e:
        _logger.error(f"Error reading {file_path}: {e}")
        raise
    return df

def load_renewable_data(country: str) -> pd.DataFrame:
    """
    Load renewable percentage data for a given country.

    The files are assumed to be located in greencurve/data/raw/<COUNTRY>/
    
    Args:
        country (str): Country code (e.g., "US", "FR").
        
    Returns:
        pd.DataFrame: DataFrame containing "Datetime (UTC)" and "Renewable Percentage".
    """

    folder = country.upper()
    data_dir = get_data_path("", folder)
    # Use the folder (country code) as the prefix for file names.
    filenames = [f for f in os.listdir(data_dir) if f.startswith(f"{folder}_") and f.endswith("_hourly.csv")]
    if not filenames:
        raise FileNotFoundError(f"No renewable energy data found for country: {country}")
    
    dfs = [load_data_from_file(get_data_path(fname, folder)) for fname in filenames]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    print("Successfully loaded renewable data. ")
    return df[["Datetime (UTC)", "Renewable Percentage"]]

def load_total_load_data(country: str) -> pd.DataFrame:
    """
    Load total load data for a given country.
    
    The expected filename format varies by country:
    - US (ERCOT): "ercot_load_act_hr_YYYY.csv"
    - France (FR): "FR_load_YYYY.csv"
    - Germany (DE): "DE_load_YYYY.csv"
    
    Args:
        country (str): Country code (e.g., "US", "FR", "DE").
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of total load data.
    """
    if country.upper() in ["US"]:
        folder = "US"
        data_dir = get_data_path("", folder)
        filenames = [f for f in os.listdir(data_dir) if f.startswith("ercot_load_act_hr_") and f.endswith(".csv")]
        if not filenames:
            raise FileNotFoundError(f"No total load data found for country: {country}")
        print("Successfully loaded total load data. ")
        dfs = [load_data_from_file(get_data_path(fname, folder)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        # Convert timestamp to datetime
        df["Datetime (UTC)"] = pd.to_datetime(df["UTC Timestamp (Interval Ending)"])

    else:
        # For France/Germany etc.
        folder = country.upper()
        data_dir = get_data_path("", folder)
        filenames = [f for f in os.listdir(data_dir) if f.startswith("Actual Generation per Production Type_") and f.endswith(f"_{country}.csv")]
        if not filenames:
            raise FileNotFoundError(f"No total load data found for country: {country}")
        print("Successfully loaded total load data. ")
        dfs = [load_data_from_file(get_data_path(fname, folder)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        if "MTU" in df.columns:
            df = df.rename(columns={"MTU": "Datetime (UTC)"})
            # Convert timestamp (MTU) to proper datetime
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"].str.split(" - ").str[0], format="%d.%m.%Y %H:%M")
            # Compute total generation (sum of all sources)
            generation_cols = [col for col in df.columns if col.endswith("Aggregated [MW]")]
            df["TOTAL Actual Load (MW)"] = df[generation_cols].replace("n/e", np.nan).astype(float).sum(axis=1)

    # Only keep relevant columns for the total load prediction
    df = df[["Datetime (UTC)", "TOTAL Actual Load (MW)"]]
    return df

def load_historical_data(country: str) -> pd.DataFrame:
    """
    For country "US", load both the renewable percentage data and the total load data,
    and merge them on timestamp.
    For "ercot", load only the ERCOT load data.
    """
    if country.upper() in ["US","FR","PL","NL"]:
        renewable_df = load_renewable_data(country)
        load_df = load_total_load_data(country)
        
        # Convert timestamps to datetime
        renewable_df['Datetime (UTC)'] = pd.to_datetime(renewable_df['Datetime (UTC)'])
        load_df['Datetime (UTC)'] = pd.to_datetime(load_df['Datetime (UTC)'])
        
        # Sort data by time for merge_asof
        renewable_df = renewable_df.sort_values("Datetime (UTC)")
        load_df = load_df.sort_values("Datetime (UTC)")
        
        # Use merge_asof to join the total load to the renewable data.
        # We use a tolerance (e.g., 30 minutes) to allow for slight differences in timestamps.
        merged = pd.merge_asof(
            renewable_df,
            load_df[['Datetime (UTC)', 'TOTAL Actual Load (MW)']],
            on="Datetime (UTC)",
            direction="nearest",
            tolerance=pd.Timedelta("30min")
        )
        return merged
    else:
        raise ValueError("Unsupported country. ")

def impute_total_load(data: pd.DataFrame, total_load_col: str) -> pd.DataFrame:
    """
    Impute missing total load values using Prophet.
    
    Train a Prophet model on rows where total load is available,
    then forecast missing total load values.
    """
    data = data.copy()
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    
    valid_mask = data[total_load_col].notna()
    if valid_mask.sum() < 2:
        return data

    valid_data = data.loc[valid_mask, ['Datetime (UTC)', total_load_col]].rename(
        columns={'Datetime (UTC)': 'ds', total_load_col: 'y'}
    )
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    m.fit(valid_data)
    
    missing_mask = ~valid_mask
    if missing_mask.sum() > 0:
        missing_data = data.loc[missing_mask, ['Datetime (UTC)']].rename(
            columns={'Datetime (UTC)': 'ds'}
        )
        forecast = m.predict(missing_data)
        data.loc[missing_mask, total_load_col] = forecast['yhat'].values
    return data

def prepare_features(data: pd.DataFrame, total_load_col: str = None) -> pd.DataFrame:
    """
    Prepare data for forecasting.
    
    For US data, ensure we have both "Renewable Percentage" and "TOTAL Actual Load (MW)".
    Impute missing total load values if needed, and compute "Renewable MW".
    """
    _logger.info("Preparing US features")
    data.columns = data.columns.str.strip()
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    # Ensure the renewable percentage column exists and drop rows where it's missing.
    data = data.dropna(subset=["Renewable Percentage"])
    if total_load_col and total_load_col in data.columns:
        # Impute missing total load values.
        data = impute_total_load(data, total_load_col)
        # Drop any remaining rows missing total load.
        data = data.dropna(subset=[total_load_col])
        # Compute renewable production in MW.
        data['Renewable MW'] = (data['Renewable Percentage'] / 100) * data[total_load_col]
    else:
        data['Renewable MW'] = data['Renewable Percentage']
    ts_data = data[['Datetime (UTC)', 'Renewable MW']].rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable MW': 'y'}
    )
    _logger.debug("US time series data prepared with shape: %s", ts_data.shape)
    return ts_data

def prepare_ercot_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare ERCOT data for time series forecasting using Prophet."""
    _logger.info("Preparing ERCOT features")
    data = data.dropna(subset=["Coast Actual Load (MW)", "East Actual Load (MW)", "Far West Actual Load (MW)"])
    data.columns = data.columns.str.strip()
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    data['Renewable MW'] = (data['Coast Actual Load (MW)'] +
                            data['East Actual Load (MW)'] +
                            data['Far West Actual Load (MW)'])
    ts_data = data[['Datetime (UTC)', 'Renewable MW']].rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable MW': 'y'}
    )
    _logger.debug("ERCOT time series data prepared with shape: %s", ts_data.shape)
    return ts_data

def train_prophet_model(data: pd.DataFrame) -> Prophet:
    """Train a Prophet model on the provided time series data."""
    _logger.info("Training Prophet model")
    model = Prophet()
    model.add_seasonality(name='daily', period=1, fourier_order=10)
    valid_data = data.dropna(subset=['y'])
    if valid_data.shape[0] < 2:
        raise ValueError("Not enough valid data to train Prophet model.")
    model.fit(valid_data)
    _logger.info("Prophet model trained successfully")
    return model

def plot_forecast(historical_data_original, forecast, extra_data, plot_start, plot_end, country, days):
    """
    Plot the historical data, modified historical data, user overrides, and forecast.

    Args:
        historical_data_original (pd.DataFrame): The original observed data before modifications.
        forecast (pd.DataFrame): The predicted energy curve.
        extra_data (Dict[str, List[float]]): User-provided additional renewable energy percentages.
        plot_start (pd.Timestamp): The starting date for the plot.
        plot_end (pd.Timestamp): The end date for the plot.
        country (str): The country name for labeling.
    """
    plt.figure(figsize=(14, 7))

    # Filter data for plotting range
    historical_data_original = historical_data_original[
        (historical_data_original["ds"] >= plot_start) & (historical_data_original["ds"] <= plot_end)
    ]
    forecast = forecast[(forecast["ds"] >= plot_start) & (forecast["ds"] <= plot_end)]

    # Plot Historical Data (Before User Modifications)
    plt.plot(historical_data_original['ds'], historical_data_original['y'], color='blue', label="Historical Data (Observed)", alpha=0.6)

    # Plot Forecasted Data
    plt.plot(forecast['ds'], forecast['yhat'], linestyle="dashed", color="orange", label="Predicted Data (Prophet)")

    # Highlight User Override Data
    first_label = True  
    for date_str, values in extra_data.items():
        date_dt = pd.to_datetime(date_str)
        override_times = [date_dt + pd.Timedelta(hours=i) for i in range(24)]
        override_values = [(values[i] / 100) * forecast['yhat'].mean() for i in range(24)]  # Use forecast mean as approximation
        if first_label:
            plt.scatter(override_times, override_values, color="green", label="User Override Data", marker="x")
            first_label = False  
        else:
            plt.scatter(override_times, override_values, color="green", marker="x")

    # Ensure labels and legend
    plt.xlabel("Date")
    plt.ylabel("Renewable Energy Production (MW)")
    plt.title(f"Renewable Energy Prediction for {country}")
    plt.legend()
    plt.grid(True)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=30)

    # Highlight the plot start and plot end date in RED on the x-axis
    plt.axvline(x=plot_end - pd.Timedelta(days=days), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(x=plot_end, color='red', linestyle='dashed', linewidth=1)

    plt.show()

def parse_days(period_str: str) -> int:
    """Convert a string like '730 days' into an integer number of days."""
    try:
        parts = period_str.split()
        return int(parts[0])
    except Exception:
        raise ValueError("Invalid period string format: {}".format(period_str))

def tune_prophet_model(data: pd.DataFrame, param_grid: Dict[str, List[float]], 
    cv_initial: str = '730 days', cv_period: str = '180 days', cv_horizon: str = '365 days') -> Prophet:
    """
    Perform grid search over Prophet hyperparameters using cross-validation.
    Adjusts cv_horizon if necessary so that enough data exists after the initial window.
    Args:
        data (pd.DataFrame): Training data with columns ['ds', 'y'].
        param_grid (Dict[str, List[float]]): Dictionary containing keys like:
            - 'changepoint_prior_scale': list of floats
            - 'seasonality_prior_scale': list of floats
            - 'seasonality_mode': list of strings (e.g. ['additive', 'multiplicative'])
            - 'daily_fourier_order': list of ints (e.g. [3, 5, 10])
        cv_initial (str): Initial training period for cross-validation.
        cv_period (str): Period between cutoffs.
        cv_horizon (str): Forecast horizon.
    
    Returns:
        Prophet: The best model trained on all data using the best found parameters,
                 or a default model if tuning fails.
    """
    best_rmse = float('inf')
    best_params = {}
    best_model = None

    # Calculate available data range in days.
    available_days = (data['ds'].max() - data['ds'].min()).days
    initial_days = parse_days(cv_initial)
    horizon_days = parse_days(cv_horizon)
    if available_days - initial_days < horizon_days:
        new_horizon = available_days - initial_days - 1  # ensure at least one day gap
        if new_horizon < 1:
            _logger.error("Not enough data available for cross validation. Available: %d days, initial: %d days.",
                          available_days, initial_days)
            raise ValueError("Not enough data available for cross validation.")
        else:
            _logger.info("Adjusting cv_horizon from '%s' to '%d days' because available data is only %d days.",
                         cv_horizon, new_horizon, available_days)
            cv_horizon = f"{new_horizon} days"

    for cps in param_grid.get('changepoint_prior_scale', [0.05]):
        for sps in param_grid.get('seasonality_prior_scale', [10.0]):
            for mode in param_grid.get('seasonality_mode', ['additive']):
                for fourier_order in param_grid.get('daily_fourier_order', [10]):
                    _logger.info("Tuning with cps=%.4f, sps=%.4f, mode=%s, daily_fourier_order=%d", 
                                 cps, sps, mode, fourier_order)
                    model = Prophet(
                        changepoint_prior_scale=cps,
                        seasonality_prior_scale=sps,
                        seasonality_mode=mode,
                        daily_seasonality=False,  # disable built-in daily seasonality
                        weekly_seasonality=True,
                        yearly_seasonality=True
                    )
                    # Add custom daily seasonality with the specified Fourier order.
                    model.add_seasonality(name='daily', period=1, fourier_order=fourier_order)
                    
                    try:
                        model.fit(data)
                    except Exception as e:
                        _logger.error("Model failed to fit: %s", e)
                        continue

                    try:
                        df_cv = cross_validation(model, initial=cv_initial, period=cv_period, horizon=cv_horizon, parallel="processes")
                        df_p = performance_metrics(df_cv)
                        rmse = df_p['rmse'].mean()
                        _logger.info("RMSE: %.4f", rmse)
                    except Exception as e:
                        _logger.error("Cross validation failed: %s", e)
                        continue

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            'changepoint_prior_scale': cps,
                            'seasonality_prior_scale': sps,
                            'seasonality_mode': mode,
                            'daily_fourier_order': fourier_order
                        }
                        best_model = model

    if best_model is None:
        _logger.warning("No valid model could be tuned with the given parameter grid. Falling back to default Prophet model.")
        best_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        best_model.add_seasonality(name='daily', period=1, fourier_order=10)
        valid_data = data.dropna(subset=['y'])
        if valid_data.shape[0] < 2:
            raise ValueError("Not enough valid data to train Prophet model.")
        best_model.fit(valid_data)
    else:
        _logger.info("Best hyperparameters: %s with RMSE: %.4f", best_params, best_rmse)

    return best_model

def predict_energy_curve(country: str, extra_data: Dict[str, List[float]], current_date: str, days: int = 2, 
    plot: bool = False, plot_date: str = None, tune: bool = False, param_grid: Optional[Dict[str, List[float]]] = None,
    cv_initial: str = '730 days', cv_period: str = '180 days', cv_horizon: str = '365 days') -> pd.DataFrame:
    """
    Predict the renewable energy production curve (in MW) for the requested number of days.

    - If user provides extra_data, it modifies the historical data before forecasting.
    - The model forecasts from the last available data point (after adding user data) up to `current_date + days`.
    - Optionally, the model can be tuned via cross-validation to select the best hyperparameters.
    - The plot shows historical, modified, and forecasted data separately.

    Args:
        country (str): Country code ("US", "FR", etc.).
        extra_data (Dict[str, List[float]]): Mapping of dates to lists of 24 hourly percentages.
        current_date (str): Forecast start date in "YYYY-MM-DD" format.
        days (int): Number of days to forecast from `current_date`.
        plot_date (str, optional): The starting date for the plot. Defaults to "2023-12-20".
        plot (bool): Whether to display a plot of the forecast.
        tune (bool): Whether to perform hyperparameter tuning.
        param_grid (Optional[Dict[str, List[float]]]): Grid of hyperparameters to search over.
        cv_initial (str): Initial period for cross-validation.
        cv_period (str): Period between cutoffs in cross-validation.
        cv_horizon (str): Forecast horizon in cross-validation.

    Returns:
        pd.DataFrame: Forecast DataFrame with columns 'ds' (datetime) and 'yhat' (predicted MW).
    """
    _logger.info("Starting energy prediction for country: %s, date: %s", country, current_date)

    # Load historical data and prepare features
    if country.upper() in ["US", "FR", "PL", "NL"]:
        historical_data = load_historical_data(country)
        data_ts = prepare_features(historical_data, total_load_col='TOTAL Actual Load (MW)')
        avg_total_load = historical_data['TOTAL Actual Load (MW)'].mean() if 'TOTAL Actual Load (MW)' in historical_data.columns else 1
    else:
        raise ValueError("Unsupported country. Available options: US, FR, PL, NL.")

    # Store original historical data before modifications for plotting
    historical_data_original = data_ts.copy()

    # Apply user overrides: modify historical data where user provides extra_data
    for date_str, values in extra_data.items():
        date_dt = pd.to_datetime(date_str)
        for hour, value in enumerate(values):
            time_point = date_dt + pd.Timedelta(hours=hour)
            if time_point in data_ts['ds'].values:
                data_ts.loc[data_ts['ds'] == time_point, 'y'] = (value / 100) * avg_total_load
            else:
                new_row = pd.DataFrame({'ds': [time_point], 'y': [(value / 100) * avg_total_load]})
                data_ts = pd.concat([data_ts, new_row], ignore_index=True)

    data_ts = data_ts.sort_values('ds').drop_duplicates('ds')

    # Train Prophet model – optionally tuning hyperparameters
    if tune:
        if param_grid is None:
            # Provide default grid if none is given
            param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'daily_fourier_order': [3, 5, 10]
            }
        _logger.info("Performing hyperparameter tuning...")
        model = tune_prophet_model(data_ts, param_grid, cv_initial, cv_period, cv_horizon)
    else:
        model = Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True)
        model.add_seasonality(name='daily', period=1, fourier_order=10)
        valid_data = data_ts.dropna(subset=['y'])
        if valid_data.shape[0] < 2:
            raise ValueError("Not enough valid data to train Prophet model.")
        model.fit(valid_data)
        _logger.info("Prophet model trained successfully with default parameters.")

    # Determine forecast range: from the last available date (after adding user data) to `current_date + days`
    last_available_date = data_ts['ds'].max()
    forecast_start = last_available_date + pd.Timedelta(hours=1)
    forecast_end = pd.to_datetime(current_date) + pd.Timedelta(days=days)
    future = pd.DataFrame({"ds": pd.date_range(start=forecast_start, end=forecast_end, freq="H")})
    _logger.info("Forecasting for %d future time points", len(future))

    forecast = model.predict(future)

    # Apply extra data overrides in forecast results
    forecast_dates = forecast['ds'].dt.strftime("%Y-%m-%d")
    for idx, date_str in forecast_dates.iteritems():
        if date_str in extra_data:
            override_list = extra_data[date_str]
            if len(override_list) == 24:
                hour = forecast.loc[idx, 'ds'].hour
                adjusted_value = (override_list[hour] / 100) * avg_total_load
                forecast.at[idx, 'yhat'] = adjusted_value
            else:
                _logger.warning("Override list for %s does not contain 24 values.", date_str)

    # Determine plot start date
    plot_start = pd.to_datetime(plot_date) if plot_date else pd.to_datetime("2023-12-20")
    plot_end = forecast['ds'].max()

    if plot:
        plot_forecast(historical_data_original, forecast, extra_data, plot_start, plot_end, country, days)

    _logger.info("Prediction completed")
    return forecast[['ds', 'yhat']]
