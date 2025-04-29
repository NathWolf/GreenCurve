#!/usr/bin/env python3
import logging
import warnings
from typing import List, Dict, Optional, Tuple
import os
import json
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pmdarima.arima import ARIMA

from xgboost import XGBRegressor

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)  # Hide non-critical logs

_logger = logging.getLogger(__name__)

# -----------------------
# Timeout helper for model fitting
# -----------------------

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout while fitting the model")

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs the given function, but raises TimeoutException if it takes longer than `timeout` seconds.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result

# -----------------------
# Helper functions for file paths
# -----------------------

def get_data_path(filename: str, folder: str = "US") -> str:
    package_dir = os.path.dirname(__file__)
    return os.path.join(package_dir, "data", "raw", folder, filename)

def get_test_data_path(filename: str, folder: str = "US") -> str:
    package_dir = os.path.dirname(__file__)
    return os.path.join(package_dir, "data", "test", folder, filename)

# -----------------------
# Data Loading Functions (Training Data)
# -----------------------

def load_data_from_file(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, delimiter=',')
        df.columns = df.columns.str.strip()
        _ = os.path.getsize(file_path)  # file_size is read but not used
    except Exception as e:
        _logger.error(f"Error reading {file_path}: {e}")
        raise
    return df

def load_renewable_data(country: str) -> pd.DataFrame:
    folder = country.upper()
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw", folder)
    filenames = [f for f in os.listdir(data_dir) if f.startswith(f"{folder}_") and f.endswith("_hourly.csv")]
    if not filenames:
        raise FileNotFoundError(f"No renewable energy data found for {country}")
    dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"]).dt.tz_localize(None)
    return df[["Datetime (UTC)", "Renewable Percentage"]]

def load_total_load_data(country: str) -> pd.DataFrame:
    folder = country.upper()
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw", folder)
    if country.upper() == "US":
        filenames = [f for f in os.listdir(data_dir) if f.startswith("ercot_load_act_hr_") and f.endswith(".csv")]
        if not filenames:
            raise FileNotFoundError(f"No total load data found for {country}")
        dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        df["Datetime (UTC)"] = pd.to_datetime(df["UTC Timestamp (Interval Ending)"]).dt.tz_localize(None)
    else:
        filenames = [f for f in os.listdir(data_dir) if f.startswith("Actual Generation per Production Type_") and f.endswith(f"_{country}.csv")]
        if not filenames:
            raise FileNotFoundError(f"No total load data found for {country}")
        dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        if "MTU" in df.columns:
            df = df.rename(columns={"MTU": "Datetime (UTC)"})
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"].str.split(" - ").str[0], format="%d.%m.%Y %H:%M").dt.tz_localize(None)
            generation_cols = [col for col in df.columns if col.endswith("Aggregated [MW]")]
            df["TOTAL Actual Load (MW)"] = df[generation_cols].replace("n/e", np.nan).astype(float).sum(axis=1)
    df = df[["Datetime (UTC)", "TOTAL Actual Load (MW)"]]
    return df 

def load_historical_data(country: str) -> pd.DataFrame:
    if country.upper() in ["US", "FR", "PL", "NL"]:
        renewable_df = load_renewable_data(country)
        load_df = load_total_load_data(country)
        renewable_df['Datetime (UTC)'] = pd.to_datetime(renewable_df['Datetime (UTC)'])
        load_df['Datetime (UTC)'] = pd.to_datetime(load_df['Datetime (UTC)'])
        renewable_df = renewable_df.sort_values("Datetime (UTC)")
        load_df = load_df.sort_values("Datetime (UTC)")
        merged = pd.merge_asof(
            renewable_df,
            load_df[['Datetime (UTC)', 'TOTAL Actual Load (MW)']],
            on="Datetime (UTC)",
            direction="nearest",
            tolerance=pd.Timedelta("30min")
        )
        return merged
    else:
        raise ValueError("Country not supported.")

# -----------------------
# Data Loading Functions (Real/Test Data)
# -----------------------

def load_real_renewable_data(country: str) -> pd.DataFrame:
    folder = country.upper()
    data_dir = os.path.join(os.path.dirname(__file__), "data", "test", folder)
    filenames = [f for f in os.listdir(data_dir) if f.startswith(f"{folder}_") and f.endswith("_hourly.csv")]
    if not filenames:
        raise FileNotFoundError(f"No real renewable energy data found for {country}")
    dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"]).dt.tz_localize(None)
    return df[["Datetime (UTC)", "Renewable Percentage"]]

def load_real_total_load_data(country: str) -> pd.DataFrame:
    folder = country.upper()
    data_dir = os.path.join(os.path.dirname(__file__), "data", "test", folder)
    if country.upper() == "US":
        filenames = [f for f in os.listdir(data_dir) if f.startswith("ercot_load_act_hr_") and f.endswith(".csv")]
        if not filenames:
            raise FileNotFoundError(f"No real total load data found for {country}")
        dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        df["Datetime (UTC)"] = pd.to_datetime(df["UTC Timestamp (Interval Ending)"]).dt.tz_localize(None)
    else:
        filenames = [f for f in os.listdir(data_dir) if f.startswith("Actual Generation per Production Type_") and f.endswith(f"_{country}.csv")]
        if not filenames:
            raise FileNotFoundError(f"No real total load data found for {country}")
        dfs = [load_data_from_file(os.path.join(data_dir, fname)) for fname in filenames]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        if "MTU" in df.columns:
            df = df.rename(columns={"MTU": "Datetime (UTC)"})
            df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"].str.split(" - ").str[0], format="%d.%m.%Y %H:%M").dt.tz_localize(None)
            generation_cols = [col for col in df.columns if col.endswith("Aggregated [MW]")]
            df["TOTAL Actual Load (MW)"] = df[generation_cols].replace("n/e", np.nan).astype(float).sum(axis=1)
    df = df[["Datetime (UTC)", "TOTAL Actual Load (MW)"]]
    return df

def load_real_historical_data(country: str) -> pd.DataFrame:
    if country.upper() in ["US", "FR", "PL", "NL"]:
        renewable_df = load_real_renewable_data(country)
        load_df = load_real_total_load_data(country)
        renewable_df['Datetime (UTC)'] = pd.to_datetime(renewable_df['Datetime (UTC)'])
        load_df['Datetime (UTC)'] = pd.to_datetime(load_df['Datetime (UTC)'])
        renewable_df = renewable_df.sort_values("Datetime (UTC)")
        load_df = load_df.sort_values("Datetime (UTC)")
        merged = pd.merge_asof(
            renewable_df,
            load_df[['Datetime (UTC)', 'TOTAL Actual Load (MW)']],
            on="Datetime (UTC)",
            direction="nearest",
            tolerance=pd.Timedelta("30min")
        )
        return merged
    else:
        raise ValueError("Country not supported.")

# -----------------------
# Preparation for Prophet
# -----------------------

def impute_total_load(data: pd.DataFrame, total_load_col: str) -> pd.DataFrame:
    data = data.copy()
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    valid_mask = data[total_load_col].notna()
    if valid_mask.sum() < 2:
        return data
    valid_data = data.loc[valid_mask, ['Datetime (UTC)', total_load_col]].rename(
        columns={'Datetime (UTC)': 'ds', total_load_col: 'y'}
    )
    m = Prophet(
        daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
        n_changepoints=15, changepoint_prior_scale=0.1
    )
    m.fit(valid_data)
    missing_mask = ~valid_mask
    if missing_mask.sum() > 0:
        missing_data = data.loc[missing_mask, ['Datetime (UTC)']].rename(columns={'Datetime (UTC)': 'ds'})
        forecast = m.predict(missing_data)
        data.loc[missing_mask, total_load_col] = forecast['yhat'].values
    return data

def prepare_features(data: pd.DataFrame, total_load_col: str = None) -> pd.DataFrame:
    _logger.info("Preparing features for data")
    data = data.copy()
    data.columns = data.columns.str.strip()
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    data = data.dropna(subset=["Renewable Percentage"])
    if total_load_col and total_load_col in data.columns:
        data = impute_total_load(data, total_load_col)
        data = data.dropna(subset=[total_load_col])
        data['Renewable MW'] = (data['Renewable Percentage'] / 100) * data[total_load_col]
    else:
        data['Renewable MW'] = data['Renewable Percentage']
    ts_data = data[['Datetime (UTC)', 'Renewable MW']].rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable MW': 'y'}
    )
    _logger.debug("Data prepared with shape: %s", ts_data.shape)
    return ts_data

def train_prophet_model(data: pd.DataFrame) -> Prophet:
    _logger.info("Training improved Prophet model")

    model = Prophet(
        n_changepoints=15,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=10)
    model.add_seasonality(name='weekly', period=7, fourier_order=6)
    # Optionally add monthly and yearly seasonality if data span permits:
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=8)
    
    valid_data = data.dropna(subset=['y'])
    if valid_data.shape[0] < 2:
        raise ValueError("Not enough valid data to train Prophet model.")
    model.fit(valid_data)
    _logger.info("Prophet model trained with improved seasonality and changepoints")
    return model

def tune_prophet_model(data: pd.DataFrame, param_grid: Dict[str, List[float]],
                       cv_initial: str = '730 days', cv_period: str = '180 days', cv_horizon: str = '365 days') -> Prophet:
    best_rmse = float('inf')
    best_params = {}
    best_model = None
    available_days = (data['ds'].max() - data['ds'].min()).days
    initial_days = int(cv_initial.split()[0])
    horizon_days = int(cv_horizon.split()[0])
    if available_days - initial_days < horizon_days:
        new_horizon = available_days - initial_days - 1
        if new_horizon < 1:
            _logger.error("Not enough data for cross-validation. Available: %d days, initial: %d days.",
                          available_days, initial_days)
            raise ValueError("Not enough data for cross-validation.")
        else:
            _logger.info("Adjusting cv_horizon from '%s' to '%d days' due to limited data (%d days available).",
                         cv_horizon, new_horizon, available_days)
            cv_horizon = f"{new_horizon} days"
    for cps in param_grid.get('changepoint_prior_scale', [0.1]):
        for sps in param_grid.get('seasonality_prior_scale', [5.0]):
            for mode in param_grid.get('seasonality_mode', ['additive']):
                for daily_fourier_order in param_grid.get('daily_fourier_order', [10]):
                    for weekly_fourier_order in param_grid.get('weekly_fourier_order', [6]):
                        _logger.info("Tuning with cps=%.4f, sps=%.4f, mode=%s, daily_fourier_order=%d, weekly_fourier_order=%d", 
                                     cps, sps, mode, daily_fourier_order, weekly_fourier_order)
                        model = Prophet(
                            changepoint_prior_scale=cps,
                            seasonality_prior_scale=sps,
                            seasonality_mode=mode,
                            daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True,
                            interval_width=0.95
                        )
                        model.add_seasonality(name='daily', period=1, fourier_order=daily_fourier_order)
                        model.add_seasonality(name='weekly', period=7, fourier_order=weekly_fourier_order)
                        try:
                            model.fit(data)
                        except Exception as e:
                            _logger.error("Model failed to train: %s", e)
                            continue
                        try:
                            df_cv = cross_validation(model, initial=cv_initial, period=cv_period, horizon=cv_horizon)
                            df_p = performance_metrics(df_cv)
                            cur_rmse = df_p['rmse'].mean()
                            _logger.info("RMSE: %.4f", cur_rmse)
                        except Exception as e:
                            _logger.error("Cross-validation failed: %s", e)
                            continue
                        if cur_rmse < best_rmse:
                            best_rmse = cur_rmse
                            best_params = {
                                'changepoint_prior_scale': cps,
                                'seasonality_prior_scale': sps,
                                'seasonality_mode': mode,
                                'daily_fourier_order': daily_fourier_order,
                                'weekly_fourier_order': weekly_fourier_order
                            }
                            best_model = model
    if best_model is None:
        _logger.warning("Could not tune the model; using default Prophet model.")
        best_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            interval_width=0.95
        )
        best_model.add_seasonality(name='daily', period=1, fourier_order=10)
        best_model.add_seasonality(name='weekly', period=7, fourier_order=6)
        valid_data = data.dropna(subset=['y'])
        if valid_data.shape[0] < 2:
            raise ValueError("Not enough valid data to train Prophet model.")
        best_model.fit(valid_data)
    else:
        _logger.info("Best hyperparameters: %s with RMSE: %.4f", best_params, best_rmse)
        print("Best hyperparameters:", best_params)
    return best_model

def plot_forecast(historical_data_original, forecast, extra_data, plot_start, plot_end, country, days, real_data=None):
    plt.figure(figsize=(14, 7))
    hist_plot = historical_data_original[(historical_data_original["ds"] >= plot_start) & 
                                         (historical_data_original["ds"] <= plot_end)]
    forecast_plot = forecast[(forecast["ds"] >= plot_start) & (forecast["ds"] <= plot_end)]
    plt.plot(hist_plot['ds'], hist_plot['y'], color='blue', label="Historical Training Data", alpha=0.6)
    plt.plot(forecast_plot['ds'], forecast_plot['renewable_mw'], linestyle="dashed", color="orange", label="Forecast Renewable (MW)")
    plt.plot(forecast_plot['ds'], forecast_plot['total_load'], linestyle="dotted", color="red", label="Forecast Total Load (MW)")
    if real_data is not None:
        plt.plot(real_data['ds'], real_data['TOTAL Actual Load (MW)'], color="magenta", label="Real Total Load", marker="o")
        plt.plot(real_data['ds'], real_data['Renewable Percentage'] / 100 * real_data['TOTAL Actual Load (MW)'], 
                 color="green", label="Real Renewable MW", marker="o")
    for date_str, values in extra_data.items():
        date_dt = pd.to_datetime(date_str)
        override_times = [date_dt + pd.Timedelta(hours=i) for i in range(24)]
        override_values = [(values[i] / 100) * forecast_plot['total_load'].mean() for i in range(24)]
        plt.scatter(override_times, override_values, color="green", label="User Override", marker="x")
    plt.xlabel("Datetime")
    plt.ylabel("Energy (MW)")
    plt.title(f"Renewable Energy and Total Load Forecast for {country}")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=30)
    plt.axvline(x=plot_end - pd.Timedelta(days=days), color='red', linestyle='dashed', linewidth=1)
    plt.axvline(x=plot_end, color='red', linestyle='dashed', linewidth=1)
    plt.show()

def parse_days(period_str: str) -> int:
    try:
        parts = period_str.split()
        return int(parts[0])
    except Exception:
        raise ValueError("Invalid period string format: {}".format(period_str))

# -----------------------
# XGBoost Helper Functions: Create time features with lag variables and iterative forecasting
# -----------------------

def create_time_features_from_index(dates: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame({
        'hour': dates.hour,
        'dayofweek': dates.dayofweek,
        'day': dates.day,
        'month': dates.month,
        'quarter': dates.quarter,
        'sin_hour': np.sin(2 * np.pi * dates.hour / 24),
        'cos_hour': np.cos(2 * np.pi * dates.hour / 24),
        'sin_dayofweek': np.sin(2 * np.pi * dates.dayofweek / 7),
        'cos_dayofweek': np.cos(2 * np.pi * dates.dayofweek / 7),
        'time_idx': (dates - dates[0]).total_seconds() / 3600  # hours since first date in the series
    }, index=dates)
    return features

def create_features_with_lags(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Assumes data is indexed by datetime and sorted
    df_features = create_time_features_from_index(data.index)
    df_features['lag_1'] = data[target_col].shift(1)
    df_features['lag_24'] = data[target_col].shift(24)
    df_features = df_features.dropna()
    return df_features

def create_time_features_for_single(timestamp: pd.Timestamp, base_time: pd.Timestamp) -> Dict:
    features = {
        'hour': timestamp.hour,
        'dayofweek': timestamp.dayofweek,
        'day': timestamp.day,
        'month': timestamp.month,
        'quarter': (timestamp.month - 1) // 3 + 1,
        'sin_hour': np.sin(2 * np.pi * timestamp.hour / 24),
        'cos_hour': np.cos(2 * np.pi * timestamp.hour / 24),
        'sin_dayofweek': np.sin(2 * np.pi * timestamp.dayofweek / 7),
        'cos_dayofweek': np.cos(2 * np.pi * timestamp.dayofweek / 7),
        'time_idx': (timestamp - base_time).total_seconds() / 3600.0
    }
    return features

def iterative_xgb_forecast(model, last_train: pd.DataFrame, forecast_steps: int, target_col: str) -> Tuple[pd.DatetimeIndex, List[float]]:
    """
    Implements an iterative (recursive) forecast using lag features.
    `last_train` is the training DataFrame (indexed by datetime) containing the target.
    It uses lag_1 and lag_24 features.
    """
    forecast_times = pd.date_range(start=last_train.index.max() + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    base_time = last_train.index[0]
    forecast_vals = []
    # Create a dictionary of known values from training
    known_values = last_train[target_col].to_dict()
    for t in forecast_times:
        feats = create_time_features_for_single(t, base_time)
        # Get lag_1: value at t - 1 hour; if not in training, use previous forecast
        lag1_time = t - pd.Timedelta(hours=1)
        if lag1_time in known_values:
            lag1 = known_values[lag1_time]
        else:
            lag1 = forecast_vals[-1] if forecast_vals else list(known_values.values())[-1]
        # Get lag_24: if available from training, else from forecast if already predicted 24 hours ahead
        lag24_time = t - pd.Timedelta(hours=24)
        if lag24_time in known_values:
            lag24 = known_values[lag24_time]
        else:
            # For the first 24 forecast points, if lag24 is not available, fallback to last training values (if available)
            if len(forecast_vals) < 24:
                # Attempt to get from training by offsetting from the last training timestamp
                candidate_time = last_train.index.max() - pd.Timedelta(hours=24 - len(forecast_vals))
                lag24 = known_values.get(candidate_time, lag1)
            else:
                lag24 = forecast_vals[-24]
        feats['lag_1'] = lag1
        feats['lag_24'] = lag24
        X_feat = pd.DataFrame([feats])
        y_pred = model.predict(X_feat)[0]
        forecast_vals.append(y_pred)
        # Add the new predicted value to known_values so it can be used as lag in future steps
        known_values[t] = y_pred
    return forecast_times, forecast_vals

# -----------------------
# Forecasts Based on Prophet
# -----------------------

def predict_renewable_pct_prophet(country: str, forecast_steps: int, 
                                  extra_data: Optional[pd.DataFrame] = None, 
                                  history_days: int = 400, start_date: Optional[str] = None,
                                  tune: bool = False, param_grid: Optional[Dict[str, List[float]]] = None
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
         renewable_data = pd.concat([renewable_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    renewable_data.set_index("Datetime (UTC)", inplace=True)

    length = str(history_days) + "D"
    data_subset = renewable_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
        
    _logger.info("Renewable % Training data range: %s to %s", train.index.min(), train.index.max())
    if not test.empty:
        _logger.info("Renewable % Test data range: %s to %s", test.index.min(), test.index.max())
    else:
        _logger.info("No test data (future forecast)")

    # Rename columns for Prophet:
    renewable_ts = train.reset_index().rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable Percentage': 'y'}
    )
    
    if tune:
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1],
                'seasonality_prior_scale': [1.0, 5.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative'],
                'daily_fourier_order': [5, 10],
                'weekly_fourier_order': [3, 6]
            }
        _logger.info("Tuning Prophet model for renewable percentage forecast...")
        model = tune_prophet_model(renewable_ts, param_grid)
    else:
        model = train_prophet_model(renewable_ts)
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = train.index.max() + pd.Timedelta(hours=1)
    future = pd.DataFrame({
        "ds": pd.date_range(start=start_dt, periods=forecast_steps, freq="H")
    })
    try:
        forecast = model.predict(future, uncertainty_samples=0)
    except TypeError:
        forecast = model.predict(future)
    forecast_series = forecast['yhat'].values
    forecast_df = pd.DataFrame({
        'ds': future['ds'],
        'renewable_pct_prophet': forecast_series
    }).set_index('ds')
    return forecast_df, test


def predict_total_load_prophet(country: str, forecast_steps: int, 
                               extra_data: Optional[pd.DataFrame] = None, 
                               history_days: int = 400, start_date: Optional[str] = None,
                               tune: bool = False, param_grid: Optional[Dict[str, List[float]]] = None
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
         total_load_data = pd.concat([total_load_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    total_load_data.set_index("Datetime (UTC)", inplace=True)

    length = str(history_days) + "D"
    data_subset = total_load_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]

    _logger.info("Training total load range: %s to %s", train.index.min(), train.index.max())
    if not test.empty:
        _logger.info("Test total load range: %s to %s", test.index.min(), test.index.max())
    else:
        _logger.info("No test data (future forecast)")

    # Prepare the training data with the expected Prophet column names:
    total_load_ts = train.reset_index().rename(
        columns={'Datetime (UTC)': 'ds', 'TOTAL Actual Load (MW)': 'y'}
    )
    
    # If tuning is requested, call the tuning function. Otherwise use the standard training.
    if tune:
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1],
                'seasonality_prior_scale': [1.0, 5.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative'],
                'daily_fourier_order': [5, 10],
                'weekly_fourier_order': [3, 6]
            }
        _logger.info("Tuning Prophet model for total load forecast...")
        model = tune_prophet_model(total_load_ts, param_grid)
    else:
        model = train_prophet_model(total_load_ts)

    if start_date:
        start_dt = pd.to_datetime(start_date)
    else:
        start_dt = train.index.max() + pd.Timedelta(hours=1)
    future = pd.DataFrame({
        "ds": pd.date_range(start=start_dt, periods=forecast_steps, freq="H")
    })
    try:
        forecast = model.predict(future, uncertainty_samples=0)
    except TypeError:
        forecast = model.predict(future)
    forecast_series = forecast['yhat'].values
    forecast_df = pd.DataFrame({
        'ds': future['ds'],
        'total_load_prophet': forecast_series
    }).set_index('ds')
    return forecast_df, test


# -----------------------
# Forecasts Based on SARIMAX using ARIMA (via pmdarima)
# -----------------------

def predict_total_load_sarimax(country: str, forecast_steps: int, 
                               exog: Optional[pd.DataFrame] = None, 
                               extra_data: Optional[pd.DataFrame] = None, 
                               history_days: int = 400, start_date: Optional[str] = None,
                               fit_timeout: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast total load using an enhanced ARIMA model that explores various candidate orders
    and enforces a maximum time limit on the model fitting.

    Parameters:
        country (str): Country code for data loading.
        forecast_steps (int): Number of forecast steps (hours).
        exog (Optional[pd.DataFrame]): Exogenous regressors (if any).
        extra_data (Optional[pd.DataFrame]): Extra data to append.
        history_days (int): Number of days of historical data to use.
        start_date (Optional[str]): If provided, forecast starting from this date.
        fit_timeout (int): Maximum seconds allowed for a single ARIMA fit.

    Returns:
        forecast_df (pd.DataFrame): Forecasted total load with datetime index.
        test (pd.DataFrame): Test data (empty if forecasting into the future).
    """
    # Load and prepare total load data (helper function must be defined)
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    
    if extra_data is not None and not extra_data.empty:
        extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
        total_load_data = pd.concat([total_load_data, extra_data])\
                              .sort_values("Datetime (UTC)")\
                              .drop_duplicates(subset=["Datetime (UTC)"])
    
    total_load_data.set_index("Datetime (UTC)", inplace=True)
    data_subset = total_load_data.last(str(history_days) + "D")
    
    # Split data into train and test sets or use all data if forecasting future values.
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    # Align exogenous regressors if provided.
    exog_train = exog.iloc[-len(train):] if exog is not None else None
    exog_test  = exog.iloc[-forecast_steps:] if exog is not None else None

    # Define candidate grids for nonseasonal (p, d, q) and seasonal orders.
    candidate_orders = [
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2)
    ]
    candidate_seasonal_orders = [
        (0, 0, 0, 24),   # No seasonal components.
        (1, 1, 1, 24)    # Allow daily seasonality.
    ]
    
    best_aic = float('inf')
    best_model = None

    # Loop over all combinations with a time limit on each fit.
    for order in candidate_orders:
        for s_order in candidate_seasonal_orders:
            try:
                model = ARIMA(order=order, seasonal_order=s_order, approximation=False)
                # Fit model with a timeout limit.
                run_with_timeout(model.fit, fit_timeout, train['TOTAL Actual Load (MW)'], exogenous=exog_train)
                current_aic = model.aic()
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_model = model
            except TimeoutException:
                # Skip candidate if it exceeds the time limit.
                print(f"Timeout reached when fitting ARIMA order {order} seasonal {s_order}. Skipping...")
                continue
            except Exception as e:
                # Skip candidates that fail due to other errors.
                # Optionally, log the exception here.
                continue

    if best_model is None:
        raise ValueError("ARIMA model fitting failed for all candidate orders.")

    # Define forecast index.
    if start_date:
        ds_index = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        ds_index = test.index

    forecast_series = best_model.predict(n_periods=forecast_steps, exogenous=exog_test)
    forecast_df = pd.DataFrame({'ds': ds_index, 'total_load_sarimax': forecast_series}).set_index('ds')
    
    return forecast_df, test


def predict_renewable_pct_sarimax(country: str, forecast_steps: int, 
                                  exog: Optional[pd.DataFrame] = None, 
                                  extra_data: Optional[pd.DataFrame] = None, 
                                  history_days: int = 400, start_date: Optional[str] = None,
                                  fit_timeout: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast renewable percentage using an enhanced ARIMA model with a time-limited fit.

    Parameters:
        country (str): Country code for data selection.
        forecast_steps (int): Number of forecast steps (hours).
        exog (Optional[pd.DataFrame]): Exogenous regressors (unused in this function but kept for consistency).
        extra_data (Optional[pd.DataFrame]): Extra data to append.
        history_days (int): Number of days of historical data to use.
        start_date (Optional[str]): If provided, forecast starting from this date.
        fit_timeout (int): Maximum seconds allowed for a single ARIMA fit.

    Returns:
        forecast_df (pd.DataFrame): Forecasted renewable percentage with datetime index.
        test (pd.DataFrame): Test data (empty if forecasting future values).
    """
    # Load and prepare renewable data (helper function must be defined)
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    
    if extra_data is not None and not extra_data.empty:
        extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
        renewable_data = pd.concat([renewable_data, extra_data])\
                           .sort_values("Datetime (UTC)")\
                           .drop_duplicates(subset=["Datetime (UTC)"])
    
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    data_subset = renewable_data.last(str(history_days) + "D")
    
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    candidate_orders = [
        (0, 1, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2)
    ]
    candidate_seasonal_orders = [
        (0, 0, 0, 24),
        (1, 1, 1, 24)
    ]
    
    best_aic = float('inf')
    best_model = None

    for order in candidate_orders:
        for s_order in candidate_seasonal_orders:
            try:
                model = ARIMA(order=order, seasonal_order=s_order, approximation=False)
                run_with_timeout(model.fit, fit_timeout, train["Renewable Percentage"])
                current_aic = model.aic()
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_model = model
            except TimeoutException:
                print(f"Timeout reached when fitting ARIMA order {order} seasonal {s_order} for renewable pct. Skipping...")
                continue
            except Exception as e:
                continue

    if best_model is None:
        raise ValueError("ARIMA model fitting failed for all candidate orders.")

    if start_date:
        ds_index = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        ds_index = test.index

    forecast_series = best_model.predict(n_periods=forecast_steps)
    forecast_df = pd.DataFrame({'ds': ds_index, 'renewable_pct_sarimax': forecast_series}).set_index('ds')
    
    return forecast_df, test

# -----------------------
# Forecasts Based on XGBoost with Iterative Forecasting using Lag Features
# -----------------------

def predict_total_load_xgboost(country: str, forecast_steps: int, 
                               extra_data: Optional[pd.DataFrame] = None, 
                               history_days: int = 400, start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
         total_load_data = pd.concat([total_load_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    total_load_data.set_index("Datetime (UTC)", inplace=True)
    total_load_data = total_load_data.resample("H").mean().dropna()
    
    length = str(history_days) + "D"
    data_subset = total_load_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    df_train_feats = create_features_with_lags(train, 'TOTAL Actual Load (MW)')
    X_train = df_train_feats
    y_train = train['TOTAL Actual Load (MW)'].loc[df_train_feats.index]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, 
                          objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    if start_date:
        forecast_times = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        forecast_times = pd.date_range(start=train.index.max() + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    forecast_values = iterative_xgb_forecast(model, train, forecast_steps, 'TOTAL Actual Load (MW)')[1]
    forecast_df = pd.DataFrame({'ds': forecast_times, 'total_load_xgboost': forecast_values}).set_index('ds')
    return forecast_df, test


def predict_renewable_pct_xgboost(country: str, forecast_steps: int, 
                                  extra_data: Optional[pd.DataFrame] = None, 
                                  history_days: int = 400, start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
         renewable_data = pd.concat([renewable_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    length = str(history_days) + "D"
    data_subset = renewable_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    df_train_feats = create_features_with_lags(train, 'Renewable Percentage')
    X_train = df_train_feats
    y_train = train['Renewable Percentage'].loc[df_train_feats.index]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, 
                          objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    if start_date:
        forecast_times = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        forecast_times = pd.date_range(start=train.index.max() + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    forecast_values = iterative_xgb_forecast(model, train, forecast_steps, 'Renewable Percentage')[1]
    forecast_df = pd.DataFrame({'ds': forecast_times, 'renewable_pct_xgboost': forecast_values}).set_index('ds')
    return forecast_df, test

# -----------------------
# Forecasts Based on Moving Average
# -----------------------

def predict_total_load_moving_average(country: str, forecast_steps: int, 
                                      history_days: int = 400, start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    total_load_data.set_index("Datetime (UTC)", inplace=True)
    total_load_data = total_load_data.resample("H").mean().dropna()
    
    length = str(history_days) + "D"
    data_subset = total_load_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    hourly_means = train.groupby(train.index.hour)['TOTAL Actual Load (MW)'].mean()
    daily_avg = train['TOTAL Actual Load (MW)'].resample("D").mean()
    days = (daily_avg.index - daily_avg.index[0]).days.astype(float)
    slope = np.polyfit(days, daily_avg.values, 1)[0] if len(days) > 1 else 0.0
    
    if start_date:
        future_times = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        future_times = pd.date_range(start=train.index.max() + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    
    forecast_series = []
    for ts in future_times:
        dt_days = (ts - train.index.max()).total_seconds() / 86400.0
        base_value = hourly_means.get(ts.hour, train['TOTAL Actual Load (MW)'].mean())
        forecast_series.append(base_value + slope * dt_days)
    
    forecast_df = pd.DataFrame({'ds': future_times, 'total_load_moving_average': forecast_series})
    forecast_df.set_index('ds', inplace=True)
    return forecast_df, test


def predict_renewable_pct_moving_average(country: str, forecast_steps: int, 
                                         history_days: int = 400, start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    
    length = str(history_days) + "D"
    data_subset = renewable_data.last(length)
    if start_date:
        train = data_subset
        test = pd.DataFrame()
    else:
        train = data_subset.iloc[:-forecast_steps]
        test = data_subset.iloc[-forecast_steps:]
    
    hourly_means = train.groupby(train.index.hour)['Renewable Percentage'].mean()
    daily_avg = train['Renewable Percentage'].resample("D").mean()
    days = (daily_avg.index - daily_avg.index[0]).days.astype(float)
    slope = np.polyfit(days, daily_avg.values, 1)[0] if len(days) > 1 else 0.0
    
    if start_date:
        future_times = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_steps, freq="H")
    else:
        future_times = pd.date_range(start=train.index.max() + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    
    forecast_series = []
    for ts in future_times:
        dt_days = (ts - train.index.max()).total_seconds() / 86400.0
        base_value = hourly_means.get(ts.hour, train['Renewable Percentage'].mean())
        forecast_series.append(base_value + slope * dt_days)
    
    forecast_df = pd.DataFrame({'ds': future_times, 'renewable_pct_moving_average': forecast_series})
    forecast_df.set_index('ds', inplace=True)
    return forecast_df, test

# -----------------------
# Forecast Comparison with Real Data Measurements
# -----------------------

def calc_rmse(series1: pd.Series, series2: pd.Series) -> float:
    return np.sqrt(np.mean((series1 - series2) ** 2))

def calc_mae(series1: pd.Series, series2: pd.Series) -> float:
    return np.mean(np.abs(series1 - series2))

def calc_mape(real: pd.Series, forecast: pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).
       Adding a small constant (1e-8) to avoid division by zero."""
    return np.mean(np.abs((real - forecast) / (real + 1e-8))) * 100

def calc_r2(real: pd.Series, forecast: pd.Series) -> float:
    """Calculate the coefficient of determination (R²)."""
    ss_res = np.sum((real - forecast) ** 2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    return 1 - ss_res / ss_tot

def calc_rmspe(real: pd.Series, forecast: pd.Series) -> float:
    """Calculate Root Mean Square Percentage Error (RMSPE)."""
    return np.sqrt(np.mean(((real - forecast) / (real + 1e-8)) ** 2)) * 100

# -----------------------
# Energy Curve Forecast with Override and Extra Data Integration
# -----------------------

def predict_energy_curve(
    country: str,
    extra_data: Dict[str, List[float]],
    current_date: str,
    forecast_load: str = "Prophet",
    forecast_percentage: str = "Prophet",
    history_days: int = 400,
    days: int = 2,
    plot: bool = False,
    plot_date: Optional[str] = None,
    tune: bool = False,
    param_grid: Optional[Dict[str, List[float]]] = None,
    cv_initial: str = '730 days',
    cv_period: str = '180 days',
    cv_horizon: str = '365 days'
) -> pd.DataFrame:
    """
    Forecast energy curve using specified models for load and percentage over a given horizon.

    Returns a DataFrame with ['ds', 'total_load', 'renewable_percentage', 'renewable_mw'].
    """
    forecast_steps = days * 24

    load_methods = {
        "prophet": predict_total_load_prophet,
        "sarimax": predict_total_load_sarimax,
        "xgboost": predict_total_load_xgboost,
        "moving_average": predict_total_load_moving_average
    }
    pct_methods = {
        "prophet": predict_renewable_pct_prophet,
        "sarimax": predict_renewable_pct_sarimax,
        "xgboost": predict_renewable_pct_xgboost,
        "moving_average": predict_renewable_pct_moving_average
    }

    key_load = forecast_load.lower().replace("-", "_")
    key_pct  = forecast_percentage.lower().replace("-", "_")

    if key_load not in load_methods:
        raise ValueError(f"Unsupported forecast_load method: {forecast_load}")
    if key_pct not in pct_methods:
        raise ValueError(f"Unsupported forecast_percentage method: {forecast_percentage}")

    # --- total load forecast ---
    if key_load in ("prophet", "sarimax"):
        total_df, _ = load_methods[key_load](
            country=country,
            forecast_steps=forecast_steps,
            extra_data=None,
            history_days=history_days,
            start_date=current_date,
            tune=tune,
            param_grid=param_grid
        )
    elif key_load == "xgboost":
        total_df, _ = load_methods[key_load](
            country=country,
            forecast_steps=forecast_steps,
            extra_data=None,
            history_days=history_days,
            start_date=current_date
        )
    else:  # moving_average
        total_df, _ = load_methods[key_load](
            country=country,
            forecast_steps=forecast_steps,
            history_days=history_days,
            start_date=current_date
        )

    # --- renewable percentage forecast ---
    if key_pct in ("prophet", "sarimax"):
        pct_df, _ = pct_methods[key_pct](
            country=country,
            forecast_steps=forecast_steps,
            extra_data=None,
            history_days=history_days,
            start_date=current_date,
            tune=tune,
            param_grid=param_grid
        )
    elif key_pct == "xgboost":
        pct_df, _ = pct_methods[key_pct](
            country=country,
            forecast_steps=forecast_steps,
            extra_data=None,
            history_days=history_days,
            start_date=current_date
        )
    else:  # moving_average
        pct_df, _ = pct_methods[key_pct](
            country=country,
            forecast_steps=forecast_steps,
            history_days=history_days,
            start_date=current_date
        )

    # Combine and compute renewable_mw
    df = pd.DataFrame({
        'ds': total_df.index,
        'total_load': total_df.iloc[:, 0].values,
        'renewable_percentage': pct_df.iloc[:, 0].reindex(total_df.index).values
    }, index=total_df.index)
    df['renewable_mw'] = df['total_load'] * df['renewable_percentage'] / 100

    return df

# -----------------------
# Entry Point
# -----------------------

if __name__ == "__main__":
    country = "NL"
    horizon_days = 3
    forecast_steps = horizon_days * 24  # hours
    history_days = 0
    start_date = "2024-01-01 00:00:00"
    
    # Load extra data from training (raw) data; only include data before the test start.
    renewable_extra = load_renewable_data(country)
    total_load_extra = load_total_load_data(country)
    renewable_extra = renewable_extra[pd.to_datetime(renewable_extra["Datetime (UTC)"]) < pd.to_datetime(start_date)]
    total_load_extra = total_load_extra[pd.to_datetime(total_load_extra["Datetime (UTC)"]) < pd.to_datetime(start_date)]
    
