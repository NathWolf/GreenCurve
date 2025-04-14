#!/usr/bin/env python3
import logging
import warnings
from typing import List, Dict, Optional
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX

from typing import Tuple 
import xgboost as xgb
from xgboost import XGBRegressor

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)  # Hide non-critical logs

_logger = logging.getLogger(__name__)

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
    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
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
    _logger.info("Training Prophet model")
    model = Prophet(
        n_changepoints=10,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    valid_data = data.dropna(subset=['y'])
    if valid_data.shape[0] < 2:
        raise ValueError("Not enough valid data to train Prophet model.")
    model.fit(valid_data)
    _logger.info("Prophet model trained successfully")
    return model

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

def tune_prophet_model(data: pd.DataFrame, param_grid: Dict[str, List[float]],
                       cv_initial: str = '730 days', cv_period: str = '180 days', cv_horizon: str = '365 days') -> Prophet:
    best_rmse = float('inf')
    best_params = {}
    best_model = None
    available_days = (data['ds'].max() - data['ds'].min()).days
    initial_days = parse_days(cv_initial)
    horizon_days = parse_days(cv_horizon)
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
    for cps in param_grid.get('changepoint_prior_scale', [0.05]):
        for sps in param_grid.get('seasonality_prior_scale', [10.0]):
            for mode in param_grid.get('seasonality_mode', ['additive']):
                for daily_fourier_order in param_grid.get('daily_fourier_order', [10]):
                    for weekly_fourier_order in param_grid.get('weekly_fourier_order', [5]):
                        _logger.info("Tuning with cps=%.4f, sps=%.4f, mode=%s, daily_fourier_order=%d, weekly_fourier_order=%d", 
                                     cps, sps, mode, daily_fourier_order, weekly_fourier_order)
                        model = Prophet(
                            changepoint_prior_scale=cps,
                            seasonality_prior_scale=sps,
                            seasonality_mode=mode,
                            daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True
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
            yearly_seasonality=True
        )
        best_model.add_seasonality(name='daily', period=1, fourier_order=10)
        best_model.add_seasonality(name='weekly', period=7, fourier_order=5)
        valid_data = data.dropna(subset=['y'])
        if valid_data.shape[0] < 2:
            raise ValueError("Not enough valid data to train Prophet model.")
        best_model.fit(valid_data)
    else:
        _logger.info("Best hyperparameters: %s with RMSE: %.4f", best_params, best_rmse)
        print("Best hyperparameters:", best_params)
    return best_model

# -----------------------
# Forecasts Based on Prophet
# (Extra data parameter is added to inject known data before the test period)
# -----------------------

def predict_renewable_pct_prophet(country: str, forecast_steps: int, extra_data: Optional[pd.DataFrame] = None, history_days: int = 400):
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    # Inject extra data if available.
    if extra_data is not None and not extra_data.empty:
        extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
        renewable_data = pd.concat([renewable_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()

    length = str(history_days) + "D"
    train = renewable_data.last(length).iloc[:-forecast_steps]
    test = renewable_data.last(length).iloc[-forecast_steps:]

    _logger.info("Renewable % Training data range: %s to %s", train.index.min(), train.index.max())
    _logger.info("Renewable % Test data range: %s to %s", test.index.min(), test.index.max())

    model = Prophet(
        n_changepoints=10,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)

    renewable_ts = train.reset_index().rename(
        columns={'Datetime (UTC)': 'ds', 'Renewable Percentage': 'y'}
    )
    model.fit(renewable_ts)
    future = pd.DataFrame({
        "ds": pd.date_range(start=train.index.max() + pd.Timedelta(hours=1),
                             periods=forecast_steps, freq="H")
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

def predict_total_load_prophet(country: str, forecast_steps: int, extra_data: Optional[pd.DataFrame] = None, history_days: int = 400):
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    # Inject extra data if available.
    if extra_data is not None and not extra_data.empty:
         extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
         total_load_data = pd.concat([total_load_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    total_load_data.set_index("Datetime (UTC)", inplace=True)

    length = str(history_days) + "D"
    train = total_load_data.last(length).iloc[:-forecast_steps]
    test = total_load_data.last(length).iloc[-forecast_steps:]

    _logger.info("Training total load range: %s to %s", train.index.min(), train.index.max())
    _logger.info("Test total load range: %s to %s", test.index.min(), test.index.max())

    model = Prophet(
        n_changepoints=10,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)

    total_load_ts = train.reset_index().rename(
        columns={'Datetime (UTC)': 'ds', 'TOTAL Actual Load (MW)': 'y'}
    )
    model.fit(total_load_ts)
    future = pd.DataFrame({
        "ds": pd.date_range(start=train.index.max() + pd.Timedelta(hours=1),
                             periods=forecast_steps, freq="H")
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
# Forecasts Based on SARIMAX
# (Extra data parameter added similarly)
# -----------------------

def predict_renewable_pct_sarimax(country: str, forecast_steps: int, exog: pd.DataFrame = None, extra_data: Optional[pd.DataFrame] = None, history_days: int = 400):
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    # Inject extra data if available.
    if extra_data is not None and not extra_data.empty:
        extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
        renewable_data = pd.concat([renewable_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    
    # Use last x days for training.
    length = str(history_days) + "D"
    train = renewable_data.last(length).iloc[:-forecast_steps]
    test = renewable_data.last(length).iloc[-forecast_steps:]
    
    model_fit = SARIMAX(train["Renewable Percentage"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)).fit(disp=False)
    forecast_obj = model_fit.get_forecast(steps=forecast_steps)
    forecast_series = forecast_obj.predicted_mean
    forecast_df = pd.DataFrame({
         'ds': test.index,
         'renewable_pct_sarimax': forecast_series.values
    }).set_index('ds')
    return forecast_df, test

def predict_total_load_sarimax(country: str, forecast_steps: int, exog: pd.DataFrame = None, extra_data: Optional[pd.DataFrame] = None, history_days: int = 400):
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    # Inject extra data if available.
    if extra_data is not None and not extra_data.empty:
         extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
         total_load_data = pd.concat([total_load_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    total_load_data.set_index("Datetime (UTC)", inplace=True)

    # Use last x days for training.
    length = str(history_days) + "D"
    train = total_load_data.last(length).iloc[:-forecast_steps]
    test = total_load_data.last(length).iloc[-forecast_steps:]
    
    exog_train = exog.iloc[-len(train):] if exog is not None else None
    exog_test = exog.iloc[-forecast_steps:] if exog is not None else None

    model_fit = fit_sarimax_model(train['TOTAL Actual Load (MW)'], exog=exog_train,
                                  order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
    forecast, _ = forecast_sarimax(model_fit, steps=forecast_steps, exog_future=exog_test)
    forecast_df = pd.DataFrame({
        'ds': test.index,
        'total_load_sarimax': forecast.values
    }).set_index('ds')
    return forecast_df, test

# Auxiliary SARIMAX functions
def fit_sarimax_model(endog: pd.Series, exog: pd.DataFrame = None,
                      order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)) -> SARIMAX:
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order, exog=exog)
    model_fit = model.fit(disp=False, method='bfgs', maxiter=300)
    _logger.info("SARIMAX summary:\n%s", model_fit.summary())
    return model_fit

def forecast_sarimax(model_fit: SARIMAX, steps: int, exog_future: pd.DataFrame = None):
    forecast_obj = model_fit.get_forecast(steps=steps, exog=exog_future)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    return forecast, conf_int

# -----------------------
# Forecasts Based on XGBoost
# -----------------------

def create_time_features_from_index(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create time-based features from a DatetimeIndex.
    Features include:
      - hour, dayofweek, day, month, quarter
      - cyclical representations for hour and dayofweek
    """
    features = pd.DataFrame({
        'hour': dates.hour,
        'dayofweek': dates.dayofweek,
        'day': dates.day,
        'month': dates.month,
        'quarter': dates.quarter,
        'sin_hour': np.sin(2 * np.pi * dates.hour / 24),
        'cos_hour': np.cos(2 * np.pi * dates.hour / 24),
        'sin_dayofweek': np.sin(2 * np.pi * dates.dayofweek / 7),
        'cos_dayofweek': np.cos(2 * np.pi * dates.dayofweek / 7)
    }, index=dates)
    return features

def predict_total_load_xgboost(country: str, forecast_steps: int,
                               extra_data: Optional[pd.DataFrame] = None,
                               history_days: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast the total load using XGBoost.
    
    - Loads the total load data and (optionally) extra data.
    - Resamples the data to hourly frequency.
    - Uses the last `history_days` days to train a model that uses time-based features.
    - Forecasts total load for the given number of forecast_steps (hours).
    
    Returns a tuple of:
      (forecast_df, test_set)
    """
    # Load data and inject extra data if available.
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data['Datetime (UTC)'] = pd.to_datetime(extra_data['Datetime (UTC)'])
         total_load_data = pd.concat([total_load_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    
    # Set datetime index and resample to hourly values.
    total_load_data.set_index("Datetime (UTC)", inplace=True)
    total_load_data = total_load_data.resample("H").mean().dropna()
    
    # Split the data into training and test sets.
    length = str(history_days) + "D"
    train = total_load_data.last(length).iloc[:-forecast_steps]
    test = total_load_data.last(length).iloc[-forecast_steps:]
    
    # Create time features.
    X_train = create_time_features_from_index(train.index)
    y_train = train['TOTAL Actual Load (MW)']
    
    # Train the XGBoost regressor.
    model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    # Prepare the future dataframe.
    last_train_time = train.index.max()
    future_times = pd.date_range(start=last_train_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    X_future = create_time_features_from_index(future_times)
    
    # Generate the forecast.
    forecast_values = model.predict(X_future)
    forecast_df = pd.DataFrame({'ds': future_times, 'total_load_xgboost': forecast_values}).set_index('ds')
    
    return forecast_df, test

def predict_renewable_pct_xgboost(country: str, forecast_steps: int,
                                  extra_data: Optional[pd.DataFrame] = None,
                                  history_days: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast the renewable percentage using XGBoost.
    
    - Loads the renewable percentage data and (optionally) extra data.
    - Resamples the data to hourly frequency.
    - Uses the last `history_days` days of data to train the regressor with time-based features.
    - Forecasts renewable percentage for forecast_steps hours.
    
    Returns a tuple of:
      (forecast_df, test_set)
    """
    # Load data and inject extra data if available.
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    if extra_data is not None and not extra_data.empty:
         extra_data["Datetime (UTC)"] = pd.to_datetime(extra_data["Datetime (UTC)"])
         renewable_data = pd.concat([renewable_data, extra_data]).sort_values("Datetime (UTC)").drop_duplicates(subset=["Datetime (UTC)"])
    
    # Set datetime index and resample to hourly values.
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    
    # Split into training and test sets.
    length = str(history_days) + "D"
    train = renewable_data.last(length).iloc[:-forecast_steps]
    test = renewable_data.last(length).iloc[-forecast_steps:]
    
    # Create time features.
    X_train = create_time_features_from_index(train.index)
    y_train = train['Renewable Percentage']
    
    # Train the XGBoost regressor.
    model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    # Prepare future times.
    last_train_time = train.index.max()
    future_times = pd.date_range(start=last_train_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    X_future = create_time_features_from_index(future_times)
    
    # Generate forecast.
    forecast_values = model.predict(X_future)
    forecast_df = pd.DataFrame({'ds': future_times, 'renewable_pct_xgboost': forecast_values}).set_index('ds')
    
    return forecast_df, test

# -----------------------
# Forecasts Based on Moving Average
# -----------------------

def predict_total_load_moving_average(country: str, forecast_steps: int, history_days: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast the total load using a moving average computed by hour-of-day.
    For each future timestamp, it uses the mean total load of training data that falls in the same hour.
    
    Returns a tuple (forecast_df, test_set)
    """
    total_load_data = load_total_load_data(country)
    total_load_data['Datetime (UTC)'] = pd.to_datetime(total_load_data['Datetime (UTC)'])
    total_load_data = total_load_data.sort_values("Datetime (UTC)")
    total_load_data.set_index("Datetime (UTC)", inplace=True)
    total_load_data = total_load_data.resample("H").mean().dropna()
    
    length = str(history_days) + "D"
    train = total_load_data.last(length).iloc[:-forecast_steps]
    test = total_load_data.last(length).iloc[-forecast_steps:]
    
    last_train_time = train.index.max()
    future_times = pd.date_range(start=last_train_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    
    # Compute the mean total load for each hour of the day in the training data.
    hourly_means = train.groupby(train.index.hour)['TOTAL Actual Load (MW)'].mean()
    
    forecast_series = []
    for ts in future_times:
        # Use the hourly average; if missing, fall back to overall training mean.
        forecast_value = hourly_means.get(ts.hour, train['TOTAL Actual Load (MW)'].mean())
        forecast_series.append(forecast_value)
    
    forecast_df = pd.DataFrame({'ds': future_times, 'total_load_moving_average': forecast_series})
    forecast_df.set_index('ds', inplace=True)
    return forecast_df, test

def predict_renewable_pct_moving_average(country: str, forecast_steps: int, history_days: int = 400) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast the renewable percentage using a moving average computed by hour-of-day.
    For each future timestamp, it uses the mean renewable percentage of training data that falls in the same hour.
    
    Returns a tuple (forecast_df, test_set)
    """
    renewable_data = load_renewable_data(country)
    renewable_data["Datetime (UTC)"] = pd.to_datetime(renewable_data["Datetime (UTC)"])
    renewable_data = renewable_data.sort_values("Datetime (UTC)")
    renewable_data.set_index("Datetime (UTC)", inplace=True)
    renewable_data = renewable_data.resample("H").mean().dropna()
    
    length = str(history_days) + "D"
    train = renewable_data.last(length).iloc[:-forecast_steps]
    test = renewable_data.last(length).iloc[-forecast_steps:]
    
    last_train_time = train.index.max()
    future_times = pd.date_range(start=last_train_time + pd.Timedelta(hours=1), periods=forecast_steps, freq="H")
    
    # Compute the mean renewable percentage for each hour of the day in the training data.
    hourly_means = train.groupby(train.index.hour)['Renewable Percentage'].mean()
    
    forecast_series = []
    for ts in future_times:
        forecast_value = hourly_means.get(ts.hour, train['Renewable Percentage'].mean())
        forecast_series.append(forecast_value)
    
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

# Updated compare_forecasts now accepts extra data for both total load and renewable pct.
def compare_forecasts(country: str, forecast_steps: int = 48, start_date: Optional[pd.Timestamp] = None,
                      extra_total_load: Optional[pd.DataFrame] = None,
                      extra_renewable: Optional[pd.DataFrame] = None,
                      history_days: int = 400):
    # Get forecasts providing the extra data if available.
    prophet_total_load_forecast, _ = predict_total_load_prophet(country, forecast_steps, extra_data=extra_total_load, history_days=history_days)
    sarimax_total_load_forecast, _ = predict_total_load_sarimax(country, forecast_steps, extra_data=extra_total_load, history_days=history_days)
    xgb_total_load_forecast, _ = predict_total_load_xgboost(country, forecast_steps, extra_data=extra_total_load, history_days=history_days)
    ma_total_load_forecast, _ = predict_total_load_moving_average(country, forecast_steps, history_days=history_days)
    
    prophet_pct_forecast, _ = predict_renewable_pct_prophet(country, forecast_steps, extra_data=extra_renewable, history_days=history_days)
    sarimax_pct_forecast, _ = predict_renewable_pct_sarimax(country, forecast_steps, extra_data=extra_renewable, history_days=history_days)
    xgb_pct_forecast, _ = predict_renewable_pct_xgboost(country, forecast_steps, extra_data=extra_renewable, history_days=history_days)
    ma_pct_forecast, _ = predict_renewable_pct_moving_average(country, forecast_steps, history_days=history_days)

    # Determine the forecast starting time.
    if start_date is None:
        forecast_start = prophet_total_load_forecast.index.min()
    else:
        forecast_start = pd.to_datetime(start_date)

    # Create a common index: hourly timestamps starting at forecast_start for forecast_steps periods.
    common_index = pd.date_range(start=forecast_start, periods=forecast_steps, freq="H")

    # Load real data (for example, 2024 real data) from the test folder.
    real_hist = load_real_historical_data(country)
    real_hist = real_hist.rename(columns={"Datetime (UTC)": "ds"})
    real_hist = real_hist.sort_values("ds")
    # Restrict real data to the forecast period.
    real_data = real_hist[(real_hist["ds"] >= common_index[0]) & (real_hist["ds"] <= common_index[-1])].set_index("ds")
    
    # Align real series to the common index using forward fill.
    real_total = real_data["TOTAL Actual Load (MW)"].reindex(common_index, method='ffill')
    real_pct = real_data["Renewable Percentage"].reindex(common_index, method='ffill')
    real_renewable_mw = (real_pct / 100) * real_total

    # Extract and align forecast series to the common index.
    prophet_total = prophet_total_load_forecast["total_load_prophet"].reindex(common_index, method='ffill')
    sarimax_total = sarimax_total_load_forecast["total_load_sarimax"].reindex(common_index, method='ffill')
    xgb_total = xgb_total_load_forecast["total_load_xgboost"].reindex(common_index, method='ffill')
    ma_total = ma_total_load_forecast["total_load_moving_average"].reindex(common_index, method='ffill')

    prophet_pct = prophet_pct_forecast["renewable_pct_prophet"].reindex(common_index, method='ffill')
    sarimax_pct = sarimax_pct_forecast["renewable_pct_sarimax"].reindex(common_index, method='ffill')
    xgb_pct = xgb_pct_forecast["renewable_pct_xgboost"].reindex(common_index, method='ffill')
    ma_pct = ma_pct_forecast["renewable_pct_moving_average"].reindex(common_index, method='ffill')

    # Compute renewable MW forecasts.
    prophet_renewable_mw = (prophet_pct / 100) * prophet_total
    sarimax_renewable_mw = (sarimax_pct / 100) * sarimax_total
    xgb_renewable_mw = (xgb_pct / 100) * xgb_total
    ma_renewable_mw = (ma_pct / 100) * ma_total

    # Calculate error metrics.
    # For Total Load:
    rmse_total_prophet = calc_rmse(prophet_total, real_total)
    mae_total_prophet = calc_mae(prophet_total, real_total)
    rmse_total_sarimax = calc_rmse(sarimax_total, real_total)
    mae_total_sarimax = calc_mae(sarimax_total, real_total)
    rmse_total_xgb = calc_rmse(xgb_total, real_total)
    mae_total_xgb = calc_mae(xgb_total, real_total)
    rmse_total_ma = calc_rmse(ma_total, real_total)
    mae_total_ma = calc_mae(ma_total, real_total)
    
    # For Renewable Percentage:
    rmse_pct_prophet = calc_rmse(prophet_pct, real_pct)
    mae_pct_prophet = calc_mae(prophet_pct, real_pct)
    rmse_pct_sarimax = calc_rmse(sarimax_pct, real_pct)
    mae_pct_sarimax = calc_mae(sarimax_pct, real_pct)
    rmse_pct_xgb = calc_rmse(xgb_pct, real_pct)
    mae_pct_xgb = calc_mae(xgb_pct, real_pct)
    rmse_pct_ma = calc_rmse(ma_pct, real_pct)
    mae_pct_ma = calc_mae(ma_pct, real_pct)
    
    # For Renewable MW:
    rmse_mw_prophet = calc_rmse(prophet_renewable_mw, real_renewable_mw)
    mae_mw_prophet = calc_mae(prophet_renewable_mw, real_renewable_mw)
    rmse_mw_sarimax = calc_rmse(sarimax_renewable_mw, real_renewable_mw)
    mae_mw_sarimax = calc_mae(sarimax_renewable_mw, real_renewable_mw)
    rmse_mw_xgb = calc_rmse(xgb_renewable_mw, real_renewable_mw)
    mae_mw_xgb = calc_mae(xgb_renewable_mw, real_renewable_mw)
    rmse_mw_ma = calc_rmse(ma_renewable_mw, real_renewable_mw)
    mae_mw_ma = calc_mae(ma_renewable_mw, real_renewable_mw)
    
    print("Error Metrics (Forecast vs Real Data) over the forecast period:")
    print("Total Load:")
    print(f"  Prophet: RMSE = {rmse_total_prophet:.2f}, MAE = {mae_total_prophet:.2f}")
    print(f"  SARIMAX: RMSE = {rmse_total_sarimax:.2f}, MAE = {mae_total_sarimax:.2f}")
    print(f"  XGBoost: RMSE = {rmse_total_xgb:.2f}, MAE = {mae_total_xgb:.2f}")
    print(f"  Moving Avg: RMSE = {rmse_total_ma:.2f}, MAE = {mae_total_ma:.2f}\n")
    
    print("Renewable %:")
    print(f"  Prophet: RMSE = {rmse_pct_prophet:.2f}, MAE = {mae_pct_prophet:.2f}")
    print(f"  SARIMAX: RMSE = {rmse_pct_sarimax:.2f}, MAE = {mae_pct_sarimax:.2f}")
    print(f"  XGBoost: RMSE = {rmse_pct_xgb:.2f}, MAE = {mae_pct_xgb:.2f}")
    print(f"  Moving Avg: RMSE = {rmse_pct_ma:.2f}, MAE = {mae_pct_ma:.2f}\n")
    
    print("Renewable MW:")
    print(f"  Prophet: RMSE = {rmse_mw_prophet:.2f}, MAE = {mae_mw_prophet:.2f}")
    print(f"  SARIMAX: RMSE = {rmse_mw_sarimax:.2f}, MAE = {mae_mw_sarimax:.2f}")
    print(f"  XGBoost: RMSE = {rmse_mw_xgb:.2f}, MAE = {mae_mw_xgb:.2f}")
    print(f"  Moving Avg: RMSE = {rmse_mw_ma:.2f}, MAE = {mae_mw_ma:.2f}")

    # Main plot: forecasts vs. real data.
    plt.figure(figsize=(14, 16))
    
    # Subplot 1: Total Load Comparison.
    plt.subplot(4, 1, 1)
    plt.plot(prophet_total.index, prophet_total, label="Prophet Total Load", marker="x")
    plt.plot(sarimax_total.index, sarimax_total, label="SARIMAX Total Load", marker="^")
    plt.plot(xgb_total.index, xgb_total, label="XGBoost Total Load", marker="s")
    plt.plot(ma_total.index, ma_total, label="Moving Avg Total Load", marker="d")
    plt.plot(real_total.index, real_total, label="Real Total Load", marker="o", color="magenta")
    plt.xlabel("Datetime")
    plt.ylabel("Total Load (MW)")
    plt.title(f"Total Load Comparison for {country}")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Percentage Renewable Comparison.
    plt.subplot(4, 1, 2)
    plt.plot(prophet_pct.index, prophet_pct, label="Prophet % Renewable", marker="x")
    plt.plot(sarimax_pct.index, sarimax_pct, label="SARIMAX % Renewable", marker="^")
    plt.plot(xgb_pct.index, xgb_pct, label="XGBoost % Renewable", marker="s")
    plt.plot(ma_pct.index, ma_pct, label="Moving Avg % Renewable", marker="d")
    plt.plot(real_pct.index, real_pct, label="Real % Renewable", marker="o", color="magenta")
    plt.xlabel("Datetime")
    plt.ylabel("% Renewable")
    plt.title(f"Percentage Renewable Comparison for {country}")
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Renewable MW Comparison.
    plt.subplot(4, 1, 3)
    plt.plot(prophet_total.index, prophet_renewable_mw, label="Prophet Renewable MW", marker="x")
    plt.plot(sarimax_total.index, sarimax_renewable_mw, label="SARIMAX Renewable MW", marker="^")
    plt.plot(xgb_total.index, xgb_renewable_mw, label="XGBoost Renewable MW", marker="s")
    plt.plot(ma_total.index, ma_renewable_mw, label="Moving Avg Renewable MW", marker="d")
    plt.plot(real_renewable_mw.index, real_renewable_mw, label="Real Renewable MW", marker="o", color="magenta")
    plt.xlabel("Datetime")
    plt.ylabel("Renewable MW")
    plt.title(f"Renewable MW Comparison for {country}")
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Percentage Renewable Errors (Real - Forecast).
    plt.subplot(4, 1, 4)
    error_pct_prophet = real_pct - prophet_pct
    error_pct_sarimax = real_pct - sarimax_pct
    error_pct_xgb = real_pct - xgb_pct
    error_pct_ma = real_pct - ma_pct
    plt.plot(common_index, error_pct_prophet, label="Error % Prophet", marker="x")
    plt.plot(common_index, error_pct_sarimax, label="Error % SARIMAX", marker="^")
    plt.plot(common_index, error_pct_xgb, label="Error % XGBoost", marker="s")
    plt.plot(common_index, error_pct_ma, label="Error % Moving Avg", marker="d")
    plt.xlabel("Datetime")
    plt.ylabel("Error (%)")
    plt.title(f"% Renewable Error for {country}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Basic assertions to ensure forecasts are not empty.
    assert not prophet_total_load_forecast.empty, "Prophet total load forecast is empty."
    assert not sarimax_total_load_forecast.empty, "SARIMAX total load forecast is empty."
    assert not xgb_total_load_forecast.empty, "XGBoost total load forecast is empty."
    assert not ma_total_load_forecast.empty, "Moving Avg total load forecast is empty."
    assert not prophet_pct_forecast.empty, "Prophet % renewable forecast is empty."
    assert not sarimax_pct_forecast.empty, "SARIMAX % renewable forecast is empty."
    assert not xgb_pct_forecast.empty, "XGBoost % renewable forecast is empty."
    assert not ma_pct_forecast.empty, "Moving Avg % renewable forecast is empty."

# -----------------------
# Energy Curve Forecast with Override and Extra Data Integration
# -----------------------

def predict_energy_curve(country: str, extra_data: Dict[str, List[float]], current_date: str,
                         days: int = 2, plot: bool = False, plot_date: str = None,
                         tune: bool = False, param_grid: Optional[Dict[str, List[float]]] = None,
                         cv_initial: str = '730 days', cv_period: str = '180 days', cv_horizon: str = '365 days') -> pd.DataFrame:
    _logger.info("Starting energy forecast for: %s", country)
    if country.upper() in ["US", "FR", "PL", "NL"]:
        historical_data = load_historical_data(country)
        data_ts = prepare_features(historical_data, total_load_col='TOTAL Actual Load (MW)')
    else:
        raise ValueError("Country not supported. Options: US, FR, PL, NL.")
    
    historical_data_original = data_ts.copy()

    # Forecast total load using Prophet.
    _logger.info("Training Prophet model for total load forecast...")
    total_load_model = Prophet(
        n_changepoints=10,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    total_load_model.add_seasonality(name='daily', period=1, fourier_order=5)
    total_load_model.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    total_load_ts = historical_data[['Datetime (UTC)', 'TOTAL Actual Load (MW)']].rename(
        columns={'Datetime (UTC)': 'ds', 'TOTAL Actual Load (MW)': 'y'}
    ).dropna()
    total_load_model.fit(total_load_ts)
    last_available_date = data_ts['ds'].max()
    forecast_start = last_available_date + pd.Timedelta(hours=1)
    forecast_steps = days * 24
    future = pd.DataFrame({"ds": pd.date_range(start=forecast_start, periods=forecast_steps, freq="H")})
    _logger.info("Forecasting total load for %d future points", len(future))
    try:
        total_load_forecast = total_load_model.predict(future, uncertainty_samples=0)
    except TypeError:
        total_load_forecast = total_load_model.predict(future)
    total_load_forecast = total_load_forecast[['ds', 'yhat']].rename(columns={'yhat': 'total_load'})
    data_ts = pd.merge_asof(data_ts, total_load_forecast, on="ds", direction="nearest")
    data_ts['total_load'] = data_ts['total_load'].ffill()
    
    # Apply override if provided.
    for date_str, values in extra_data.items():
        date_dt = pd.to_datetime(date_str)
        for hour, value in enumerate(values):
            time_point = date_dt + pd.Timedelta(hours=hour)
            forecasted_total_load = total_load_forecast.loc[total_load_forecast['ds'] == time_point, 'total_load'].values
            if len(forecasted_total_load) > 0:
                forecasted_total_load = forecasted_total_load[0]
            else:
                forecasted_total_load = data_ts['total_load'].ffill().iloc[-1]
            renewable_mw = (value / 100) * forecasted_total_load
            if time_point in data_ts['ds'].values:
                data_ts.loc[data_ts['ds'] == time_point, 'y'] = renewable_mw
            else:
                new_row = pd.DataFrame({'ds': [time_point], 'y': [renewable_mw], 'total_load': [forecasted_total_load]})
                data_ts = pd.concat([data_ts, new_row], ignore_index=True)
    data_ts = data_ts.sort_values('ds').drop_duplicates('ds')
    
    _logger.info("Training Prophet model for renewable energy forecast...")
    model = Prophet(
        n_changepoints=10,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=0.1,
        seasonality_mode='additive',
        daily_seasonality=False,
        weekly_seasonality=False,
        interval_width=0.95
    )
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    
    valid_data = data_ts.dropna(subset=['y'])
    if valid_data.shape[0] < 2:
        raise ValueError("Not enough valid data to train Prophet model.")
    model.fit(valid_data)
    try:
        renewable_forecast = model.predict(future, uncertainty_samples=0)
    except TypeError:
        renewable_forecast = model.predict(future)
    renewable_forecast = renewable_forecast[['ds', 'yhat']].rename(columns={'yhat': 'renewable_mw'})
    final_forecast = pd.merge(renewable_forecast, total_load_forecast, on="ds", how="left")
    final_forecast['renewable_percentage'] = (final_forecast['renewable_mw'] / final_forecast['total_load']) * 100
    final_forecast = final_forecast[['ds', 'renewable_mw', 'total_load', 'renewable_percentage']]
    
    plot_start = pd.to_datetime(plot_date) if plot_date else last_available_date - pd.Timedelta(days=7)
    plot_end = final_forecast['ds'].max()
    if plot:
        plot_forecast(historical_data_original, final_forecast, extra_data, plot_start, plot_end, country, days)
    _logger.info("Forecast complete")
    return final_forecast

# -----------------------
# Entry Point
# -----------------------

if __name__ == "__main__":
    country = "NL"
    horizon_days = 3
    forecast_steps = horizon_days * 24  #hours
    history_days = 100
    test_start = "2024-05-01 00:00:00"
    
    # Load extra data from the training (raw) data, selecting records before the test start.
    renewable_extra = load_renewable_data(country)
    total_load_extra = load_total_load_data(country)
    # Filter extra data: only include data before the test start.
    renewable_extra = renewable_extra[pd.to_datetime(renewable_extra["Datetime (UTC)"]) < pd.to_datetime(test_start)]
    total_load_extra = total_load_extra[pd.to_datetime(total_load_extra["Datetime (UTC)"]) < pd.to_datetime(test_start)]
    
    # Now, call compare_forecasts with the extra data and the defined test start.
    compare_forecasts(country, forecast_steps, start_date=test_start,
                      extra_total_load=total_load_extra, extra_renewable=renewable_extra,
                      history_days=history_days)
