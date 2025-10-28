"""
Forecasting helpers for time series prediction.

This module provides a flexible forecasting function that supports multiple
models, including Exponential Smoothing, ARIMA, and Prophet.
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _STATS_AVAILABLE = True
except ImportError:
    _STATS_AVAILABLE = False

# We avoid importing heavy/compiled optional dependencies at module import time
# because compiled extensions (pmdarima, prophet) can raise binary-compatibility
# errors if they were built against a different numpy ABI. We'll import them
# lazily inside the branches that need them and raise friendly errors if they
# fail to load.
_ARIMA_AVAILABLE = None
_PROPHET_AVAILABLE = None


def forecast_series(
    series: pd.Series,
    model: str = 'ets',
    periods: int = 12,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Generates a forecast for a time series using the specified model.

    Args:
        series: The time series data to forecast.
        model: The forecasting model to use ('ets', 'arima', or 'prophet').
        periods: The number of periods to forecast into the future.
        **kwargs: Additional arguments for the forecasting model.

    Returns:
        A DataFrame containing the forecast, including upper and lower bounds if available.

    Raises:
        ValueError: If the selected model is not supported or its dependencies are not installed.
        Exception: For any errors during the forecasting process.
    """
    s = series.dropna().astype(float)
    if len(s) < 3:
        raise ValueError('Series is too short for forecasting.')

    if model == 'ets':
        if not _STATS_AVAILABLE:
            raise ValueError("statsmodels is not installed. Please run 'pip install statsmodels'.")
        return _forecast_ets(s, periods, **kwargs)
    elif model == 'arima':
        # Lazy import of pmdarima to avoid import-time binary ABI errors
        global _ARIMA_AVAILABLE
        if _ARIMA_AVAILABLE is None:
            try:
                from pmdarima import auto_arima  # type: ignore
                _ARIMA_AVAILABLE = True
            except Exception as e:
                _ARIMA_AVAILABLE = False
                msg = (
                    "pmdarima could not be imported. This often means the installed pmdarima "
                    "wheel was compiled against a different numpy ABI.\n"
                    "Try reinstalling numpy and pmdarima (e.g., `pip install --upgrade --force-reinstall numpy pmdarima`)\n"
                    "or install pmdarima from source: `pip install --no-binary=pmdarima pmdarima`.\n"
                )
                raise ValueError(f"{msg}Original error: {e}")
        return _forecast_arima(s, periods, **kwargs)
    elif model == 'prophet':
        # Lazy import of Prophet (fbprophet/prophet) with friendly guidance
        global _PROPHET_AVAILABLE
        if _PROPHET_AVAILABLE is None:
            try:
                from prophet import Prophet  # type: ignore
                _PROPHET_AVAILABLE = True
            except Exception as e:
                _PROPHET_AVAILABLE = False
                raise ValueError(
                    "prophet could not be imported. Install it with `pip install prophet`\n"
                    "or follow package-specific install instructions for your platform.\n"
                    f"Original error: {e}")
        return _forecast_prophet(s, periods, **kwargs)
    else:
        raise ValueError(f"Unsupported forecast model: {model}. Choose from 'ets', 'arima', or 'prophet'.")


def _forecast_ets(series: pd.Series, periods: int, **kwargs: Any) -> pd.DataFrame:
    """Forecasts using Exponential Smoothing."""
    try:
        seasonal_periods = kwargs.get('seasonal_periods', 12 if kwargs.get('seasonal') else None)
        model = ExponentialSmoothing(
            series,
            seasonal=kwargs.get('seasonal', 'add'),
            trend=kwargs.get('trend', 'add'),
            seasonal_periods=seasonal_periods
        )
        fit = model.fit(optimized=True)
        pred = fit.forecast(periods)
        return pd.DataFrame({'forecast': pred})
    except Exception as e:
        logger.exception('Exponential Smoothing forecast failed.')
        raise


def _forecast_arima(series: pd.Series, periods: int, **kwargs: Any) -> pd.DataFrame:
    """Forecasts using auto-ARIMA."""
    try:
        model = auto_arima(series, **kwargs)
        pred, conf_int = model.predict(n_periods=periods, return_conf_int=True)
        return pd.DataFrame({
            'forecast': pred,
            'lower_bound': conf_int[:, 0],
            'upper_bound': conf_int[:, 1]
        }, index=pd.date_range(start=series.index[-1] + series.index.freq, periods=periods, freq=series.index.freq))
    except Exception as e:
        logger.exception('ARIMA forecast failed.')
        raise


def _forecast_prophet(series: pd.Series, periods: int, **kwargs: Any) -> pd.DataFrame:
    """Forecasts using Prophet."""
    try:
        df = series.reset_index()
        df.columns = ['ds', 'y']
        
        model = Prophet(**kwargs)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods, freq=series.index.freq)
        forecast = model.predict(future)
        
        return pd.DataFrame({
            'forecast': forecast['yhat'].values[-periods:],
            'lower_bound': forecast['yhat_lower'].values[-periods:],
            'upper_bound': forecast['yhat_upper'].values[-periods:]
        }, index=future['ds'].values[-periods:])
    except Exception as e:
        logger.exception('Prophet forecast failed.')
        raise