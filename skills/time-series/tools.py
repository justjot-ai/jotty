"""
Time Series Skill for Jotty
===========================

Time series analysis and forecasting tools.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("time-series")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def timeseries_decompose_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompose a time series into trend, seasonal, and residual components.

    Args:
        params: Dict with keys:
            - data: DataFrame or Series with time series data
            - column: Column name if DataFrame
            - period: Seasonal period (default auto-detect)
            - model: 'additive' or 'multiplicative' (default 'additive')

    Returns:
        Dict with trend, seasonal, and residual components
    """
    status.set_callback(params.pop('_status_callback', None))

    from statsmodels.tsa.seasonal import seasonal_decompose

    logger.info("[TimeSeries] Decomposing time series...")

    data = params.get('data')
    column = params.get('column')
    period = params.get('period')
    model = params.get('model', 'additive')

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            series = data[column]
        else:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            series = data[numeric_cols[0]]
    else:
        series = pd.Series(data)

    # Auto-detect period if not specified
    if period is None:
        period = min(len(series) // 2, 12)  # Default to 12 or half the length

    # Decompose
    result = seasonal_decompose(series, model=model, period=period)

    logger.info(f"[TimeSeries] Decomposed with period={period}, model={model}")

    return {
        'success': True,
        'trend': result.trend.dropna().tolist(),
        'seasonal': result.seasonal.dropna().tolist(),
        'residual': result.resid.dropna().tolist(),
        'period': period,
        'model': model,
    }


@async_tool_wrapper()
async def timeseries_forecast_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast time series using ARIMA/SARIMA.

    Args:
        params: Dict with keys:
            - data: DataFrame or Series with time series data
            - column: Column name if DataFrame
            - horizon: Number of periods to forecast
            - order: ARIMA order (p, d, q) - auto if None
            - seasonal_order: Seasonal order (P, D, Q, s) - optional

    Returns:
        Dict with forecasts and confidence intervals
    """
    status.set_callback(params.pop('_status_callback', None))

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    logger.info("[TimeSeries] Forecasting time series...")

    data = params.get('data')
    column = params.get('column')
    horizon = params.get('horizon', 10)
    order = params.get('order')
    seasonal_order = params.get('seasonal_order')

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            series = data[column]
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            series = data[numeric_cols[0]]
    else:
        series = pd.Series(data)

    series = series.dropna()

    # Auto-detect order if not provided
    if order is None:
        # Simple auto-detection based on stationarity
        adf_result = adfuller(series)
        d = 0 if adf_result[1] < 0.05 else 1
        order = (1, d, 1)

    # Fit ARIMA
    try:
        if seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(series, order=order)

        fitted = model.fit()

        # Forecast
        forecast = fitted.get_forecast(steps=horizon)
        predictions = forecast.predicted_mean
        conf_int = forecast.conf_int()

        logger.info(f"[TimeSeries] Forecast generated for {horizon} periods")

        return {
            'success': True,
            'forecast': predictions.tolist(),
            'lower_ci': conf_int.iloc[:, 0].tolist(),
            'upper_ci': conf_int.iloc[:, 1].tolist(),
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': fitted.aic,
            'bic': fitted.bic,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@async_tool_wrapper()
async def timeseries_features_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from time series data.

    Args:
        params: Dict with keys:
            - data: DataFrame or Series
            - column: Column name if DataFrame
            - features: List of features to extract (default: all)

    Returns:
        Dict with extracted features
    """
    status.set_callback(params.pop('_status_callback', None))

    from scipy import stats

    logger.info("[TimeSeries] Extracting time series features...")

    data = params.get('data')
    column = params.get('column')
    features_to_extract = params.get('features', ['all'])

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            series = data[column].dropna()
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            series = data[numeric_cols[0]].dropna()
    else:
        series = pd.Series(data).dropna()

    features = {}

    # Basic statistics
    features['mean'] = float(series.mean())
    features['std'] = float(series.std())
    features['min'] = float(series.min())
    features['max'] = float(series.max())
    features['median'] = float(series.median())
    features['skewness'] = float(series.skew())
    features['kurtosis'] = float(series.kurtosis())

    # Range and variation
    features['range'] = float(series.max() - series.min())
    features['iqr'] = float(series.quantile(0.75) - series.quantile(0.25))
    features['cv'] = float(series.std() / series.mean()) if series.mean() != 0 else 0

    # Trend features
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
    features['trend_slope'] = float(slope)
    features['trend_strength'] = float(r_value ** 2)

    # Autocorrelation
    if len(series) > 1:
        features['autocorr_lag1'] = float(series.autocorr(lag=1)) if len(series) > 1 else 0
        features['autocorr_lag5'] = float(series.autocorr(lag=5)) if len(series) > 5 else 0

    # Change features
    diff = series.diff().dropna()
    features['mean_abs_change'] = float(diff.abs().mean())
    features['mean_change'] = float(diff.mean())

    # Peaks and troughs
    features['num_peaks'] = int(((series.shift(1) < series) & (series.shift(-1) < series)).sum())
    features['num_troughs'] = int(((series.shift(1) > series) & (series.shift(-1) > series)).sum())

    logger.info(f"[TimeSeries] Extracted {len(features)} features")

    return {
        'success': True,
        'features': features,
        'series_length': len(series),
    }


@async_tool_wrapper()
async def timeseries_anomaly_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect anomalies in time series data.

    Args:
        params: Dict with keys:
            - data: DataFrame or Series
            - column: Column name if DataFrame
            - method: 'zscore', 'iqr', 'isolation_forest', or 'mad' (default 'zscore')
            - threshold: Threshold for detection (default depends on method)

    Returns:
        Dict with anomaly indices and values
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[TimeSeries] Detecting anomalies...")

    data = params.get('data')
    column = params.get('column')
    method = params.get('method', 'zscore')
    threshold = params.get('threshold')

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            series = data[column]
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            series = data[numeric_cols[0]]
    else:
        series = pd.Series(data)

    anomalies = []
    anomaly_indices = []

    if method == 'zscore':
        threshold = threshold or 3
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores > threshold

    elif method == 'iqr':
        threshold = threshold or 1.5
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        mask = (series < Q1 - threshold * IQR) | (series > Q3 + threshold * IQR)

    elif method == 'mad':  # Median Absolute Deviation
        threshold = threshold or 3
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / mad
        mask = np.abs(modified_z) > threshold

    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        contamination = threshold or 0.1
        clf = IsolationForest(contamination=contamination, random_state=42)
        values = series.values.reshape(-1, 1)
        preds = clf.fit_predict(values)
        mask = preds == -1

    else:
        return {'success': False, 'error': f'Unknown method: {method}'}

    anomaly_indices = series[mask].index.tolist()
    anomalies = series[mask].tolist()

    logger.info(f"[TimeSeries] Found {len(anomalies)} anomalies using {method}")

    return {
        'success': True,
        'method': method,
        'anomaly_indices': anomaly_indices,
        'anomaly_values': anomalies,
        'num_anomalies': len(anomalies),
        'anomaly_percent': round(len(anomalies) / len(series) * 100, 2),
    }


@async_tool_wrapper()
async def timeseries_crossval_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Time series cross-validation for model evaluation.

    Args:
        params: Dict with keys:
            - data: DataFrame or Series
            - column: Column name if DataFrame
            - model: Model to evaluate
            - n_splits: Number of CV splits (default 5)
            - gap: Gap between train and test (default 0)

    Returns:
        Dict with cross-validation scores
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    logger.info("[TimeSeries] Performing time series cross-validation...")

    data = params.get('data')
    column = params.get('column')
    model = params.get('model')
    n_splits = params.get('n_splits', 5)
    gap = params.get('gap', 0)

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            series = data[column].dropna()
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            series = data[numeric_cols[0]].dropna()
    else:
        series = pd.Series(data).dropna()

    # Create lagged features
    X = pd.DataFrame()
    for lag in range(1, 6):
        X[f'lag_{lag}'] = series.shift(lag)
    X = X.dropna()
    y = series.iloc[5:]  # Align with X

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    mse_scores = []
    mae_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model is not None:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            # Simple baseline: predict mean of training
            preds = np.full(len(y_test), y_train.mean())

        mse_scores.append(mean_squared_error(y_test, preds))
        mae_scores.append(mean_absolute_error(y_test, preds))

    logger.info(f"[TimeSeries] CV completed with {n_splits} splits")

    return {
        'success': True,
        'mse_mean': float(np.mean(mse_scores)),
        'mse_std': float(np.std(mse_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'mae_std': float(np.std(mae_scores)),
        'rmse_mean': float(np.sqrt(np.mean(mse_scores))),
        'n_splits': n_splits,
        'fold_scores': {'mse': mse_scores, 'mae': mae_scores},
    }
