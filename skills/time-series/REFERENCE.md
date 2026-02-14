# Time Series Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`timeseries_decompose_tool`](#timeseries_decompose_tool) | Decompose a time series into trend, seasonal, and residual components. |
| [`timeseries_forecast_tool`](#timeseries_forecast_tool) | Forecast time series using ARIMA/SARIMA. |
| [`timeseries_features_tool`](#timeseries_features_tool) | Extract features from time series data. |
| [`timeseries_anomaly_tool`](#timeseries_anomaly_tool) | Detect anomalies in time series data. |
| [`timeseries_crossval_tool`](#timeseries_crossval_tool) | Time series cross-validation for model evaluation. |

---

## `timeseries_decompose_tool`

Decompose a time series into trend, seasonal, and residual components.

**Parameters:**

- **data**: DataFrame or Series with time series data
- **column**: Column name if DataFrame
- **period**: Seasonal period (default auto-detect)
- **model**: 'additive' or 'multiplicative' (default 'additive')

**Returns:** Dict with trend, seasonal, and residual components

---

## `timeseries_forecast_tool`

Forecast time series using ARIMA/SARIMA.

**Parameters:**

- **data**: DataFrame or Series with time series data
- **column**: Column name if DataFrame
- **horizon**: Number of periods to forecast
- **order**: ARIMA order (p, d, q) - auto if None
- **seasonal_order**: Seasonal order (P, D, Q, s) - optional

**Returns:** Dict with forecasts and confidence intervals

---

## `timeseries_features_tool`

Extract features from time series data.

**Parameters:**

- **data**: DataFrame or Series
- **column**: Column name if DataFrame
- **features**: List of features to extract (default: all)

**Returns:** Dict with extracted features

---

## `timeseries_anomaly_tool`

Detect anomalies in time series data.

**Parameters:**

- **data**: DataFrame or Series
- **column**: Column name if DataFrame
- **method**: 'zscore', 'iqr', 'isolation_forest', or 'mad' (default 'zscore')
- **threshold**: Threshold for detection (default depends on method)

**Returns:** Dict with anomaly indices and values

---

## `timeseries_crossval_tool`

Time series cross-validation for model evaluation.

**Parameters:**

- **data**: DataFrame or Series
- **column**: Column name if DataFrame
- **model**: Model to evaluate
- **n_splits**: Number of CV splits (default 5)
- **gap**: Gap between train and test (default 0)

**Returns:** Dict with cross-validation scores
