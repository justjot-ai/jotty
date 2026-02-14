# Model Metrics Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`metrics_classify_tool`](#metrics_classify_tool) | Calculate comprehensive classification metrics. |
| [`metrics_regress_tool`](#metrics_regress_tool) | Calculate comprehensive regression metrics. |
| [`metrics_crossval_tool`](#metrics_crossval_tool) | Perform cross-validation and return metrics. |
| [`metrics_calibration_tool`](#metrics_calibration_tool) | Analyze and improve probability calibration. |
| [`metrics_threshold_tool`](#metrics_threshold_tool) | Find optimal classification threshold. |

---

## `metrics_classify_tool`

Calculate comprehensive classification metrics.

**Parameters:**

- **y_true**: True labels
- **y_pred**: Predicted labels
- **y_proba**: Optional predicted probabilities
- **average**: 'binary', 'macro', 'micro', 'weighted' (default 'weighted')
- **labels**: Optional list of label names

**Returns:** Dict with all classification metrics

---

## `metrics_regress_tool`

Calculate comprehensive regression metrics.

**Parameters:**

- **y_true**: True values
- **y_pred**: Predicted values

**Returns:** Dict with all regression metrics

---

## `metrics_crossval_tool`

Perform cross-validation and return metrics.

**Parameters:**

- **model**: Model to evaluate
- **data**: DataFrame with features
- **target**: Target column name
- **cv**: Number of folds (default 5)
- **scoring**: Scoring metric (default depends on task)
- **task**: 'classification' or 'regression'

**Returns:** Dict with cross-validation scores

---

## `metrics_calibration_tool`

Analyze and improve probability calibration.

**Parameters:**

- **y_true**: True labels
- **y_proba**: Predicted probabilities
- **n_bins**: Number of bins for calibration curve (default 10)
- **method**: Calibration method if recalibrating ('isotonic', 'sigmoid')

**Returns:** Dict with calibration metrics and curve

---

## `metrics_threshold_tool`

Find optimal classification threshold.

**Parameters:**

- **y_true**: True labels
- **y_proba**: Predicted probabilities
- **metric**: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
- **min_precision**: Minimum precision constraint
- **min_recall**: Minimum recall constraint

**Returns:** Dict with optimal threshold and metrics at that threshold
