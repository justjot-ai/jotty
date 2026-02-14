# PyCaret AutoML Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`pycaret_classify_tool`](#pycaret_classify_tool) | AutoML classification using PyCaret. |
| [`pycaret_regress_tool`](#pycaret_regress_tool) | AutoML regression using PyCaret. |
| [`pycaret_tune_tool`](#pycaret_tune_tool) | Tune a model using PyCaret's hyperparameter optimization. |
| [`pycaret_ensemble_tool`](#pycaret_ensemble_tool) | Create ensemble models using PyCaret. |
| [`pycaret_predict_tool`](#pycaret_predict_tool) | Generate predictions using a trained PyCaret model. |

---

## `pycaret_classify_tool`

AutoML classification using PyCaret.

**Parameters:**

- **data**: DataFrame or path to CSV
- **target**: Target column name
- **features**: Optional list of feature columns
- **exclude_models**: Optional list of models to exclude
- **n_select**: Number of top models to return (default 5)

**Returns:** Dict with best models, scores, and comparison results

---

## `pycaret_regress_tool`

AutoML regression using PyCaret.

**Parameters:**

- **data**: DataFrame or path to CSV
- **target**: Target column name
- **features**: Optional list of feature columns
- **exclude_models**: Optional list of models to exclude
- **n_select**: Number of top models to return (default 5)

**Returns:** Dict with best models, scores, and comparison results

---

## `pycaret_tune_tool`

Tune a model using PyCaret's hyperparameter optimization.

**Parameters:**

- **model**: Model object or model name
- **data**: DataFrame
- **target**: Target column
- **task**: 'classification' or 'regression'
- **n_iter**: Number of iterations (default 50)
- **optimize**: Metric to optimize

**Returns:** Dict with tuned model and best parameters

---

## `pycaret_ensemble_tool`

Create ensemble models using PyCaret.

**Parameters:**

- **models**: List of model objects
- **data**: DataFrame
- **target**: Target column
- **task**: 'classification' or 'regression'
- **method**: 'Bagging', 'Boosting', 'Blending', or 'Stacking'

**Returns:** Dict with ensemble model and performance

---

## `pycaret_predict_tool`

Generate predictions using a trained PyCaret model.

**Parameters:**

- **model**: Trained model object
- **data**: DataFrame to predict on
- **task**: 'classification' or 'regression'

**Returns:** Dict with predictions
