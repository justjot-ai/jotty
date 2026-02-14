# Ensemble Builder Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`ensemble_stack_tool`](#ensemble_stack_tool) | Create a stacking ensemble with meta-learner. |
| [`ensemble_blend_tool`](#ensemble_blend_tool) | Create a blending ensemble with holdout validation. |
| [`ensemble_vote_tool`](#ensemble_vote_tool) | Create a voting ensemble. |
| [`ensemble_weighted_tool`](#ensemble_weighted_tool) | Create a weighted average ensemble with automatic weight optimization. |
| [`ensemble_diversity_tool`](#ensemble_diversity_tool) | Analyze diversity of ensemble models. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`objective`](#objective) | No description available. |

---

## `ensemble_stack_tool`

Create a stacking ensemble with meta-learner.

**Parameters:**

- **models**: List of (name, model) tuples or dict
- **data**: DataFrame with features
- **target**: Target column name
- **meta_learner**: Meta-learner model (default: LogisticRegression)
- **task**: 'classification' or 'regression'
- **cv_folds**: Number of CV folds for OOF predictions (default 5)

**Returns:** Dict with stacking ensemble and performance

---

## `ensemble_blend_tool`

Create a blending ensemble with holdout validation.

**Parameters:**

- **models**: List of (name, model) tuples or dict
- **data**: DataFrame with features
- **target**: Target column name
- **meta_learner**: Meta-learner model
- **task**: 'classification' or 'regression'
- **holdout_ratio**: Ratio for holdout set (default 0.2)

**Returns:** Dict with blending ensemble and performance

---

## `ensemble_vote_tool`

Create a voting ensemble.

**Parameters:**

- **models**: List of (name, model) tuples or dict
- **data**: DataFrame with features
- **target**: Target column name
- **voting**: 'hard' or 'soft' (default 'soft')
- **weights**: Optional weights for each model

**Returns:** Dict with voting ensemble and performance

---

## `ensemble_weighted_tool`

Create a weighted average ensemble with automatic weight optimization.

**Parameters:**

- **predictions**: Dict of {model_name: predictions}
- **y_true**: True labels
- **task**: 'classification' or 'regression'
- **optimize_weights**: Whether to optimize weights (default True)

**Returns:** Dict with optimal weights and ensemble predictions

---

## `ensemble_diversity_tool`

Analyze diversity of ensemble models.

**Parameters:**

- **predictions**: Dict of {model_name: predictions}
- **y_true**: True labels

**Returns:** Dict with diversity metrics

---

## `objective`

No description available.

**Parameters:**

- **weights**
