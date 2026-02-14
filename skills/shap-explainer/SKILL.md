---
name: explaining-shap
description: "**Description:** Model explainability and interpretation using SHAP values. Use when the user wants to explain model, feature importance, shap values."
---

# SHAP Explainer Skill

**Description:** Model explainability and interpretation using SHAP values.


## Type
derived

## Base Skills
- data-profiler


## Capabilities
- analyze

## Tools
- `shap_explain_tool`: Generate SHAP explanations for a model
- `shap_importance_tool`: Get feature importance from SHAP
- `shap_local_explain_tool`: Explain individual predictions

## Dependencies
- shap
- scikit-learn
- matplotlib

## Tags
explainability, shap, interpretation, feature-importance, xai

## Triggers
- "shap explainer"
- "explain model"
- "feature importance"
- "shap values"

## Category
data-science
