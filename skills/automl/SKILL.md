# AutoML Skill

**Description:** Automatic machine learning - model selection, training, and optimization.

## Capabilities
- Automatic model selection from 10+ algorithms
- Cross-validation with stratified folds
- Ensemble creation (voting, stacking, blending)
- Feature importance analysis
- Model comparison and ranking

## Tools
- `automl_classify`: Auto-select and train best classifier
- `automl_ensemble`: Create ensemble from top models
- `automl_evaluate`: Evaluate model performance

## Usage
```python
result = await automl_classify({
    'data_path': 'train.csv',
    'target': 'Survived',
    'task': 'classification'
})
```

## Tags
machine-learning, automl, classification, regression, ensemble
