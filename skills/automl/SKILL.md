# AutoML Skill

**Description:** Automatic machine learning - model selection, training, and optimization.


## Type
derived

## Base Skills
- claude-cli-llm


## Capabilities
- analyze

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
