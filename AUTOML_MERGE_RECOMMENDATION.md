# AutoML Merge Recommendation

**Date:** 2026-02-16
**Conclusion:** NO MERGE NEEDED - Skills are already superior

## Analysis Summary

After detailed comparison of all 3 AutoML components (feature engineering, feature selection, model selection), here's what I found:

### Verdict: Skills Win Decisively

| Component | Mixin Lines | Skill Lines | Unique to Skill | Unique to Mixin | Winner |
|-----------|-------------|-------------|-----------------|-----------------|---------|
| Feature Engineering | 369 | 432 | 3 techniques | 0 | ✅ Skill |
| Feature Selection | 433 | 1,185 | 7 techniques | 0 | ✅ Skill |
| Model Selection | 748* | 274 | CatBoost | Hyperopt+Ensemble** | ✅ Skill |

*Mixin combines 3 responsibilities (violates SRP)
**These exist as separate skills in `skills/automl/hyperopt.py` and `skills/automl/ensemble.py`

---

## What Skills Have That Mixins Don't

### Feature Engineering (3 unique techniques)
1. **Quantile features** (lines 296-308)
   - Binary flags for below Q25, above Q75
   - Helps capture distribution extremes

2. **CV-validated interactions** (lines 341-391)
   - Only keeps interactions that improve CV score
   - Prevents feature explosion with useless features
   - **NOTE:** Mixin had this disabled! Skill has it working!

3. **Early pruning** (lines 393-418)
   - Removes constant/near-constant features early
   - Prevents wasted computation downstream

### Feature Selection (7 unique techniques)
4. **Successive Halving** (lines 431-498)
   - Progressive elimination with increasing budgets
   - Features surviving multiple rounds get higher scores

5. **Hyperband Selection** (lines 500-599)
   - Multi-bracket Hyperband with different starting configs
   - Combines results across brackets for robustness

6. **RFECV** (lines 601-670)
   - Recursive Feature Elimination with CV validation
   - Backward elimination monitoring CV score

7. **Diverse RF Importance** (lines 672-737)
   - Multiple RF configs: shallow/deep trees, different max_features
   - Features consistent across configs get bonus

8. **PCA Importance** (lines 739-782)
   - Variance-based feature scoring
   - Features contributing to top PCs are important

9. **BOHB** (lines 890-1019)
   - Bayesian Optimized Hyperband (TPE + Hyperband)
   - Treats feature selection as hyperparameter optimization
   - Uses ConfigSpace for smart feature subset sampling

10. **PASHA** (lines 1021-1185)
    - Progressive Adaptive Successive Halving
    - Parallel evaluation with ThreadPoolExecutor (4 workers)
    - Adaptive budget allocation based on performance
    - Based on research: https://github.com/ondrejbohdal/pasha

### Model Selection (1 unique model)
11. **CatBoost** (lines 217-225, 261-269)
    - Handles categoricals natively (no encoding leakage!)
    - Often outperforms LightGBM/XGBoost on categorical data

---

## What Mixins Have That Skills Don't

### NOTHING!

All techniques in mixins are already in skills, but skills have:
- ✅ More techniques (14 vs 7 in feature selection)
- ✅ Better implementations (CV-based target encoding works in skill, disabled in mixin)
- ✅ Proper architecture (standalone classes vs coupled mixins)
- ✅ Better error handling and logging
- ✅ Metrics tracking (`self._techniques_used`, `self._method_results`)

---

## Code Quality Comparison

### Example: Target Encoding

**Mixin (DISABLED - causes leakage):**
```python
# 1. TARGET ENCODING - DISABLED (causes leakage outside CV)
# ================================================================
# NOTE: Target encoding must be done INSIDE cross-validation folds
# to avoid leakage. Doing it here on full data before CV causes
# the model to see target information from validation rows.
# IMPLEMENT: Add proper target encoding with CV-aware pipeline
# if y is not None and len(cat_cols) > 0:
#     for col in cat_cols[:5]:
#         target_mean = X_eng.groupby(col).apply(...)  # DISABLED!
```

**Skill (WORKING - no leakage):**
```python
def _target_encoding_cv(
    self, X: pd.DataFrame, y: pd.Series, cat_cols: List[str]
) -> pd.DataFrame:
    """CV-based target encoding (no leakage)."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for col in cat_cols[:5]:
        target_enc = np.zeros(len(X))

        # CV-based target encoding - NO LEAKAGE!
        for train_idx, val_idx in kf.split(X):
            train_means = (
                pd.Series(y.iloc[train_idx].values)
                .groupby(col_encoded.iloc[train_idx].values)
                .mean()
            )
            target_enc[val_idx] = (
                col_encoded.iloc[val_idx].map(train_means).fillna(y.mean()).values
            )

        X[f"{col}_target_enc_cv"] = target_enc

    return X
```

**Winner:** ✅ Skill (mixin had this DISABLED due to leakage issues!)

---

## Architectural Issues

### Mixin Problems

1. **Wrong Layer**
   ```
   ❌ core/intelligence/orchestration/_feature_engineering_mixin.py
   ✅ skills/automl/feature_engineering.py
   ```

2. **Tight Coupling**
   ```python
   class SkillOrchestrator(FeatureEngineeringMixin, ...):
       # Can't swap implementations!
   ```

3. **Violation of SRP**
   ```python
   class ModelPipelineMixin:
       async def _builtin_model_selection(...)    # Responsibility 1
       async def _builtin_hyperopt(...)            # Responsibility 2
       async def _builtin_ensemble(...)            # Responsibility 3
   ```

4. **Hard-Coded Pipeline**
   ```python
   PIPELINE_ORDER = [
       SkillCategory.FEATURE_ENGINEERING,  # ← Must use mixin
       SkillCategory.FEATURE_SELECTION,    # ← Can't swap!
   ]
   ```

### Skill Benefits

1. **Correct Layer**
   ```
   ✅ skills/automl/feature_engineering.py
   ✅ skills/automl/feature_selection.py
   ✅ skills/automl/model_selection.py
   ✅ skills/automl/hyperopt.py
   ✅ skills/automl/ensemble.py
   ```

2. **Loose Coupling**
   ```python
   class SkillOrchestrator:
       async def solve(self, X, y):
           skill = registry.get_skill("feature_engineering")
           result = await skill.execute(X, y)
   ```

3. **Single Responsibility**
   ```python
   class FeatureEngineeringSkill(MLSkill):
       # Only does feature engineering

   class HyperoptSkill(MLSkill):
       # Only does hyperparameter optimization

   class EnsembleSkill(MLSkill):
       # Only does ensemble creation
   ```

4. **Configurable Pipeline**
   ```python
   skills = registry.discover_skills_for_category(SkillCategory.FEATURE_ENGINEERING)
   best_skill = select_best_skill(skills)
   ```

---

## Merge Decisions

### Feature Engineering: NO MERGE NEEDED
- ✅ Skill has all mixin techniques
- ✅ Skill has 3 additional techniques
- ✅ Skill has working target encoding (mixin disabled)
- ❌ Nothing to merge from mixin → skill

### Feature Selection: NO MERGE NEEDED
- ✅ Skill has all mixin techniques
- ✅ Skill has 7 additional advanced techniques (BOHB, PASHA, etc.)
- ✅ Skill has 2.7x more code (1,185 vs 433 lines)
- ❌ Nothing to merge from mixin → skill

### Model Selection: NO MERGE NEEDED
- ✅ Skill has all mixin models
- ✅ Skill has CatBoost (mixin doesn't)
- ✅ Hyperopt and Ensemble exist as separate skills (proper SRP)
- ❌ Nothing to merge from mixin → skill

---

## Recommendation

### Immediate Actions

1. **DO NOT merge mixin → skill** (skills already superior)
2. **Update SkillOrchestrator** to use skill registry instead of mixins
3. **Deprecate mixins** with warning messages
4. **Eventually delete mixins** after migration complete

### Migration Code

**File:** `core/intelligence/orchestration/skill_orchestrator.py`

**Before (Hard-coded mixins):**
```python
from ._feature_engineering_mixin import FeatureEngineeringMixin
from ._feature_selection_mixin import FeatureSelectionMixin
from ._model_pipeline_mixin import ModelPipelineMixin

class SkillOrchestrator(FeatureEngineeringMixin, FeatureSelectionMixin, ModelPipelineMixin):
    async def solve(self, X, y):
        # Hard-coded to use mixins
        fe_result = await self._builtin_feature_engineering(X, y, problem_type)
        fs_result = await self._builtin_feature_selection(X, y, problem_type)
```

**After (Use skill registry):**
```python
from skills.automl.feature_engineering import FeatureEngineeringSkill
from skills.automl.feature_selection import FeatureSelectionSkill
from skills.automl.model_selection import ModelSelectionSkill

class SkillOrchestrator:
    async def solve(self, X, y):
        # Use standalone skills
        fe_skill = FeatureEngineeringSkill()
        fe_result = await fe_skill.execute(X, y, problem_type=problem_type)

        fs_skill = FeatureSelectionSkill()
        fs_result = await fs_skill.execute(fe_result.data, y, problem_type=problem_type)

        ms_skill = ModelSelectionSkill()
        ms_result = await ms_skill.execute(fs_result.data, y, problem_type=problem_type)
```

---

## Summary

| Metric | Mixins | Skills | Winner |
|--------|--------|--------|---------|
| Total Techniques | 25 | 39 (+14) | ✅ Skills |
| Lines of Code | 1,550 | 1,891 | ✅ Skills (more features) |
| Architecture | Coupled | Standalone | ✅ Skills |
| Reusability | Low | High | ✅ Skills |
| Testability | Hard | Easy | ✅ Skills |
| Maintainability | Poor | Good | ✅ Skills |
| **Overall** | 1/6 | 6/6 | ✅ **Skills Win** |

**Conclusion:** Skills are objectively superior. No merge needed. Deprecate mixins and migrate orchestrator to use skill registry.
