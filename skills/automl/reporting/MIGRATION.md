# ML Report Generation - Migration to AutoML Skill

**Date:** February 16, 2026

## What Changed

Moved ML report generation code from `core/intelligence/orchestration/templates/` to `skills/automl/reporting/` for better architectural separation.

## Rationale

**Problem:** ML-specific reporting logic was in the core orchestration layer, violating clean architecture principles.

**Solution:** Moved ML report generation to the automl skill where it belongs.

- ✅ Core orchestration should be domain-agnostic
- ✅ ML-specific code belongs in ML-specific skills
- ✅ Better separation of concerns
- ✅ Easier to maintain and test

## Files Moved

From `core/intelligence/orchestration/templates/` → `skills/automl/reporting/`:

1. `ml_report_generator.py` (91KB) - Main report generator class
2. `_analysis_sections_mixin.py` (85KB) - Analysis sections (performance, features, etc.)
3. `_rendering_mixin.py` (36KB) - PDF/HTML rendering
4. `_report_mixin.py` (33KB) - Report structure
5. `_visualization_mixin.py` (91KB) - Charts and visualizations
6. `_drift_mixin.py` (32KB) - Model drift detection
7. `_fairness_mixin.py` (21KB) - Fairness analysis
8. `_interpretability_mixin.py` (50KB) - SHAP, LIME, interpretability
9. `_error_analysis_mixin.py` (21KB) - Error analysis
10. `_deployment_mixin.py` (15KB) - Deployment monitoring
11. `_mlflow_mixin.py` (11KB) - MLflow integration
12. `_telegram_mixin.py` (7KB) - Telegram notifications
13. `_world_class_report_mixin.py` (29KB) - World-class report formatting
14. `_protocols.py` (2KB) - Type protocols

**Total:** 14 files, ~500KB

## Files NOT Moved (Stayed in templates/)

- `base.py` - General swarm template base (used by all templates)
- `registry.py` - Template registry (general orchestration)
- `swarm_ml_comprehensive.py` - ML swarm template (orchestration, not reporting)
- `swarm_ml.py` - ML swarm template
- `swarm_lean.py` - Lean swarm template
- `swarm_science.py` - Science teaching template

## Import Changes

### New Import Path

```python
# ✅ NEW - Import from automl skill
from Jotty.skills.automl.reporting import ProfessionalMLReport

# ❌ OLD - Don't use anymore
from Jotty.core.intelligence.orchestration.templates.ml_report_generator import ProfessionalMLReport
```

### Backward Compatibility

No backward compatibility shim needed - no external code was importing from the old location.

## Next Steps

1. ✅ Move files to new location
2. ✅ Verify imports work
3. ⏳ Split large files (>1,000 lines of code):
   - `_analysis_sections_mixin.py` (1,109 code lines) → Split into 2-3 smaller mixins
   - `_visualization_mixin.py` (if >1,000 code lines)
4. ⏳ Update any documentation referencing old paths
5. ⏳ Add tests for reporting module

## Testing

Run AutoML skill tests:
```bash
pytest skills/automl/tests/ -v
```

Verify import works:
```python
from Jotty.skills.automl.reporting import ProfessionalMLReport
report = ProfessionalMLReport(output_dir="test_output")
```
