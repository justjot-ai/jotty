"""
ML Report Generation Module

Comprehensive ML model reporting with:
- Performance analysis
- Feature importance (SHAP, permutation)
- Model diagnostics (calibration, drift, fairness)
- Visualization and PDF generation

Moved from core/intelligence/orchestration/templates/ to skills/automl/reporting/
for better architectural separation (Feb 2026).
"""

from .ml_report_generator import ProfessionalMLReport

__all__ = ["ProfessionalMLReport"]
