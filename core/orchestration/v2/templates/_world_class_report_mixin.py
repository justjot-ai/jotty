"""World-Class Report Mixin - Professional and world-class report generation."""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None  # type: ignore
    np = None  # type: ignore

logger = logging.getLogger(__name__)


class WorldClassReportMixin:
    def generate_professional_report(
        self,
        results: Dict[str, Any],
        y_true=None,
        y_pred=None,
        y_prob=None,
        X_sample=None,
        shap_values=None,
        title: str = "ML Analysis Report",
        context: str = "",
        filename: str = None
    ) -> Optional[str]:
        """
        Generate a professional-grade PDF report using Pandoc + LaTeX.

        This creates publication-quality reports with:
        - Elegant typography
        - Professional visualizations
        - Proper tables and charts
        - Table of contents

        Args:
            results: Dict with keys like 'metrics', 'model_scores', 'feature_importance', etc.
            y_true: True labels for classification metrics
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            X_sample: Sample data for SHAP analysis
            shap_values: Pre-computed SHAP values
            title: Report title
            context: Business context description
            filename: Output filename

        Returns:
            Path to generated PDF or None if failed
        """
        try:
            from .ml_report_generator import ProfessionalMLReport

            # Use configured output dir or default
            output_dir = getattr(self, '_report_config', ReportConfig()).output_dir

            report = ProfessionalMLReport(output_dir=output_dir)

            # Set metadata
            report.set_metadata(
                title=title,
                subtitle=context[:100] if context else "Automated ML Analysis",
                author="Jotty SwarmMLComprehensive",
                dataset=results.get('dataset', 'Unknown'),
                problem_type=results.get('problem_type', 'Classification')
            )

            # Extract components from results
            metrics = results.get('metrics', {})
            if not metrics and 'final_score' in results:
                metrics = {'accuracy': results['final_score']}

            best_model = results.get('best_model', 'Unknown')
            feature_importance = results.get('feature_importance', {})
            model_scores = results.get('model_scores', {})
            n_features = results.get('n_features', len(feature_importance))

            # Add sections
            report.add_executive_summary(
                metrics=metrics,
                best_model=best_model,
                n_features=n_features,
                context=context
            )

            # Data profile if available
            if 'data_profile' in results:
                dp = results['data_profile']
                report.add_data_profile(
                    shape=dp.get('shape', (0, n_features)),
                    dtypes=dp.get('dtypes', {}),
                    missing=dp.get('missing', {}),
                    recommendations=dp.get('recommendations', [])
                )

            # Feature importance
            if feature_importance:
                report.add_feature_importance(feature_importance)

            # Model benchmarking
            if model_scores:
                report.add_model_benchmarking(model_scores)

            # Classification metrics
            if y_true is not None and y_pred is not None:
                n_classes = len(np.unique(y_true))
                default_labels = [f'Class {i}' for i in range(n_classes)]
                labels = results.get('labels', default_labels)
                report.add_confusion_matrix(y_true, y_pred, labels)

                if y_prob is not None:
                    report.add_roc_analysis(y_true, y_prob)
                    report.add_precision_recall(y_true, y_prob)

            # SHAP analysis
            if shap_values is not None and X_sample is not None:
                feature_names = list(feature_importance.keys()) if feature_importance else []
                report.add_shap_analysis(shap_values, feature_names, X_sample)

            # Baseline comparison
            if 'baseline_score' in results:
                report.add_baseline_comparison(
                    results['baseline_score'],
                    metrics.get('accuracy', results.get('final_score', 0))
                )

            # Recommendations
            recommendations = results.get('recommendations', [
                f"Best model {best_model} achieved strong performance",
                "Consider hyperparameter tuning for further improvement",
                "Monitor model performance over time for drift",
                "Regular retraining recommended as data patterns evolve"
            ])
            report.add_recommendations(recommendations)

            # Generate PDF
            pdf_path = report.generate(filename)

            if pdf_path:
                logger.info(f"Professional report generated: {pdf_path}")

            return pdf_path

        except ImportError as e:
            logger.warning(f"Professional report generator not available: {e}")
            # Fallback to basic report
            return self.generate_report(filename)
        except Exception as e:
            logger.error(f"Failed to generate professional report: {e}")
            return None

    def generate_world_class_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        results: Dict[str, Any],
        y_pred=None,
        y_prob=None,
        shap_values=None,
        title: str = "Comprehensive ML Analysis",
        context: str = "",
        filename: str = None,
        include_all: bool = True,
        theme: str = "professional",
        # New parameters
        generate_html: bool = False,
        llm_narrative: bool = False,
        sensitive_features: Dict[str, Any] = None,
        X_reference=None,
        pipeline_steps: List[Dict] = None,
        study_or_trials=None,
        validation_datasets: Dict = None,
        trained_models: Dict = None,
    ) -> Optional[str]:
        """
        Generate the world's most comprehensive ML report.

        Themes:
        - 'professional': Modern blue, sans-serif, bold headers (default)
        - 'goldman': Goldman Sachs style, navy/serif, uppercase headers, institutional

        Includes ALL advanced analysis:
        - Data Quality Analysis (outliers, missing patterns, distributions)
        - Correlation & Multicollinearity (VIF analysis)
        - Learning Curves & Bias-Variance Analysis
        - Calibration Analysis
        - Lift & Gain Charts (KS statistic)
        - Cross-Validation Detailed Analysis
        - Error Analysis (misclassification patterns)
        - SHAP Deep Dive (summary, dependence, waterfall)
        - Threshold Optimization (cost-benefit)
        - Full Reproducibility Section
        - Data Drift Monitoring (if X_reference provided)
        - Fairness & Bias Audit (if sensitive_features provided)
        - Hyperparameter Search Visualization (if study_or_trials provided)
        - Multi-Dataset Validation (if validation_datasets provided)
        - Model Comparison with overlaid ROC & radar (if trained_models provided)
        - Confidence-Calibrated Predictions
        - Pipeline DAG Visualization (if pipeline_steps provided)
        - Deep Learning Analysis (if model is neural network)
        - LLM-Generated Narrative Insights (if llm_narrative=True)
        - Interactive HTML Report (if generate_html=True)

        Args:
            X: Feature DataFrame
            y: Target Series
            model: Trained model
            results: Results dictionary
            y_pred: Predictions
            y_prob: Probabilities
            shap_values: SHAP values
            title: Report title
            context: Business context
            filename: Output filename
            include_all: Include all sections (True) or only essential (False)
            theme: Report theme ('professional' or 'goldman')
            generate_html: Also generate interactive HTML report
            llm_narrative: Enable LLM-generated narrative insights
            sensitive_features: Dict mapping feature_name -> column data for fairness audit
            X_reference: Reference dataset for drift analysis
            pipeline_steps: List of pipeline step dicts for DAG visualization
            study_or_trials: Optuna study or List[Dict] for hyperparameter viz
            validation_datasets: Dict[name -> (X, y)] for cross-dataset validation
            trained_models: Dict[name -> trained model] for side-by-side model comparison

        Returns:
            Path to generated PDF
        """
        try:
            from .ml_report_generator import ProfessionalMLReport

            # Use configured output dir
            output_dir = getattr(self, '_report_config', ReportConfig()).output_dir
            report = ProfessionalMLReport(output_dir=output_dir, theme=theme,
                                          llm_narrative=llm_narrative, html_enabled=generate_html)

            logger.info(f"  Theme: {report.theme['name']}")

            # Set metadata
            report.set_metadata(
                title=title,
                subtitle=context[:100] if context else "World-Class ML Analysis",
                author="Jotty SwarmMLComprehensive",
                dataset=results.get('dataset', 'Custom Dataset'),
                problem_type=results.get('problem_type', 'Classification')
            )

            # Extract components
            metrics = results.get('metrics', {})
            if not metrics and 'final_score' in results:
                metrics = {'accuracy': results['final_score']}

            best_model = str(results.get('best_model', type(model).__name__))
            feature_importance = results.get('feature_importance', {})
            model_scores = results.get('model_scores', {})
            feature_names = list(X.columns)

            problem_type = results.get('problem_type', 'Classification')
            is_regression = problem_type.lower() == 'regression'

            logger.info("Generating world-class comprehensive report...")

            # ==== SECTION 1: EXECUTIVE SUMMARY ====
            report.add_executive_summary(
                metrics=metrics,
                best_model=best_model,
                n_features=len(feature_names),
                context=context
            )

            # ==== SECTION 1.5: EXECUTIVE DASHBOARD (NEW) ====
            if include_all and metrics:
                logger.info("  - Adding executive dashboard...")
                self._guarded_section('executive_dashboard', report, 'add_executive_dashboard',
                    metrics=metrics, model_name=best_model,
                    dataset_name=results.get('dataset', ''))

            # ==== SECTION 2: DATA QUALITY ANALYSIS ====
            if include_all:
                logger.info("  - Adding data quality analysis...")
                report.add_data_quality_analysis(X, y)

            # ==== SECTION 2.5: CLASS DISTRIBUTION (NEW) ====
            if include_all and not is_regression:
                logger.info("  - Adding class distribution analysis...")
                self._guarded_section('class_distribution', report, 'add_class_distribution',
                    y, y_pred, results.get('labels', None))

            # ==== SECTION 3: CORRELATION ANALYSIS ====
            if include_all and len(X.select_dtypes(include=[np.number]).columns) >= 2:
                logger.info("  - Adding correlation analysis...")
                report.add_correlation_analysis(X)

            # ==== SECTION 4: DATA PROFILE ====
            if 'data_profile' in results:
                dp = results['data_profile']
                report.add_data_profile(
                    shape=dp.get('shape', X.shape),
                    dtypes=dp.get('dtypes', dict(X.dtypes.value_counts())),
                    missing=dp.get('missing', dict(X.isnull().sum())),
                    recommendations=dp.get('recommendations', [])
                )

            # ==== SECTION 4.5: PIPELINE DAG VISUALIZATION (NEW) ====
            if include_all and pipeline_steps:
                logger.info("  - Adding pipeline visualization...")
                self._guarded_section('pipeline_visualization', report, 'add_pipeline_visualization',
                    pipeline_steps)

            # ==== SECTION 5: FEATURE IMPORTANCE ====
            if feature_importance:
                report.add_feature_importance(feature_importance)

            # ==== SECTION 5.5: PERMUTATION IMPORTANCE (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding permutation importance...")
                self._guarded_section('permutation_importance', report, 'add_permutation_importance',
                    model, X, y)

            # ==== SECTION 5.7: PARTIAL DEPENDENCE PLOTS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding partial dependence plots...")
                self._guarded_section('partial_dependence', report, 'add_partial_dependence',
                    model, X, feature_names)

            # ==== SECTION 6: MODEL BENCHMARKING ====
            if model_scores:
                report.add_model_benchmarking(model_scores)

            # ==== SECTION 6.3: MODEL COMPARISON (NEW — Round 4) ====
            comparison_models = trained_models or results.get('trained_models')
            if include_all and comparison_models and isinstance(comparison_models, dict) and len(comparison_models) >= 2:
                logger.info("  - Adding model comparison...")
                self._guarded_section('model_comparison', report, 'add_model_comparison',
                    comparison_models, X, y)

            # ==== SECTION 6.5: MULTI-DATASET VALIDATION (NEW) ====
            if include_all and validation_datasets:
                logger.info("  - Adding cross-dataset validation...")
                self._guarded_section('cross_dataset_validation', report, 'add_cross_dataset_validation',
                    validation_datasets, model)

            # ==== SECTION 7: LEARNING CURVES ====
            if include_all and model is not None:
                logger.info("  - Adding learning curves...")
                self._guarded_section('learning_curves', report, 'add_learning_curves',
                    model, X, y)

            # ==== SECTION 8: CV DETAILED ANALYSIS ====
            if include_all and model is not None:
                logger.info("  - Adding CV analysis...")
                self._guarded_section('cv_detailed_analysis', report, 'add_cv_detailed_analysis',
                    model, X, y)

            # ==== SECTION 8.5: STATISTICAL SIGNIFICANCE (NEW) ====
            if include_all and y_pred is not None:
                logger.info("  - Adding statistical significance tests...")
                self._guarded_section('statistical_tests', report, 'add_statistical_tests',
                    y, y_pred, y_prob)

            # ==== SECTION 9/9R: CLASSIFICATION METRICS or REGRESSION ANALYSIS ====
            if is_regression and y_pred is not None:
                # ==== SECTION 9R: REGRESSION ANALYSIS (NEW) ====
                logger.info("  - Adding regression analysis...")
                self._guarded_section('regression_analysis', report, 'add_regression_analysis',
                    y, y_pred)
            elif y_pred is not None:
                n_cls = len(np.unique(y))
                default_labels = ['Negative', 'Positive'] if n_cls <= 2 else [f'Class {i}' for i in range(n_cls)]
                labels = results.get('labels', default_labels)
                report.add_confusion_matrix(y, y_pred, labels)

                if y_prob is not None:
                    report.add_roc_analysis(y, y_prob)
                    report.add_precision_recall(y, y_prob)

                    # ==== SECTION 10: CALIBRATION ====
                    if include_all:
                        logger.info("  - Adding calibration analysis...")
                        report.add_calibration_analysis(y, y_prob)

                    # ==== SECTION 10.5: CONFIDENCE-CALIBRATED PREDICTIONS (NEW) ====
                    if include_all:
                        logger.info("  - Adding prediction confidence analysis...")
                        self._guarded_section('prediction_confidence', report,
                            'add_prediction_confidence_analysis', X, y, y_pred, y_prob)

                    # ==== SECTION 11: LIFT & GAIN ====
                    if include_all:
                        logger.info("  - Adding lift/gain analysis...")
                        report.add_lift_gain_analysis(y, y_prob)

                    # ==== SECTION 12: THRESHOLD OPTIMIZATION ====
                    if include_all:
                        logger.info("  - Adding threshold optimization...")
                        report.add_threshold_optimization(y, y_prob)

                    # ==== SECTION 12.5: SCORE DISTRIBUTION (NEW) ====
                    if include_all:
                        logger.info("  - Adding score distribution...")
                        self._guarded_section('score_distribution', report, 'add_score_distribution',
                            y, y_prob, labels)

                # ==== SECTION 13: ERROR ANALYSIS ====
                if include_all:
                    logger.info("  - Adding error analysis...")
                    report.add_error_analysis(X, y, y_pred, y_prob)

            # ==== SECTION 13.5: DATA DRIFT MONITORING (NEW) ====
            if include_all and X_reference is not None:
                logger.info("  - Adding drift analysis...")
                self._guarded_section('drift_analysis', report, 'add_drift_analysis',
                    X_reference, X, feature_importance=feature_importance)

            # ==== SECTION 14: SHAP DEEP DIVE ====
            if shap_values is not None:
                logger.info("  - Adding SHAP deep analysis...")
                X_array = X.values if hasattr(X, 'values') else X
                report.add_shap_deep_analysis(shap_values, feature_names, X_array, model)

                # ==== SECTION 14.5: FEATURE INTERACTIONS (NEW) ====
                if include_all:
                    logger.info("  - Adding feature interactions...")
                    self._guarded_section('feature_interactions', report, 'add_feature_interactions',
                        shap_values, feature_names, X_array, model)

            # ==== SECTION 14.6: INTERPRETABILITY ANALYSIS (NEW) ====
            if include_all and model is not None and y_pred is not None:
                logger.info("  - Adding interpretability analysis...")
                X_array = X.values if hasattr(X, 'values') else X
                self._guarded_section('interpretability_analysis', report, 'add_interpretability_analysis',
                    model, X_array, y_pred, feature_names, top_n=5)

            # ==== SECTION 14.7: DEEP LEARNING ANALYSIS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding deep learning analysis...")
                X_array = X.values if hasattr(X, 'values') else X
                self._guarded_section('deep_learning_analysis', report, 'add_deep_learning_analysis',
                    model, X_array, training_history=results.get('training_history'))

            # ==== SECTION 15: BASELINE COMPARISON ====
            if 'baseline_score' in results:
                report.add_baseline_comparison(
                    results['baseline_score'],
                    metrics.get('accuracy', results.get('final_score', 0))
                )

            # ==== SECTION 16: RECOMMENDATIONS ====
            recommendations = self._generate_smart_recommendations(results, model, feature_importance)
            report.add_recommendations(recommendations)

            # ==== SECTION 17: REPRODUCIBILITY ====
            if include_all:
                logger.info("  - Adding reproducibility section...")
                report.add_reproducibility_section(
                    model,
                    params=results.get('best_params', {}),
                    random_state=42
                )

            # ==== SECTION 17.2: DEPLOYMENT READINESS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding deployment readiness...")
                X_array = X.values if hasattr(X, 'values') else X
                self._guarded_section('deployment_readiness', report, 'add_deployment_readiness',
                    model, X_array)

            # ==== SECTION 17.5: HYPERPARAMETER SEARCH VIZ (NEW) ====
            if include_all and study_or_trials is not None:
                logger.info("  - Adding hyperparameter visualization...")
                self._guarded_section('hyperparameter_visualization', report,
                    'add_hyperparameter_visualization', study_or_trials)

            # ==== SECTION 18: MODEL CARD (NEW) ====
            if include_all:
                logger.info("  - Adding model card...")
                self._guarded_section('model_card', report, 'add_model_card',
                    model=model, results=results,
                    intended_use=results.get('intended_use', ''),
                    limitations=results.get('limitations', ''),
                    ethical=results.get('ethical_considerations', ''))

            # ==== SECTION 18.5: FAIRNESS & BIAS AUDIT (NEW) ====
            if include_all and sensitive_features is not None:
                logger.info("  - Adding fairness audit...")
                self._guarded_section('fairness_audit', report, 'add_fairness_audit',
                    X, y, y_pred, y_prob, sensitive_features)

            # ==== INSIGHT PRIORITIZATION (LAST — reads all section data) ====
            if include_all:
                logger.info("  - Adding insight prioritization...")
                self._guarded_section('insight_prioritization', report, 'add_insight_prioritization')

            # Generate PDF
            logger.info("  - Generating PDF...")
            pdf_path = report.generate(filename)

            if pdf_path:
                logger.info(f"World-class report generated: {pdf_path}")
                # Get file size
                import os
                size_kb = os.path.getsize(pdf_path) / 1024
                logger.info(f"  - Size: {size_kb:.1f} KB")

            # Generate HTML report if requested
            if generate_html:
                logger.info("  - Generating interactive HTML report...")
                html_filename = filename.replace('.pdf', '.html') if filename else None
                html_path = report.generate_html(html_filename)
                if html_path:
                    logger.info(f"  - HTML report: {html_path}")

            # Auto-send to Telegram if configured
            if pdf_path:
                try:
                    if not getattr(self, '_telegram_available', False):
                        self.init_telegram()
                    if self._telegram_available:
                        logger.info("  - Sending report to Telegram...")
                        sent = self.send_telegram_report(pdf_path, results=results)
                        if sent:
                            logger.info("  - Report sent to Telegram")
                except Exception as e:
                    logger.debug(f"Telegram send failed: {e}")

            return pdf_path

        except Exception as e:
            logger.error(f"Failed to generate world-class report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_smart_recommendations(self, results: Dict, model, feature_importance: Dict) -> List[str]:
        """Generate intelligent recommendations based on results."""
        recommendations = []

        # Model performance
        score = results.get('final_score', results.get('metrics', {}).get('accuracy', 0))
        if score >= 0.95:
            recommendations.append(f"Excellent performance ({score:.1%}) achieved - monitor for overfitting")
        elif score >= 0.85:
            recommendations.append(f"Good performance ({score:.1%}) - consider ensemble methods for improvement")
        elif score >= 0.75:
            recommendations.append(f"Moderate performance ({score:.1%}) - feature engineering may help")
        else:
            recommendations.append(f"Performance needs improvement ({score:.1%}) - consider more complex models or better features")

        # Model type recommendation
        model_name = type(model).__name__ if model else 'Unknown'
        if 'Logistic' in model_name:
            recommendations.append("Logistic Regression provides good interpretability - ideal for regulated industries")
        elif 'Forest' in model_name or 'Gradient' in model_name:
            recommendations.append("Tree-based model captures non-linear patterns well")

        # Feature importance insights
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            top_names = [f[0] for f in top_features]
            recommendations.append(f"Top predictive features: {', '.join(top_names)}")

            # Concentration check
            total_imp = sum(feature_importance.values())
            top3_imp = sum(f[1] for f in top_features)
            if top3_imp / total_imp > 0.6:
                recommendations.append("High feature importance concentration - model relies heavily on few features")

        # AUC-based recommendation
        auc = results.get('metrics', {}).get('auc_roc', results.get('auc_roc', 0))
        if auc > 0:
            if auc >= 0.9:
                recommendations.append(f"Excellent discrimination (AUC={auc:.3f}) - suitable for production")
            elif auc >= 0.8:
                recommendations.append(f"Good discrimination (AUC={auc:.3f}) - threshold tuning recommended")
            else:
                recommendations.append(f"Moderate discrimination (AUC={auc:.3f}) - consider feature engineering")

        # General recommendations
        recommendations.extend([
            "Monitor model performance over time for concept drift",
            "Validate on held-out data before production deployment",
            "Document model decisions for regulatory compliance",
        ])

        return recommendations

    # =========================================================================
    # TELEGRAM NOTIFICATION
    # =========================================================================

