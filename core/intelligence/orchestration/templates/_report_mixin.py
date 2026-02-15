"""Report Mixin - PDF report generation with sections and figures."""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.intelligence.orchestration.templates.swarm_ml_comprehensive import ReportConfig

logger = logging.getLogger(__name__)


class ReportMixin:
    def init_report(self, config: "ReportConfig" = None) -> None:
        """
        Initialize PDF report generation.

        Args:
            config: Report configuration (uses default if None)
        """
        from Jotty.core.intelligence.orchestration.templates.swarm_ml_comprehensive import ReportConfig
        self._report_config = config or ReportConfig()
        self._report_data = {
            'sections': [],
            'figures': [],
            'tables': [],
            'metrics': {},
            'timestamp': datetime.now(),
        }
        self._report_available = False

        if not self._report_config.enabled:
            return

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            self._reportlab = {
                'colors': colors,
                'letter': letter,
                'A4': A4,
                'SimpleDocTemplate': SimpleDocTemplate,
                'Paragraph': Paragraph,
                'Spacer': Spacer,
                'Table': Table,
                'TableStyle': TableStyle,
                'Image': Image,
                'PageBreak': PageBreak,
                'getSampleStyleSheet': getSampleStyleSheet,
                'ParagraphStyle': ParagraphStyle,
                'inch': inch,
            }

            self._report_available = True

            # Create output directory
            os.makedirs(self._report_config.output_dir, exist_ok=True)

            logger.info(f"Report generation initialized: output_dir={self._report_config.output_dir}")

        except ImportError:
            logger.warning("reportlab not installed. Install with: pip install reportlab")
        except Exception as e:
            logger.warning(f"Report initialization failed: {e}")

    def add_executive_summary(self, results: Dict[str, Any], context: str = "") -> None:
        """Add executive summary section to report."""
        if not self._report_available:
            return

        summary = {
            'type': 'executive_summary',
            'title': 'Executive Summary',
            'content': {
                'context': context,
                'final_score': results.get('final_score', 0),
                'best_model': str(results.get('best_model', 'N/A')),
                'n_features_used': results.get('n_features', 0),
                'iterations': self._learning_state.iteration,
                'improvement': self._calculate_improvement(),
                'key_findings': self._extract_key_findings(results),
            }
        }
        self._report_data['sections'].append(summary)

    def add_data_profile(self, eda_insights: Dict[str, Any]) -> None:
        """Add data profiling section to report."""
        if not self._report_available or not self._report_config.include_data_profile:
            return

        profile = {
            'type': 'data_profile',
            'title': 'Data Profile & EDA',
            'content': {
                'shape': eda_insights.get('shape', {}),
                'dtypes': eda_insights.get('dtypes', {}),
                'missing': eda_insights.get('missing_values', {}),
                'statistics': eda_insights.get('statistics', {}),
                'correlations': eda_insights.get('correlations', {}),
                'recommendations': eda_insights.get('recommendations', []),
            }
        }
        self._report_data['sections'].append(profile)

    def add_feature_importance(self, importance: Dict[str, float], top_n: int = None) -> None:
        """Add feature importance section to report."""
        if not self._report_available or not self._report_config.include_feature_importance:
            return

        top_n = top_n or self._report_config.max_features_in_report
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        section = {
            'type': 'feature_importance',
            'title': 'Feature Importance Analysis',
            'content': {
                'importance_ranking': sorted_imp,
                'total_features': len(importance),
                'top_n_shown': top_n,
            }
        }
        self._report_data['sections'].append(section)

        # Create importance figure
        self._create_importance_figure(sorted_imp)

    def add_model_benchmarking(self, model_scores: Dict[str, Dict[str, float]]) -> None:
        """Add model benchmarking comparison to report."""
        if not self._report_available or not self._report_config.include_model_benchmarking:
            return

        section = {
            'type': 'model_benchmarking',
            'title': 'Model Benchmarking',
            'content': {
                'models': model_scores,
                'best_model': max(model_scores.items(), key=lambda x: x[1].get('cv_score', 0))[0] if model_scores else None,
            }
        }
        self._report_data['sections'].append(section)

    def add_confusion_matrix_report(self, y_true: Any, y_pred: Any, labels: Any = None) -> None:
        """Add confusion matrix to report."""
        if not self._report_available or not self._report_config.include_confusion_matrix:
            return

        try:
            from sklearn.metrics import confusion_matrix, classification_report

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

            section = {
                'type': 'confusion_matrix',
                'title': 'Confusion Matrix & Classification Report',
                'content': {
                    'matrix': cm.tolist(),
                    'labels': labels.tolist() if hasattr(labels, 'tolist') else labels,
                    'classification_report': report,
                }
            }
            self._report_data['sections'].append(section)

            # Create confusion matrix figure
            self._create_confusion_matrix_figure(cm, labels)

        except Exception as e:
            logger.debug(f"Failed to add confusion matrix: {e}")

    def add_roc_analysis(self, y_true: Any, y_prob: Any, pos_label: Any = 1) -> None:
        """Add ROC curve analysis to report."""
        if not self._report_available or not self._report_config.include_roc_curves:
            return

        try:
            from sklearn.metrics import roc_curve, auc, roc_auc_score

            fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            section = {
                'type': 'roc_analysis',
                'title': 'ROC Curve Analysis',
                'content': {
                    'auc': roc_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'optimal_threshold': self._find_optimal_threshold(fpr, tpr, thresholds),
                }
            }
            self._report_data['sections'].append(section)

            # Create ROC figure
            self._create_roc_figure(fpr, tpr, roc_auc)

        except Exception as e:
            logger.debug(f"Failed to add ROC analysis: {e}")

    def add_precision_recall_analysis(self, y_true: Any, y_prob: Any, pos_label: Any = 1) -> None:
        """Add precision-recall curve analysis to report."""
        if not self._report_available or not self._report_config.include_precision_recall:
            return

        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
            avg_precision = average_precision_score(y_true, y_prob, pos_label=pos_label)

            section = {
                'type': 'precision_recall',
                'title': 'Precision-Recall Analysis',
                'content': {
                    'average_precision': avg_precision,
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                }
            }
            self._report_data['sections'].append(section)

            # Create PR curve figure
            self._create_pr_figure(precision, recall, avg_precision)

        except Exception as e:
            logger.debug(f"Failed to add precision-recall analysis: {e}")

    def add_shap_analysis(self, shap_values: Any, feature_names: List[str], X_sample: Any = None) -> None:
        """Add SHAP analysis to report."""
        if not self._report_available or not self._report_config.include_shap_analysis:
            return

        try:
            import shap

            # Calculate mean absolute SHAP values per feature
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = dict(zip(feature_names, mean_shap.tolist()))

            section = {
                'type': 'shap_analysis',
                'title': 'SHAP Feature Analysis',
                'content': {
                    'shap_importance': shap_importance,
                    'top_features': sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:20],
                }
            }
            self._report_data['sections'].append(section)

            # Create SHAP figures
            self._create_shap_figures(shap_values, feature_names, X_sample)

        except Exception as e:
            logger.debug(f"Failed to add SHAP analysis: {e}")

    def add_baseline_comparison(self, baseline_score: float, final_score: float, baseline_model: str = 'DummyClassifier') -> Any:
        """Add baseline vs final model comparison."""
        if not self._report_available or not self._report_config.include_baseline_comparison:
            return

        improvement = final_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

        section = {
            'type': 'baseline_comparison',
            'title': 'Baseline Comparison',
            'content': {
                'baseline_model': baseline_model,
                'baseline_score': baseline_score,
                'final_score': final_score,
                'improvement': improvement,
                'improvement_percent': improvement_pct,
            }
        }
        self._report_data['sections'].append(section)

    def add_recommendations(self, recommendations: List[str]) -> None:
        """Add recommendations section to report."""
        if not self._report_available or not self._report_config.include_recommendations:
            return

        section = {
            'type': 'recommendations',
            'title': 'Recommendations & Next Steps',
            'content': {
                'recommendations': recommendations,
            }
        }
        self._report_data['sections'].append(section)

    def generate_report(self, filename: str = None) -> Optional[str]:
        """
        Generate the final PDF report.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated PDF file, or None if generation failed
        """
        if not self._report_available:
            logger.warning("Report generation not available")
            return None

        try:
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ml_report_{timestamp}.pdf"

            filepath = os.path.join(self._report_config.output_dir, filename)

            # Build PDF document
            doc = self._reportlab['SimpleDocTemplate'](
                filepath,
                pagesize=self._reportlab['letter'],
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Build story (content)
            story = self._build_report_story()

            # Generate PDF
            doc.build(story)

            logger.info(f"Report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

    def _build_report_story(self) -> List:
        """Build the report content (story) for reportlab."""
        styles = self._reportlab['getSampleStyleSheet']()
        story = []

        # Title page
        story.append(self._reportlab['Paragraph'](
            "Machine Learning Analysis Report",
            styles['Title']
        ))
        story.append(self._reportlab['Spacer'](1, 12))
        story.append(self._reportlab['Paragraph'](
            f"Generated by Jotty SwarmML Comprehensive v{self.version}",
            styles['Normal']
        ))
        story.append(self._reportlab['Paragraph'](
            f"Date: {self._report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        story.append(self._reportlab['Spacer'](1, 24))

        # Process each section
        for section in self._report_data['sections']:
            story.extend(self._render_section(section, styles))
            story.append(self._reportlab['Spacer'](1, 12))

        # Add figures
        for fig_path in self._report_data['figures']:
            if os.path.exists(fig_path):
                try:
                    img = self._reportlab['Image'](fig_path, width=6*self._reportlab['inch'],
                                                   height=4*self._reportlab['inch'])
                    story.append(img)
                    story.append(self._reportlab['Spacer'](1, 12))
                except Exception:
                    pass

        return story

    def _render_section(self, section: Dict, styles: Any) -> List:
        """Render a report section to reportlab elements."""
        elements = []

        # Section title
        elements.append(self._reportlab['Paragraph'](
            section['title'],
            styles['Heading1']
        ))
        elements.append(self._reportlab['Spacer'](1, 6))

        content = section['content']
        section_type = section['type']

        if section_type == 'executive_summary':
            elements.extend(self._render_executive_summary(content, styles))
        elif section_type == 'data_profile':
            elements.extend(self._render_data_profile(content, styles))
        elif section_type == 'feature_importance':
            elements.extend(self._render_feature_importance(content, styles))
        elif section_type == 'model_benchmarking':
            elements.extend(self._render_model_benchmarking(content, styles))
        elif section_type == 'confusion_matrix':
            elements.extend(self._render_confusion_matrix(content, styles))
        elif section_type == 'roc_analysis':
            elements.extend(self._render_roc_analysis(content, styles))
        elif section_type == 'precision_recall':
            elements.extend(self._render_precision_recall(content, styles))
        elif section_type == 'baseline_comparison':
            elements.extend(self._render_baseline_comparison(content, styles))
        elif section_type == 'recommendations':
            elements.extend(self._render_recommendations(content, styles))

        return elements

    def _render_executive_summary(self, content: Dict, styles: Any) -> List:
        """Render executive summary section."""
        elements = []

        if content.get('context'):
            elements.append(self._reportlab['Paragraph'](
                f"<b>Context:</b> {content['context']}",
                styles['Normal']
            ))

        elements.append(self._reportlab['Paragraph'](
            f"<b>Final Score:</b> {content.get('final_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Best Model:</b> {content.get('best_model', 'N/A')}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Features Used:</b> {content.get('n_features_used', 0)}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Iterations:</b> {content.get('iterations', 0)}",
            styles['Normal']
        ))

        if content.get('key_findings'):
            elements.append(self._reportlab['Spacer'](1, 6))
            elements.append(self._reportlab['Paragraph']("<b>Key Findings:</b>", styles['Normal']))
            for finding in content['key_findings']:
                elements.append(self._reportlab['Paragraph'](f"• {finding}", styles['Normal']))

        return elements

    def _render_data_profile(self, content: Dict, styles: Any) -> List:
        """Render data profile section."""
        elements = []

        shape = content.get('shape', {})
        elements.append(self._reportlab['Paragraph'](
            f"<b>Dataset Shape:</b> {shape.get('rows', 0)} rows × {shape.get('columns', 0)} columns",
            styles['Normal']
        ))

        if content.get('missing'):
            missing = content['missing']
            if any(v > 0 for v in missing.values()):
                elements.append(self._reportlab['Paragraph']("<b>Missing Values:</b>", styles['Normal']))
                for col, count in list(missing.items())[:10]:
                    if count > 0:
                        elements.append(self._reportlab['Paragraph'](f"  • {col}: {count}", styles['Normal']))

        if content.get('recommendations'):
            elements.append(self._reportlab['Spacer'](1, 6))
            elements.append(self._reportlab['Paragraph']("<b>EDA Recommendations:</b>", styles['Normal']))
            for rec in content['recommendations'][:5]:
                elements.append(self._reportlab['Paragraph'](f"• {rec}", styles['Normal']))

        return elements

    def _render_feature_importance(self, content: Dict, styles: Any) -> List:
        """Render feature importance section."""
        elements = []

        ranking = content.get('importance_ranking', [])
        elements.append(self._reportlab['Paragraph'](
            f"<b>Total Features:</b> {content.get('total_features', 0)}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Top {content.get('top_n_shown', 20)} Features:</b>",
            styles['Normal']
        ))

        # Create table
        table_data = [['Rank', 'Feature', 'Importance']]
        for rank, (feat, imp) in enumerate(ranking[:15], 1):
            table_data.append([str(rank), feat[:30], f"{imp:.4f}"])

        table = self._reportlab['Table'](table_data, colWidths=[40, 250, 80])
        table.setStyle(self._reportlab['TableStyle']([
            ('BACKGROUND', (0, 0), (-1, 0), self._reportlab['colors'].grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), self._reportlab['colors'].whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, self._reportlab['colors'].black),
        ]))
        elements.append(table)

        return elements

    def _render_model_benchmarking(self, content: Dict, styles: Any) -> List:
        """Render model benchmarking section."""
        elements = []

        models = content.get('models', {})
        if not models:
            return elements

        elements.append(self._reportlab['Paragraph'](
            f"<b>Best Model:</b> {content.get('best_model', 'N/A')}",
            styles['Normal']
        ))

        # Create comparison table
        table_data = [['Model', 'CV Score', 'Std', 'Train Time']]
        for model_name, scores in models.items():
            table_data.append([
                model_name[:25],
                f"{scores.get('cv_score', 0):.4f}",
                f"{scores.get('cv_std', 0):.4f}",
                f"{scores.get('train_time', 0):.2f}s"
            ])

        table = self._reportlab['Table'](table_data, colWidths=[150, 80, 80, 80])
        table.setStyle(self._reportlab['TableStyle']([
            ('BACKGROUND', (0, 0), (-1, 0), self._reportlab['colors'].grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), self._reportlab['colors'].whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, self._reportlab['colors'].black),
        ]))
        elements.append(table)

        return elements

    def _render_confusion_matrix(self, content: Dict, styles: Any) -> List:
        """Render confusion matrix section."""
        elements = []

        report = content.get('classification_report', {})
        if report:
            elements.append(self._reportlab['Paragraph']("<b>Classification Metrics:</b>", styles['Normal']))

            for cls, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    elements.append(self._reportlab['Paragraph'](
                        f"  Class {cls}: Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}",
                        styles['Normal']
                    ))

        return elements

    def _render_roc_analysis(self, content: Dict, styles: Any) -> List:
        """Render ROC analysis section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>AUC Score:</b> {content.get('auc', 0):.4f}",
            styles['Normal']
        ))

        if content.get('optimal_threshold'):
            elements.append(self._reportlab['Paragraph'](
                f"<b>Optimal Threshold:</b> {content['optimal_threshold']:.4f}",
                styles['Normal']
            ))

        return elements

    def _render_precision_recall(self, content: Dict, styles: Any) -> List:
        """Render precision-recall analysis section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>Average Precision:</b> {content.get('average_precision', 0):.4f}",
            styles['Normal']
        ))

        return elements

    def _render_baseline_comparison(self, content: Dict, styles: Any) -> List:
        """Render baseline comparison section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>Baseline Model:</b> {content.get('baseline_model', 'N/A')}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Baseline Score:</b> {content.get('baseline_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Final Score:</b> {content.get('final_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Improvement:</b> {content.get('improvement', 0):.4f} "
            f"({content.get('improvement_percent', 0):.1f}%)",
            styles['Normal']
        ))

        return elements

    def _render_recommendations(self, content: Dict, styles: Any) -> List:
        """Render recommendations section."""
        elements = []

        for rec in content.get('recommendations', []):
            elements.append(self._reportlab['Paragraph'](f"• {rec}", styles['Normal']))

        return elements

    def _create_importance_figure(self, sorted_importance: List[Tuple[str, float]]) -> Any:
        """Create and save feature importance figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            features = [x[0] for x in sorted_importance[:20]]
            values = [x[1] for x in sorted_importance[:20]]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), values[::-1], color='steelblue')
            plt.yticks(range(len(features)), features[::-1], fontsize=8)
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create importance figure: {e}")

    def _create_confusion_matrix_figure(self, cm: Any, labels: Any) -> Any:
        """Create and save confusion matrix figure."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            import tempfile

            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap='Blues', ax=plt.gca())
            plt.title('Confusion Matrix')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create confusion matrix figure: {e}")

    def _create_roc_figure(self, fpr: Any, tpr: Any, roc_auc: Any) -> Any:
        """Create and save ROC curve figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create ROC figure: {e}")

    def _create_pr_figure(self, precision: Any, recall: Any, avg_precision: Any) -> Any:
        """Create and save precision-recall curve figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create PR figure: {e}")

    def _create_shap_figures(self, shap_values: Any, feature_names: Any, X_sample: Any) -> Any:
        """Create and save SHAP analysis figures."""
        try:
            import shap
            import matplotlib.pyplot as plt
            import tempfile

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                            plot_type="bar", show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create SHAP figures: {e}")

    def _calculate_improvement(self) -> float:
        """Calculate score improvement over iterations."""
        history = self._learning_state.score_history
        if len(history) < 2:
            return 0.0
        return history[-1] - history[0]

    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from results for executive summary."""
        findings = []

        # Best score finding
        score = results.get('final_score', 0)
        if score > 0.9:
            findings.append(f"Achieved excellent performance with {score:.1%} accuracy")
        elif score > 0.8:
            findings.append(f"Achieved good performance with {score:.1%} accuracy")
        else:
            findings.append(f"Model achieved {score:.1%} accuracy")

        # Feature importance finding
        importance = results.get('feature_importance', {})
        if importance:
            top_feat = max(importance.items(), key=lambda x: x[1])
            findings.append(f"Most important feature: {top_feat[0]} ({top_feat[1]:.2%} importance)")

        # Improvement finding
        improvement = self._calculate_improvement()
        if improvement > 0.01:
            findings.append(f"Improved {improvement:.1%} through iterative refinement")

        # Learning finding
        n_patterns = len(self._learning_state.learned_patterns)
        if n_patterns > 0:
            findings.append(f"Leveraged {n_patterns} learned patterns from historical sessions")

        return findings

    def _find_optimal_threshold(self, fpr: Any, tpr: Any, thresholds: Any) -> float:
        """Find optimal classification threshold using Youden's J statistic."""
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # =========================================================================
    # PROFESSIONAL PDF REPORT (Pandoc + LaTeX)
    # =========================================================================

