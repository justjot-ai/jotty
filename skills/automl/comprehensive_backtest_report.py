"""
Comprehensive Backtest Report Generator
=======================================

Generates world-class PDF reports from ComprehensiveBacktestResult.

Includes:
- Executive Summary with confidence intervals
- Walk-Forward Analysis
- Monte Carlo Simulation Results
- Risk Metrics (VaR, CVaR, Stress Tests)
- Statistical Significance
- Regime Analysis
- Factor Exposure
- Position Sizing Recommendations
- Transaction Cost Analysis
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backtest_engine import (
    ComprehensiveBacktestResult,
    FactorExposure,
    MonteCarloResult,
    PositionSizing,
    RegimeAnalysis,
    RiskMetrics,
    StatisticalTests,
    TransactionCosts,
    WalkForwardResult,
)

logger = logging.getLogger(__name__)


class ComprehensiveBacktestChartGenerator:
    """Generate advanced charts for comprehensive backtest reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_monte_carlo_distribution(
        self, result: ComprehensiveBacktestResult, filename: str = "monte_carlo_dist.png"
    ) -> Optional[Path]:
        """Generate Monte Carlo return distribution chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            mc = result.monte_carlo
            if mc.n_simulations == 0:
                return None

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 1. Return Distribution
            ax1 = axes[0]
            x = np.linspace(mc.return_5th - 10, mc.return_95th + 10, 100)
            # Approximate normal distribution
            from scipy import stats

            pdf = stats.norm.pdf(x, mc.mean_return, mc.std_return)
            ax1.fill_between(x, pdf, alpha=0.3, color="#2196F3")
            ax1.axvline(
                mc.mean_return,
                color="#1565C0",
                linestyle="-",
                linewidth=2,
                label=f"Mean: {mc.mean_return:.1f}%",
            )
            ax1.axvline(
                mc.return_5th,
                color="#E53935",
                linestyle="--",
                linewidth=1.5,
                label=f"5th: {mc.return_5th:.1f}%",
            )
            ax1.axvline(
                mc.return_95th,
                color="#4CAF50",
                linestyle="--",
                linewidth=1.5,
                label=f"95th: {mc.return_95th:.1f}%",
            )
            ax1.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
            ax1.set_xlabel("Total Return (%)", fontweight="bold")
            ax1.set_ylabel("Probability Density")
            ax1.set_title("Return Distribution (Monte Carlo)", fontweight="bold")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 2. Sharpe Distribution
            ax2 = axes[1]
            x = np.linspace(mc.sharpe_5th - 0.5, mc.sharpe_95th + 0.5, 100)
            pdf = stats.norm.pdf(x, mc.mean_sharpe, mc.std_sharpe)
            ax2.fill_between(x, pdf, alpha=0.3, color="#9C27B0")
            ax2.axvline(
                mc.mean_sharpe,
                color="#7B1FA2",
                linestyle="-",
                linewidth=2,
                label=f"Mean: {mc.mean_sharpe:.2f}",
            )
            ax2.axvline(1.0, color="#4CAF50", linestyle="--", linewidth=1.5, label="Sharpe = 1")
            ax2.axvline(0, color="#E53935", linestyle="--", linewidth=1.5, label="Sharpe = 0")
            ax2.set_xlabel("Sharpe Ratio", fontweight="bold")
            ax2.set_ylabel("Probability Density")
            ax2.set_title("Sharpe Ratio Distribution", fontweight="bold")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 3. Probability Metrics
            ax3 = axes[2]
            probs = [mc.prob_positive, mc.prob_beat_benchmark, mc.prob_sharpe_above_1]
            labels = ["P(Return > 0)", "P(Beat Benchmark)", "P(Sharpe > 1)"]
            colors = ["#4CAF50" if p > 0.5 else "#E53935" for p in probs]

            bars = ax3.bar(
                labels,
                [p * 100 for p in probs],
                color=colors,
                alpha=0.7,
                edgecolor="white",
                linewidth=2,
            )
            ax3.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax3.set_ylabel("Probability (%)", fontweight="bold")
            ax3.set_title("Success Probabilities", fontweight="bold")
            ax3.set_ylim(0, 100)

            for bar, p in zip(bars, probs):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{p*100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax3.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate Monte Carlo chart: {e}")
            return None

    def generate_walk_forward_chart(
        self, result: ComprehensiveBacktestResult, filename: str = "walk_forward.png"
    ) -> Optional[Path]:
        """Generate walk-forward analysis chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            wf_results = result.walk_forward_results
            if not wf_results:
                return None

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            windows = range(1, len(wf_results) + 1)

            # 1. IS vs OOS Returns
            ax1 = axes[0, 0]
            is_returns = [wf.is_return for wf in wf_results]
            oos_returns = [wf.oos_return for wf in wf_results]

            x = np.arange(len(windows))
            width = 0.35
            ax1.bar(x - width / 2, is_returns, width, label="In-Sample", color="#2196F3", alpha=0.8)
            ax1.bar(
                x + width / 2, oos_returns, width, label="Out-of-Sample", color="#FF9800", alpha=0.8
            )
            ax1.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax1.set_xlabel("Window", fontweight="bold")
            ax1.set_ylabel("Return (%)", fontweight="bold")
            ax1.set_title("In-Sample vs Out-of-Sample Returns", fontweight="bold")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"W{i}" for i in windows])
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis="y")

            # 2. IS vs OOS Sharpe
            ax2 = axes[0, 1]
            is_sharpes = [wf.is_sharpe for wf in wf_results]
            oos_sharpes = [wf.oos_sharpe for wf in wf_results]

            ax2.bar(x - width / 2, is_sharpes, width, label="In-Sample", color="#9C27B0", alpha=0.8)
            ax2.bar(
                x + width / 2, oos_sharpes, width, label="Out-of-Sample", color="#E91E63", alpha=0.8
            )
            ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax2.axhline(1, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7)
            ax2.set_xlabel("Window", fontweight="bold")
            ax2.set_ylabel("Sharpe Ratio", fontweight="bold")
            ax2.set_title("In-Sample vs Out-of-Sample Sharpe", fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"W{i}" for i in windows])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis="y")

            # 3. Degradation
            ax3 = axes[1, 0]
            degradation = [wf.sharpe_degradation for wf in wf_results]
            colors = ["#E53935" if d > 0 else "#4CAF50" for d in degradation]

            ax3.bar(windows, degradation, color=colors, alpha=0.8)
            ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax3.set_xlabel("Window", fontweight="bold")
            ax3.set_ylabel("Degradation (%)", fontweight="bold")
            ax3.set_title("Sharpe Degradation (IS - OOS)", fontweight="bold")
            ax3.grid(True, alpha=0.3, axis="y")

            # 4. Summary Stats
            ax4 = axes[1, 1]
            ax4.axis("off")

            summary_text = f"""
WALK-FORWARD SUMMARY

Windows Tested: {len(wf_results)}

Average OOS Return: {result.wf_avg_oos_return:.2f}%
Average OOS Sharpe: {result.wf_avg_oos_sharpe:.2f}

Average Degradation: {result.wf_degradation:.1f}%

Consistency Score: {sum(1 for wf in wf_results if wf.oos_return > 0) / len(wf_results) * 100:.0f}%
(% of windows with positive OOS return)
"""
            ax4.text(
                0.1,
                0.5,
                summary_text,
                fontsize=12,
                fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate walk-forward chart: {e}")
            return None

    def generate_risk_metrics_chart(
        self, result: ComprehensiveBacktestResult, filename: str = "risk_metrics.png"
    ) -> Optional[Path]:
        """Generate comprehensive risk metrics chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            rm = result.risk_metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. VaR and CVaR
            ax1 = axes[0, 0]
            metrics = ["VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"]
            values = [rm.var_95, rm.var_99, rm.cvar_95, rm.cvar_99]
            colors = ["#FF9800", "#E53935", "#FF5722", "#B71C1C"]

            bars = ax1.barh(metrics, values, color=colors, alpha=0.8)
            ax1.axvline(0, color="black", linestyle="-", linewidth=0.5)
            ax1.set_xlabel("Daily Loss (%)", fontweight="bold")
            ax1.set_title("Value at Risk Analysis", fontweight="bold")
            ax1.grid(True, alpha=0.3, axis="x")

            for bar, val in zip(bars, values):
                ax1.text(
                    val - 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%",
                    ha="right",
                    va="center",
                    fontweight="bold",
                    color="white",
                )

            # 2. Distribution Moments
            ax2 = axes[0, 1]
            moments = ["Volatility\n(Ann.)", "Skewness", "Excess\nKurtosis", "Tail\nRatio"]
            moment_values = [rm.volatility, rm.skewness, rm.kurtosis, rm.tail_ratio]

            # Normalize for visualization
            normalized = [
                rm.volatility / 30,
                rm.skewness / 2 + 0.5,
                rm.kurtosis / 5 + 0.5,
                rm.tail_ratio / 3,
            ]
            colors = [
                "#2196F3",
                "#4CAF50" if rm.skewness > 0 else "#E53935",
                "#FF9800" if rm.kurtosis > 0 else "#9C27B0",
                "#607D8B",
            ]

            bars = ax2.bar(moments, normalized, color=colors, alpha=0.8)
            ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax2.set_ylabel("Normalized Value", fontweight="bold")
            ax2.set_title("Distribution Characteristics", fontweight="bold")
            ax2.set_ylim(0, 1.5)

            # Add actual values as text
            for bar, val in zip(bars, moment_values):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            # 3. Drawdown Analysis
            ax3 = axes[1, 0]
            dd_metrics = ["Max\nDrawdown", "Avg\nDrawdown", "Ulcer\nIndex", "Pain\nIndex"]
            dd_values = [abs(rm.max_drawdown), abs(rm.avg_drawdown), rm.ulcer_index, rm.pain_index]

            colors = plt.cm.Reds([0.3, 0.5, 0.7, 0.9])
            bars = ax3.bar(dd_metrics, dd_values, color=colors)
            ax3.set_ylabel("Percentage (%)", fontweight="bold")
            ax3.set_title("Drawdown Metrics", fontweight="bold")

            for bar, val in zip(bars, dd_values):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # 4. Risk Summary Dashboard
            ax4 = axes[1, 1]
            ax4.axis("off")

            # Risk score (simplified)
            risk_score = min(10, abs(rm.max_drawdown) / 5 + abs(rm.cvar_99) + abs(rm.kurtosis) / 2)

            summary_text = f"""
RISK DASHBOARD

Volatility (Annualized): {rm.volatility:.1f}%
Maximum Drawdown: {rm.max_drawdown:.1f}%
Max DD Duration: {rm.max_drawdown_duration} days

Daily VaR (99%): {rm.var_99:.2f}%
Expected Shortfall (99%): {rm.cvar_99:.2f}%

Skewness: {rm.skewness:.2f} {'(Negative - left tail risk)' if rm.skewness < 0 else '(Positive - right skewed)'}
Kurtosis: {rm.kurtosis:.2f} {'(Fat tails)' if rm.kurtosis > 0 else '(Thin tails)'}

Downside Deviation: {rm.downside_deviation:.1f}%

RISK SCORE: {risk_score:.1f}/10
{'🔴 HIGH RISK' if risk_score > 7 else '🟡 MEDIUM RISK' if risk_score > 4 else '🟢 LOW RISK'}
"""
            ax4.text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate risk metrics chart: {e}")
            return None

    def generate_regime_analysis_chart(
        self, result: ComprehensiveBacktestResult, filename: str = "regime_analysis.png"
    ) -> Optional[Path]:
        """Generate regime analysis chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            ra = result.regime_analysis
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 1. Time in Regimes
            ax1 = axes[0]
            regimes = ["Bull", "Bear", "Crisis"]
            times = [ra.time_in_bull, ra.time_in_bear, ra.time_in_crisis]
            colors = ["#4CAF50", "#E53935", "#9C27B0"]

            wedges, texts, autotexts = ax1.pie(
                times,
                labels=regimes,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                explode=(0.02, 0.02, 0.05),
            )
            ax1.set_title("Time in Market Regimes", fontweight="bold")

            # 2. Performance by Regime
            ax2 = axes[1]
            x = np.arange(3)
            width = 0.35

            returns = [ra.return_bull, ra.return_bear, ra.return_crisis]
            sharpes = [ra.sharpe_bull, ra.sharpe_bear, ra.sharpe_crisis]

            ax2.bar(x - width / 2, returns, width, label="Return (%)", color="#2196F3", alpha=0.8)
            ax2.bar(
                x + width / 2,
                [s * 10 for s in sharpes],
                width,
                label="Sharpe (x10)",
                color="#FF9800",
                alpha=0.8,
            )
            ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(regimes)
            ax2.set_ylabel("Value", fontweight="bold")
            ax2.set_title("Performance by Regime", fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis="y")

            # 3. Regime Summary
            ax3 = axes[2]
            ax3.axis("off")

            summary_text = f"""
REGIME ANALYSIS

Current Regime: {ra.current_regime.upper()}
Total Regime Changes: {ra.regime_changes}

BULL MARKET:
  Time: {ra.time_in_bull:.1f}%
  Return: {ra.return_bull:+.1f}%
  Sharpe: {ra.sharpe_bull:.2f}

BEAR MARKET:
  Time: {ra.time_in_bear:.1f}%
  Return: {ra.return_bear:+.1f}%
  Sharpe: {ra.sharpe_bear:.2f}

CRISIS:
  Time: {ra.time_in_crisis:.1f}%
  Return: {ra.return_crisis:+.1f}%
  Sharpe: {ra.sharpe_crisis:.2f}
"""
            ax3.text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate regime analysis chart: {e}")
            return None

    def generate_statistical_significance_chart(
        self, result: ComprehensiveBacktestResult, filename: str = "statistical_significance.png"
    ) -> Optional[Path]:
        """Generate statistical significance chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy import stats

            st = result.statistical_tests
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 1. Confidence Intervals
            ax1 = axes[0]

            # Sharpe CI
            sharpe_point = result.sharpe_ratio
            ax1.errorbar(
                1,
                sharpe_point,
                yerr=[[sharpe_point - st.sharpe_ci_lower], [st.sharpe_ci_upper - sharpe_point]],
                fmt="o",
                color="#2196F3",
                markersize=12,
                capsize=10,
                capthick=2,
                elinewidth=2,
            )
            ax1.axhline(0, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.7)
            ax1.axhline(1, color="#4CAF50", linestyle="--", linewidth=1.5, alpha=0.7)

            ax1.set_xlim(0.5, 1.5)
            ax1.set_xticks([1])
            ax1.set_xticklabels(["Sharpe Ratio"])
            ax1.set_ylabel("Value", fontweight="bold")
            ax1.set_title("95% Confidence Interval", fontweight="bold")
            ax1.grid(True, alpha=0.3, axis="y")

            ax1.text(1.1, sharpe_point, f"{sharpe_point:.2f}", fontweight="bold")
            ax1.text(1.1, st.sharpe_ci_lower, f"{st.sharpe_ci_lower:.2f}", fontsize=9, color="gray")
            ax1.text(1.1, st.sharpe_ci_upper, f"{st.sharpe_ci_upper:.2f}", fontsize=9, color="gray")

            # 2. P-value and Significance
            ax2 = axes[1]

            significance_levels = [0.01, 0.05, 0.10]
            sig_labels = [
                "p < 0.01\n(Very Significant)",
                "p < 0.05\n(Significant)",
                "p < 0.10\n(Marginally Sig.)",
            ]

            for i, (level, label) in enumerate(zip(significance_levels, sig_labels)):
                color = "#4CAF50" if st.p_value < level else "#E0E0E0"
                ax2.bar(i, 1, color=color, alpha=0.8, edgecolor="white", linewidth=2)
                ax2.text(
                    i,
                    0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if st.p_value < level else "normal",
                )

            ax2.axhline(0, color="black")
            ax2.set_xlim(-0.5, 2.5)
            ax2.set_ylim(0, 1.2)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f"Statistical Significance (p = {st.p_value:.4f})", fontweight="bold")

            # 3. Summary Dashboard
            ax3 = axes[2]
            ax3.axis("off")

            significance = "SIGNIFICANT" if st.p_value < 0.05 else "NOT SIGNIFICANT"
            sig_color = "🟢" if st.p_value < 0.05 else "🔴"

            summary_text = f"""
STATISTICAL ANALYSIS

T-Statistic: {st.t_statistic:.2f}
P-Value: {st.p_value:.4f}
Result: {sig_color} {significance}

Sharpe Ratio: {result.sharpe_ratio:.2f}
95% CI: [{st.sharpe_ci_lower:.2f}, {st.sharpe_ci_upper:.2f}]

P(Sharpe > 0): {st.prob_sharpe_zero*100:.1f}%

Deflated Sharpe: {st.deflated_sharpe:.2f}
(Adjusted for multiple testing)

Min Track Record: {st.min_track_record} days
(for statistical significance)
"""
            ax3.text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate statistical significance chart: {e}")
            return None


class ComprehensiveBacktestReportGenerator:
    """Generate world-class PDF reports from ComprehensiveBacktestResult."""

    def __init__(self) -> None:
        self.report_dir = Path.home() / "jotty" / "comprehensive_backtest_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    async def generate_report(
        self, result: ComprehensiveBacktestResult, template_name: str = "quantitative"
    ) -> Tuple[Path, Path]:
        """Generate comprehensive markdown and PDF report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.symbol}_comprehensive_{timestamp}"

        # Generate charts
        charts_dir = self.report_dir / "charts" / base_name
        chart_gen = ComprehensiveBacktestChartGenerator(charts_dir)

        chart_files = []

        mc_chart = chart_gen.generate_monte_carlo_distribution(result)
        if mc_chart:
            chart_files.append(mc_chart)

        wf_chart = chart_gen.generate_walk_forward_chart(result)
        if wf_chart:
            chart_files.append(wf_chart)

        risk_chart = chart_gen.generate_risk_metrics_chart(result)
        if risk_chart:
            chart_files.append(risk_chart)

        regime_chart = chart_gen.generate_regime_analysis_chart(result)
        if regime_chart:
            chart_files.append(regime_chart)

        stat_chart = chart_gen.generate_statistical_significance_chart(result)
        if stat_chart:
            chart_files.append(stat_chart)

        # Generate markdown
        markdown = self._generate_markdown(result)

        md_path = self.report_dir / f"{base_name}.md"
        with open(md_path, "w") as f:
            f.write(markdown)

        # Generate PDF
        pdf_path = await self._generate_pdf(markdown, base_name, chart_files, template_name)

        return md_path, pdf_path

    def _generate_markdown(self, result: ComprehensiveBacktestResult) -> str:
        """Generate comprehensive markdown report."""
        r = result
        rm = r.risk_metrics
        st = r.statistical_tests
        ra = r.regime_analysis
        fe = r.factor_exposure
        mc = r.monte_carlo
        ps = r.position_sizing
        tc = r.transaction_costs

        # Determine overall verdict
        is_significant = st.p_value < 0.05
        beats_benchmark = r.total_return > 0
        good_sharpe = r.sharpe_ratio > 1

        if is_significant and beats_benchmark and good_sharpe:
            verdict = "STRONG BUY"
            verdict_emoji = "🟢"
        elif is_significant and beats_benchmark:
            verdict = "BUY"
            verdict_emoji = "🟢"
        elif beats_benchmark:
            verdict = "HOLD"
            verdict_emoji = "🟡"
        else:
            verdict = "AVOID"
            verdict_emoji = "🔴"

        md = f"""
# Comprehensive ML Backtest Report

## {result.symbol} | {verdict_emoji} **{verdict}**

**Strategy:** {result.strategy_name or 'ML Strategy'} | **Period:** {result.start_date} to {result.end_date}

---

## Executive Summary

### Performance Overview

| Metric | Gross | Net (After Costs) | Benchmark |
|--------|------:|------------------:|----------:|
| Total Return | {r.total_return:+.2f}% | {r.total_return_net:+.2f}% | - |
| Annual Return | {r.annual_return:+.2f}% | {r.annual_return_net:+.2f}% | - |
| Sharpe Ratio | {r.sharpe_ratio:.2f} | {r.sharpe_ratio_net:.2f} | - |
| Sortino Ratio | {r.sortino_ratio:.2f} | - | - |
| Calmar Ratio | {r.calmar_ratio:.2f} | - | - |

### Confidence Intervals (95%)

| Metric | Point Estimate | Lower Bound | Upper Bound |
|--------|---------------:|------------:|------------:|
| Sharpe Ratio | {r.sharpe_ratio:.2f} | {st.sharpe_ci_lower:.2f} | {st.sharpe_ci_upper:.2f} |
| Annual Return | {r.annual_return:.2f}% | {st.return_ci_lower:.2f}% | {st.return_ci_upper:.2f}% |

### Statistical Significance

| Test | Value | Interpretation |
|------|------:|----------------|
| T-Statistic | {st.t_statistic:.2f} | {'Significant' if abs(st.t_statistic) > 1.96 else 'Not Significant'} |
| P-Value | {st.p_value:.4f} | {'Reject H0' if st.p_value < 0.05 else 'Fail to Reject H0'} |
| Probability Sharpe > 0 | {st.prob_sharpe_zero*100:.1f}% | {'High Confidence' if st.prob_sharpe_zero > 0.95 else 'Uncertain'} |
| Deflated Sharpe | {st.deflated_sharpe:.2f} | Adjusted for multiple testing |

---

## Monte Carlo Simulation

**Simulations Run:** {mc.n_simulations:,}

### Return Distribution

| Percentile | Value |
|------------|------:|
| 5th (Worst Case) | {mc.return_5th:+.2f}% |
| 25th | {(mc.mean_return - 0.67*mc.std_return):+.2f}% |
| 50th (Median) | {mc.median_return:+.2f}% |
| 75th | {(mc.mean_return + 0.67*mc.std_return):+.2f}% |
| 95th (Best Case) | {mc.return_95th:+.2f}% |

### Probability Metrics

| Outcome | Probability |
|---------|------------:|
| Positive Return | {mc.prob_positive*100:.1f}% |
| Beat Benchmark | {mc.prob_beat_benchmark*100:.1f}% |
| Sharpe > 1 | {mc.prob_sharpe_above_1*100:.1f}% |

### Sharpe Ratio Distribution

| Statistic | Value |
|-----------|------:|
| Mean | {mc.mean_sharpe:.2f} |
| Std Dev | {mc.std_sharpe:.2f} |
| 5th Percentile | {mc.sharpe_5th:.2f} |
| 95th Percentile | {mc.sharpe_95th:.2f} |

---

## Walk-Forward Analysis

**Windows Tested:** {len(r.walk_forward_results)}

### Summary Statistics

| Metric | Value |
|--------|------:|
| Average OOS Return | {r.wf_avg_oos_return:+.2f}% |
| Average OOS Sharpe | {r.wf_avg_oos_sharpe:.2f} |
| Average Degradation | {r.wf_degradation:.1f}% |
"""

        if r.walk_forward_results:
            md += """
### Window Details

| Window | IS Return | OOS Return | IS Sharpe | OOS Sharpe | Degradation |
|--------|----------:|-----------:|----------:|-----------:|------------:|
"""
            for i, wf in enumerate(r.walk_forward_results, 1):
                md += f"| W{i} | {wf.is_return:+.1f}% | {wf.oos_return:+.1f}% | {wf.is_sharpe:.2f} | {wf.oos_sharpe:.2f} | {wf.sharpe_degradation:.1f}% |\n"

        md += f"""
---

## Risk Analysis

### Value at Risk (VaR)

| Confidence Level | VaR (Daily) | CVaR (Expected Shortfall) |
|------------------|------------:|--------------------------:|
| 95% | {rm.var_95:.2f}% | {rm.cvar_95:.2f}% |
| 99% | {rm.var_99:.2f}% | {rm.cvar_99:.2f}% |

### Drawdown Metrics

| Metric | Value |
|--------|------:|
| Maximum Drawdown | {rm.max_drawdown:.2f}% |
| Average Drawdown | {rm.avg_drawdown:.2f}% |
| Max DD Duration | {rm.max_drawdown_duration} days |
| Ulcer Index | {rm.ulcer_index:.2f}% |
| Pain Index | {rm.pain_index:.2f}% |

### Distribution Characteristics

| Metric | Value | Interpretation |
|--------|------:|----------------|
| Volatility (Ann.) | {rm.volatility:.2f}% | {'High' if rm.volatility > 25 else 'Moderate' if rm.volatility > 15 else 'Low'} |
| Skewness | {rm.skewness:.2f} | {'Negative (left tail risk)' if rm.skewness < -0.5 else 'Positive (favorable)' if rm.skewness > 0.5 else 'Symmetric'} |
| Excess Kurtosis | {rm.kurtosis:.2f} | {'Fat tails (tail risk)' if rm.kurtosis > 1 else 'Thin tails' if rm.kurtosis < -1 else 'Normal'} |
| Tail Ratio | {rm.tail_ratio:.2f} | Ratio of gains to losses at extremes |
| Downside Deviation | {rm.downside_deviation:.2f}% | Risk of negative returns |

---

## Regime Analysis

### Time Allocation

| Regime | Time (%) | Return | Sharpe |
|--------|--------:|-------:|-------:|
| Bull Market | {ra.time_in_bull:.1f}% | {ra.return_bull:+.1f}% | {ra.sharpe_bull:.2f} |
| Bear Market | {ra.time_in_bear:.1f}% | {ra.return_bear:+.1f}% | {ra.sharpe_bear:.2f} |
| Crisis | {ra.time_in_crisis:.1f}% | {ra.return_crisis:+.1f}% | {ra.sharpe_crisis:.2f} |

**Current Regime:** {ra.current_regime.upper()}
**Regime Changes:** {ra.regime_changes}

---

## Factor Exposure

### Market Model

| Factor | Beta | Interpretation |
|--------|-----:|----------------|
| Market | {fe.market_beta:.2f} | {'Aggressive' if fe.market_beta > 1.2 else 'Defensive' if fe.market_beta < 0.8 else 'Neutral'} |
| Momentum | {fe.momentum_beta:.2f} | {'Trend follower' if fe.momentum_beta > 0.3 else 'Mean reverter' if fe.momentum_beta < -0.3 else 'Neutral'} |
| Volatility | {fe.volatility_beta:.2f} | {'Vol seeker' if fe.volatility_beta > 0.3 else 'Vol avoider' if fe.volatility_beta < -0.3 else 'Neutral'} |

### Alpha Analysis

| Metric | Value |
|--------|------:|
| Jensen's Alpha (Ann.) | {fe.alpha:+.2f}% |
| Alpha T-Statistic | {fe.alpha_t_stat:.2f} |
| Alpha P-Value | {fe.alpha_p_value:.4f} |
| R-Squared | {fe.r_squared:.2f} |
| Information Ratio | {fe.information_ratio:.2f} |
| Tracking Error | {fe.tracking_error:.2f}% |

---

## Position Sizing

### Optimal Allocation

| Method | Recommended |
|--------|------------:|
| Kelly Criterion | {ps.kelly_fraction*100:.1f}% |
| Half Kelly (Conservative) | {ps.half_kelly*100:.1f}% |
| Optimal Leverage | {ps.optimal_leverage:.2f}x |

### Volatility Targeting

| Metric | Value |
|--------|------:|
| Target Volatility | {ps.target_volatility*100:.1f}% |
| Current Volatility | {ps.current_volatility*100:.1f}% |
| Volatility Scalar | {ps.vol_scalar:.2f}x |

---

## Transaction Costs

### Cost Model

| Component | Rate |
|-----------|-----:|
| Commission | {tc.commission_pct*100:.3f}% |
| Slippage | {tc.slippage_pct*100:.3f}% |
| Market Impact | {tc.market_impact_pct*100:.3f}% |
| **Total Round-Trip** | **{tc.total_cost_pct()*100:.3f}%** |

### Impact Analysis

| Metric | Gross | Net | Impact |
|--------|------:|----:|-------:|
| Total Return | {r.total_return:+.2f}% | {r.total_return_net:+.2f}% | {r.total_return - r.total_return_net:.2f}% |
| Sharpe Ratio | {r.sharpe_ratio:.2f} | {r.sharpe_ratio_net:.2f} | {r.sharpe_ratio - r.sharpe_ratio_net:.2f} |
| Total Trades | {r.total_trades:,} | - | - |
| Total Costs Paid | - | {r.total_costs_paid*100:.2f}% | - |

---

## Trade Statistics

| Metric | Value |
|--------|------:|
| Total Trades | {r.total_trades:,} |
| Win Rate | {r.win_rate:.1f}% |
| Profit Factor | {r.profit_factor:.2f} |
| Expectancy | {r.expectancy:+.3f}% |

---

## Investment Recommendation

### Rating: {verdict}

### Key Strengths
"""

        if is_significant:
            md += "✓ Statistically significant alpha\n"
        if good_sharpe:
            md += f"✓ Strong risk-adjusted returns (Sharpe = {r.sharpe_ratio:.2f})\n"
        if mc.prob_positive > 0.8:
            md += f"✓ High probability of positive return ({mc.prob_positive*100:.0f}%)\n"
        if r.wf_degradation < 30:
            md += "✓ Consistent out-of-sample performance\n"
        if abs(rm.max_drawdown) < 15:
            md += f"✓ Controlled drawdown ({rm.max_drawdown:.1f}%)\n"

        md += """
### Key Risks
"""

        if not is_significant:
            md += f"⚠ Not statistically significant (p = {st.p_value:.3f})\n"
        if r.wf_degradation > 50:
            md += f"⚠ High walk-forward degradation ({r.wf_degradation:.0f}%)\n"
        if abs(rm.max_drawdown) > 20:
            md += f"⚠ Large maximum drawdown ({rm.max_drawdown:.1f}%)\n"
        if rm.skewness < -0.5:
            md += f"⚠ Negative skewness indicates tail risk\n"
        if rm.kurtosis > 2:
            md += "⚠ Fat tails increase extreme event risk\n"

        md += f"""
---

## Disclaimer

This comprehensive backtest report is generated by Jotty ML for informational purposes only.
Past performance is not indicative of future results. Backtests may suffer from overfitting,
look-ahead bias, and survivorship bias. Monte Carlo simulations assume returns are IID.
Walk-forward analysis provides more realistic OOS estimates but cannot guarantee future performance.

**Report Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}

**Analyst:** Jotty AI Quantitative Research

---

*© {datetime.now().year} Jotty Research. All rights reserved.*
"""

        return md

    async def _generate_pdf(
        self, markdown: str, base_name: str, chart_files: List[Path], template_name: str
    ) -> Path:
        """Convert markdown to PDF with charts."""
        try:
            import base64

            from ..research.templates import TemplateRegistry

            # Get template
            template = TemplateRegistry.get(template_name) or TemplateRegistry.get("quantitative")
            if template is None:
                from ..research.templates.backtest_templates import QuantitativeTemplate

                template = QuantitativeTemplate()

            # Convert markdown to HTML
            try:
                import markdown as md_lib

                md_converter = md_lib.Markdown(
                    extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
                )
                html_content = md_converter.convert(markdown)
            except ImportError:
                html_content = f"<pre>{markdown}</pre>"

            # Enhance HTML
            import re

            html_content = re.sub(r"<h2>", '<div class="page-break"></div>\n<h2>', html_content)
            html_content = html_content.replace('<div class="page-break"></div>\n<h2>', "<h2>", 1)

            # Embed charts
            if chart_files:
                charts_html = '<div class="page-break"></div>\n<h2>Charts & Visualizations</h2>\n'
                for chart_path in chart_files:
                    if chart_path and chart_path.exists():
                        try:
                            with open(chart_path, "rb") as f:
                                data = base64.b64encode(f.read()).decode("utf-8")
                            chart_name = chart_path.stem.replace("_", " ").title()
                            charts_html += f"""
<div class="chart-container">
    <h3>{chart_name}</h3>
    <img src="data:image/png;base64,{data}" style="width:100%;max-width:100%;height:auto;"/>
</div>
"""
                        except Exception as e:
                            logger.warning(f"Failed to embed chart: {e}")

                if "## Disclaimer" in html_content:
                    html_content = html_content.replace(
                        "<h2>Disclaimer</h2>",
                        charts_html + '<div class="page-break"></div>\n<h2>Disclaimer</h2>',
                    )
                else:
                    html_content += charts_html

            # Wrap with template
            html = template.get_html_wrapper(html_content, f"Comprehensive Backtest - {base_name}")

            # Generate PDF
            pdf_path = self.report_dir / f"{base_name}.pdf"

            try:
                from weasyprint import HTML

                HTML(string=html).write_pdf(str(pdf_path))
            except ImportError:
                logger.warning("WeasyPrint not installed")
                return None

            return pdf_path

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def send_to_telegram(self, pdf_path: Path, result: "ComprehensiveBacktestResult") -> bool:
        """Send comprehensive report to Telegram."""
        try:
            import os

            from telegram import Bot

            # Telegram config from environment variables
            BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
            if not BOT_TOKEN:
                logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
                return

            REPORTS_CHANNEL = os.getenv("TELEGRAM_REPORTS_CHANNEL", "-1001723570817")

            bot = Bot(token=BOT_TOKEN)

            # Build caption
            verdict = (
                "BUY"
                if result.sharpe_ratio > 1 and result.statistical_tests.p_value < 0.05
                else "HOLD" if result.total_return > 0 else "AVOID"
            )
            caption = f"""📊 **Comprehensive ML Backtest: {result.symbol}**

📈 Return: {result.total_return:+.1f}% (Net: {result.total_return_net:+.1f}%)
📉 Sharpe: {result.sharpe_ratio:.2f} | Sortino: {result.sortino_ratio:.2f}
🎯 P(Positive): {result.monte_carlo.prob_positive*100:.0f}%
📊 Rating: **{verdict}**

✅ Walk-Forward | Monte Carlo | VaR/CVaR
✅ Regime Analysis | Factor Exposure
✅ Statistical Tests | Position Sizing

🤖 Generated by Jotty AI"""

            with open(pdf_path, "rb") as f:
                await bot.send_document(
                    chat_id=REPORTS_CHANNEL, document=f, caption=caption, parse_mode="Markdown"
                )

            logger.info(f"Sent {result.symbol} report to Telegram")
            return True

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
