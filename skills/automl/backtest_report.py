"""
ML Backtest Report Generator
============================

World-class ML backtesting PDF reports inspired by top quantitative research firms
(Two Sigma, Renaissance Technologies, AQR, Man Group, Citadel).

Generates comprehensive PDF reports with:
- Executive Summary with key metrics
- Strategy Performance Analysis
- Risk-Adjusted Returns
- Drawdown Analysis
- Feature Importance
- Model Comparison
- Trade Statistics
- Equity Curve Charts
- Rolling Performance Charts
- Distribution Analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Core backtest performance metrics."""

    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    romad: float = 0.0  # Return Over Max Drawdown
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sqn: float = 0.0  # System Quality Number


@dataclass
class TradeStatistics:
    """Trade-level statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    expectancy: float = 0.0  # Expected value per trade


@dataclass
class ModelResults:
    """Results for a single ML model."""

    name: str = ""
    accuracy: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    r2: float = 0.0
    rmse: float = 0.0
    is_best: bool = False
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest result container."""

    symbol: str = ""
    target_type: str = ""
    target_days: int = 1
    timeframe: str = "day"
    problem_type: str = "classification"

    # Date range
    start_date: str = ""
    end_date: str = ""
    trading_days: int = 0

    # Strategy metrics
    strategy_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    benchmark_metrics: BacktestMetrics = field(default_factory=BacktestMetrics)

    # Trade stats
    trade_stats: TradeStatistics = field(default_factory=TradeStatistics)

    # Model results
    models: List[ModelResults] = field(default_factory=list)
    best_model: str = ""

    # Feature importance (from best model)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Equity curve data
    equity_curve: List[Dict] = field(default_factory=list)

    # Rolling metrics
    rolling_sharpe: List[Dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    yearly_returns: Dict[str, float] = field(default_factory=dict)


class BacktestChartGenerator:
    """Generate charts for backtest reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_equity_curve(
        self, result: BacktestResult, filename: str = "equity_curve.png"
    ) -> Optional[Path]:
        """Generate equity curve chart comparing strategy vs benchmark."""
        try:
            from datetime import datetime

            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            if not result.equity_curve:
                return None

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
            )

            dates = [
                (
                    datetime.strptime(d["date"], "%Y-%m-%d")
                    if isinstance(d["date"], str)
                    else d["date"]
                )
                for d in result.equity_curve
            ]
            strategy = [d.get("strategy", 1.0) for d in result.equity_curve]
            benchmark = [d.get("benchmark", 1.0) for d in result.equity_curve]
            drawdown = [d.get("drawdown", 0.0) for d in result.equity_curve]

            # Equity curve
            ax1.fill_between(dates, 1, strategy, alpha=0.3, color="#2196F3", label="Strategy")
            ax1.plot(
                dates,
                strategy,
                color="#1565C0",
                linewidth=2,
                label=f"Strategy ({(strategy[-1]-1)*100:.1f}%)",
            )
            ax1.plot(
                dates,
                benchmark,
                color="#757575",
                linewidth=1.5,
                linestyle="--",
                label=f"Buy & Hold ({(benchmark[-1]-1)*100:.1f}%)",
            )

            ax1.set_ylabel("Cumulative Return", fontsize=11, fontweight="bold")
            ax1.legend(loc="upper left", frameon=True, fancybox=True)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(
                f"{result.symbol} - Strategy Performance", fontsize=14, fontweight="bold", pad=10
            )

            # Drawdown
            ax2.fill_between(dates, 0, drawdown, color="#E53935", alpha=0.6)
            ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Date", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            fig.autofmt_xdate()

            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate equity curve: {e}")
            return None

    def generate_monthly_returns_heatmap(
        self, result: BacktestResult, filename: str = "monthly_returns.png"
    ) -> Optional[Path]:
        """Generate monthly returns heatmap."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not result.monthly_returns:
                return None

            # Parse monthly returns into year/month grid
            months = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            years = sorted(set(k[:4] for k in result.monthly_returns.keys()))

            data = np.zeros((len(years), 12))
            data[:] = np.nan

            for key, value in result.monthly_returns.items():
                year_idx = years.index(key[:4])
                month_idx = int(key[5:7]) - 1
                data[year_idx, month_idx] = value

            fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.5 + 1)))

            # Custom colormap: red for negative, green for positive
            cmap = plt.cm.RdYlGn
            vmax = (
                max(abs(np.nanmin(data)), abs(np.nanmax(data)))
                if not np.all(np.isnan(data))
                else 10
            )

            im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

            ax.set_xticks(np.arange(12))
            ax.set_yticks(np.arange(len(years)))
            ax.set_xticklabels(months)
            ax.set_yticklabels(years)

            # Add text annotations
            for i in range(len(years)):
                for j in range(12):
                    if not np.isnan(data[i, j]):
                        color = "white" if abs(data[i, j]) > vmax * 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{data[i, j]:.1f}%",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                            fontweight="bold",
                        )

            plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.8)
            ax.set_title(
                f"{result.symbol} - Monthly Returns Heatmap", fontsize=14, fontweight="bold", pad=10
            )

            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate monthly returns heatmap: {e}")
            return None

    def generate_feature_importance(
        self, result: BacktestResult, filename: str = "feature_importance.png"
    ) -> Optional[Path]:
        """Generate feature importance horizontal bar chart."""
        try:
            import matplotlib.pyplot as plt

            if not result.feature_importance:
                return None

            # Get top 20 features
            sorted_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:20]
            features = [f[0] for f in sorted_features][::-1]
            importance = [f[1] for f in sorted_features][::-1]

            fig, ax = plt.subplots(figsize=(10, 8))

            colors = plt.cm.Blues([(i / len(features)) * 0.6 + 0.4 for i in range(len(features))])
            bars = ax.barh(features, importance, color=colors, edgecolor="#1565C0", linewidth=0.5)

            ax.set_xlabel("Importance (%)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{result.symbol} - Feature Importance ({result.best_model})",
                fontsize=14,
                fontweight="bold",
                pad=10,
            )
            ax.grid(True, alpha=0.3, axis="x")

            # Add value labels
            for bar, val in zip(bars, importance):
                ax.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=9,
                )

            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate feature importance chart: {e}")
            return None

    def generate_returns_distribution(
        self, result: BacktestResult, filename: str = "returns_distribution.png"
    ) -> Optional[Path]:
        """Generate returns distribution histogram."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not result.equity_curve or len(result.equity_curve) < 2:
                return None

            # Calculate daily returns from equity curve
            strategy_values = [d.get("strategy", 1.0) for d in result.equity_curve]
            returns = [
                (strategy_values[i] / strategy_values[i - 1] - 1) * 100
                for i in range(1, len(strategy_values))
            ]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram
            n, bins, patches = ax.hist(
                returns, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.5
            )

            # Color negative returns red
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor("#E53935")

            # Add statistics
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)

            ax.axvline(
                mean_ret,
                color="#1B5E20",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_ret:.2f}%",
            )
            ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            ax.set_xlabel("Daily Return (%)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{result.symbol} - Returns Distribution", fontsize=14, fontweight="bold", pad=10
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # Add text box with stats
            stats_text = (
                f"Mean: {mean_ret:.2f}%\nStd: {std_ret:.2f}%\nSkew: {self._calc_skew(returns):.2f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate returns distribution: {e}")
            return None

    def generate_model_comparison(
        self, result: BacktestResult, filename: str = "model_comparison.png"
    ) -> Optional[Path]:
        """Generate model comparison radar chart."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not result.models or len(result.models) < 2:
                return None

            is_classification = result.problem_type == "classification"

            if is_classification:
                # Radar chart for classification metrics
                categories = ["Accuracy", "F1 Score", "AUC", "Win Rate", "Profit Factor"]

                fig, ax = plt.subplots(figsize=(10, 6))

                x = np.arange(len(categories))
                width = 0.8 / len(result.models)

                colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63"]

                for i, model in enumerate(result.models):
                    values = [
                        model.accuracy * 100,
                        model.f1_score * 100,
                        model.auc * 100,
                        (
                            result.strategy_metrics.win_rate
                            if model.is_best
                            else result.strategy_metrics.win_rate * 0.9
                        ),
                        (
                            min(result.strategy_metrics.profit_factor * 20, 100)
                            if model.is_best
                            else min(result.strategy_metrics.profit_factor * 18, 90)
                        ),
                    ]

                    bars = ax.bar(
                        x + i * width - width * len(result.models) / 2,
                        values,
                        width,
                        label=f'{model.name}{"*" if model.is_best else ""}',
                        color=colors[i % len(colors)],
                        alpha=0.8,
                    )

                ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.legend(loc="upper right")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3, axis="y")
                ax.set_title(
                    f"{result.symbol} - Model Comparison (Classification)",
                    fontsize=14,
                    fontweight="bold",
                    pad=10,
                )
            else:
                # Bar chart for regression metrics
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                models = [m.name for m in result.models]
                r2_scores = [m.r2 for m in result.models]
                rmse_scores = [m.rmse for m in result.models]

                colors = ["#4CAF50" if m.is_best else "#2196F3" for m in result.models]

                ax1.barh(models, r2_scores, color=colors, alpha=0.8)
                ax1.set_xlabel("R² Score", fontsize=11, fontweight="bold")
                ax1.set_title("R² Comparison", fontsize=12, fontweight="bold")
                ax1.grid(True, alpha=0.3, axis="x")

                ax2.barh(models, rmse_scores, color=colors, alpha=0.8)
                ax2.set_xlabel("RMSE", fontsize=11, fontweight="bold")
                ax2.set_title("RMSE Comparison (lower is better)", fontsize=12, fontweight="bold")
                ax2.grid(True, alpha=0.3, axis="x")

                fig.suptitle(
                    f"{result.symbol} - Model Comparison (Regression)",
                    fontsize=14,
                    fontweight="bold",
                )

            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate model comparison: {e}")
            return None

    def generate_rolling_metrics(
        self, result: BacktestResult, filename: str = "rolling_metrics.png"
    ) -> Optional[Path]:
        """Generate rolling Sharpe ratio and volatility chart."""
        try:
            from datetime import datetime

            import matplotlib.pyplot as plt
            import numpy as np

            if not result.equity_curve or len(result.equity_curve) < 60:
                return None

            dates = [
                (
                    datetime.strptime(d["date"], "%Y-%m-%d")
                    if isinstance(d["date"], str)
                    else d["date"]
                )
                for d in result.equity_curve
            ]
            strategy = [d.get("strategy", 1.0) for d in result.equity_curve]

            # Calculate rolling returns and metrics (60-day window)
            window = 60
            returns = [(strategy[i] / strategy[i - 1] - 1) for i in range(1, len(strategy))]

            rolling_mean = []
            rolling_std = []
            rolling_sharpe = []

            for i in range(window, len(returns)):
                window_returns = returns[i - window : i]
                mean_ret = np.mean(window_returns) * 252  # Annualized
                std_ret = np.std(window_returns) * np.sqrt(252)  # Annualized
                sharpe = mean_ret / std_ret if std_ret > 0 else 0

                rolling_mean.append(mean_ret * 100)
                rolling_std.append(std_ret * 100)
                rolling_sharpe.append(sharpe)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            plot_dates = dates[window + 1 :]

            # Rolling Sharpe
            ax1.fill_between(
                plot_dates,
                0,
                rolling_sharpe,
                where=[s > 0 for s in rolling_sharpe],
                color="#4CAF50",
                alpha=0.3,
            )
            ax1.fill_between(
                plot_dates,
                0,
                rolling_sharpe,
                where=[s <= 0 for s in rolling_sharpe],
                color="#E53935",
                alpha=0.3,
            )
            ax1.plot(plot_dates, rolling_sharpe, color="#1565C0", linewidth=1.5)
            ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax1.axhline(
                y=1, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7, label="Sharpe = 1"
            )
            ax1.axhline(
                y=2, color="#1B5E20", linestyle="--", linewidth=1, alpha=0.7, label="Sharpe = 2"
            )
            ax1.set_ylabel("Rolling Sharpe (60d)", fontsize=11, fontweight="bold")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)
            ax1.set_title(
                f"{result.symbol} - Rolling Risk Metrics", fontsize=14, fontweight="bold", pad=10
            )

            # Rolling Volatility
            ax2.fill_between(plot_dates, 0, rolling_std, color="#FF9800", alpha=0.4)
            ax2.plot(plot_dates, rolling_std, color="#E65100", linewidth=1.5)
            ax2.set_ylabel("Rolling Volatility (60d, Ann.)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Date", fontsize=11)
            ax2.grid(True, alpha=0.3)

            fig.autofmt_xdate()
            plt.tight_layout()

            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()

            return filepath

        except Exception as e:
            logger.error(f"Failed to generate rolling metrics: {e}")
            return None

    def _calc_skew(self, returns: List[float]) -> float:
        """Calculate skewness of returns."""
        import numpy as np

        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((np.array(returns) - mean) / std) ** 3)


class BacktestReportGenerator:
    """Generate comprehensive ML backtest PDF reports."""

    def __init__(self) -> None:
        self.report_dir = Path.home() / "jotty" / "backtest_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    async def generate_report(
        self, result: BacktestResult, template_name: str = "quantitative"
    ) -> Tuple[Path, Path]:
        """Generate markdown and PDF backtest report.

        Returns:
            Tuple of (markdown_path, pdf_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.symbol}_backtest_{timestamp}"

        # Generate charts
        charts_dir = self.report_dir / "charts" / base_name
        chart_gen = BacktestChartGenerator(charts_dir)

        chart_files = []

        # Generate all charts
        equity_chart = chart_gen.generate_equity_curve(result)
        if equity_chart:
            chart_files.append(equity_chart)

        monthly_chart = chart_gen.generate_monthly_returns_heatmap(result)
        if monthly_chart:
            chart_files.append(monthly_chart)

        feature_chart = chart_gen.generate_feature_importance(result)
        if feature_chart:
            chart_files.append(feature_chart)

        dist_chart = chart_gen.generate_returns_distribution(result)
        if dist_chart:
            chart_files.append(dist_chart)

        model_chart = chart_gen.generate_model_comparison(result)
        if model_chart:
            chart_files.append(model_chart)

        rolling_chart = chart_gen.generate_rolling_metrics(result)
        if rolling_chart:
            chart_files.append(rolling_chart)

        # Generate markdown
        markdown = self._generate_markdown(result)

        md_path = self.report_dir / f"{base_name}.md"
        with open(md_path, "w") as f:
            f.write(markdown)

        # Generate PDF
        pdf_path = await self._generate_pdf(markdown, base_name, chart_files, template_name)

        return md_path, pdf_path

    def _generate_markdown(self, result: BacktestResult) -> str:
        """Generate markdown content for backtest report."""
        sm = result.strategy_metrics
        bm = result.benchmark_metrics
        ts = result.trade_stats

        # Performance verdict
        outperformance = sm.total_return - bm.total_return
        verdict = "OUTPERFORMS" if outperformance > 0 else "UNDERPERFORMS"
        verdict_emoji = "" if outperformance > 5 else "" if outperformance > -5 else ""

        md = f"""
# ML Backtest Report: {result.symbol}

### {result.symbol} | {verdict_emoji} **{verdict}** Benchmark by {abs(outperformance):.1f}%

**Strategy:** ML {result.problem_type.title()} | **Target:** {result.target_days}-Day | **Timeframe:** {result.timeframe.title()}

---

## Executive Summary

| **Strategy Metrics** | Value | **Benchmark (B&H)** | Value |
|---------------------|------:|---------------------|------:|
| Total Return | {sm.total_return:+.2f}% | Total Return | {bm.total_return:+.2f}% |
| Annual Return | {sm.annual_return:+.2f}% | Annual Return | {bm.annual_return:+.2f}% |
| Sharpe Ratio | {sm.sharpe_ratio:.2f} | Sharpe Ratio | {bm.sharpe_ratio:.2f} |
| Max Drawdown | {sm.max_drawdown:.2f}% | Max Drawdown | {bm.max_drawdown:.2f}% |
| ROMAD | {sm.romad:.2f} | ROMAD | {bm.romad:.2f} |

### Performance Snapshot

| Metric | Strategy | Benchmark | Alpha |
|--------|--------:|--------:|------:|
| Total Return | {sm.total_return:+.2f}% | {bm.total_return:+.2f}% | {outperformance:+.2f}% |
| Volatility (Ann.) | {sm.volatility:.2f}% | {bm.volatility:.2f}% | {sm.volatility - bm.volatility:+.2f}% |
| Sharpe Ratio | {sm.sharpe_ratio:.2f} | {bm.sharpe_ratio:.2f} | {sm.sharpe_ratio - bm.sharpe_ratio:+.2f} |
| Sortino Ratio | {sm.sortino_ratio:.2f} | {bm.sortino_ratio:.2f} | {sm.sortino_ratio - bm.sortino_ratio:+.2f} |

---

## Strategy Overview

### Configuration

| Parameter | Value |
|-----------|------:|
| Symbol | {result.symbol} |
| Target Type | {result.target_type} |
| Prediction Horizon | {result.target_days} days |
| Problem Type | {result.problem_type.title()} |
| Timeframe | {result.timeframe.title()} |
| Period | {result.start_date} to {result.end_date} |
| Trading Days | {result.trading_days:,} |
| Best Model | {result.best_model} |

---

## Risk-Adjusted Performance

### Key Ratios

| Ratio | Strategy | Benchmark | Interpretation |
|-------|--------:|--------:|----------------|
| Sharpe Ratio | {sm.sharpe_ratio:.2f} | {bm.sharpe_ratio:.2f} | {"Excellent" if sm.sharpe_ratio > 2 else "Good" if sm.sharpe_ratio > 1 else "Fair" if sm.sharpe_ratio > 0.5 else "Poor"} |
| Sortino Ratio | {sm.sortino_ratio:.2f} | {bm.sortino_ratio:.2f} | {"Excellent" if sm.sortino_ratio > 2 else "Good" if sm.sortino_ratio > 1 else "Fair"} |
| Calmar Ratio | {sm.calmar_ratio:.2f} | {bm.calmar_ratio:.2f} | Return/MaxDD |
| ROMAD | {sm.romad:.2f} | {bm.romad:.2f} | Return over Max Adverse Drawdown |

### Risk Metrics

| Metric | Strategy | Benchmark |
|--------|--------:|--------:|
| Max Drawdown | {sm.max_drawdown:.2f}% | {bm.max_drawdown:.2f}% |
| Avg Drawdown | {sm.avg_drawdown:.2f}% | {bm.avg_drawdown:.2f}% |
| Volatility (Ann.) | {sm.volatility:.2f}% | {bm.volatility:.2f}% |
| Win Rate | {sm.win_rate:.1f}% | - |
| Profit Factor | {sm.profit_factor:.2f} | - |

---

## Trade Statistics

### Overview

| Metric | Value |
|--------|------:|
| Total Trades | {ts.total_trades:,} |
| Winning Trades | {ts.winning_trades:,} |
| Losing Trades | {ts.losing_trades:,} |
| Win Rate | {sm.win_rate:.1f}% |

### Trade Analysis

| Metric | Value |
|--------|------:|
| Average Win | {ts.avg_win:+.2f}% |
| Average Loss | {ts.avg_loss:.2f}% |
| Largest Win | {ts.largest_win:+.2f}% |
| Largest Loss | {ts.largest_loss:.2f}% |
| Expectancy | {ts.expectancy:+.3f}% |
| Profit Factor | {sm.profit_factor:.2f} |

### Consistency Metrics

| Metric | Value |
|--------|------:|
| Consecutive Wins | {ts.consecutive_wins} |
| Consecutive Losses | {ts.consecutive_losses} |
| SQN (System Quality) | {sm.sqn:.2f} |

---

## Model Performance

### Model Comparison
"""

        if result.problem_type == "classification":
            md += """
| Model | Accuracy | F1 Score | AUC | Best |
|-------|--------:|--------:|----:|:----:|
"""
            for model in result.models:
                best_marker = "" if model.is_best else ""
                md += f"| {model.name} | {model.accuracy*100:.2f}% | {model.f1_score*100:.2f}% | {model.auc*100:.2f}% | {best_marker} |\n"
        else:
            md += """
| Model | R² Score | RMSE | Best |
|-------|--------:|-----:|:----:|
"""
            for model in result.models:
                best_marker = "" if model.is_best else ""
                md += f"| {model.name} | {model.r2:.4f} | {model.rmse:.4f} | {best_marker} |\n"

        md += f"""
### Best Model: {result.best_model}

---

## Feature Importance

### Top 15 Predictive Features

| Rank | Feature | Importance |
|-----:|---------|----------:|
"""

        sorted_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:15]
        for i, (feat, imp) in enumerate(sorted_features, 1):
            md += f"| {i} | {feat} | {imp:.2f}% |\n"

        md += """
---

## Scenario Analysis

### Performance Scenarios

| Scenario | Probability | Expected Return | Risk Level |
|----------|:-----------:|---------------:|:----------:|
"""

        # Calculate scenarios based on volatility
        base_return = sm.annual_return
        vol = sm.volatility

        bull_return = base_return + vol * 1.5
        bear_return = base_return - vol * 1.5

        md += f"| **BULL** | 25% | {bull_return:+.1f}% | Low |\n"
        md += f"| **BASE** | 50% | {base_return:+.1f}% | Medium |\n"
        md += f"| **BEAR** | 25% | {bear_return:+.1f}% | High |\n"

        weighted_return = 0.25 * bull_return + 0.50 * base_return + 0.25 * bear_return
        md += f"| **Weighted Avg** | 100% | **{weighted_return:+.1f}%** | - |\n"

        md += f"""
---

## Investment Recommendation

### Rating: {"BUY" if outperformance > 10 and sm.sharpe_ratio > 1 else "HOLD" if outperformance > 0 else "AVOID"}

### Key Takeaways

1. **Performance:** Strategy {"outperforms" if outperformance > 0 else "underperforms"} buy-and-hold by {abs(outperformance):.1f}%
2. **Risk:** {"Lower" if sm.max_drawdown > bm.max_drawdown else "Higher"} maximum drawdown than benchmark ({sm.max_drawdown:.1f}% vs {bm.max_drawdown:.1f}%)
3. **Consistency:** Win rate of {sm.win_rate:.1f}% with profit factor of {sm.profit_factor:.2f}
4. **Best Model:** {result.best_model} achieved highest predictive accuracy

### Strategy Strengths

"""

        if sm.sharpe_ratio > 1:
            md += " Strong risk-adjusted returns (Sharpe > 1)\n"
        if sm.win_rate > 50:
            md += f" Positive win rate ({sm.win_rate:.1f}%)\n"
        if sm.profit_factor > 1.5:
            md += f" Good profit factor ({sm.profit_factor:.2f})\n"
        if outperformance > 0:
            md += f" Outperforms benchmark by {outperformance:.1f}%\n"
        if sm.max_drawdown > bm.max_drawdown * 0.8:
            md += " Controlled drawdown vs benchmark\n"

        md += """
### Key Risks

"""

        if sm.max_drawdown < -20:
            md += f" Significant drawdown risk ({sm.max_drawdown:.1f}%)\n"
        if sm.win_rate < 50:
            md += f" Win rate below 50% ({sm.win_rate:.1f}%)\n"
        if sm.sharpe_ratio < 0.5:
            md += " Low risk-adjusted returns\n"
        if outperformance < 0:
            md += f" Underperforms buy-and-hold by {abs(outperformance):.1f}%\n"

        md += f"""
---

## Disclaimer

This backtest report is generated by Jotty ML for informational purposes only and does not constitute
investment advice. Past performance is not indicative of future results. Backtests may suffer from
overfitting, look-ahead bias, and other statistical artifacts.

**Report Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}

**Analyst:** Jotty AI Research

---

* {datetime.now().year} Jotty Research. All rights reserved.*
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
                # Fallback to default quantitative template
                from ..research.templates.backtest_templates import QuantitativeTemplate

                template = QuantitativeTemplate()

            # Convert markdown to HTML first
            try:
                import markdown as md_lib

                md_converter = md_lib.Markdown(
                    extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
                )
                html_content = md_converter.convert(markdown)
            except ImportError:
                # Basic fallback conversion
                html_content = self._basic_markdown_to_html(markdown)

            # Enhance HTML with page breaks and styling
            html_content = self._enhance_backtest_html(html_content)

            # Embed charts as base64
            if chart_files:
                charts_html = '<div class="page-break"></div>\n<h2>Charts & Visualizations</h2>\n'
                for chart_path in chart_files:
                    if chart_path and chart_path.exists() and chart_path.stat().st_size > 1000:
                        try:
                            with open(chart_path, "rb") as f:
                                data = base64.b64encode(f.read()).decode("utf-8")
                            chart_name = chart_path.stem.replace("_", " ").title()
                            charts_html += f"""
<div class="chart-container">
    <h3>{chart_name}</h3>
    <img src="data:image/png;base64,{data}" alt="{chart_name}" style="width:100%;max-width:100%;height:auto;"/>
</div>
"""
                        except Exception as e:
                            logger.warning(f"Failed to embed chart {chart_path}: {e}")

                # Insert charts before disclaimer
                if "## Disclaimer" in html_content:
                    html_content = html_content.replace(
                        "<h2>Disclaimer</h2>",
                        charts_html + '<div class="page-break"></div>\n<h2>Disclaimer</h2>',
                    )
                else:
                    html_content += charts_html

            # Wrap with template
            html = template.get_html_wrapper(html_content, f"Backtest Report - {base_name}")

            # Generate PDF
            pdf_path = self.report_dir / f"{base_name}.pdf"

            try:
                from weasyprint import HTML

                HTML(string=html).write_pdf(str(pdf_path))
            except ImportError:
                logger.warning("WeasyPrint not installed. PDF generation skipped.")
                return None

            return pdf_path

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _basic_markdown_to_html(self, markdown: str) -> str:
        """Basic markdown to HTML conversion fallback."""
        import re

        html = markdown

        # Headers
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Links
        html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)

        # Horizontal rules
        html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r"\n\n", "</p><p>", html)
        html = f"<p>{html}</p>"

        return html

    def _enhance_backtest_html(self, html: str) -> str:
        """Enhance HTML with backtest-specific styling."""
        import re

        # Add page breaks before major sections
        html = re.sub(r"<h2>", '<div class="page-break"></div>\n<h2>', html)
        html = html.replace('<div class="page-break"></div>\n<h2>', "<h2>", 1)  # Remove first

        # Style positive/negative values
        html = re.sub(r"(\+\d+\.?\d*%)", r'<span class="positive">\1</span>', html)
        html = re.sub(r"(-\d+\.?\d*%)", r'<span class="negative">\1</span>', html)

        # Add avoid-break to tables
        html = re.sub(r"<table>", '<table class="avoid-break">', html)

        return html


class BacktestReportSkill:
    """Jotty skill for generating backtest reports."""

    name = "backtest_report"
    description = "Generate comprehensive ML backtest PDF reports"

    def __init__(self) -> None:
        self.generator = BacktestReportGenerator()

    async def execute(
        self, result: BacktestResult, template: str = "quantitative", send_telegram: bool = True
    ) -> Dict[str, Any]:
        """Execute the skill to generate a backtest report.

        Args:
            result: BacktestResult containing all backtest data
            template: Template name for PDF styling
            send_telegram: Whether to send the report to Telegram

        Returns:
            Dict with paths to generated files and status
        """
        try:
            md_path, pdf_path = await self.generator.generate_report(result, template)

            telegram_sent = False
            if send_telegram and pdf_path:
                telegram_sent = await self._send_to_telegram(pdf_path, result)

            return {
                "status": "success",
                "markdown_path": str(md_path) if md_path else None,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "symbol": result.symbol,
                "total_return": result.strategy_metrics.total_return,
                "sharpe_ratio": result.strategy_metrics.sharpe_ratio,
                "telegram_sent": telegram_sent,
            }
        except Exception as e:
            logger.error(f"Backtest report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def _send_to_telegram(self, pdf_path: Path, result: BacktestResult) -> bool:
        """Send report to Telegram."""
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

            registry = get_skills_registry()
            registry.init()

            telegram = registry.get_skill("telegram-sender")
            if telegram:
                send_tool = telegram.tools.get("send_telegram_file_tool")
                if send_tool:
                    sm = result.strategy_metrics
                    bm = result.benchmark_metrics
                    outperformance = sm.total_return - bm.total_return

                    caption = (
                        f" ML Backtest Report: {result.symbol}\n\n"
                        f"Strategy Return: {sm.total_return:+.1f}%\n"
                        f"Sharpe Ratio: {sm.sharpe_ratio:.2f}\n"
                        f"{'Outperforms' if outperformance > 0 else 'Underperforms'} B&H by: {abs(outperformance):+.1f}%\n"
                        f"Win Rate: {sm.win_rate:.1f}%"
                    )

                    import inspect

                    if inspect.iscoroutinefunction(send_tool):
                        send_result = await send_tool(
                            {
                                "file_path": str(pdf_path),
                                "caption": caption,
                            }
                        )
                    else:
                        send_result = send_tool(
                            {
                                "file_path": str(pdf_path),
                                "caption": caption,
                            }
                        )

                    return send_result.get("success", False)

            return False

        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False
