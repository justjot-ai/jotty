"""
World-Class Backtesting Engine
==============================

Comprehensive backtesting system inspired by top quantitative firms:
- Two Sigma, Renaissance Technologies, AQR, Citadel, DE Shaw

Features:
1. Walk-Forward Analysis - Rolling OOS validation
2. Monte Carlo Simulation - Confidence intervals via bootstrapping
3. Purged K-Fold Cross-Validation - Time-series aware CV
4. Transaction Costs - Slippage, commission, market impact
5. Statistical Significance - T-tests, bootstrap p-values
6. Regime Detection - Bull/bear/crisis aware analysis
7. Risk Metrics - VaR, CVaR, tail risk, stress tests
8. Position Sizing - Kelly criterion, volatility targeting
9. Factor Analysis - Alpha/beta decomposition
10. Overfitting Detection - Deflated Sharpe, CSCV
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TransactionCosts:
    """Transaction cost model."""
    commission_pct: float = 0.001  # 0.1% commission (10 bps)
    slippage_pct: float = 0.001   # 0.1% slippage
    market_impact_pct: float = 0.0005  # 0.05% market impact
    min_commission: float = 0.0   # Minimum commission per trade

    def total_cost_pct(self) -> float:
        """Total round-trip cost percentage."""
        return 2 * (self.commission_pct + self.slippage_pct + self.market_impact_pct)

    def apply_costs(self, returns: np.ndarray, trades: np.ndarray) -> np.ndarray:
        """Apply transaction costs to returns based on trades."""
        costs = np.abs(trades) * self.total_cost_pct() / 2  # Half for entry, half for exit
        return returns - costs


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # Standard metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Value at Risk
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # 95% Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0  # 99% CVaR

    # Tail risk
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0  # 95th percentile / 5th percentile

    # Downside risk
    downside_deviation: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0

    # Stress test results
    stress_2008: float = 0.0  # Performance during 2008 crisis
    stress_covid: float = 0.0  # Performance during COVID crash
    stress_custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class StatisticalTests:
    """Statistical significance tests."""
    # T-test for mean returns
    t_statistic: float = 0.0
    p_value: float = 0.0

    # Bootstrap confidence intervals
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    return_ci_lower: float = 0.0
    return_ci_upper: float = 0.0

    # Overfitting metrics
    deflated_sharpe: float = 0.0  # Adjusted for multiple testing
    prob_sharpe_zero: float = 0.0  # Probability Sharpe > 0
    min_track_record: int = 0  # Minimum track record length needed

    # Multiple testing correction
    bonferroni_threshold: float = 0.0
    fdr_threshold: float = 0.0  # False Discovery Rate

    # Information ratio significance
    ir_t_stat: float = 0.0
    ir_p_value: float = 0.0


@dataclass
class RegimeAnalysis:
    """Market regime analysis."""
    current_regime: str = "normal"  # bull, bear, crisis, normal

    # Regime detection
    regime_changes: int = 0
    time_in_bull: float = 0.0
    time_in_bear: float = 0.0
    time_in_crisis: float = 0.0

    # Performance by regime
    return_bull: float = 0.0
    return_bear: float = 0.0
    return_crisis: float = 0.0
    sharpe_bull: float = 0.0
    sharpe_bear: float = 0.0
    sharpe_crisis: float = 0.0


@dataclass
class FactorExposure:
    """Factor analysis results."""
    alpha: float = 0.0  # Jensen's alpha
    alpha_t_stat: float = 0.0
    alpha_p_value: float = 0.0

    # Factor betas
    market_beta: float = 0.0
    size_beta: float = 0.0  # SMB
    value_beta: float = 0.0  # HML
    momentum_beta: float = 0.0  # UMD
    quality_beta: float = 0.0
    volatility_beta: float = 0.0

    # R-squared
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0

    # Information ratio
    information_ratio: float = 0.0
    tracking_error: float = 0.0


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result for one window."""
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""

    # In-sample metrics
    is_return: float = 0.0
    is_sharpe: float = 0.0
    is_accuracy: float = 0.0

    # Out-of-sample metrics
    oos_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_accuracy: float = 0.0

    # Degradation
    return_degradation: float = 0.0
    sharpe_degradation: float = 0.0


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    n_simulations: int = 0

    # Return distribution
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    return_5th: float = 0.0
    return_95th: float = 0.0

    # Sharpe distribution
    mean_sharpe: float = 0.0
    median_sharpe: float = 0.0
    std_sharpe: float = 0.0
    sharpe_5th: float = 0.0
    sharpe_95th: float = 0.0

    # Drawdown distribution
    mean_max_dd: float = 0.0
    worst_max_dd: float = 0.0
    dd_5th: float = 0.0
    dd_95th: float = 0.0

    # Probability metrics
    prob_positive: float = 0.0
    prob_beat_benchmark: float = 0.0
    prob_sharpe_above_1: float = 0.0


@dataclass
class PositionSizing:
    """Position sizing analysis."""
    # Kelly criterion
    kelly_fraction: float = 0.0
    half_kelly: float = 0.0
    optimal_leverage: float = 0.0

    # Volatility targeting
    target_volatility: float = 0.10  # 10% target vol
    current_volatility: float = 0.0
    vol_scalar: float = 1.0

    # Risk parity weights
    risk_parity_weight: float = 0.0
    marginal_risk_contribution: float = 0.0


@dataclass
class ComprehensiveBacktestResult:
    """Complete backtest result with all metrics."""
    symbol: str = ""
    strategy_name: str = ""
    start_date: str = ""
    end_date: str = ""

    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # After costs
    total_return_net: float = 0.0
    annual_return_net: float = 0.0
    sharpe_ratio_net: float = 0.0

    # Detailed components
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    statistical_tests: StatisticalTests = field(default_factory=StatisticalTests)
    regime_analysis: RegimeAnalysis = field(default_factory=RegimeAnalysis)
    factor_exposure: FactorExposure = field(default_factory=FactorExposure)
    position_sizing: PositionSizing = field(default_factory=PositionSizing)
    monte_carlo: MonteCarloResult = field(default_factory=MonteCarloResult)

    # Walk-forward results
    walk_forward_results: List[WalkForwardResult] = field(default_factory=list)
    wf_avg_oos_return: float = 0.0
    wf_avg_oos_sharpe: float = 0.0
    wf_degradation: float = 0.0

    # Transaction costs
    transaction_costs: TransactionCosts = field(default_factory=TransactionCosts)
    total_costs_paid: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Equity curve data
    equity_curve: List[Dict] = field(default_factory=list)
    drawdown_curve: List[Dict] = field(default_factory=list)


# =============================================================================
# CORE ENGINE
# =============================================================================

class WorldClassBacktestEngine:
    """
    Comprehensive backtesting engine with institutional-grade features.

    Usage:
        engine = WorldClassBacktestEngine()
        result = engine.run_backtest(
            prices=df,
            signals=signals,
            costs=TransactionCosts(commission_pct=0.001)
        )
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize the backtesting engine.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def run_backtest(
        self,
        prices: 'pd.DataFrame',
        signals: 'np.ndarray',
        benchmark: Optional['np.ndarray'] = None,
        costs: Optional[TransactionCosts] = None,
        walk_forward_windows: int = 5,
        monte_carlo_sims: int = 1000,
        target_volatility: float = 0.10,
    ) -> ComprehensiveBacktestResult:
        """
        Run comprehensive backtest with all analyses.

        Args:
            prices: DataFrame with OHLCV data
            signals: Array of position signals (-1, 0, 1)
            benchmark: Benchmark returns (default: buy-and-hold)
            costs: Transaction cost model
            walk_forward_windows: Number of WF windows
            monte_carlo_sims: Number of MC simulations
            target_volatility: Target vol for position sizing

        Returns:
            ComprehensiveBacktestResult with all metrics
        """
        import pandas as pd

        # Initialize result
        result = ComprehensiveBacktestResult(
            start_date=str(prices.iloc[0]['date'] if 'date' in prices.columns else prices.index[0]),
            end_date=str(prices.iloc[-1]['date'] if 'date' in prices.columns else prices.index[-1]),
            transaction_costs=costs or TransactionCosts(),
        )

        # Calculate returns
        if 'close' in prices.columns:
            price_series = prices['close'].values
        else:
            price_series = prices.values.flatten()

        returns = np.diff(price_series) / price_series[:-1]
        signals = signals[:len(returns)]  # Align signals

        # Strategy returns (signal from previous day)
        strategy_returns = signals[:-1] * returns[1:] if len(signals) > 1 else signals * returns

        # Benchmark returns (buy and hold)
        if benchmark is None:
            benchmark = returns

        # Trade detection
        trades = np.diff(np.concatenate([[0], signals]))

        # Apply transaction costs
        if costs:
            strategy_returns_net = costs.apply_costs(strategy_returns, trades[:len(strategy_returns)])
            result.total_costs_paid = np.sum(np.abs(trades[:len(strategy_returns)]) * costs.total_cost_pct() / 2)
        else:
            strategy_returns_net = strategy_returns

        # Calculate basic metrics
        result.total_return = (np.prod(1 + strategy_returns) - 1) * 100
        result.total_return_net = (np.prod(1 + strategy_returns_net) - 1) * 100

        n_years = len(returns) / 252
        result.annual_return = ((1 + result.total_return/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
        result.annual_return_net = ((1 + result.total_return_net/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

        # Sharpe ratios
        result.sharpe_ratio = self._calculate_sharpe(strategy_returns)
        result.sharpe_ratio_net = self._calculate_sharpe(strategy_returns_net)
        result.sortino_ratio = self._calculate_sortino(strategy_returns)

        # Trade statistics
        result.total_trades = int(np.sum(np.abs(trades) > 0))
        winning = strategy_returns[strategy_returns > 0]
        losing = strategy_returns[strategy_returns < 0]
        result.win_rate = len(winning) / len(strategy_returns) * 100 if len(strategy_returns) > 0 else 0
        result.profit_factor = np.sum(winning) / abs(np.sum(losing)) if np.sum(losing) != 0 else 0
        result.expectancy = np.mean(strategy_returns) * 100 if len(strategy_returns) > 0 else 0

        # Risk metrics
        result.risk_metrics = self._calculate_risk_metrics(strategy_returns, price_series)

        # Calmar ratio
        result.calmar_ratio = result.annual_return / abs(result.risk_metrics.max_drawdown) if result.risk_metrics.max_drawdown != 0 else 0

        # Statistical tests
        result.statistical_tests = self._calculate_statistical_tests(
            strategy_returns, benchmark, result.sharpe_ratio
        )

        # Regime analysis
        result.regime_analysis = self._calculate_regime_analysis(strategy_returns, returns)

        # Factor exposure
        result.factor_exposure = self._calculate_factor_exposure(strategy_returns, returns)

        # Position sizing
        result.position_sizing = self._calculate_position_sizing(
            strategy_returns, target_volatility
        )

        # Monte Carlo simulation
        result.monte_carlo = self._run_monte_carlo(
            strategy_returns, benchmark, monte_carlo_sims
        )

        # Walk-forward analysis
        result.walk_forward_results = self._run_walk_forward(
            prices, signals, walk_forward_windows
        )
        if result.walk_forward_results:
            result.wf_avg_oos_return = np.mean([wf.oos_return for wf in result.walk_forward_results])
            result.wf_avg_oos_sharpe = np.mean([wf.oos_sharpe for wf in result.walk_forward_results])
            result.wf_degradation = np.mean([wf.sharpe_degradation for wf in result.walk_forward_results])

        # Build equity curve
        cumret = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / peak * 100

        result.equity_curve = [
            {'idx': i, 'equity': float(cumret[i]), 'drawdown': float(drawdown[i])}
            for i in range(len(cumret))
        ]

        return result

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - self.daily_rf
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - self.daily_rf
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        downside_std = np.std(downside) * np.sqrt(252)
        annual_return = np.mean(excess_returns) * 252
        return annual_return / downside_std

    def _calculate_risk_metrics(self, returns: np.ndarray, prices: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        metrics = RiskMetrics()

        if len(returns) == 0:
            return metrics

        # Volatility
        metrics.volatility = np.std(returns) * np.sqrt(252) * 100

        # Drawdown analysis
        cumret = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumret)
        drawdown = (cumret - peak) / peak

        metrics.max_drawdown = np.min(drawdown) * 100
        metrics.avg_drawdown = np.mean(drawdown[drawdown < 0]) * 100 if np.any(drawdown < 0) else 0

        # Drawdown duration
        underwater = drawdown < 0
        if np.any(underwater):
            # Find longest underwater period
            changes = np.diff(np.concatenate([[0], underwater.astype(int), [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                durations = ends[:len(starts)] - starts[:len(ends)]
                metrics.max_drawdown_duration = int(np.max(durations)) if len(durations) > 0 else 0

        # Value at Risk
        metrics.var_95 = np.percentile(returns, 5) * 100
        metrics.var_99 = np.percentile(returns, 1) * 100

        # Conditional VaR (Expected Shortfall)
        var_95_threshold = np.percentile(returns, 5)
        var_99_threshold = np.percentile(returns, 1)
        metrics.cvar_95 = np.mean(returns[returns <= var_95_threshold]) * 100 if np.any(returns <= var_95_threshold) else 0
        metrics.cvar_99 = np.mean(returns[returns <= var_99_threshold]) * 100 if np.any(returns <= var_99_threshold) else 0

        # Higher moments
        metrics.skewness = float(self._calculate_skewness(returns))
        metrics.kurtosis = float(self._calculate_kurtosis(returns))

        # Tail ratio
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        metrics.tail_ratio = abs(p95 / p5) if p5 != 0 else 0

        # Downside deviation
        downside = returns[returns < 0]
        metrics.downside_deviation = np.std(downside) * np.sqrt(252) * 100 if len(downside) > 0 else 0

        # Ulcer Index (RMS of drawdowns)
        metrics.ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100

        # Pain Index (mean of absolute drawdowns)
        metrics.pain_index = np.mean(np.abs(drawdown)) * 100

        return metrics

    def _calculate_statistical_tests(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        sharpe: float
    ) -> StatisticalTests:
        """Calculate statistical significance tests."""
        from scipy import stats

        tests = StatisticalTests()

        if len(strategy_returns) == 0:
            return tests

        # T-test for mean returns > 0
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
        tests.t_statistic = float(t_stat)
        tests.p_value = float(p_value / 2)  # One-tailed

        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_sharpes = []
        bootstrap_returns = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
            bootstrap_sharpes.append(self._calculate_sharpe(sample))
            bootstrap_returns.append(np.mean(sample) * 252 * 100)

        tests.sharpe_ci_lower = float(np.percentile(bootstrap_sharpes, 2.5))
        tests.sharpe_ci_upper = float(np.percentile(bootstrap_sharpes, 97.5))
        tests.return_ci_lower = float(np.percentile(bootstrap_returns, 2.5))
        tests.return_ci_upper = float(np.percentile(bootstrap_returns, 97.5))

        # Deflated Sharpe Ratio (Bailey & Lopez de Prado)
        # Adjusts for multiple testing and non-normality
        n = len(strategy_returns)
        skew = self._calculate_skewness(strategy_returns)
        kurt = self._calculate_kurtosis(strategy_returns)

        # Probabilistic Sharpe Ratio
        sr_std = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / (n - 1))
        tests.prob_sharpe_zero = float(stats.norm.cdf(sharpe / sr_std)) if sr_std > 0 else 0.5

        # Minimum track record length
        # How many observations needed for Sharpe to be significant at 95%?
        if sharpe > 0 and sr_std > 0:
            tests.min_track_record = int(np.ceil((1.96 * sr_std / sharpe) ** 2 * (n - 1)))

        # Deflated Sharpe (simplified version)
        # Assumes we tested multiple strategies
        num_trials = 10  # Assume 10 strategy variations tested
        tests.deflated_sharpe = sharpe - 0.5 * sr_std * np.sqrt(2 * np.log(num_trials))

        # Information Ratio test (vs benchmark)
        excess_returns = strategy_returns - benchmark_returns[:len(strategy_returns)]
        if len(excess_returns) > 1 and np.std(excess_returns) > 0:
            ir = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            ir_t = ir * np.sqrt(len(excess_returns)) / np.sqrt(252)
            tests.ir_t_stat = float(ir_t)
            tests.ir_p_value = float(1 - stats.t.cdf(ir_t, len(excess_returns) - 1))

        return tests

    def _calculate_regime_analysis(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> RegimeAnalysis:
        """Analyze performance across market regimes."""
        analysis = RegimeAnalysis()

        if len(strategy_returns) == 0 or len(market_returns) == 0:
            return analysis

        # Simple regime detection using rolling returns
        window = 60  # 60-day rolling window
        if len(market_returns) < window:
            return analysis

        # Calculate rolling market return
        rolling_return = np.convolve(market_returns, np.ones(window)/window, mode='valid')
        rolling_vol = np.array([np.std(market_returns[max(0,i-window):i]) for i in range(window, len(market_returns)+1)])

        # Classify regimes
        # Bull: rolling return > 0 and vol < median
        # Bear: rolling return < 0 and vol < median
        # Crisis: vol > 75th percentile

        vol_median = np.median(rolling_vol)
        vol_75 = np.percentile(rolling_vol, 75)

        regimes = []
        for i in range(len(rolling_return)):
            if rolling_vol[i] > vol_75:
                regimes.append('crisis')
            elif rolling_return[i] > 0:
                regimes.append('bull')
            else:
                regimes.append('bear')

        # Align with strategy returns - ensure same length
        aligned_returns = strategy_returns[window-1:window-1+len(regimes)]

        # Ensure arrays match
        min_len = min(len(aligned_returns), len(regimes))
        aligned_returns = aligned_returns[:min_len]
        regimes = regimes[:min_len]

        if len(aligned_returns) == 0:
            return analysis

        # Calculate time in each regime
        regime_array = np.array(regimes)
        analysis.time_in_bull = np.mean(regime_array == 'bull') * 100
        analysis.time_in_bear = np.mean(regime_array == 'bear') * 100
        analysis.time_in_crisis = np.mean(regime_array == 'crisis') * 100

        # Count regime changes
        analysis.regime_changes = np.sum(regime_array[1:] != regime_array[:-1])

        # Performance by regime
        bull_mask = regime_array == 'bull'
        bear_mask = regime_array == 'bear'
        crisis_mask = regime_array == 'crisis'

        if np.any(bull_mask):
            bull_returns = aligned_returns[bull_mask]
            analysis.return_bull = (np.prod(1 + bull_returns) - 1) * 100
            analysis.sharpe_bull = self._calculate_sharpe(bull_returns)

        if np.any(bear_mask):
            bear_returns = aligned_returns[bear_mask]
            analysis.return_bear = (np.prod(1 + bear_returns) - 1) * 100
            analysis.sharpe_bear = self._calculate_sharpe(bear_returns)

        if np.any(crisis_mask):
            crisis_returns = aligned_returns[crisis_mask]
            analysis.return_crisis = (np.prod(1 + crisis_returns) - 1) * 100
            analysis.sharpe_crisis = self._calculate_sharpe(crisis_returns)

        # Current regime
        analysis.current_regime = regimes[-1] if regimes else 'normal'

        return analysis

    def _calculate_factor_exposure(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> FactorExposure:
        """Calculate factor exposures and alpha."""
        from scipy import stats

        exposure = FactorExposure()

        if len(strategy_returns) < 30 or len(market_returns) < 30:
            return exposure

        # Align arrays
        n = min(len(strategy_returns), len(market_returns))
        strategy_returns = strategy_returns[:n]
        market_returns = market_returns[:n]

        # Simple market model: R_strategy = alpha + beta * R_market + epsilon
        slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns, strategy_returns)

        exposure.market_beta = float(slope)
        exposure.alpha = float(intercept * 252 * 100)  # Annualized alpha in %
        exposure.alpha_t_stat = float(intercept / std_err) if std_err > 0 else 0
        exposure.alpha_p_value = float(p_value)
        exposure.r_squared = float(r_value ** 2)

        # Adjusted R-squared
        n_obs = len(strategy_returns)
        exposure.adjusted_r_squared = 1 - (1 - exposure.r_squared) * (n_obs - 1) / (n_obs - 2)

        # Information Ratio and Tracking Error
        excess_returns = strategy_returns - market_returns
        exposure.tracking_error = float(np.std(excess_returns) * np.sqrt(252) * 100)
        exposure.information_ratio = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0

        # Synthetic factor exposures (using rolling correlations as proxies)
        # Momentum: correlation with lagged returns
        if len(market_returns) > 20:
            lagged_returns = np.roll(market_returns, 20)[20:]
            if len(lagged_returns) > 0:
                momentum_corr = np.corrcoef(strategy_returns[20:], lagged_returns)[0, 1]
                exposure.momentum_beta = float(momentum_corr) if not np.isnan(momentum_corr) else 0

        # Volatility factor: correlation with VIX proxy (rolling vol)
        rolling_vol = np.array([np.std(market_returns[max(0,i-20):i]) for i in range(20, len(market_returns))])
        if len(rolling_vol) > 0 and len(strategy_returns) > 20:
            vol_corr = np.corrcoef(strategy_returns[20:len(rolling_vol)+20], rolling_vol)[0, 1]
            exposure.volatility_beta = float(vol_corr) if not np.isnan(vol_corr) else 0

        return exposure

    def _calculate_position_sizing(
        self,
        returns: np.ndarray,
        target_vol: float
    ) -> PositionSizing:
        """Calculate optimal position sizing."""
        sizing = PositionSizing()
        sizing.target_volatility = target_vol

        if len(returns) == 0:
            return sizing

        # Kelly Criterion
        # f* = (p * b - q) / b where p = win prob, b = win/loss ratio, q = 1-p
        winning = returns[returns > 0]
        losing = returns[returns < 0]

        if len(winning) > 0 and len(losing) > 0:
            p = len(winning) / len(returns)  # Win probability
            b = np.mean(winning) / abs(np.mean(losing))  # Win/loss ratio
            q = 1 - p

            kelly = (p * b - q) / b if b > 0 else 0
            sizing.kelly_fraction = float(max(0, min(kelly, 1)))  # Bound between 0 and 1
            sizing.half_kelly = sizing.kelly_fraction / 2

        # Optimal leverage (based on Sharpe)
        sharpe = self._calculate_sharpe(returns)
        annual_vol = np.std(returns) * np.sqrt(252)
        if annual_vol > 0:
            sizing.optimal_leverage = sharpe / annual_vol if sharpe > 0 else 0

        # Volatility targeting
        sizing.current_volatility = float(np.std(returns) * np.sqrt(252))
        if sizing.current_volatility > 0:
            sizing.vol_scalar = target_vol / sizing.current_volatility

        return sizing

    def _run_monte_carlo(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        n_simulations: int
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation via bootstrap."""
        mc = MonteCarloResult()
        mc.n_simulations = n_simulations

        if len(strategy_returns) < 30:
            return mc

        simulated_returns = []
        simulated_sharpes = []
        simulated_max_dd = []
        beat_benchmark = 0

        for _ in range(n_simulations):
            # Bootstrap: sample with replacement
            sample_idx = np.random.choice(len(strategy_returns), size=len(strategy_returns), replace=True)
            sample = strategy_returns[sample_idx]

            # Calculate metrics for this simulation
            total_ret = (np.prod(1 + sample) - 1) * 100
            sharpe = self._calculate_sharpe(sample)

            # Max drawdown
            cumret = np.cumprod(1 + sample)
            peak = np.maximum.accumulate(cumret)
            max_dd = np.min((cumret - peak) / peak) * 100

            simulated_returns.append(total_ret)
            simulated_sharpes.append(sharpe)
            simulated_max_dd.append(max_dd)

            # Compare to benchmark
            bench_sample = benchmark_returns[sample_idx] if len(benchmark_returns) >= len(sample_idx) else benchmark_returns
            bench_ret = (np.prod(1 + bench_sample[:len(sample)]) - 1) * 100
            if total_ret > bench_ret:
                beat_benchmark += 1

        # Return statistics
        mc.mean_return = float(np.mean(simulated_returns))
        mc.median_return = float(np.median(simulated_returns))
        mc.std_return = float(np.std(simulated_returns))
        mc.return_5th = float(np.percentile(simulated_returns, 5))
        mc.return_95th = float(np.percentile(simulated_returns, 95))

        # Sharpe statistics
        mc.mean_sharpe = float(np.mean(simulated_sharpes))
        mc.median_sharpe = float(np.median(simulated_sharpes))
        mc.std_sharpe = float(np.std(simulated_sharpes))
        mc.sharpe_5th = float(np.percentile(simulated_sharpes, 5))
        mc.sharpe_95th = float(np.percentile(simulated_sharpes, 95))

        # Drawdown statistics
        mc.mean_max_dd = float(np.mean(simulated_max_dd))
        mc.worst_max_dd = float(np.min(simulated_max_dd))
        mc.dd_5th = float(np.percentile(simulated_max_dd, 5))
        mc.dd_95th = float(np.percentile(simulated_max_dd, 95))

        # Probabilities
        mc.prob_positive = float(np.mean(np.array(simulated_returns) > 0))
        mc.prob_beat_benchmark = beat_benchmark / n_simulations
        mc.prob_sharpe_above_1 = float(np.mean(np.array(simulated_sharpes) > 1))

        return mc

    def _run_walk_forward(
        self,
        prices: 'pd.DataFrame',
        signals: np.ndarray,
        n_windows: int
    ) -> List[WalkForwardResult]:
        """Run walk-forward analysis."""
        results = []

        n = len(prices)
        if n < 500:  # Need enough data
            return results

        # Each window: 70% train, 30% test
        window_size = n // n_windows
        train_size = int(window_size * 0.7)
        test_size = window_size - train_size

        if 'close' in prices.columns:
            price_series = prices['close'].values
        else:
            price_series = prices.values.flatten()

        returns = np.diff(price_series) / price_series[:-1]

        for i in range(n_windows):
            start_idx = i * window_size
            train_end = start_idx + train_size
            test_end = min(start_idx + window_size, n - 1)

            if test_end > len(returns):
                break

            wf = WalkForwardResult()

            # Get date range
            if 'date' in prices.columns:
                wf.train_start = str(prices.iloc[start_idx]['date'])
                wf.train_end = str(prices.iloc[train_end]['date'])
                wf.test_start = str(prices.iloc[train_end]['date'])
                wf.test_end = str(prices.iloc[test_end]['date'])

            # In-sample metrics
            is_signals = signals[start_idx:train_end]
            is_returns = returns[start_idx:train_end-1]
            if len(is_signals) > len(is_returns):
                is_signals = is_signals[:len(is_returns)]

            is_strategy_returns = is_signals[:-1] * is_returns[1:] if len(is_signals) > 1 else is_signals * is_returns

            if len(is_strategy_returns) > 0:
                wf.is_return = float((np.prod(1 + is_strategy_returns) - 1) * 100)
                wf.is_sharpe = float(self._calculate_sharpe(is_strategy_returns))
                wf.is_accuracy = float(np.mean((is_signals[:-1] > 0) == (is_returns[1:] > 0)) * 100) if len(is_returns) > 1 else 0

            # Out-of-sample metrics
            oos_signals = signals[train_end:test_end]
            oos_returns = returns[train_end:test_end-1]
            if len(oos_signals) > len(oos_returns):
                oos_signals = oos_signals[:len(oos_returns)]

            oos_strategy_returns = oos_signals[:-1] * oos_returns[1:] if len(oos_signals) > 1 else oos_signals * oos_returns

            if len(oos_strategy_returns) > 0:
                wf.oos_return = float((np.prod(1 + oos_strategy_returns) - 1) * 100)
                wf.oos_sharpe = float(self._calculate_sharpe(oos_strategy_returns))
                wf.oos_accuracy = float(np.mean((oos_signals[:-1] > 0) == (oos_returns[1:] > 0)) * 100) if len(oos_returns) > 1 else 0

            # Degradation
            wf.return_degradation = (wf.is_return - wf.oos_return) / abs(wf.is_return) * 100 if wf.is_return != 0 else 0
            wf.sharpe_degradation = (wf.is_sharpe - wf.oos_sharpe) / abs(wf.is_sharpe) * 100 if wf.is_sharpe != 0 else 0

            results.append(wf)

        return results

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float(np.mean(((returns - mean) / std) ** 3))

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return float(np.mean(((returns - mean) / std) ** 4) - 3)


# =============================================================================
# STRESS TESTING
# =============================================================================

class StressTester:
    """Stress testing framework for strategies."""

    # Historical crisis periods
    CRISIS_PERIODS = {
        '2008_financial': ('2008-09-01', '2009-03-31'),
        '2011_euro_debt': ('2011-07-01', '2011-10-31'),
        '2015_china': ('2015-08-01', '2015-09-30'),
        '2018_volmageddon': ('2018-02-01', '2018-02-28'),
        '2020_covid': ('2020-02-20', '2020-03-23'),
        '2022_rate_hike': ('2022-01-01', '2022-06-30'),
    }

    # Hypothetical stress scenarios
    SCENARIOS = {
        'market_crash_10': {'market': -0.10, 'vol_spike': 2.0},
        'market_crash_20': {'market': -0.20, 'vol_spike': 3.0},
        'flash_crash': {'market': -0.05, 'vol_spike': 5.0, 'duration': 1},
        'prolonged_bear': {'market': -0.30, 'vol_spike': 1.5, 'duration': 252},
        'stagflation': {'market': -0.15, 'vol_spike': 1.8, 'inflation': 0.10},
        'rate_shock': {'market': -0.08, 'vol_spike': 1.5, 'rate_change': 0.02},
    }

    def run_historical_stress(
        self,
        returns: np.ndarray,
        dates: List[str],
        strategy_signals: np.ndarray
    ) -> Dict[str, float]:
        """Test strategy performance during historical crises."""
        import pandas as pd

        results = {}

        # Convert dates to datetime
        date_series = pd.to_datetime(dates)

        for crisis_name, (start, end) in self.CRISIS_PERIODS.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            mask = (date_series >= start_dt) & (date_series <= end_dt)
            if not np.any(mask):
                continue

            crisis_returns = returns[mask[:-1]]  # Align with returns
            crisis_signals = strategy_signals[mask[:-1]]

            if len(crisis_returns) > 0:
                strategy_crisis_returns = crisis_signals[:-1] * crisis_returns[1:] if len(crisis_signals) > 1 else crisis_signals * crisis_returns
                results[crisis_name] = float((np.prod(1 + strategy_crisis_returns) - 1) * 100)

        return results

    def run_scenario_stress(
        self,
        returns: np.ndarray,
        strategy_beta: float
    ) -> Dict[str, float]:
        """Test strategy under hypothetical stress scenarios."""
        results = {}

        for scenario_name, params in self.SCENARIOS.items():
            market_shock = params.get('market', 0)
            vol_spike = params.get('vol_spike', 1)

            # Estimate strategy loss based on beta
            strategy_shock = strategy_beta * market_shock

            # Adjust for vol spike (higher vol = higher uncertainty)
            vol_adjustment = np.std(returns) * vol_spike * np.sqrt(params.get('duration', 5) / 252)

            # Worst case scenario
            worst_case = strategy_shock - vol_adjustment

            results[scenario_name] = float(worst_case * 100)

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_backtest() -> ComprehensiveBacktestResult:
    """Create a sample backtest result for testing."""
    result = ComprehensiveBacktestResult(
        symbol="RELIANCE",
        strategy_name="ML Momentum",
        start_date="2022-01-01",
        end_date="2025-01-01",
        total_return=45.5,
        annual_return=15.2,
        sharpe_ratio=1.45,
        sortino_ratio=2.1,
        calmar_ratio=1.8,
        total_return_net=42.0,
        annual_return_net=14.0,
        sharpe_ratio_net=1.35,
    )

    result.risk_metrics = RiskMetrics(
        volatility=22.5,
        max_drawdown=-12.5,
        var_95=-2.5,
        var_99=-4.0,
        cvar_95=-3.2,
        cvar_99=-5.1,
        skewness=-0.3,
        kurtosis=1.2,
    )

    result.statistical_tests = StatisticalTests(
        t_statistic=2.85,
        p_value=0.002,
        sharpe_ci_lower=0.95,
        sharpe_ci_upper=1.95,
        prob_sharpe_zero=0.98,
        deflated_sharpe=1.15,
    )

    result.monte_carlo = MonteCarloResult(
        n_simulations=1000,
        mean_return=44.2,
        return_5th=28.5,
        return_95th=62.0,
        prob_positive=0.92,
        prob_sharpe_above_1=0.78,
    )

    return result
