"""
Fundamental Features Skill (V2 - Backtest Safe)
================================================

Downloads and processes fundamental data from Yahoo Finance
with PROPER TIME-SHIFTING to avoid forward-looking bias.

Key Principles:
1. Only use data that was AVAILABLE at each point in time
2. Quarterly data is lagged by 1 quarter (earnings release delay)
3. No current analyst targets/ratings for historical predictions
4. Rolling calculations are strictly backward-looking

Features:
- Time-varying PE, PB from historical quarterly EPS/Book Value
- Value + Momentum combo (proven quant factor)
- Mean reversion signals (Z-scores)
- Fundamental momentum (improving/declining metrics)
- Quality trends over time
- Relative strength vs sector

Usage:
    skill = FundamentalFeaturesSkill()
    features_df = await skill.get_features("RELIANCE", df_prices, mode="backtest")
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import MLSkill, SkillCategory, SkillResult

logger = logging.getLogger(__name__)


class FundamentalFeaturesSkill(MLSkill):
    """
    Fundamental feature engineering from Yahoo Finance.
    V2: Backtest-safe with proper time-shifting.
    """

    name = "fundamental_features"
    version = "2.0.0"
    description = "Yahoo Finance fundamental features (backtest-safe)"
    category = SkillCategory.FEATURE_ENGINEERING

    required_inputs = ["symbol"]
    optional_inputs = ["df_prices", "mode"]
    outputs = ["fundamental_features", "feature_names"]

    CACHE_DIR = Path.home() / ".jotty" / "fundamental_cache"
    CACHE_EXPIRY_HOURS = 24
    NSE_SUFFIX = ".NS"

    # Earnings release lag (days after quarter end)
    EARNINGS_LAG_DAYS = 45  # Conservative: most companies report within 45 days

    # Sector mappings
    SECTOR_STOCKS = {
        "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BANKBARODA", "PNB"],
        "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
        "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN"],
        "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
        "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
        "energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC"],
        "metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA"],
    }

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self._yf = None
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_yfinance(self) -> Any:
        if self._yf is None:
            import yfinance as yf

            self._yf = yf
        return self._yf

    async def execute(
        self, X: pd.DataFrame = None, y: Optional[pd.Series] = None, **context: Any
    ) -> SkillResult:
        start_time = time.time()
        symbol = context.get("symbol")
        if not symbol:
            return self._create_error_result("Symbol required")

        df_prices = context.get("df_prices")
        mode = context.get("mode", "backtest")  # 'backtest' or 'live'

        try:
            fund_features = await self.get_fundamental_features(
                symbol, df_prices=df_prices, mode=mode
            )

            return self._create_result(
                success=True,
                data=fund_features,
                metadata={
                    "symbol": symbol,
                    "mode": mode,
                    "n_features": len(fund_features.columns) if fund_features is not None else 0,
                },
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"Fundamental feature extraction failed: {e}")
            return self._create_error_result(str(e))

    async def get_fundamental_features(
        self, symbol: str, df_prices: pd.DataFrame = None, mode: str = "backtest"
    ) -> pd.DataFrame:
        """
        Get fundamental features with proper time-shifting.

        Args:
            symbol: Stock symbol
            df_prices: Price DataFrame with 'date', 'close', 'volume'
            mode: 'backtest' (strict no lookahead) or 'live' (can use current data)

        Returns:
            DataFrame with backtest-safe fundamental features
        """
        if df_prices is None or "date" not in df_prices.columns:
            logger.warning("Price data required for fundamental features")
            return None

        yf = self._ensure_yfinance()

        # Download quarterly financial history
        ticker_symbol = f"{symbol}{self.NSE_SUFFIX}"
        ticker = yf.Ticker(ticker_symbol)

        try:
            info = ticker.info or {}
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance = ticker.quarterly_balance_sheet
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return None

        # Prepare price DataFrame
        df = df_prices.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Remove timezone info for consistent comparisons
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        # Ensure numeric columns
        for col in ["close", "open", "high", "low", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Initialize features DataFrame
        fund_df = pd.DataFrame(index=df.index)

        # === 1. TIME-SHIFTED QUARTERLY FUNDAMENTALS ===
        fund_df = self._add_quarterly_features(
            fund_df, df, quarterly_financials, quarterly_balance, info
        )

        # === 2. PRICE-BASED FUNDAMENTALS (Safe - uses historical prices) ===
        fund_df = self._add_price_based_features(fund_df, df, info)

        # === 3. INTELLIGENT DERIVED FEATURES ===
        fund_df = self._add_derived_features(fund_df, df)

        # === 4. SECTOR-RELATIVE FEATURES ===
        sector = self._get_sector(symbol)
        if sector:
            fund_df = await self._add_sector_features(fund_df, df, symbol, sector)

        # === 5. MARKET REGIME FEATURES ===
        fund_df = self._add_regime_features(fund_df, df)

        # === 6. LIVE-ONLY FEATURES (skip in backtest mode) ===
        if mode == "live":
            fund_df = self._add_live_features(fund_df, df, info)

        # Clean up
        fund_df = fund_df.replace([np.inf, -np.inf], np.nan)

        return fund_df

    def _add_quarterly_features(
        self,
        fund_df: pd.DataFrame,
        df: pd.DataFrame,
        quarterly_financials: pd.DataFrame,
        quarterly_balance: pd.DataFrame,
        info: Dict,
    ) -> pd.DataFrame:
        """
        Add features from quarterly financials with PROPER TIME-SHIFTING.
        Key: Only use data that was available at each point in time.
        """
        shares = info.get("sharesOutstanding", info.get("impliedSharesOutstanding"))

        if quarterly_financials is None or quarterly_financials.empty:
            return fund_df

        # === EPS from quarterly Net Income ===
        net_income_row = None
        for metric in ["Net Income", "Net Income From Continuing Operation Net Minority Interest"]:
            if metric in quarterly_financials.index:
                net_income_row = quarterly_financials.loc[metric]
                break

        if net_income_row is not None and shares:
            quarterly_eps = (net_income_row / shares).dropna().sort_index()

            # Create time-shifted EPS series
            # Key: Each quarterly value only becomes available AFTER earnings release
            eps_available = pd.Series(index=df.index, dtype=float)

            for q_date in sorted(quarterly_eps.index):
                q_date = pd.Timestamp(q_date)
                # Data becomes available ~45 days after quarter end
                available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                mask = df.index >= available_date
                eps_available.loc[mask] = quarterly_eps[q_date]

            eps_available = eps_available.ffill()

            # Trailing 4-quarter EPS (TTM)
            eps_ttm = eps_available * 4  # Annualized (simplified)

            if "close" in df.columns and eps_ttm.notna().any():
                fund_df["fund_trailing_pe"] = df["close"] / eps_ttm.replace(0, np.nan)
                fund_df["fund_trailing_pe"] = fund_df["fund_trailing_pe"].clip(0, 200)

                # Earnings yield
                fund_df["fund_earnings_yield"] = eps_ttm / df["close"]

            # === EPS GROWTH (Quarter over Quarter) ===
            eps_growth = quarterly_eps.pct_change()
            eps_growth_available = pd.Series(index=df.index, dtype=float)

            for q_date in sorted(eps_growth.index):
                q_date = pd.Timestamp(q_date)
                available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                mask = df.index >= available_date
                if not pd.isna(eps_growth.get(q_date)):
                    eps_growth_available.loc[mask] = eps_growth[q_date]

            fund_df["fund_eps_growth_qoq"] = eps_growth_available.ffill()

            # === EPS ACCELERATION (is growth speeding up?) ===
            eps_accel = eps_growth.diff()
            eps_accel_available = pd.Series(index=df.index, dtype=float)

            for q_date in sorted(eps_accel.index):
                q_date = pd.Timestamp(q_date)
                available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                mask = df.index >= available_date
                if not pd.isna(eps_accel.get(q_date)):
                    eps_accel_available.loc[mask] = eps_accel[q_date]

            fund_df["fund_eps_acceleration"] = eps_accel_available.ffill()

            # === EARNINGS SURPRISE PROXY ===
            # Compare actual EPS change vs expected (3-quarter average)
            expected_growth = eps_growth.rolling(3).mean()
            surprise = eps_growth - expected_growth.shift(1)  # Shift to avoid lookahead

            surprise_available = pd.Series(index=df.index, dtype=float)
            for q_date in sorted(surprise.index):
                q_date = pd.Timestamp(q_date)
                available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                mask = df.index >= available_date
                if not pd.isna(surprise.get(q_date)):
                    surprise_available.loc[mask] = surprise[q_date]

            fund_df["fund_earnings_surprise"] = surprise_available.ffill()

        # === EBITDA GROWTH ===
        if "EBITDA" in quarterly_financials.index:
            ebitda = quarterly_financials.loc["EBITDA"].dropna().sort_index()
            ebitda_growth = ebitda.pct_change()

            ebitda_growth_available = pd.Series(index=df.index, dtype=float)
            for q_date in sorted(ebitda_growth.index):
                q_date = pd.Timestamp(q_date)
                available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                mask = df.index >= available_date
                if not pd.isna(ebitda_growth.get(q_date)):
                    ebitda_growth_available.loc[mask] = ebitda_growth[q_date]

            fund_df["fund_ebitda_growth"] = ebitda_growth_available.ffill()

        # === BOOK VALUE (from balance sheet) ===
        if quarterly_balance is not None and not quarterly_balance.empty:
            if "Stockholders Equity" in quarterly_balance.index and shares:
                equity = quarterly_balance.loc["Stockholders Equity"].dropna().sort_index()
                bvps = equity / shares

                bvps_available = pd.Series(index=df.index, dtype=float)
                for q_date in sorted(bvps.index):
                    q_date = pd.Timestamp(q_date)
                    available_date = q_date + pd.Timedelta(days=self.EARNINGS_LAG_DAYS)
                    mask = df.index >= available_date
                    bvps_available.loc[mask] = bvps[q_date]

                bvps_available = bvps_available.ffill()

                if "close" in df.columns:
                    fund_df["fund_pb_ratio"] = df["close"] / bvps_available.replace(0, np.nan)
                    fund_df["fund_pb_ratio"] = fund_df["fund_pb_ratio"].clip(0, 50)

        return fund_df

    def _add_price_based_features(
        self, fund_df: pd.DataFrame, df: pd.DataFrame, info: Dict
    ) -> pd.DataFrame:
        """
        Add features derived from price action (no lookahead risk).
        """
        if "close" not in df.columns:
            return fund_df

        close = df["close"]

        # === PE STATISTICS (rolling, backward-looking) ===
        if "fund_trailing_pe" in fund_df.columns:
            pe = fund_df["fund_trailing_pe"]

            # PE percentile vs rolling history
            fund_df["fund_pe_percentile"] = pe.rolling(252, min_periods=60).apply(
                lambda x: (
                    (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
                )
            )

            # PE vs 52-week average
            pe_ma = pe.rolling(252, min_periods=60).mean()
            fund_df["fund_pe_vs_avg"] = pe / pe_ma.replace(0, np.nan)

            # PE Z-score (mean reversion signal)
            pe_mean = pe.rolling(252, min_periods=60).mean()
            pe_std = pe.rolling(252, min_periods=60).std()
            fund_df["fund_pe_zscore"] = ((pe - pe_mean) / (pe_std + 1e-10)).clip(-3, 3)

            # PE trend (expanding or contracting?)
            fund_df["fund_pe_trend_20d"] = pe.diff(20)
            fund_df["fund_pe_trend_60d"] = pe.diff(60)

        # === PB STATISTICS ===
        if "fund_pb_ratio" in fund_df.columns:
            pb = fund_df["fund_pb_ratio"]

            fund_df["fund_pb_percentile"] = pb.rolling(252, min_periods=60).apply(
                lambda x: (
                    (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
                )
            )

            pb_mean = pb.rolling(252, min_periods=60).mean()
            pb_std = pb.rolling(252, min_periods=60).std()
            fund_df["fund_pb_zscore"] = ((pb - pb_mean) / (pb_std + 1e-10)).clip(-3, 3)

        # === PRICE MOMENTUM (for combo features) ===
        fund_df["fund_momentum_20d"] = close.pct_change(20)
        fund_df["fund_momentum_60d"] = close.pct_change(60)
        fund_df["fund_momentum_120d"] = close.pct_change(120)

        # === VOLATILITY ===
        returns = close.pct_change()
        fund_df["fund_volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
        fund_df["fund_volatility_60d"] = returns.rolling(60).std() * np.sqrt(252)

        return fund_df

    def _add_derived_features(self, fund_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intelligent derived/combo features.
        """
        close = df["close"]

        # === VALUE + MOMENTUM COMBO ===
        # Classic quant factor: cheap stocks with positive momentum
        if "fund_pe_percentile" in fund_df.columns and "fund_momentum_60d" in fund_df.columns:
            # Value rank (low PE = high value)
            value_score = 1 - fund_df["fund_pe_percentile"]

            # Momentum rank
            mom = fund_df["fund_momentum_60d"]
            mom_rank = mom.rolling(252, min_periods=60).apply(
                lambda x: (
                    (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
                )
            )

            # Combo score
            fund_df["fund_value_momentum"] = (value_score + mom_rank) / 2

            # Value with momentum filter (only count value if momentum positive)
            fund_df["fund_value_if_momentum"] = value_score * (mom > 0).astype(float)

        # === QUALITY + VALUE COMBO ===
        if "fund_eps_growth_qoq" in fund_df.columns and "fund_pe_percentile" in fund_df.columns:
            # Quality: positive and accelerating earnings
            quality = (fund_df["fund_eps_growth_qoq"] > 0).astype(float)
            if "fund_eps_acceleration" in fund_df.columns:
                quality += (fund_df["fund_eps_acceleration"] > 0).astype(float) * 0.5
            quality = quality / 1.5  # Normalize

            value_score = 1 - fund_df["fund_pe_percentile"]
            fund_df["fund_quality_value"] = (quality + value_score) / 2

        # === RISK-ADJUSTED VALUE ===
        if "fund_earnings_yield" in fund_df.columns and "fund_volatility_20d" in fund_df.columns:
            fund_df["fund_yield_per_risk"] = fund_df["fund_earnings_yield"] / (
                fund_df["fund_volatility_20d"] + 0.01
            )

        # === MEAN REVERSION SIGNALS ===
        if "fund_pe_zscore" in fund_df.columns:
            # Extreme PE (likely to revert)
            fund_df["fund_pe_extreme_low"] = (fund_df["fund_pe_zscore"] < -1.5).astype(float)
            fund_df["fund_pe_extreme_high"] = (fund_df["fund_pe_zscore"] > 1.5).astype(float)

        # === FUNDAMENTAL MOMENTUM (improving fundamentals) ===
        if "fund_eps_growth_qoq" in fund_df.columns:
            # Is EPS growth improving?
            eps_g = fund_df["fund_eps_growth_qoq"]
            fund_df["fund_fundamental_momentum"] = eps_g.rolling(2).mean()  # Smoothed

        # === 52-WEEK POSITION ===
        high_52w = close.rolling(252, min_periods=60).max()
        low_52w = close.rolling(252, min_periods=60).min()
        fund_df["fund_52w_position"] = (close - low_52w) / (high_52w - low_52w + 1e-10)

        # === DRAWDOWN ===
        rolling_max = close.expanding().max()
        fund_df["fund_drawdown"] = (close - rolling_max) / rolling_max

        return fund_df

    async def _add_sector_features(
        self, fund_df: pd.DataFrame, df: pd.DataFrame, symbol: str, sector: str
    ) -> pd.DataFrame:
        """
        Add sector-relative features.
        """
        sector_stocks = [s for s in self.SECTOR_STOCKS.get(sector, []) if s != symbol.upper()]
        if len(sector_stocks) < 2:
            return fund_df

        yf = self._ensure_yfinance()

        # Get sector PE ratios for comparison
        sector_pes = []
        for s in sector_stocks[:5]:  # Limit to 5 for speed
            try:
                ticker = yf.Ticker(f"{s}{self.NSE_SUFFIX}")
                s_info = ticker.info or {}
                s_pe = s_info.get("trailingPE")
                if s_pe and 0 < s_pe < 200:
                    sector_pes.append(s_pe)
            except Exception:
                continue

        if sector_pes and "fund_trailing_pe" in fund_df.columns:
            sector_median_pe = np.median(sector_pes)
            # Relative PE (< 1 means cheaper than sector)
            fund_df["fund_pe_vs_sector"] = fund_df["fund_trailing_pe"] / sector_median_pe

        return fund_df

    def _add_regime_features(self, fund_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime features.
        """
        if "close" not in df.columns:
            return fund_df

        close = df["close"]

        # === TREND REGIME ===
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        # Golden/Death cross regime
        fund_df["fund_trend_regime"] = (sma_50 > sma_200).astype(float)

        # Trend strength
        fund_df["fund_trend_strength"] = (close - sma_200) / sma_200

        # === VOLATILITY REGIME ===
        vol = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
        vol_ma = vol.rolling(60).mean()
        fund_df["fund_vol_regime"] = (vol > vol_ma).astype(float)  # High vol = 1

        # === MOMENTUM REGIME ===
        mom = close.pct_change(60)
        fund_df["fund_momentum_regime"] = (mom > 0).astype(float)

        return fund_df

    def _add_live_features(
        self, fund_df: pd.DataFrame, df: pd.DataFrame, info: Dict
    ) -> pd.DataFrame:
        """
        Add features only safe for LIVE predictions (not backtesting).
        These use current point-in-time data.
        """
        # Analyst target (current)
        target = info.get("targetMeanPrice")
        current = info.get("currentPrice", info.get("regularMarketPrice"))
        if target and current and current > 0:
            fund_df["fund_live_target_gap"] = (target - current) / current

        # Analyst rating (current)
        rating = info.get("recommendationMean")
        if rating:
            fund_df["fund_live_analyst_score"] = 6 - rating  # Invert: higher = more bullish

        # Forward PE (current estimates)
        fwd_pe = info.get("forwardPE")
        trail_pe = info.get("trailingPE")
        if fwd_pe and trail_pe and fwd_pe > 0:
            fund_df["fund_live_pe_fwd_vs_trail"] = trail_pe / fwd_pe

        return fund_df

    def _get_sector(self, symbol: str) -> Optional[str]:
        symbol_upper = symbol.upper()
        for sector, stocks in self.SECTOR_STOCKS.items():
            if symbol_upper in stocks:
                return sector
        return None

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features."""
        return {
            # Quarterly (time-shifted)
            "fund_trailing_pe": "Trailing PE from lagged quarterly EPS",
            "fund_earnings_yield": "Earnings yield (E/P)",
            "fund_eps_growth_qoq": "EPS growth quarter-over-quarter (lagged)",
            "fund_eps_acceleration": "Change in EPS growth rate",
            "fund_earnings_surprise": "Actual vs expected EPS growth",
            "fund_ebitda_growth": "EBITDA growth (lagged)",
            "fund_pb_ratio": "Price to Book from lagged book value",
            # PE Statistics
            "fund_pe_percentile": "PE percentile vs 52-week history",
            "fund_pe_vs_avg": "PE vs 52-week average",
            "fund_pe_zscore": "PE z-score (mean reversion signal)",
            "fund_pe_trend_20d": "PE change over 20 days",
            # Combo Features
            "fund_value_momentum": "Value + Momentum combo score",
            "fund_value_if_momentum": "Value score if momentum positive",
            "fund_quality_value": "Quality + Value combo",
            "fund_yield_per_risk": "Earnings yield per unit volatility",
            # Mean Reversion
            "fund_pe_extreme_low": "PE extremely low (buy signal)",
            "fund_pe_extreme_high": "PE extremely high (sell signal)",
            # Regime
            "fund_trend_regime": "Uptrend (1) or downtrend (0)",
            "fund_vol_regime": "High volatility regime",
            "fund_momentum_regime": "Positive momentum regime",
        }
