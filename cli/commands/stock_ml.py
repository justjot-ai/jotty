"""
Stock ML Command
================

Machine learning for stock market prediction with custom targets.

Usage:
    /stock-ml RELIANCE                          # Predict next day up/down
    /stock-ml RELIANCE --target next_5d_up      # Predict 5-day direction
    /stock-ml RELIANCE --target return_30d      # Predict 30-day return
    /stock-ml RELIANCE --timeframe 15minute     # Use 15-minute data
    /stock-ml --list                            # List available stocks

Target Types:
    next_Nd_up      - Binary: price up after N days (classification)
    return_Nd       - Continuous: N-day return percentage (regression)
    above_threshold - Binary: return > X% in N days
    volatility_Nd   - Predict N-day volatility (regression)

Timeframes:
    day, 15minute, 30minute, 60minute, week
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
from pathlib import Path
import asyncio
import warnings
import os
import glob
import json

warnings.filterwarnings('ignore')

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


from ._stock_ml_training import StockMLTrainingMixin
from ._stock_ml_swarm import StockMLSwarmMixin

class StockMLCommand(StockMLTrainingMixin, StockMLSwarmMixin, BaseCommand):
    """ML for stock market prediction with custom targets."""

    name = "stock-ml"
    aliases = ["sml", "stockml"]
    description = "Stock market ML with custom targets (next N days, returns, etc.)"
    usage = "/stock-ml <symbol> [--target <type>] [--timeframe <tf>] [--years <n>]"
    category = "ml"

    # Base data path
    DATA_BASE = Path("/var/www/sites/personal/stock_market/common/Data/NSE")

    # Timeframe mappings
    TIMEFRAMES = {
        "day": "DayData",
        "daily": "DayData",
        "15min": "15minuteData",
        "15minute": "15minuteData",
        "30min": "30minuteData",
        "30minute": "30minuteData",
        "60min": "60minuteData",
        "60minute": "60minuteData",
        "hourly": "60minuteData",
        "week": "WeekData",
        "weekly": "WeekData",
    }

    # Target type definitions
    TARGET_TYPES = {
        "next_1d_up": {"type": "classification", "days": 1, "desc": "Price up tomorrow"},
        "next_5d_up": {"type": "classification", "days": 5, "desc": "Price up in 5 days"},
        "next_10d_up": {"type": "classification", "days": 10, "desc": "Price up in 10 days"},
        "next_20d_up": {"type": "classification", "days": 20, "desc": "Price up in 20 days"},
        "next_30d_up": {"type": "classification", "days": 30, "desc": "Price up in 30 days"},
        "return_5d": {"type": "regression", "days": 5, "desc": "5-day return %"},
        "return_10d": {"type": "regression", "days": 10, "desc": "10-day return %"},
        "return_20d": {"type": "regression", "days": 20, "desc": "20-day return %"},
        "return_30d": {"type": "regression", "days": 30, "desc": "30-day return %"},
        "volatility_20d": {"type": "regression", "days": 20, "desc": "20-day volatility"},
    }

    # Popular stock sets for quick sweeps
    STOCK_SETS = {
        # Market Cap based
        "nifty50": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
                    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
                    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "BAJFINANCE",
                    "WIPRO", "ULTRACEMCO", "NESTLEIND", "HCLTECH", "TECHM", "ADANIENT",
                    "POWERGRID", "NTPC", "ONGC", "M&M", "TATAMOTORS", "JSWSTEEL", "COALINDIA"],
        "nifty_next50": ["ADANIPORTS", "GRASIM", "BAJAJFINSV", "DIVISLAB", "DRREDDY",
                         "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA", "DABUR",
                         "GODREJCP", "SIEMENS", "HAVELLS", "ABB", "BOSCHLTD"],
        "top10": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                  "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK"],
        "top20": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
                  "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
                  "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "BAJFINANCE",
                  "WIPRO", "HCLTECH", "TECHM"],

        # Sector based
        "nifty_bank": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                       "BANKBARODA", "INDUSINDBK", "PNB", "FEDERALBNK", "IDFCFIRSTB",
                       "AUBANK", "BANDHANBNK"],
        "nifty_psu_bank": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK",
                           "INDIANB", "IOB", "CENTRALBK", "MAHABANK", "BANKINDIA"],
        "nifty_it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS",
                     "COFORGE", "PERSISTENT", "LTTS"],
        "nifty_pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA",
                         "BIOCON", "LUPIN", "TORNTPHARM", "ALKEM", "GLENMARK"],
        "nifty_auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
                       "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BHARATFORG", "MOTHERSON"],
        "nifty_fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
                       "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "VBL"],
        "nifty_metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA",
                        "NMDC", "SAIL", "JINDALSTEL", "NATIONALUM", "HINDZINC"],
        "nifty_energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC",
                         "GAIL", "ADANIGREEN", "TATAPOWER", "ADANIENSOL"],
        "nifty_infra": ["LT", "ADANIPORTS", "ULTRACEMCO", "GRASIM", "SHREECEM",
                        "ACC", "AMBUJACEM", "DLF", "GODREJPROP", "OBEROIRLTY"],
        "nifty_realty": ["DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE",
                         "BRIGADE", "SOBHA", "SUNTECK", "MAHLIFE"],
        "nifty_finance": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                          "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI"],

        # Thematic
        "nifty_pse": ["NTPC", "POWERGRID", "ONGC", "COALINDIA", "IOC", "BPCL",
                      "GAIL", "NHPC", "NMDC", "SAIL", "BHEL", "BEL"],
        "nifty_cpse": ["NTPC", "POWERGRID", "ONGC", "COALINDIA", "IOC", "NHPC",
                       "NMDC", "SAIL", "BHEL", "CONCOR", "IRCTC"],
        "nifty_defence": ["HAL", "BEL", "BHARATFORGE", "COCHINSHIP", "MAZAGON",
                          "GRSE", "BDL", "DATAPATTNS"],
        "nifty_consumption": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TITAN",
                              "DABUR", "MARICO", "GODREJCP", "COLPAL", "JUBLFOOD"],

        # Legacy aliases
        "banks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                  "BANKBARODA", "INDUSINDBK", "PNB", "FEDERALBNK", "IDFCFIRSTB"],
        "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE"],
        "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "BIOCON"],
        "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
        "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
    }

    # Indices data source
    INDICES_JSON = Path("/var/www/sites/personal/stock_market/planmyinvesting.com/src/indices_data.json")
    INDICES_CACHE_DIR = Path.home() / ".jotty" / "indices_cache"

    # Sweep results file
    SWEEP_RESULTS_FILE = Path.home() / ".jotty" / "stock_ml_sweep_results.json"

    # SwarmML learning state file
    SWARM_ML_STATE_FILE = Path.home() / ".jotty" / "swarm_ml_learnings.json"
    Q_TABLE_PATH = Path.home() / ".jotty" / "stock_ml_q_table.json"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute stock ML pipeline."""
        import pandas as pd
        import numpy as np

        # Check for list flag
        if "list" in args.flags or "ls" in args.flags:
            return await self._list_stocks(args, cli)

        if "targets" in args.flags:
            return self._show_targets(cli)

        # Check for compare mode
        if "compare" in args.flags or "benchmark" in args.flags:
            symbol = args.positional[0].upper() if args.positional else None
            if not symbol:
                cli.renderer.error("Symbol required for comparison")
                return CommandResult.fail("Symbol required")
            return await self._run_comparison(args, cli, symbol)

        # Check for sweep mode (multi-stock, multi-target grid search)
        if "sweep" in args.flags or "grid" in args.flags:
            return await self._run_sweep(args, cli)

        # Check for leaderboard
        if "leaderboard" in args.flags or "lb" in args.flags:
            return self._show_leaderboard(cli)

        # Check for stock sets info
        if "sets" in args.flags:
            return self._show_stock_sets(cli)

        # Check for world-class comprehensive backtest
        if "wc" in args.flags or "world-class" in args.flags or "comprehensive-backtest" in args.flags:
            return await self._run_world_class_backtest(args, cli)

        # Check for unified/cross-stock training mode
        if "unified" in args.flags or "cross-stock" in args.flags:
            return await self._run_unified_training(args, cli)

        # Check for swarm-learn mode (multi-agent learning from results)
        if "swarm-learn" in args.flags or "auto-learn" in args.flags:
            return await self._run_swarm_learning(args, cli)

        # Check for insights mode (analyze learnings)
        if "insights" in args.flags:
            return self._show_swarm_insights(cli)

        # Check for cross-stock normalized training
        if "cross-stock" in args.flags or "normalized" in args.flags:
            return await self._run_cross_stock_normalized(args, cli)

        # Check for comprehensive features mode (uses /ml skills)
        if "comprehensive" in args.flags or "llm-features" in args.flags:
            return await self._run_with_comprehensive_features(args, cli)

        # Parse arguments
        symbol = args.positional[0].upper() if args.positional else None
        timeframe = args.flags.get("timeframe", args.flags.get("tf", "day"))
        target_type = args.flags.get("target", args.flags.get("t", "next_1d_up"))
        years = int(args.flags.get("years", args.flags.get("y", "5")))
        iterations = int(args.flags.get("iterations", args.flags.get("i", "2")))
        # MLflow is now ON by default, use --no-mlflow to disable
        use_mlflow = "no-mlflow" not in args.flags and "no_mlflow" not in args.flags
        experiment_name = args.flags.get("experiment", f"stock_ml_{symbol}")
        # Fundamental features from Yahoo Finance
        use_fundamentals = "fundamentals" in args.flags or "fund" in args.flags or "yf" in args.flags

        # Custom target parsing (e.g., next_30d_up, return_15d)
        target_config = self._parse_target(target_type)

        if not symbol:
            cli.renderer.error("Stock symbol required.")
            cli.renderer.info("")
            cli.renderer.info("Usage: /stock-ml <SYMBOL> [options]")
            cli.renderer.info("")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /stock-ml RELIANCE                    # Default: next day prediction")
            cli.renderer.info("  /stock-ml RELIANCE --target next_30d_up   # 30-day direction")
            cli.renderer.info("  /stock-ml RELIANCE --compare              # Compare all targets")
            cli.renderer.info("  /stock-ml --sweep --stocks top10          # Sweep top 10 stocks")
            cli.renderer.info("  /stock-ml --unified --stocks nifty_bank   # Cross-stock unified model")
            cli.renderer.info("  /stock-ml --leaderboard                   # Show sweep leaderboard")
            cli.renderer.info("  /stock-ml --list                      # List available stocks")
            cli.renderer.info("  /stock-ml --sets                      # Show stock sets")
            return CommandResult.fail("Symbol required")

        # Load stock data
        cli.renderer.info(f"Loading {symbol} ({timeframe} data, last {years} years)...")
        try:
            df = await self._load_stock_data(symbol, timeframe, years, cli)
            if df is None or len(df) < 100:
                cli.renderer.error(f"Insufficient data for {symbol}")
                return CommandResult.fail("Insufficient data")
        except Exception as e:
            cli.renderer.error(f"Failed to load data: {e}")
            return CommandResult.fail(str(e))

        cli.renderer.info(f"Loaded {len(df)} records ({df['date'].min().date()} to {df['date'].max().date()})")

        # Generate features and target
        cli.renderer.info(f"Target: {target_config['desc']} ({target_config['type']})")
        X, y, feature_names = self._create_features_and_target(df, target_config)

        if X is None or len(X) < 100:
            cli.renderer.error("Insufficient data after feature engineering")
            return CommandResult.fail("Insufficient data")

        cli.renderer.info(f"Technical Features: {len(feature_names)}")

        # Add fundamental features from Yahoo Finance if enabled
        if use_fundamentals:
            try:
                import pandas as pd
                from Jotty.core.skills.ml import FundamentalFeaturesSkill
                cli.renderer.status(f"Downloading fundamental data for {symbol}...")

                fund_skill = FundamentalFeaturesSkill()
                fund_features = await fund_skill.get_fundamental_features(symbol, df_prices=df)

                if fund_features is not None and not fund_features.empty:
                    # Remove duplicate dates, keep last value
                    fund_features = fund_features[~fund_features.index.duplicated(keep='last')]

                    # Get dates corresponding to X's index from original df
                    dates_series = pd.to_datetime(df.loc[X.index, 'date'])

                    # Map fundamental features to X's rows by date
                    fund_aligned = pd.DataFrame(index=X.index)
                    for col in fund_features.columns:
                        # Create a mapping from date to value
                        date_to_val = fund_features[col].to_dict()
                        fund_aligned[col] = dates_series.map(date_to_val)

                    # Forward/backward fill any missing values
                    fund_aligned = fund_aligned.ffill().bfill()

                    # Only add features with variance (time-varying)
                    added_features = []
                    for col in fund_aligned.columns:
                        n_unique = fund_aligned[col].dropna().nunique()
                        if n_unique > 10:  # Has meaningful variance
                            X[col] = fund_aligned[col].values
                            added_features.append(col)

                    if added_features:
                        feature_names = list(feature_names) + added_features
                        cli.renderer.info(f"Fundamental Features: {len(added_features)} (time-varying)")
                    else:
                        cli.renderer.info("Note: No time-varying fundamental features")
                else:
                    cli.renderer.info("Note: No fundamental data available")
            except Exception as e:
                cli.renderer.info(f"Note: Fundamental features skipped: {e}")

        cli.renderer.info(f"Total Features: {len(feature_names)}, Samples: {len(X)}")

        # Check if backtesting is enabled
        run_backtest = "backtest" in args.flags or "bt" in args.flags

        # Check if PDF report generation is enabled
        generate_report = "report" in args.flags or "pdf" in args.flags
        report_template = args.flags.get("template", "quantitative")

        # Check if comprehensive (world-class) backtest is enabled
        comprehensive_backtest = "world-class" in args.flags or "wc" in args.flags or "comprehensive-report" in args.flags

        # Run ML pipeline
        cli.renderer.info("")
        cli.renderer.header(f"Stock ML: {symbol}")
        cli.renderer.info(f"Target: {target_type}")
        cli.renderer.info(f"Problem: {target_config['type']}")

        try:
            result = await self._run_stock_ml(
                X, y, feature_names, target_config,
                symbol, iterations, cli,
                use_mlflow=use_mlflow,
                experiment_name=experiment_name,
                df_ohlcv=df,
                run_backtest=run_backtest,
                timeframe=timeframe,
                generate_report=generate_report,
                report_template=report_template,
                comprehensive_backtest=comprehensive_backtest
            )
            return CommandResult.ok(data=result)
        except Exception as e:
            cli.renderer.error(f"ML pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return CommandResult.fail(str(e))

    def _parse_target(self, target_type: str) -> Dict[str, Any]:
        """Parse target type string into config."""
        import re

        # Check predefined targets
        if target_type in self.TARGET_TYPES:
            return self.TARGET_TYPES[target_type].copy()

        # Parse custom targets like next_15d_up, return_45d
        match = re.match(r'next_(\d+)d_up', target_type)
        if match:
            days = int(match.group(1))
            return {"type": "classification", "days": days, "desc": f"Price up in {days} days"}

        match = re.match(r'return_(\d+)d', target_type)
        if match:
            days = int(match.group(1))
            return {"type": "regression", "days": days, "desc": f"{days}-day return %"}

        match = re.match(r'volatility_(\d+)d', target_type)
        if match:
            days = int(match.group(1))
            return {"type": "regression", "days": days, "desc": f"{days}-day volatility", "volatility": True}

        # Default to next day up/down
        return {"type": "classification", "days": 1, "desc": "Price up tomorrow"}

    async def _load_stock_data(self, symbol: str, timeframe: str, years: int, cli: Any) -> Optional['pd.DataFrame']:
        """Load stock data from files."""
        import pandas as pd
        from datetime import datetime

        # Get timeframe directory
        tf_dir = self.TIMEFRAMES.get(timeframe.lower(), "DayData")
        data_path = self.DATA_BASE / tf_dir

        if not data_path.exists():
            cli.renderer.error(f"Data directory not found: {data_path}")
            return None

        # Find matching files
        current_year = datetime.now().year
        start_year = current_year - years

        files = []
        for year in range(start_year, current_year + 1):
            pattern = f"{year}-*-{symbol}-*.csv.gz"
            matches = list(data_path.glob(pattern))
            files.extend(matches)

        if not files:
            cli.renderer.error(f"No data files found for {symbol}")
            # Try to find similar symbols
            all_files = list(data_path.glob(f"*{symbol}*.csv.gz"))[:5]
            if all_files:
                cli.renderer.info("Did you mean one of these?")
                for f in all_files:
                    cli.renderer.info(f"  {f.name}")
            return None

        # Load and concatenate
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f, compression='gzip', on_bad_lines='skip')
                df = df.loc[:, ~df.columns.duplicated()]
                dfs.append(df)
            except Exception as e:
                continue

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()

        # Handle duplicate columns (like date.1)
        df = df.loc[:, ~df.columns.duplicated()]

        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'].astype(str).str[:10], format='%Y-%m-%d', errors='coerce')

        # Keep only needed columns
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [c for c in cols if c in df.columns]
        if len(available_cols) < 5:
            cli.renderer.error(f"Missing required columns. Found: {list(df.columns)[:10]}")
            return None
        df = df[available_cols]

        # Clean
        df = df.dropna(subset=['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna().sort_values('date').reset_index(drop=True)

        return df

    def _create_features_and_target(self, df: 'pd.DataFrame', target_config: Dict) -> tuple:
        """Create features and target variable."""
        import pandas as pd
        import numpy as np

        df = df.copy()

        # ============ Technical Indicators ============

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / loss))

        # RSI variations
        for period in [7, 21]:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            df[f'rsi_{period}'] = 100 - (100 / (1 + gain / loss))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['close'] * 100

        # Stochastic
        for period in [14]:
            low_n = df['low'].rolling(period).min()
            high_n = df['high'].rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_n) / (high_n - low_n)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'close_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}'] * 100

        # Price momentum / Returns
        for period in [1, 2, 3, 5, 10, 20, 60]:
            df[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # Volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_change'] = df['volume'].pct_change()

        # Price patterns
        df['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['open_close_range'] = (df['close'] - df['open']) / df['open'] * 100
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close'] * 100
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close'] * 100

        # Trend indicators
        df['adx'] = self._calculate_adx(df)

        # ============ Create Target ============
        days = target_config['days']
        target_type = target_config['type']

        if target_config.get('volatility'):
            # Volatility target
            df['target'] = df['return_1d'].shift(-days).rolling(days).std()
        elif target_type == 'classification':
            # Binary up/down
            future_close = df['close'].shift(-days)
            df['target'] = (future_close > df['close']).astype(int)
        else:
            # Return regression
            df['target'] = ((df['close'].shift(-days) - df['close']) / df['close'] * 100)

        # Drop NaN and select features
        df = df.dropna()

        # Feature columns (exclude date, OHLCV, target)
        exclude = ['date', 'open', 'high', 'low', 'close', 'volume', 'target',
                   'bb_upper_20', 'bb_lower_20', 'sma_5', 'sma_10', 'sma_20',
                   'sma_50', 'sma_100', 'sma_200', 'ema_5', 'ema_10', 'ema_20',
                   'ema_50', 'ema_100', 'ema_200', 'volume_sma_20']

        feature_cols = [c for c in df.columns if c not in exclude]

        X = df[feature_cols]
        y = df['target']

        return X, y, feature_cols

    def _calculate_adx(self, df: 'pd.DataFrame', period: int = 14) -> 'pd.Series':
        """Calculate Average Directional Index."""
        import pandas as pd
        import numpy as np

        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def _run_backtest(self, df_ohlcv: Any, X: Any, y: Any, model: Any, target_config: Any, symbol: Any, cli: Any) -> Dict[str, Any]:
        """Run comprehensive backtesting with Buy & Hold comparison."""
        import pandas as pd
        import numpy as np

        cli.renderer.info("")
        cli.renderer.header("Backtesting Results")

        # Align OHLCV data with features
        df = df_ohlcv.copy()
        df = df.loc[X.index]

        if len(df) < 50:
            cli.renderer.error("Insufficient data for backtesting")
            return None

        # Generate predictions
        predictions = model.predict(X.values)
        probabilities = model.predict_proba(X.values)[:, 1] if hasattr(model, 'predict_proba') else predictions

        df['signal'] = predictions
        df['probability'] = probabilities

        # Split into train/test (use same split as ML)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()

        if len(test_df) < 20:
            cli.renderer.error("Insufficient test data for backtesting")
            return None

        # Calculate strategy returns
        test_df['returns'] = test_df['close'].pct_change()
        test_df['strategy_returns'] = test_df['signal'].shift(1) * test_df['returns']  # Signal from previous day
        test_df['bnh_returns'] = test_df['returns']  # Buy and hold

        # Cumulative returns
        test_df['strategy_cumret'] = (1 + test_df['strategy_returns']).cumprod()
        test_df['bnh_cumret'] = (1 + test_df['bnh_returns']).cumprod()

        # Calculate metrics
        metrics = self._calculate_backtest_metrics(test_df, target_config)

        # Display results
        cli.renderer.info("")
        cli.renderer.info("┌───────────────────────────────┬────────────────┬────────────────┐")
        cli.renderer.info("│           Metric              │    Strategy    │   Buy & Hold   │")
        cli.renderer.info("├───────────────────────────────┼────────────────┼────────────────┤")

        for metric_name, (strat_val, bnh_val) in metrics['comparison'].items():
            strat_str = f"{strat_val:>12.2f}" if isinstance(strat_val, (int, float)) else f"{strat_val:>12}"
            bnh_str = f"{bnh_val:>12.2f}" if isinstance(bnh_val, (int, float)) else f"{bnh_val:>12}"
            cli.renderer.info(f"│ {metric_name:<29} │ {strat_str:>14} │ {bnh_str:>14} │")

        cli.renderer.info("└───────────────────────────────┴────────────────┴────────────────┘")

        # Strategy performance summary
        cli.renderer.info("")
        cli.renderer.info("Strategy Performance:")
        cli.renderer.info(f"  Total Return:        {metrics['strategy']['total_return']:.2f}%")
        cli.renderer.info(f"  Annualized Return:   {metrics['strategy']['annual_return']:.2f}%")
        cli.renderer.info(f"  Sharpe Ratio:        {metrics['strategy']['sharpe']:.2f}")
        cli.renderer.info(f"  Sortino Ratio:       {metrics['strategy']['sortino']:.2f}")
        cli.renderer.info(f"  Max Drawdown:        {metrics['strategy']['max_drawdown']:.2f}%")
        cli.renderer.info(f"  Win Rate:            {metrics['strategy']['win_rate']:.2f}%")
        cli.renderer.info(f"  Profit Factor:       {metrics['strategy']['profit_factor']:.2f}")
        cli.renderer.info(f"  ROMAD:               {metrics['strategy']['romad']:.2f}")

        cli.renderer.info("")
        cli.renderer.info("Buy & Hold Performance:")
        cli.renderer.info(f"  Total Return:        {metrics['bnh']['total_return']:.2f}%")
        cli.renderer.info(f"  Max Drawdown:        {metrics['bnh']['max_drawdown']:.2f}%")
        cli.renderer.info(f"  ROMAD:               {metrics['bnh']['romad']:.2f}")

        # Outperformance
        outperformance = metrics['strategy']['total_return'] - metrics['bnh']['total_return']
        romad_ratio = metrics['strategy']['romad'] / metrics['bnh']['romad'] if metrics['bnh']['romad'] != 0 else 0

        cli.renderer.info("")
        if outperformance > 0:
            cli.renderer.info(f"Strategy OUTPERFORMS Buy & Hold by {outperformance:.2f}%")
        else:
            cli.renderer.info(f"Strategy UNDERPERFORMS Buy & Hold by {abs(outperformance):.2f}%")
        cli.renderer.info(f"ROMAD Ratio (Strategy/B&H): {romad_ratio:.2f}")

        # Trade statistics
        trades = metrics.get('trades', {})
        if trades:
            cli.renderer.info("")
            cli.renderer.info("Trade Statistics:")
            cli.renderer.info(f"  Total Trades:        {trades.get('total', 0)}")
            cli.renderer.info(f"  Winning Trades:      {trades.get('wins', 0)}")
            cli.renderer.info(f"  Losing Trades:       {trades.get('losses', 0)}")
            cli.renderer.info(f"  Avg Win:             {trades.get('avg_win', 0):.2f}%")
            cli.renderer.info(f"  Avg Loss:            {trades.get('avg_loss', 0):.2f}%")
            cli.renderer.info(f"  Expectancy:          {trades.get('expectancy', 0):.2f}%")

        return metrics

    def _calculate_backtest_metrics(self, df: 'pd.DataFrame', target_config: Dict) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        import numpy as np

        strategy_rets = df['strategy_returns'].dropna()
        bnh_rets = df['bnh_returns'].dropna()

        # Basic returns
        strategy_total = (df['strategy_cumret'].iloc[-1] - 1) * 100 if len(df) > 0 else 0
        bnh_total = (df['bnh_cumret'].iloc[-1] - 1) * 100 if len(df) > 0 else 0

        # Annualized returns (assuming 252 trading days)
        n_days = len(df)
        strategy_annual = ((1 + strategy_total/100) ** (252/n_days) - 1) * 100 if n_days > 0 else 0
        bnh_annual = ((1 + bnh_total/100) ** (252/n_days) - 1) * 100 if n_days > 0 else 0

        # Volatility (annualized)
        strategy_vol = strategy_rets.std() * np.sqrt(252) * 100 if len(strategy_rets) > 0 else 0
        bnh_vol = bnh_rets.std() * np.sqrt(252) * 100 if len(bnh_rets) > 0 else 0

        # Sharpe Ratio (assuming 0 risk-free rate)
        strategy_sharpe = (strategy_annual / strategy_vol) if strategy_vol > 0 else 0
        bnh_sharpe = (bnh_annual / bnh_vol) if bnh_vol > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_rets = strategy_rets[strategy_rets < 0]
        downside_std = downside_rets.std() * np.sqrt(252) * 100 if len(downside_rets) > 0 else 0
        strategy_sortino = (strategy_annual / downside_std) if downside_std > 0 else 0

        bnh_downside = bnh_rets[bnh_rets < 0]
        bnh_downside_std = bnh_downside.std() * np.sqrt(252) * 100 if len(bnh_downside) > 0 else 0
        bnh_sortino = (bnh_annual / bnh_downside_std) if bnh_downside_std > 0 else 0

        # Max Drawdown
        def calc_max_drawdown(cumret: Any) -> Any:
            peak = cumret.expanding(min_periods=1).max()
            drawdown = (cumret - peak) / peak
            return drawdown.min() * 100

        strategy_mdd = calc_max_drawdown(df['strategy_cumret'])
        bnh_mdd = calc_max_drawdown(df['bnh_cumret'])

        # ROMAD (Return Over Max Drawdown)
        strategy_romad = -strategy_total / strategy_mdd if strategy_mdd != 0 else 0
        bnh_romad = -bnh_total / bnh_mdd if bnh_mdd != 0 else 0

        # Win Rate and Profit Factor
        winning_days = strategy_rets[strategy_rets > 0]
        losing_days = strategy_rets[strategy_rets < 0]
        win_rate = len(winning_days) / len(strategy_rets) * 100 if len(strategy_rets) > 0 else 0

        gross_profit = winning_days.sum() * 100
        gross_loss = abs(losing_days.sum() * 100)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Trade statistics (based on signal changes)
        signals = df['signal']
        signal_changes = signals.diff().fillna(0)
        entries = signal_changes[signal_changes == 1]
        exits = signal_changes[signal_changes == -1]

        # Simplified trade calculation
        trades_info = {
            'total': len(entries),
            'wins': len(winning_days),
            'losses': len(losing_days),
            'avg_win': winning_days.mean() * 100 if len(winning_days) > 0 else 0,
            'avg_loss': losing_days.mean() * 100 if len(losing_days) > 0 else 0,
            'expectancy': strategy_rets.mean() * 100 if len(strategy_rets) > 0 else 0,
        }

        return {
            'strategy': {
                'total_return': strategy_total,
                'annual_return': strategy_annual,
                'volatility': strategy_vol,
                'sharpe': strategy_sharpe,
                'sortino': strategy_sortino,
                'max_drawdown': strategy_mdd,
                'romad': strategy_romad,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
            },
            'bnh': {
                'total_return': bnh_total,
                'annual_return': bnh_annual,
                'volatility': bnh_vol,
                'sharpe': bnh_sharpe,
                'sortino': bnh_sortino,
                'max_drawdown': bnh_mdd,
                'romad': bnh_romad,
            },
            'comparison': {
                'Total Return (%)': (strategy_total, bnh_total),
                'Annual Return (%)': (strategy_annual, bnh_annual),
                'Volatility (%)': (strategy_vol, bnh_vol),
                'Sharpe Ratio': (strategy_sharpe, bnh_sharpe),
                'Sortino Ratio': (strategy_sortino, bnh_sortino),
                'Max Drawdown (%)': (strategy_mdd, bnh_mdd),
                'ROMAD': (strategy_romad, bnh_romad),
            },
            'trades': trades_info,
            'outperformance': strategy_total - bnh_total,
            'romad_ratio': strategy_romad / bnh_romad if bnh_romad != 0 else 0,
        }

    async def _run_stock_ml(self, X: Any, y: Any, feature_names: Any, target_config: Any, symbol: Any, max_iterations: Any, cli: Any, use_mlflow: Any = True, experiment_name: Any = 'stock', df_ohlcv: Any = None, run_backtest: Any = False, timeframe: Any = 'day', generate_report: Any = False, report_template: Any = 'quantitative', comprehensive_backtest: Any = False) -> Any:
        """Run ML pipeline for stock prediction with auto-MLflow logging and optional backtesting."""
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error

        problem_type = target_config['type']
        is_classification = problem_type == 'classification'

        # Time series split (no shuffle - preserve order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        cli.renderer.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Auto-initialize MLflow (default ON for all runs)
        mlflow_tracker = None
        if use_mlflow:
            try:
                from Jotty.core.skills.ml import MLflowTrackerSkill
                from .ml import MLCommand

                mlflow_tracker = MLflowTrackerSkill()
                # Use stock_ml_{symbol} as experiment name for easy querying
                auto_experiment_name = f"stock_ml_{symbol}"
                await mlflow_tracker.init(experiment_name=auto_experiment_name)

                # Create descriptive run name
                run_name = f"{symbol}_{target_config['days']}d_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
                await mlflow_tracker.start_run(
                    run_name=run_name,
                    tags={
                        'symbol': symbol,
                        'target': target_config.get('desc', ''),
                        'timeframe': timeframe,
                        'problem_type': problem_type,
                    }
                )

                # Log comprehensive parameters
                await mlflow_tracker.log_params({
                    'symbol': symbol,
                    'target_type': target_config['type'],
                    'target_days': str(target_config['days']),
                    'target_desc': target_config.get('desc', ''),
                    'timeframe': timeframe,
                    'n_features': str(len(feature_names)),
                    'train_samples': str(len(X_train)),
                    'test_samples': str(len(X_test)),
                    'data_points': str(len(X)),
                    'feature_list': ','.join(feature_names[:20]),  # Top 20 features
                })
                cli.renderer.info(f"MLflow: Logging to experiment '{auto_experiment_name}'")
            except Exception as e:
                cli.renderer.info(f"Note: MLflow initialization skipped: {e}")
                mlflow_tracker = None

        # Import models
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier, CatBoostRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Timeframe-aware model selection
        # RoMaD-optimized hyperparameters from Optuna optimization (2026-02-01)
        is_intraday = timeframe.lower() in ['15min', '15minute', '30min', '30minute',
                                             '60min', '60minute', 'hourly']

        # Model zoo - Optimized hyperparameters for financial time series
        if is_classification:
            if is_intraday:
                # RoMaD-optimized params for intraday (shallow trees, slow learning, high regularization)
                # Optimized on 60min data from 2015-2025, achieved +852% RoMaD improvement
                cli.renderer.info("Using RoMaD-optimized intraday hyperparameters")
                models = {
                    'LightGBM': lgb.LGBMClassifier(
                        n_estimators=386,
                        learning_rate=0.0067,
                        max_depth=4,
                        num_leaves=32,
                        min_child_samples=7,
                        subsample=0.80,
                        colsample_bytree=0.50,
                        reg_alpha=0.013,
                        reg_lambda=0.017,
                        verbose=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    'XGBoost': xgb.XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.01,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.75,
                        colsample_bytree=0.5,
                        gamma=0.05,
                        reg_alpha=0.01,
                        reg_lambda=0.02,
                        verbosity=0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method='hist',
                    ),
                    'CatBoost': CatBoostClassifier(
                        iterations=400,
                        learning_rate=0.01,
                        depth=4,
                        l2_leaf_reg=5,
                        bagging_temperature=0.3,
                        random_strength=0.3,
                        verbose=0,
                        random_state=42,
                        thread_count=-1,
                    ),
                    'RandomForest': RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_split=15,
                        min_samples_leaf=10,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                    ),
                }
            else:
                # Daily timeframe params (deeper trees, faster learning)
                models = {
                    'LightGBM': lgb.LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.02,
                        max_depth=8,
                        num_leaves=63,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        verbose=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    'XGBoost': xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.02,
                        max_depth=7,
                        min_child_weight=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        gamma=0.1,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        verbosity=0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method='hist',
                    ),
                    'CatBoost': CatBoostClassifier(
                        iterations=500,
                        learning_rate=0.02,
                        depth=7,
                        l2_leaf_reg=3,
                        bagging_temperature=0.5,
                        random_strength=0.5,
                        verbose=0,
                        random_state=42,
                        thread_count=-1,
                    ),
                    'RandomForest': RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                    ),
                }
        else:
            # Regression models - also timeframe-aware
            if is_intraday:
                # RoMaD-optimized params for intraday regression
                cli.renderer.info("Using RoMaD-optimized intraday hyperparameters (regression)")
                models = {
                    'LightGBM': lgb.LGBMRegressor(
                        n_estimators=386,
                        learning_rate=0.0067,
                        max_depth=4,
                        num_leaves=32,
                        min_child_samples=7,
                        subsample=0.80,
                        colsample_bytree=0.50,
                        reg_alpha=0.013,
                        reg_lambda=0.017,
                        verbose=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    'XGBoost': xgb.XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.01,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.75,
                        colsample_bytree=0.5,
                        gamma=0.05,
                        reg_alpha=0.01,
                        reg_lambda=0.02,
                        verbosity=0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method='hist',
                    ),
                    'CatBoost': CatBoostRegressor(
                        iterations=400,
                        learning_rate=0.01,
                        depth=4,
                        l2_leaf_reg=5,
                        bagging_temperature=0.3,
                        random_strength=0.3,
                        verbose=0,
                        random_state=42,
                        thread_count=-1,
                    ),
                    'RandomForest': RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_split=15,
                        min_samples_leaf=10,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                    ),
                }
            else:
                # Daily timeframe regression params
                models = {
                    'LightGBM': lgb.LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.02,
                        max_depth=8,
                        num_leaves=63,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        verbose=-1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    'XGBoost': xgb.XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.02,
                        max_depth=7,
                        min_child_weight=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        gamma=0.1,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        verbosity=0,
                        random_state=42,
                        n_jobs=-1,
                        tree_method='hist',
                    ),
                    'CatBoost': CatBoostRegressor(
                        iterations=500,
                        learning_rate=0.02,
                        depth=7,
                        l2_leaf_reg=3,
                        bagging_temperature=0.5,
                        random_strength=0.5,
                        verbose=0,
                        random_state=42,
                        thread_count=-1,
                    ),
                    'RandomForest': RandomForestRegressor(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1,
                    ),
                }

        # Train and evaluate models
        cli.renderer.info("")
        cli.renderer.header("Model Training")

        results = []
        best_model = None
        best_score = -np.inf
        best_name = None

        for name, model in models.items():
            cli.renderer.status(f"Training {name}...")
            model.fit(X_train.values, y_train.values)

            pred = model.predict(X_test.values)

            if is_classification:
                proba = model.predict_proba(X_test.values)
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred)
                try:
                    auc = roc_auc_score(y_test, proba[:, 1])
                except:
                    auc = acc
                score = auc

                results.append({
                    'model': name,
                    'accuracy': acc,
                    'f1': f1,
                    'auc': auc,
                })
                cli.renderer.info(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            else:
                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                score = r2

                results.append({
                    'model': name,
                    'r2': r2,
                    'rmse': rmse,
                })
                cli.renderer.info(f"  {name}: R²={r2:.4f}, RMSE={rmse:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        # Results table
        cli.renderer.info("")
        cli.renderer.header("Results Summary")

        if is_classification:
            cli.renderer.info("┌─────────────────┬──────────┬──────────┬──────────┐")
            cli.renderer.info("│     Model       │ Accuracy │    F1    │  ROC-AUC │")
            cli.renderer.info("├─────────────────┼──────────┼──────────┼──────────┤")
            for r in results:
                marker = " " if r['model'] == best_name else " "
                cli.renderer.info(f"│{marker}{r['model']:<13} │ {r['accuracy']:^8.4f} │ {r['f1']:^8.4f} │ {r['auc']:^8.4f} │")
            cli.renderer.info("└─────────────────┴──────────┴──────────┴──────────┘")
        else:
            cli.renderer.info("┌─────────────────┬──────────┬──────────┐")
            cli.renderer.info("│     Model       │    R²    │   RMSE   │")
            cli.renderer.info("├─────────────────┼──────────┼──────────┤")
            for r in results:
                marker = " " if r['model'] == best_name else " "
                cli.renderer.info(f"│{marker}{r['model']:<13} │ {r['r2']:^8.4f} │ {r['rmse']:^8.4f} │")
            cli.renderer.info("└─────────────────┴──────────┴──────────┘")

        cli.renderer.info("")
        cli.renderer.info(f"Best Model: {best_name}")
        cli.renderer.info(f"Best Score: {best_score:.4f}")

        # Feature importance (normalized to percentages)
        if hasattr(best_model, 'feature_importances_'):
            raw_importance = best_model.feature_importances_
            total = sum(raw_importance) if sum(raw_importance) > 0 else 1
            importance = {feat: (imp / total) * 100 for feat, imp in zip(feature_names, raw_importance)}
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]

            cli.renderer.info("")
            cli.renderer.info("Top 15 Features:")
            cli.renderer.info("┌────────────────────────────────┬────────────┐")
            cli.renderer.info("│           Feature              │ Importance │")
            cli.renderer.info("├────────────────────────────────┼────────────┤")
            for feat, imp in sorted_imp:
                feat_display = feat[:30] if len(feat) > 30 else feat
                cli.renderer.info(f"│ {feat_display:<30} │ {imp:>8.2f} % │")
            cli.renderer.info("└────────────────────────────────┴────────────┘")

        # Log comprehensive metrics to MLflow
        if mlflow_tracker:
            # Log best model metrics
            metrics = {
                'best_score': best_score,
                'best_model_name': best_name,
            }

            # Log metrics for ALL models (prefixed by model name)
            for r in results:
                model_prefix = r['model'].lower().replace(' ', '_')
                if is_classification:
                    metrics[f'{model_prefix}_auc'] = r.get('auc', 0)
                    metrics[f'{model_prefix}_accuracy'] = r.get('accuracy', 0)
                    metrics[f'{model_prefix}_f1'] = r.get('f1', 0)
                else:
                    metrics[f'{model_prefix}_r2'] = r.get('r2', 0)
                    metrics[f'{model_prefix}_rmse'] = r.get('rmse', 0)

            # Add best model's metrics at top level for easy querying
            if is_classification:
                best_result = next((r for r in results if r['model'] == best_name), results[0])
                metrics['test_auc'] = best_result.get('auc', 0)
                metrics['test_accuracy'] = best_result.get('accuracy', 0)
                metrics['test_f1'] = best_result.get('f1', 0)
            else:
                best_result = next((r for r in results if r['model'] == best_name), results[0])
                metrics['test_r2'] = best_result.get('r2', 0)
                metrics['test_rmse'] = best_result.get('rmse', 0)

            await mlflow_tracker.log_metrics(metrics)

            # Log feature importance
            if hasattr(best_model, 'feature_importances_'):
                await mlflow_tracker.log_feature_importance(importance, top_n=20)

            # Log best model artifact
            model_uri = await mlflow_tracker.log_model(
                best_model,
                model_name=f"{symbol}_best",
                registered_name=f"stock_ml/{symbol}/{target_config['days']}d"
            )
            if model_uri:
                cli.renderer.info(f"Model logged to MLflow: {model_uri}")

        # Run backtesting if enabled
        backtest_results = None
        if run_backtest and df_ohlcv is not None and is_classification:
            backtest_results = self._run_backtest(
                df_ohlcv, X, y, best_model, target_config, symbol, cli
            )

            # Log backtest results to MLflow
            if mlflow_tracker and backtest_results:
                backtest_metrics = {
                    'backtest_total_return': backtest_results['strategy']['total_return'],
                    'backtest_annual_return': backtest_results['strategy']['annual_return'],
                    'backtest_sharpe': backtest_results['strategy']['sharpe'],
                    'backtest_sortino': backtest_results['strategy']['sortino'],
                    'backtest_max_drawdown': backtest_results['strategy']['max_drawdown'],
                    'backtest_romad': backtest_results['strategy']['romad'],
                    'backtest_win_rate': backtest_results['strategy']['win_rate'],
                    'backtest_profit_factor': backtest_results['strategy']['profit_factor'],
                    'bnh_total_return': backtest_results['bnh']['total_return'],
                    'bnh_sharpe': backtest_results['bnh']['sharpe'],
                    'beats_buy_hold': 1 if backtest_results.get('outperformance', 0) > 0 else 0,
                    'outperformance': backtest_results.get('outperformance', 0),
                }
                await mlflow_tracker.log_metrics(backtest_metrics)

        # Generate PDF report if enabled
        report_paths = None
        if generate_report and backtest_results:
            try:
                cli.renderer.info("")
                cli.renderer.status("Generating backtest report...")

                from Jotty.core.skills.ml.backtest_report import (
                    BacktestReportSkill,
                    BacktestResult,
                    BacktestMetrics,
                    TradeStatistics,
                    ModelResults,
                )

                # Build BacktestResult from collected data
                bt_result = BacktestResult(
                    symbol=symbol,
                    target_type=target_config.get('desc', target_config['type']),
                    target_days=target_config['days'],
                    timeframe=timeframe,
                    problem_type=target_config['type'],
                    start_date=str(df_ohlcv['date'].min().date()) if df_ohlcv is not None else "",
                    end_date=str(df_ohlcv['date'].max().date()) if df_ohlcv is not None else "",
                    trading_days=len(df_ohlcv) if df_ohlcv is not None else 0,
                )

                # Strategy metrics
                bt_result.strategy_metrics = BacktestMetrics(
                    total_return=backtest_results['strategy']['total_return'],
                    annual_return=backtest_results['strategy']['annual_return'],
                    volatility=backtest_results['strategy']['volatility'],
                    sharpe_ratio=backtest_results['strategy']['sharpe'],
                    sortino_ratio=backtest_results['strategy']['sortino'],
                    max_drawdown=backtest_results['strategy']['max_drawdown'],
                    romad=backtest_results['strategy']['romad'],
                    win_rate=backtest_results['strategy']['win_rate'],
                    profit_factor=backtest_results['strategy']['profit_factor'],
                )

                # Benchmark metrics
                bt_result.benchmark_metrics = BacktestMetrics(
                    total_return=backtest_results['bnh']['total_return'],
                    annual_return=backtest_results['bnh']['annual_return'],
                    volatility=backtest_results['bnh']['volatility'],
                    sharpe_ratio=backtest_results['bnh']['sharpe'],
                    sortino_ratio=backtest_results['bnh']['sortino'],
                    max_drawdown=backtest_results['bnh']['max_drawdown'],
                    romad=backtest_results['bnh']['romad'],
                )

                # Trade stats
                trades = backtest_results.get('trades', {})
                bt_result.trade_stats = TradeStatistics(
                    total_trades=trades.get('total', 0),
                    winning_trades=trades.get('wins', 0),
                    losing_trades=trades.get('losses', 0),
                    avg_win=trades.get('avg_win', 0),
                    avg_loss=trades.get('avg_loss', 0),
                    expectancy=trades.get('expectancy', 0),
                )

                # Model results
                bt_result.models = [
                    ModelResults(
                        name=r['model'],
                        accuracy=r.get('accuracy', 0),
                        f1_score=r.get('f1', 0),
                        auc=r.get('auc', 0),
                        r2=r.get('r2', 0),
                        rmse=r.get('rmse', 0),
                        is_best=(r['model'] == best_name),
                    )
                    for r in results
                ]
                bt_result.best_model = best_name

                # Feature importance
                if hasattr(best_model, 'feature_importances_'):
                    bt_result.feature_importance = dict(sorted_imp)

                # Build equity curve from test data
                if df_ohlcv is not None:
                    split_idx = int(len(df_ohlcv) * 0.8)
                    test_df = df_ohlcv.iloc[split_idx:].copy()
                    if len(test_df) > 20:
                        test_df['returns'] = test_df['close'].pct_change()
                        test_df['strategy_cumret'] = (1 + test_df['returns']).cumprod()
                        test_df['bnh_cumret'] = (1 + test_df['returns']).cumprod()

                        # Calculate drawdown
                        cumret = test_df['strategy_cumret']
                        peak = cumret.expanding(min_periods=1).max()
                        drawdown = ((cumret - peak) / peak) * 100

                        bt_result.equity_curve = [
                            {
                                'date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date'])[:10],
                                'strategy': row['strategy_cumret'],
                                'benchmark': row['bnh_cumret'],
                                'drawdown': dd,
                            }
                            for (idx, row), dd in zip(test_df.iterrows(), drawdown)
                            if pd.notna(row['strategy_cumret'])
                        ]

                # Generate report
                report_skill = BacktestReportSkill()
                report_result = await report_skill.execute(bt_result, template=report_template)

                if report_result.get('status') == 'success':
                    report_paths = {
                        'markdown': report_result.get('markdown_path'),
                        'pdf': report_result.get('pdf_path'),
                    }
                    cli.renderer.info(f"Report generated: {report_result.get('pdf_path')}")

                    # Try to send via Telegram
                    try:
                        from Jotty.core.skills.notification import TelegramNotifierSkill
                        telegram = TelegramNotifierSkill()
                        if report_paths.get('pdf'):
                            await telegram.send_document(
                                document_path=report_paths['pdf'],
                                caption=f" ML Backtest Report: {symbol}\n"
                                       f"Strategy Return: {bt_result.strategy_metrics.total_return:+.1f}%\n"
                                       f"Sharpe: {bt_result.strategy_metrics.sharpe_ratio:.2f}"
                            )
                            cli.renderer.info("Report sent to Telegram")
                    except Exception as te:
                        cli.renderer.info(f"Note: Telegram send skipped: {te}")

                else:
                    cli.renderer.info(f"Note: Report generation issue: {report_result.get('error', 'unknown')}")

            except Exception as e:
                cli.renderer.info(f"Note: Report generation skipped: {e}")
                import traceback
                traceback.print_exc()

        # Generate comprehensive (world-class) backtest report if enabled
        comprehensive_report_paths = None
        if comprehensive_backtest and df_ohlcv is not None and is_classification:
            try:
                cli.renderer.info("")
                cli.renderer.status("Running World-Class Backtest Engine...")

                from Jotty.core.skills.ml.backtest_engine import (
                    WorldClassBacktestEngine,
                    TransactionCosts,
                )
                from Jotty.core.skills.ml.comprehensive_backtest_report import (
                    ComprehensiveBacktestReportGenerator,
                )

                # Get predictions on full test set for signals
                split_idx = int(len(X) * 0.8)
                X_test_full = X.iloc[split_idx:]
                signals = best_model.predict(X_test_full)

                # Pad signals to match price data
                full_signals = np.zeros(len(df_ohlcv))
                full_signals[split_idx:split_idx+len(signals)] = signals

                # Create cost model
                costs = TransactionCosts(
                    commission_pct=0.001,  # 0.1%
                    slippage_pct=0.001,    # 0.1%
                    market_impact_pct=0.0005  # 0.05%
                )

                # Run comprehensive backtest
                engine = WorldClassBacktestEngine(risk_free_rate=0.05)
                comp_result = engine.run_backtest(
                    prices=df_ohlcv,
                    signals=full_signals,
                    costs=costs,
                    walk_forward_windows=5,
                    monte_carlo_sims=1000,
                    target_volatility=0.15
                )

                comp_result.symbol = symbol
                comp_result.strategy_name = f"ML {target_config.get('desc', 'Strategy')}"

                # Print comprehensive results summary
                cli.renderer.info("")
                cli.renderer.header("World-Class Backtest Results")
                cli.renderer.info(f"Total Return (Gross): {comp_result.total_return:+.2f}%")
                cli.renderer.info(f"Total Return (Net): {comp_result.total_return_net:+.2f}%")
                cli.renderer.info(f"Sharpe Ratio: {comp_result.sharpe_ratio:.2f}")
                cli.renderer.info(f"Monte Carlo P(Positive): {comp_result.monte_carlo.prob_positive*100:.1f}%")
                cli.renderer.info(f"Walk-Forward Avg OOS Sharpe: {comp_result.wf_avg_oos_sharpe:.2f}")
                cli.renderer.info(f"Statistical P-Value: {comp_result.statistical_tests.p_value:.4f}")

                # Generate comprehensive report
                cli.renderer.info("")
                cli.renderer.status("Generating Comprehensive PDF Report...")
                report_gen = ComprehensiveBacktestReportGenerator()
                md_path, pdf_path = await report_gen.generate_report(comp_result, template_name=report_template)

                if pdf_path:
                    comprehensive_report_paths = {
                        'markdown': str(md_path),
                        'pdf': str(pdf_path),
                    }
                    cli.renderer.info(f"Comprehensive Report: {pdf_path}")

                    # Send to Telegram
                    try:
                        sent = await report_gen.send_to_telegram(pdf_path, comp_result)
                        if sent:
                            cli.renderer.info("Report sent to Telegram")
                    except Exception as te:
                        cli.renderer.info(f"Note: Telegram send skipped: {te}")

            except Exception as e:
                cli.renderer.info(f"Note: Comprehensive backtest skipped: {e}")
                import traceback
                traceback.print_exc()

        # End MLflow run
        if mlflow_tracker:
            run_info = await mlflow_tracker.end_run()
            if run_info:
                from .ml import MLCommand
                auto_experiment_name = f"stock_ml_{symbol}"
                MLCommand.save_mlflow_state(auto_experiment_name, run_info['run_id'])
                cli.renderer.info(f"MLflow run: {run_info['run_id']}")

        return {
            'symbol': symbol,
            'target': target_config,
            'best_model': best_name,
            'best_score': best_score,
            'results': results,
            'feature_importance': sorted_imp[:15] if hasattr(best_model, 'feature_importances_') else [],
            'backtest': backtest_results,
            'report_paths': report_paths,
            'comprehensive_report_paths': comprehensive_report_paths,
        }

    async def _list_stocks(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """List available stocks."""
        import re

        timeframe = args.flags.get("timeframe", args.flags.get("tf", "day"))
        tf_dir = self.TIMEFRAMES.get(timeframe.lower(), "DayData")
        data_path = self.DATA_BASE / tf_dir

        cli.renderer.header(f"Available Stocks ({timeframe})")

        if not data_path.exists():
            cli.renderer.error(f"Data directory not found: {data_path}")
            return CommandResult.fail("Directory not found")

        # Get unique symbols from 2024
        files = list(data_path.glob("2024-*.csv.gz"))
        symbols = set()

        for f in files:
            # Extract symbol from filename like 2024-NSE-SPOT-RELIANCE-RELIANCE-EQ-EQ.csv.gz
            match = re.search(r'2024-[^-]+-[^-]+-([^-]+)-', f.name)
            if match:
                symbols.add(match.group(1))

        symbols = sorted(symbols)

        cli.renderer.info(f"Found {len(symbols)} stocks")
        cli.renderer.info("")

        # Display in columns
        cols = 5
        for i in range(0, len(symbols), cols):
            row = symbols[i:i+cols]
            cli.renderer.info("  " + "  ".join(f"{s:<12}" for s in row))

        return CommandResult.ok(data=list(symbols))

    def _show_targets(self, cli: "JottyCLI") -> CommandResult:
        """Show available target types."""
        cli.renderer.header("Available Target Types")
        cli.renderer.info("")
        cli.renderer.info("Predefined targets:")
        cli.renderer.info("┌─────────────────┬────────────────┬───────────────────────────────┐")
        cli.renderer.info("│     Target      │      Type      │         Description           │")
        cli.renderer.info("├─────────────────┼────────────────┼───────────────────────────────┤")
        for name, config in self.TARGET_TYPES.items():
            cli.renderer.info(f"│ {name:<15} │ {config['type']:<14} │ {config['desc']:<29} │")
        cli.renderer.info("└─────────────────┴────────────────┴───────────────────────────────┘")

        cli.renderer.info("")
        cli.renderer.info("Custom targets (dynamic):")
        cli.renderer.info("  next_Nd_up     - Binary: price up after N days")
        cli.renderer.info("  return_Nd      - Continuous: N-day return percentage")
        cli.renderer.info("  volatility_Nd  - Continuous: N-day volatility")
        cli.renderer.info("")
        cli.renderer.info("Examples:")
        cli.renderer.info("  /stock-ml RELIANCE --target next_45d_up")
        cli.renderer.info("  /stock-ml RELIANCE --target return_90d")

        return CommandResult.ok(data=self.TARGET_TYPES)

    def _add_advanced_momentum(self, X: 'pd.DataFrame', df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add advanced momentum indicators."""
        import pandas as pd
        import numpy as np

        close = df['close']

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            X[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100

        # Momentum acceleration (2nd derivative)
        X['momentum_accel'] = X.get('return_5d', close.pct_change(5)).diff()

        # Williams %R
        for period in [14, 21]:
            high_n = df['high'].rolling(period).max()
            low_n = df['low'].rolling(period).min()
            X[f'williams_r_{period}'] = -100 * (high_n - close) / (high_n - low_n)

        # Commodity Channel Index (CCI)
        tp = (df['high'] + df['low'] + close) / 3
        for period in [20]:
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            X[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)

        # Trend strength
        X['trend_strength'] = abs(X.get('close_vs_sma_20', 0)) + abs(X.get('close_vs_sma_50', 0))

        return X

    def _add_volatility_regime(self, X: 'pd.DataFrame', df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add volatility regime indicators."""
        import pandas as pd
        import numpy as np

        returns = df['close'].pct_change()

        # Historical volatility at multiple windows
        for period in [5, 10, 20, 60]:
            vol = returns.rolling(period).std() * np.sqrt(252)
            X[f'hist_vol_{period}'] = vol

        # Volatility ratio (short/long)
        if 'hist_vol_5' in X.columns and 'hist_vol_20' in X.columns:
            X['vol_ratio_5_20'] = X['hist_vol_5'] / X['hist_vol_20'].replace(0, np.nan)

        # Parkinson volatility (using high-low)
        hl_ratio = np.log(df['high'] / df['low'])
        X['parkinson_vol'] = np.sqrt(hl_ratio.rolling(20).apply(lambda x: (x**2).sum() / (4 * len(x) * np.log(2))))

        # Volatility regime (high/low)
        vol_20 = X.get('hist_vol_20', returns.rolling(20).std())
        vol_mean = vol_20.rolling(60).mean()
        X['vol_regime'] = (vol_20 > vol_mean).astype(int)

        # Volatility percentile
        X['vol_percentile'] = vol_20.rolling(252).apply(lambda x: (x.iloc[-1] > x).mean() if len(x) > 0 else 0.5)

        return X

    def _add_volume_profile(self, X: 'pd.DataFrame', df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add volume profile features."""
        import pandas as pd
        import numpy as np

        volume = df['volume'].values
        close = df['close'].values

        # On-Balance Volume (OBV)
        obv = (np.sign(np.diff(close, prepend=close[0])) * volume).cumsum()
        obv_series = pd.Series(obv, index=df.index)
        obv_ma_20 = obv_series.rolling(20).mean()

        X['obv'] = obv_series.loc[X.index].values
        X['obv_ma_20'] = obv_ma_20.loc[X.index].values
        X['obv_trend'] = (X['obv'] > X['obv_ma_20']).astype(int)

        # Volume Price Trend (VPT)
        close_series = df['close']
        volume_series = df['volume']
        vpt = (volume_series * close_series.pct_change()).cumsum()
        X['vpt'] = vpt.loc[X.index].values

        # Money Flow Index (MFI)
        tp = (df['high'] + df['low'] + close_series) / 3
        raw_mf = tp * volume_series
        pos_mf = raw_mf.where(tp > tp.shift(), 0).rolling(14).sum()
        neg_mf = raw_mf.where(tp < tp.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
        X['mfi'] = mfi.loc[X.index].values

        # Volume weighted price
        vol_close_sum = (volume_series * close_series).rolling(20).sum()
        vol_sum = volume_series.rolling(20).sum()
        vwap_ratio = close_series / vol_close_sum * vol_sum
        X['vwap_ratio'] = vwap_ratio.loc[X.index].values

        # Accumulation/Distribution
        clv = ((close_series - df['low']) - (df['high'] - close_series)) / (df['high'] - df['low']).replace(0, np.nan)
        ad_line = (clv * volume_series).cumsum()
        X['ad_line'] = ad_line.loc[X.index].values

        return X

    def _add_pattern_features(self, X: 'pd.DataFrame', df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add candlestick pattern features."""
        import pandas as pd
        import numpy as np

        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        body = c - o
        upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
        body_size = abs(body)

        # Doji (small body)
        X['is_doji'] = (body_size < (h - l) * 0.1).astype(int)

        # Hammer (long lower shadow)
        X['is_hammer'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size * 0.5)).astype(int)

        # Engulfing
        X['is_bullish_engulf'] = ((body > 0) & (body.shift() < 0) & (c > o.shift()) & (o < c.shift())).astype(int)

        # Gap up/down
        X['gap_up'] = (o > h.shift()).astype(int)
        X['gap_down'] = (o < l.shift()).astype(int)

        # Inside day
        X['inside_day'] = ((h < h.shift()) & (l > l.shift())).astype(int)

        # Higher high / Lower low streaks
        X['hh_streak'] = (h > h.shift()).rolling(5).sum()
        X['ll_streak'] = (l < l.shift()).rolling(5).sum()

        return X

    def _add_feature_interactions(self, X: 'pd.DataFrame') -> 'pd.DataFrame':
        """Add cross-feature interactions."""
        import numpy as np

        # RSI + Volume interaction
        if 'rsi_14' in X.columns and 'volume_ratio' in X.columns:
            X['rsi_volume_interaction'] = X['rsi_14'] * X['volume_ratio']

        # Momentum + Volatility interaction
        if 'return_5d' in X.columns and 'volatility_5d' in X.columns:
            X['momentum_vol_ratio'] = X['return_5d'] / X['volatility_5d'].replace(0, np.nan)

        # Trend + Momentum alignment
        if 'close_vs_sma_20' in X.columns and 'macd' in X.columns:
            X['trend_momentum_align'] = np.sign(X['close_vs_sma_20']) * np.sign(X['macd'])

        # BB position + RSI divergence
        if 'bb_position_20' in X.columns and 'rsi_14' in X.columns:
            X['bb_rsi_divergence'] = X['bb_position_20'] - X['rsi_14'] / 100

        # Multi-timeframe alignment
        ma_cols = [c for c in X.columns if 'close_vs_sma' in c]
        if len(ma_cols) >= 3:
            X['ma_alignment'] = sum(np.sign(X[c]) for c in ma_cols[:3])

        return X

    def _save_sweep_results(self, results: List[Dict]) -> None:
        """Save sweep results to file."""
        self.SWEEP_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results
        existing = []
        if self.SWEEP_RESULTS_FILE.exists():
            try:
                with open(self.SWEEP_RESULTS_FILE) as f:
                    existing = json.load(f)
            except:
                existing = []

        # Append new results
        existing.extend(results)

        # Keep only last 1000 results
        existing = existing[-1000:]

        with open(self.SWEEP_RESULTS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)

    def _show_leaderboard(self, cli: "JottyCLI") -> CommandResult:
        """Show leaderboard from saved sweep results."""
        if not self.SWEEP_RESULTS_FILE.exists():
            cli.renderer.info("No sweep results yet. Run: /stock-ml --sweep --stocks top10")
            return CommandResult.ok(data=[])

        with open(self.SWEEP_RESULTS_FILE) as f:
            results = json.load(f)

        if not results:
            cli.renderer.info("No sweep results yet.")
            return CommandResult.ok(data=[])

        # Sort by AUC
        sorted_results = sorted(results, key=lambda x: -x.get('auc', 0))

        # Get unique best per stock
        seen_stocks = set()
        unique_best = []
        for r in sorted_results:
            if r['symbol'] not in seen_stocks:
                unique_best.append(r)
                seen_stocks.add(r['symbol'])

        cli.renderer.header("Stock ML Leaderboard")
        cli.renderer.info(f"Total results: {len(results)}")
        cli.renderer.info("")

        # Top 20 overall
        cli.renderer.info("Top 20 Overall:")
        cli.renderer.info("┌──────────────┬─────────────┬───────────┬──────────┬──────────┐")
        cli.renderer.info("│    Symbol    │   Target    │ Timeframe │ Accuracy │   AUC    │")
        cli.renderer.info("├──────────────┼─────────────┼───────────┼──────────┼──────────┤")

        for i, r in enumerate(sorted_results[:20]):
            marker = "" if i == 0 else " "
            cli.renderer.info(
                f"│{marker}{r['symbol']:<12} │ {r['target']:<11} │ {r['timeframe']:<9} │ "
                f"{r.get('accuracy', 0):^8.4f} │ {r.get('auc', 0):^8.4f} │"
            )

        cli.renderer.info("└──────────────┴─────────────┴───────────┴──────────┴──────────┘")

        # Best per stock
        cli.renderer.info("")
        cli.renderer.info("Best per Stock:")
        cli.renderer.info("┌──────────────┬─────────────┬──────────┐")
        cli.renderer.info("│    Symbol    │ Best Target │   AUC    │")
        cli.renderer.info("├──────────────┼─────────────┼──────────┤")

        for r in unique_best[:15]:
            cli.renderer.info(f"│ {r['symbol']:<12} │ {r['target']:<11} │ {r.get('auc', 0):^8.4f} │")

        cli.renderer.info("└──────────────┴─────────────┴──────────┘")

        return CommandResult.ok(data=sorted_results[:20])

    def _show_stock_sets(self, cli: "JottyCLI") -> CommandResult:
        """Show available stock sets for sweep."""
        cli.renderer.header("Available Stock Sets")
        cli.renderer.info("")

        for name, stocks in self.STOCK_SETS.items():
            cli.renderer.info(f"{name} ({len(stocks)} stocks):")
            # Display in rows of 5
            for i in range(0, len(stocks), 5):
                row = stocks[i:i+5]
                cli.renderer.info("  " + ", ".join(row))
            cli.renderer.info("")

        # Show available indices from JSON
        if self.INDICES_JSON.exists():
            cli.renderer.info("")
            cli.renderer.header("Available Indices (from niftyindices.com)")
            try:
                with open(self.INDICES_JSON) as f:
                    indices = json.load(f)
                for name in sorted(indices.keys())[:15]:
                    key = name.lower().replace(" ", "_").replace("-", "_")
                    cli.renderer.info(f"  {key}")
                cli.renderer.info(f"  ... and {len(indices) - 15} more")
            except:
                pass

        cli.renderer.info("")
        cli.renderer.info("Usage:")
        cli.renderer.info("  /stock-ml --sweep --stocks top10")
        cli.renderer.info("  /stock-ml --sweep --stocks nifty_bank       # Load from niftyindices.com")
        cli.renderer.info("  /stock-ml --sweep --stocks RELIANCE,TCS,INFY")

        return CommandResult.ok(data=self.STOCK_SETS)

    def _load_index_stocks(self, index_name: str) -> List[str]:
        """Load stocks from niftyindices.com index."""
        import pandas as pd
        import requests

        # Normalize index name
        normalized = index_name.lower().replace("_", " ").replace("-", " ")

        # Load indices JSON
        if not self.INDICES_JSON.exists():
            return []

        with open(self.INDICES_JSON) as f:
            indices = json.load(f)

        # Find matching index
        url = None
        for name, link in indices.items():
            if name.lower() == normalized or name.lower().replace(" ", "_") == index_name.lower():
                url = link
                break

        if not url:
            # Fuzzy match
            for name, link in indices.items():
                if index_name.lower() in name.lower().replace(" ", "_"):
                    url = link
                    break

        if not url:
            return []

        # Check cache
        self.INDICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self.INDICES_CACHE_DIR / f"{index_name.lower()}.json"

        # Use cache if less than 1 day old
        if cache_file.exists():
            import time
            if time.time() - cache_file.stat().st_mtime < 86400:
                with open(cache_file) as f:
                    return json.load(f)

        # Fetch from URL
        try:
            df = pd.read_csv(url)
            # Find symbol column (usually 'Symbol' or 'SYMBOL')
            symbol_col = None
            for col in df.columns:
                if 'symbol' in col.lower():
                    symbol_col = col
                    break

            if symbol_col is None:
                symbol_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]

            stocks = df[symbol_col].dropna().tolist()
            stocks = [s.strip().upper() for s in stocks if isinstance(s, str)]

            # Cache results
            with open(cache_file, 'w') as f:
                json.dump(stocks, f)

            return stocks
        except Exception as e:
            return []

    def _get_stocks_for_sweep(self, stocks_input: str) -> List[str]:
        """Get stocks list from input - can be set name, index name, or comma-separated."""
        # Check predefined sets
        if stocks_input in self.STOCK_SETS:
            return self.STOCK_SETS[stocks_input]

        # Try loading from nifty indices
        index_stocks = self._load_index_stocks(stocks_input)
        if index_stocks:
            return index_stocks

        # Comma-separated list
        return [s.strip().upper() for s in stocks_input.split(",")]

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        targets = list(self.TARGET_TYPES.keys())
        timeframes = list(self.TIMEFRAMES.keys())
        stock_sets = list(self.STOCK_SETS.keys())
        flags = ["--target", "--timeframe", "--years", "--iterations", "--mlflow",
                 "--experiment", "--list", "--targets", "--compare", "--benchmark",
                 "--compare-targets", "--compare-timeframes", "--sweep", "--grid",
                 "--stocks", "--sweep-targets", "--sweep-timeframes", "--sweep-periods",
                 "--leaderboard", "--lb", "--sets", "--unified", "--cross-stock", "--holdout",
                 "--backtest", "--bt", "--swarm-learn", "--auto-learn", "--insights",
                 "--cross-stock", "--normalized", "--comprehensive", "--llm-features"]
        popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

        all_completions = targets + timeframes + stock_sets + flags + popular_stocks
        return [s for s in all_completions if s.lower().startswith(partial.lower())]

    # =========================================================================
    # SWARM ML AUTO-LEARNING SYSTEM
    # =========================================================================

    def _load_swarm_ml_state(self) -> Dict[str, Any]:
        """Load swarm ML learning state."""
        if self.SWARM_ML_STATE_FILE.exists():
            try:
                with open(self.SWARM_ML_STATE_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {
            'stock_profiles': {},
            'config_patterns': {},
            'underperformers': [],
            'recommendations': [],
            'feature_learnings': {},
            'last_updated': None,
        }

    def _save_swarm_ml_state(self, state: Dict[str, Any]) -> None:
        """Save swarm ML learning state."""
        self.SWARM_ML_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SWARM_ML_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def _diagnose_underperformance(self, profile: Dict) -> str:
        """Diagnose why a stock might be underperforming."""
        if profile['auc_std'] > 0.1:
            return 'High variance - sensitive to config; try more configs'
        if profile['best_timeframe'] == 'day' and profile['best_auc'] < 0.55:
            return 'May need intraday data (60minute) for better signals'
        if '20d' in profile['best_target'] or '30d' in profile['best_target']:
            return 'Better at longer horizons; not suitable for short-term trading'
        if profile['n_configs_tested'] < 5:
            return 'Insufficient testing; run more configs'
        return 'Low predictability; may be driven by external factors'

    def __init__(self) -> None:
        self.Q = {}  # State-Action -> (value, count, last_updated)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor (lower for stock ML - focus on immediate)
        self.epsilon = 0.15  # Exploration rate

        # Load existing Q-table
        self._load_q_table()

    def _load_q_table(self) -> Any:
        """Load Q-table from disk."""
        if self.Q_TABLE_PATH.exists():
            try:
                with open(self.Q_TABLE_PATH) as f:
                    data = json.load(f)
                    self.Q = data.get('Q', {})
                    self.alpha = data.get('alpha', 0.1)
                    self.epsilon = data.get('epsilon', 0.15)
            except:
                self.Q = {}

    def _save_q_table(self) -> Any:
        """Save Q-table to disk."""
        self.Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.Q_TABLE_PATH, 'w') as f:
            json.dump({
                'Q': self.Q,
                'alpha': self.alpha,
                'epsilon': self.epsilon,
            }, f, indent=2)

    def get_state(self, stock_profile: Dict) -> str:
        """
        Convert stock profile to Q-learning state.

        State representation:
        - sector: banking, it, pharma, etc.
        - volatility: high/medium/low
        - predictability: high/medium/low
        - best_horizon: short/medium/long (based on which targets work)
        """
        sector = stock_profile.get('sector', 'other')
        predictability = stock_profile.get('predictability', 'medium')

        # Infer volatility from AUC variance
        auc_std = stock_profile.get('auc_std', 0.05)
        if auc_std > 0.08:
            volatility = 'high'
        elif auc_std > 0.04:
            volatility = 'medium'
        else:
            volatility = 'low'

        # Infer best horizon from target sensitivity
        target_sens = stock_profile.get('target_sensitivity', {})
        if target_sens:
            best_target = max(target_sens.items(), key=lambda x: x[1])[0]
            if '1d' in best_target:
                horizon = 'short'
            elif '5d' in best_target or '10d' in best_target:
                horizon = 'medium'
            else:
                horizon = 'long'
        else:
            horizon = 'medium'

        return f"{sector}|{volatility}|{predictability}|{horizon}"

    def get_action_key(self, target: str, timeframe: str) -> str:
        """Convert action to string key."""
        return f"{target}|{timeframe}"

    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        key = f"{state}||{action}"
        if key in self.Q:
            return self.Q[key]['value']
        return 0.5  # Optimistic initialization

    def get_best_action(self, state: str, available_actions: List[str]) -> Tuple[str, float]:
        """
        Get best action using epsilon-greedy policy.

        Returns:
            (action, q_value)
        """
        import random

        # Exploration
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            return action, self.get_q_value(state, action)

        # Exploitation
        best_action = None
        best_value = -float('inf')

        for action in available_actions:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        return best_action or available_actions[0], best_value

    def compute_reward(self, result: Dict) -> float:
        """
        Compute reward from ML/backtest result.

        Reward = base_auc + sharpe_bonus - drawdown_penalty

        For 5-day trading focus:
        - AUC > 0.65 is good
        - Sharpe > 1.5 gets bonus
        - Drawdown > 20% gets penalty
        """
        auc = result.get('auc', 0.5)
        backtest = result.get('backtest', {})

        # Base reward from AUC (normalized 0-1)
        reward = (auc - 0.5) * 2  # 0.5 AUC = 0, 0.75 AUC = 0.5, 1.0 AUC = 1.0

        # Sharpe bonus (if backtesting available)
        if backtest:
            strategy = backtest.get('strategy', {})
            sharpe = strategy.get('sharpe', 0)
            if sharpe > 2.0:
                reward += 0.3
            elif sharpe > 1.5:
                reward += 0.2
            elif sharpe > 1.0:
                reward += 0.1

            # Drawdown penalty
            mdd = abs(strategy.get('max_drawdown', 0))
            if mdd > 30:
                reward -= 0.3
            elif mdd > 20:
                reward -= 0.15

            # Outperformance bonus
            outperform = backtest.get('outperformance', 0)
            if outperform > 10:
                reward += 0.2
            elif outperform > 0:
                reward += 0.1

        return max(0.0, min(1.0, reward))

    def update(self, state: str, action: str, reward: float, next_state: str = None) -> Any:
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        """
        key = f"{state}||{action}"
        current_q = self.get_q_value(state, action)

        # For stock ML, we don't have strong state transitions
        # Use simplified update: Q(s,a) = Q(s,a) + α * (r - Q(s,a))
        new_q = current_q + self.alpha * (reward - current_q)

        if key not in self.Q:
            self.Q[key] = {'value': new_q, 'count': 1, 'last_reward': reward}
        else:
            self.Q[key]['value'] = new_q
            self.Q[key]['count'] = self.Q[key].get('count', 0) + 1
            self.Q[key]['last_reward'] = reward

        # Decay exploration rate
        self.epsilon = max(0.05, self.epsilon * 0.995)

        # Save after each update
        self._save_q_table()

    def get_recommendations(self, state: str, top_k: int = 3) -> List[Dict]:
        """
        Get top-k action recommendations for a state.
        """
        # Define all possible actions
        targets = ['next_5d_up', 'next_10d_up', 'next_1d_up']
        timeframes = ['60minute', 'day']

        actions = []
        for target in targets:
            for tf in timeframes:
                action = self.get_action_key(target, tf)
                q = self.get_q_value(state, action)
                count = self.Q.get(f"{state}||{action}", {}).get('count', 0)
                actions.append({
                    'action': action,
                    'target': target,
                    'timeframe': tf,
                    'q_value': q,
                    'confidence': min(1.0, count / 10),  # More visits = more confident
                })

        # Sort by Q-value
        actions.sort(key=lambda x: -x['q_value'])
        return actions[:top_k]

    def get_transfer_learning_context(self, state: str) -> str:
        """
        Get learned context to help with new stocks.

        This is the key value of swarm learning - transfer knowledge!
        """
        # Parse state
        parts = state.split('|')
        sector = parts[0] if len(parts) > 0 else 'unknown'

        # Find similar states (same sector)
        similar_learnings = []
        for key, data in self.Q.items():
            if f"{sector}|" in key and data.get('count', 0) >= 3:
                state_part, action_part = key.split('||')
                similar_learnings.append({
                    'state': state_part,
                    'action': action_part,
                    'value': data['value'],
                    'count': data['count'],
                })

        if not similar_learnings:
            return ""

        # Sort by value and format as context
        similar_learnings.sort(key=lambda x: -x['value'])

        lines = [f"Learned patterns for {sector} sector:"]
        for learn in similar_learnings[:5]:
            action_parts = learn['action'].split('|')
            target, tf = action_parts[0], action_parts[1] if len(action_parts) > 1 else 'day'
            lines.append(f"  - {target} with {tf}: Q={learn['value']:.2f} (tested {learn['count']}x)")

        return '\n'.join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get Q-learner statistics."""
        if not self.Q:
            return {'entries': 0, 'avg_q': 0, 'max_q': 0}

        values = [v['value'] for v in self.Q.values()]
        counts = [v.get('count', 1) for v in self.Q.values()]

        return {
            'entries': len(self.Q),
            'avg_q': sum(values) / len(values),
            'max_q': max(values),
            'total_updates': sum(counts),
            'epsilon': self.epsilon,
        }
