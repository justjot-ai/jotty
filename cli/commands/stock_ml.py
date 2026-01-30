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


class StockMLCommand(BaseCommand):
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
        use_mlflow = "mlflow" in args.flags
        experiment_name = args.flags.get("experiment", f"stock_{symbol}")

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

        cli.renderer.info(f"Features: {len(feature_names)}, Samples: {len(X)}")

        # Check if backtesting is enabled
        run_backtest = "backtest" in args.flags or "bt" in args.flags

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
                run_backtest=run_backtest
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

    async def _load_stock_data(self, symbol: str, timeframe: str, years: int, cli) -> Optional['pd.DataFrame']:
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

    def _run_backtest(self, df_ohlcv, X, y, model, target_config, symbol, cli) -> Dict[str, Any]:
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
        def calc_max_drawdown(cumret):
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

    async def _run_stock_ml(self, X, y, feature_names, target_config, symbol,
                            max_iterations, cli, use_mlflow=False, experiment_name="stock",
                            df_ohlcv=None, run_backtest=False):
        """Run ML pipeline for stock prediction with optional backtesting."""
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

        # Initialize MLflow if enabled
        mlflow_tracker = None
        if use_mlflow:
            from core.skills.ml import MLflowTrackerSkill
            from .ml import MLCommand

            mlflow_tracker = MLflowTrackerSkill()
            await mlflow_tracker.init(experiment_name=experiment_name)
            run_name = f"{symbol}_{target_config['days']}d_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
            await mlflow_tracker.start_run(run_name=run_name)

            await mlflow_tracker.log_params({
                'symbol': symbol,
                'target_type': f"{target_config['days']}d_{problem_type}",
                'n_features': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
            })

        # Import models
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier, CatBoostRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Model zoo
        if is_classification:
            models = {
                'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42),
                'CatBoost': CatBoostClassifier(iterations=200, learning_rate=0.05, verbose=0, random_state=42),
                'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            }
        else:
            models = {
                'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42),
                'CatBoost': CatBoostRegressor(iterations=200, learning_rate=0.05, verbose=0, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
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
                marker = "★ " if r['model'] == best_name else "  "
                cli.renderer.info(f"│{marker}{r['model']:<13} │ {r['accuracy']:^8.4f} │ {r['f1']:^8.4f} │ {r['auc']:^8.4f} │")
            cli.renderer.info("└─────────────────┴──────────┴──────────┴──────────┘")
        else:
            cli.renderer.info("┌─────────────────┬──────────┬──────────┐")
            cli.renderer.info("│     Model       │    R²    │   RMSE   │")
            cli.renderer.info("├─────────────────┼──────────┼──────────┤")
            for r in results:
                marker = "★ " if r['model'] == best_name else "  "
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

        # Log to MLflow
        if mlflow_tracker:
            metrics = {'best_score': best_score}
            if is_classification:
                metrics.update({'accuracy': results[0]['accuracy'], 'f1': results[0]['f1'], 'auc': results[0]['auc']})
            else:
                metrics.update({'r2': results[0]['r2'], 'rmse': results[0]['rmse']})

            await mlflow_tracker.log_metrics(metrics)

            if hasattr(best_model, 'feature_importances_'):
                await mlflow_tracker.log_feature_importance(importance)

            model_uri = await mlflow_tracker.log_model(best_model, f"{symbol}_model")
            if model_uri:
                cli.renderer.info(f"Model logged to MLflow: {model_uri}")

            run_info = await mlflow_tracker.end_run()
            if run_info:
                from .ml import MLCommand
                MLCommand.save_mlflow_state(experiment_name, run_info['run_id'])
                cli.renderer.info(f"MLflow run: {run_info['run_id']}")

        # Run backtesting if enabled
        backtest_results = None
        if run_backtest and df_ohlcv is not None and is_classification:
            backtest_results = self._run_backtest(
                df_ohlcv, X, y, best_model, target_config, symbol, cli
            )

        return {
            'symbol': symbol,
            'target': target_config,
            'best_model': best_name,
            'best_score': best_score,
            'results': results,
            'feature_importance': sorted_imp[:15] if hasattr(best_model, 'feature_importances_') else [],
            'backtest': backtest_results,
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

    async def _run_comparison(self, args: ParsedArgs, cli: "JottyCLI", symbol: str) -> CommandResult:
        """Run comparison across multiple targets and timeframes."""
        import pandas as pd
        import numpy as np

        years = int(args.flags.get("years", args.flags.get("y", "3")))
        use_mlflow = "mlflow" in args.flags
        experiment_name = args.flags.get("experiment", f"stock_{symbol}_compare")

        # Define comparison configs
        targets_to_test = [
            "next_1d_up", "next_5d_up", "next_10d_up", "next_20d_up", "next_30d_up"
        ]
        timeframes_to_test = ["day"]  # Can expand: ["day", "60minute"]

        # Check if user specified specific targets or timeframes
        if "compare-targets" in args.flags:
            targets_to_test = args.flags["compare-targets"].split(",")
        if "compare-timeframes" in args.flags:
            timeframes_to_test = args.flags["compare-timeframes"].split(",")

        cli.renderer.header(f"Stock ML Comparison: {symbol}")
        cli.renderer.info(f"Testing {len(targets_to_test)} targets × {len(timeframes_to_test)} timeframes")
        cli.renderer.info("")

        all_results = []

        for timeframe in timeframes_to_test:
            # Load data once per timeframe
            cli.renderer.info(f"Loading {symbol} ({timeframe} data, last {years} years)...")
            try:
                df = await self._load_stock_data(symbol, timeframe, years, cli)
                if df is None or len(df) < 100:
                    cli.renderer.error(f"Insufficient data for {timeframe}")
                    continue
            except Exception as e:
                cli.renderer.error(f"Failed to load {timeframe} data: {e}")
                continue

            cli.renderer.info(f"Loaded {len(df)} records")

            for target_type in targets_to_test:
                target_config = self._parse_target(target_type)
                cli.renderer.status(f"Testing {target_type} ({timeframe})...")

                try:
                    X, y, feature_names = self._create_features_and_target(df.copy(), target_config)
                    if X is None or len(X) < 100:
                        continue

                    # Quick training with just best models
                    result = await self._quick_train(X, y, feature_names, target_config, cli)

                    all_results.append({
                        'timeframe': timeframe,
                        'target': target_type,
                        'days': target_config['days'],
                        'type': target_config['type'],
                        'samples': len(X),
                        'best_model': result['best_model'],
                        'accuracy': result.get('accuracy', 0),
                        'auc': result.get('auc', 0),
                        'f1': result.get('f1', 0),
                    })
                except Exception as e:
                    cli.renderer.error(f"  Failed: {e}")
                    continue

        if not all_results:
            cli.renderer.error("No successful runs")
            return CommandResult.fail("No results")

        # Display comparison table
        cli.renderer.info("")
        cli.renderer.header("Comparison Results")
        cli.renderer.info("")
        cli.renderer.info("┌─────────────┬────────────┬───────────┬──────────┬──────────┬──────────┬─────────────────┐")
        cli.renderer.info("│  Timeframe  │   Target   │   Days    │ Samples  │ Accuracy │  AUC     │   Best Model    │")
        cli.renderer.info("├─────────────┼────────────┼───────────┼──────────┼──────────┼──────────┼─────────────────┤")

        # Sort by AUC descending
        sorted_results = sorted(all_results, key=lambda x: -x['auc'])

        for r in sorted_results:
            marker = "★" if r == sorted_results[0] else " "
            cli.renderer.info(
                f"│{marker}{r['timeframe']:<11} │ {r['target']:<10} │ {r['days']:^9} │ {r['samples']:^8} │ "
                f"{r['accuracy']:^8.4f} │ {r['auc']:^8.4f} │ {r['best_model']:<15} │"
            )

        cli.renderer.info("└─────────────┴────────────┴───────────┴──────────┴──────────┴──────────┴─────────────────┘")

        # Summary
        best = sorted_results[0]
        cli.renderer.info("")
        cli.renderer.info(f"Best Configuration:")
        cli.renderer.info(f"  Target:    {best['target']} ({best['days']}-day prediction)")
        cli.renderer.info(f"  Timeframe: {best['timeframe']}")
        cli.renderer.info(f"  Model:     {best['best_model']}")
        cli.renderer.info(f"  AUC:       {best['auc']:.4f}")
        cli.renderer.info(f"  Accuracy:  {best['accuracy']:.4f}")

        # Log to MLflow if enabled
        if use_mlflow:
            from core.skills.ml import MLflowTrackerSkill
            from .ml import MLCommand

            tracker = MLflowTrackerSkill()
            await tracker.init(experiment_name=experiment_name)
            await tracker.start_run(run_name=f"{symbol}_comparison")

            await tracker.log_params({
                'symbol': symbol,
                'targets_tested': ','.join(targets_to_test),
                'timeframes_tested': ','.join(timeframes_to_test),
                'best_target': best['target'],
                'best_timeframe': best['timeframe'],
            })

            await tracker.log_metrics({
                'best_auc': best['auc'],
                'best_accuracy': best['accuracy'],
                'n_configurations': len(all_results),
            })

            run_info = await tracker.end_run()
            if run_info:
                MLCommand.save_mlflow_state(experiment_name, run_info['run_id'])
                cli.renderer.info(f"MLflow run: {run_info['run_id']}")

        return CommandResult.ok(data={
            'symbol': symbol,
            'results': sorted_results,
            'best': best,
        })

    async def _quick_train(self, X, y, feature_names, target_config, cli) -> Dict[str, Any]:
        """Quick training with fewer models for comparison mode."""
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        is_classification = target_config['type'] == 'classification'

        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Use only fast models for comparison
        import lightgbm as lgb

        if is_classification:
            model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1, random_state=42)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, verbose=-1, random_state=42)

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

            return {
                'best_model': 'LightGBM',
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
            }
        else:
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, pred)
            return {
                'best_model': 'LightGBM',
                'r2': r2,
                'accuracy': r2,  # For sorting
                'auc': r2,
                'f1': 0,
            }

    async def _run_sweep(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Run comprehensive sweep across stocks, targets, timeframes, periods."""
        import pandas as pd
        from datetime import datetime

        # Parse sweep parameters
        stocks_input = args.flags.get("stocks", args.flags.get("s", "top10"))
        targets_input = args.flags.get("sweep-targets", "next_1d_up,next_5d_up,next_10d_up,next_20d_up")
        timeframes_input = args.flags.get("sweep-timeframes", "day")
        periods_input = args.flags.get("sweep-periods", "3")
        use_mlflow = "mlflow" in args.flags
        experiment_name = args.flags.get("experiment", "stock_sweep")

        # Parse stocks (supports predefined sets, nifty indices, or comma-separated)
        stocks = self._get_stocks_for_sweep(stocks_input)
        if not stocks:
            cli.renderer.error(f"No stocks found for: {stocks_input}")
            return CommandResult.fail("No stocks")

        # Parse other params
        targets = [t.strip() for t in targets_input.split(",")]
        timeframes = [t.strip() for t in timeframes_input.split(",")]
        periods = [int(p.strip()) for p in periods_input.split(",")]

        total_configs = len(stocks) * len(targets) * len(timeframes) * len(periods)

        cli.renderer.header("Stock ML Sweep")
        cli.renderer.info(f"Stocks:     {len(stocks)} ({stocks_input})")
        cli.renderer.info(f"Targets:    {len(targets)} ({', '.join(targets)})")
        cli.renderer.info(f"Timeframes: {len(timeframes)} ({', '.join(timeframes)})")
        cli.renderer.info(f"Periods:    {len(periods)} years ({', '.join(map(str, periods))})")
        cli.renderer.info(f"Total:      {total_configs} configurations")
        cli.renderer.info("")

        all_results = []
        completed = 0

        for symbol in stocks:
            for timeframe in timeframes:
                for years in periods:
                    # Load data once per stock/timeframe/period combo
                    cli.renderer.status(f"Loading {symbol} ({timeframe}, {years}y)...")
                    try:
                        df = await self._load_stock_data(symbol, timeframe, years, cli)
                        if df is None or len(df) < 100:
                            continue
                    except:
                        continue

                    for target_type in targets:
                        completed += 1
                        target_config = self._parse_target(target_type)
                        cli.renderer.status(f"[{completed}/{total_configs}] {symbol} {target_type} {timeframe} {years}y")

                        try:
                            X, y, feature_names = self._create_features_and_target(df.copy(), target_config)
                            if X is None or len(X) < 100:
                                continue

                            result = await self._quick_train(X, y, feature_names, target_config, cli)

                            all_results.append({
                                'symbol': symbol,
                                'target': target_type,
                                'days': target_config['days'],
                                'timeframe': timeframe,
                                'years': years,
                                'samples': len(X),
                                'accuracy': result.get('accuracy', 0),
                                'auc': result.get('auc', 0),
                                'f1': result.get('f1', 0),
                                'timestamp': datetime.now().isoformat(),
                            })
                        except Exception as e:
                            continue

        if not all_results:
            cli.renderer.error("No successful runs")
            return CommandResult.fail("No results")

        # Sort by AUC
        sorted_results = sorted(all_results, key=lambda x: -x['auc'])

        # Display top results
        cli.renderer.info("")
        cli.renderer.header(f"Sweep Results (Top 20 of {len(all_results)})")
        cli.renderer.info("")
        cli.renderer.info("┌──────────────┬─────────────┬───────────┬──────────┬────────┬──────────┬──────────┐")
        cli.renderer.info("│    Symbol    │   Target    │ Timeframe │  Years   │ Samples│ Accuracy │   AUC    │")
        cli.renderer.info("├──────────────┼─────────────┼───────────┼──────────┼────────┼──────────┼──────────┤")

        for i, r in enumerate(sorted_results[:20]):
            marker = "★" if i == 0 else " "
            cli.renderer.info(
                f"│{marker}{r['symbol']:<12} │ {r['target']:<11} │ {r['timeframe']:<9} │ {r['years']:^8} │ "
                f"{r['samples']:^6} │ {r['accuracy']:^8.4f} │ {r['auc']:^8.4f} │"
            )

        cli.renderer.info("└──────────────┴─────────────┴───────────┴──────────┴────────┴──────────┴──────────┘")

        # Summary statistics
        cli.renderer.info("")
        cli.renderer.info("Summary by Stock (avg AUC):")
        stock_aucs = {}
        for r in all_results:
            if r['symbol'] not in stock_aucs:
                stock_aucs[r['symbol']] = []
            stock_aucs[r['symbol']].append(r['auc'])

        stock_avg = [(s, sum(aucs)/len(aucs)) for s, aucs in stock_aucs.items()]
        stock_avg.sort(key=lambda x: -x[1])
        for symbol, avg_auc in stock_avg[:10]:
            cli.renderer.info(f"  {symbol:<12} {avg_auc:.4f}")

        cli.renderer.info("")
        cli.renderer.info("Summary by Target (avg AUC):")
        target_aucs = {}
        for r in all_results:
            if r['target'] not in target_aucs:
                target_aucs[r['target']] = []
            target_aucs[r['target']].append(r['auc'])

        target_avg = [(t, sum(aucs)/len(aucs)) for t, aucs in target_aucs.items()]
        target_avg.sort(key=lambda x: -x[1])
        for target, avg_auc in target_avg:
            cli.renderer.info(f"  {target:<15} {avg_auc:.4f}")

        # Save results to file
        self._save_sweep_results(sorted_results)
        cli.renderer.info("")
        cli.renderer.success(f"Results saved to {self.SWEEP_RESULTS_FILE}")

        # Best configuration
        best = sorted_results[0]
        cli.renderer.info("")
        cli.renderer.info("Best Configuration:")
        cli.renderer.info(f"  Symbol:    {best['symbol']}")
        cli.renderer.info(f"  Target:    {best['target']} ({best['days']}-day)")
        cli.renderer.info(f"  Timeframe: {best['timeframe']}")
        cli.renderer.info(f"  Period:    {best['years']} years")
        cli.renderer.info(f"  AUC:       {best['auc']:.4f}")
        cli.renderer.info(f"  Accuracy:  {best['accuracy']:.4f}")

        # Log to MLflow if enabled
        if use_mlflow:
            from core.skills.ml import MLflowTrackerSkill
            from .ml import MLCommand

            tracker = MLflowTrackerSkill()
            await tracker.init(experiment_name=experiment_name)
            await tracker.start_run(run_name=f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}")

            await tracker.log_params({
                'stocks': stocks_input,
                'targets': targets_input,
                'timeframes': timeframes_input,
                'periods': periods_input,
                'total_configs': total_configs,
                'successful_configs': len(all_results),
                'best_symbol': best['symbol'],
                'best_target': best['target'],
            })

            await tracker.log_metrics({
                'best_auc': best['auc'],
                'best_accuracy': best['accuracy'],
                'avg_auc': sum(r['auc'] for r in all_results) / len(all_results),
            })

            run_info = await tracker.end_run()
            if run_info:
                MLCommand.save_mlflow_state(experiment_name, run_info['run_id'])
                cli.renderer.info(f"MLflow run: {run_info['run_id']}")

        return CommandResult.ok(data={
            'total_configs': total_configs,
            'successful': len(all_results),
            'results': sorted_results,
            'best': best,
        })

    async def _run_unified_training(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Train a unified model across multiple stocks with normalized features."""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

        # Parse parameters
        stocks_input = args.flags.get("stocks", args.flags.get("s", "nifty_bank"))
        target_type = args.flags.get("target", args.flags.get("t", "next_1d_up"))
        timeframe = args.flags.get("timeframe", args.flags.get("tf", "day"))
        years = int(args.flags.get("years", args.flags.get("y", "3")))
        holdout_pct = float(args.flags.get("holdout", "0.2"))  # % of stocks for testing
        use_mlflow = "mlflow" in args.flags
        experiment_name = args.flags.get("experiment", f"unified_{stocks_input}")

        # Get stocks
        stocks = self._get_stocks_for_sweep(stocks_input)
        if not stocks:
            cli.renderer.error(f"No stocks found for: {stocks_input}")
            return CommandResult.fail("No stocks")

        target_config = self._parse_target(target_type)
        is_classification = target_config['type'] == 'classification'

        cli.renderer.header("Unified Cross-Stock Training")
        cli.renderer.info(f"Stock Set:   {stocks_input} ({len(stocks)} stocks)")
        cli.renderer.info(f"Target:      {target_type} ({target_config['type']})")
        cli.renderer.info(f"Timeframe:   {timeframe}")
        cli.renderer.info(f"Period:      {years} years")
        cli.renderer.info(f"Holdout:     {holdout_pct*100:.0f}% stocks for generalization test")
        cli.renderer.info("")

        # Split stocks into train and holdout (for generalization testing)
        np.random.seed(42)
        n_holdout = max(1, int(len(stocks) * holdout_pct))
        holdout_stocks = list(np.random.choice(stocks, n_holdout, replace=False))
        train_stocks = [s for s in stocks if s not in holdout_stocks]

        cli.renderer.info(f"Training stocks:  {len(train_stocks)}")
        cli.renderer.info(f"Holdout stocks:   {len(holdout_stocks)} ({', '.join(holdout_stocks)})")
        cli.renderer.info("")

        # Load and normalize data from all training stocks
        cli.renderer.info("Loading and normalizing data...")
        all_train_data = []
        stock_scalers = {}

        for symbol in train_stocks:
            cli.renderer.status(f"Loading {symbol}...")
            try:
                df = await self._load_stock_data(symbol, timeframe, years, cli)
                if df is None or len(df) < 100:
                    continue

                # Create features
                X, y, feature_names = self._create_features_and_target(df.copy(), target_config)
                if X is None or len(X) < 50:
                    continue

                # Normalize features per stock (z-score)
                scaler = StandardScaler()
                X_normalized = pd.DataFrame(
                    scaler.fit_transform(X),
                    columns=feature_names,
                    index=X.index
                )
                stock_scalers[symbol] = scaler

                # Add stock-agnostic meta features (relative to stock's own history)
                X_normalized['_symbol'] = symbol
                X_normalized['_target'] = y.values

                all_train_data.append(X_normalized)

            except Exception as e:
                continue

        if len(all_train_data) < 2:
            cli.renderer.error("Not enough stocks with valid data")
            return CommandResult.fail("Insufficient data")

        # Combine all training data
        combined_train = pd.concat(all_train_data, ignore_index=True)
        cli.renderer.info(f"Combined training samples: {len(combined_train)}")

        # Extract features and target
        feature_cols = [c for c in combined_train.columns if not c.startswith('_')]
        X_train_all = combined_train[feature_cols]
        y_train_all = combined_train['_target']

        # Time-based split within combined data (80/20)
        split_idx = int(len(X_train_all) * 0.8)
        X_train = X_train_all.iloc[:split_idx]
        X_val = X_train_all.iloc[split_idx:]
        y_train = y_train_all.iloc[:split_idx]
        y_val = y_train_all.iloc[split_idx:]

        cli.renderer.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        cli.renderer.info("")

        # Train models
        cli.renderer.header("Training Unified Models")

        import lightgbm as lgb
        import xgboost as xgb

        if is_classification:
            models = {
                'LightGBM': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42),
                'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42),
            }
        else:
            models = {
                'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, random_state=42),
            }

        results = []
        best_model = None
        best_score = -np.inf
        best_name = None

        for name, model in models.items():
            cli.renderer.status(f"Training {name}...")
            model.fit(X_train.values, y_train.values)
            pred = model.predict(X_val.values)

            if is_classification:
                proba = model.predict_proba(X_val.values)
                acc = accuracy_score(y_val, pred)
                f1 = f1_score(y_val, pred)
                try:
                    auc = roc_auc_score(y_val, proba[:, 1])
                except:
                    auc = acc
                score = auc
                results.append({'model': name, 'accuracy': acc, 'f1': f1, 'auc': auc})
                cli.renderer.info(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            else:
                r2 = r2_score(y_val, pred)
                score = r2
                results.append({'model': name, 'r2': r2})
                cli.renderer.info(f"  {name}: R²={r2:.4f}")

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        # Test generalization on holdout stocks
        cli.renderer.info("")
        cli.renderer.header("Generalization Test (Holdout Stocks)")

        holdout_results = []

        for symbol in holdout_stocks:
            try:
                df = await self._load_stock_data(symbol, timeframe, years, cli)
                if df is None or len(df) < 100:
                    continue

                X, y, _ = self._create_features_and_target(df.copy(), target_config)
                if X is None or len(X) < 50:
                    continue

                # Use global scaler (average of all training scalers)
                # For simplicity, use the first scaler or fit new one
                scaler = StandardScaler()
                X_normalized = scaler.fit_transform(X)

                # Test split
                test_start = int(len(X_normalized) * 0.8)
                X_test = X_normalized[test_start:]
                y_test = y.iloc[test_start:]

                pred = best_model.predict(X_test)

                if is_classification:
                    proba = best_model.predict_proba(X_test)
                    acc = accuracy_score(y_test, pred)
                    try:
                        auc = roc_auc_score(y_test, proba[:, 1])
                    except:
                        auc = acc

                    holdout_results.append({
                        'symbol': symbol,
                        'accuracy': acc,
                        'auc': auc,
                        'samples': len(X_test)
                    })
                    cli.renderer.info(f"  {symbol}: Acc={acc:.4f}, AUC={auc:.4f} (n={len(X_test)})")
                else:
                    r2 = r2_score(y_test, pred)
                    holdout_results.append({'symbol': symbol, 'r2': r2, 'samples': len(X_test)})
                    cli.renderer.info(f"  {symbol}: R²={r2:.4f} (n={len(X_test)})")

            except Exception as e:
                cli.renderer.error(f"  {symbol}: Failed - {e}")
                continue

        # Summary
        cli.renderer.info("")
        cli.renderer.header("Results Summary")

        if is_classification:
            cli.renderer.info("Validation Results:")
            cli.renderer.info("┌─────────────────┬──────────┬──────────┬──────────┐")
            cli.renderer.info("│     Model       │ Accuracy │    F1    │  ROC-AUC │")
            cli.renderer.info("├─────────────────┼──────────┼──────────┼──────────┤")
            for r in results:
                marker = "★ " if r['model'] == best_name else "  "
                cli.renderer.info(f"│{marker}{r['model']:<13} │ {r['accuracy']:^8.4f} │ {r['f1']:^8.4f} │ {r['auc']:^8.4f} │")
            cli.renderer.info("└─────────────────┴──────────┴──────────┴──────────┘")

            if holdout_results:
                avg_holdout_auc = sum(r['auc'] for r in holdout_results) / len(holdout_results)
                cli.renderer.info("")
                cli.renderer.info(f"Holdout Generalization:")
                cli.renderer.info(f"  Average AUC on unseen stocks: {avg_holdout_auc:.4f}")
                cli.renderer.info(f"  Validation AUC:               {best_score:.4f}")
                cli.renderer.info(f"  Generalization gap:           {best_score - avg_holdout_auc:.4f}")

        cli.renderer.info("")
        cli.renderer.info(f"Best Unified Model: {best_name}")
        cli.renderer.info(f"Validation Score:   {best_score:.4f}")
        cli.renderer.info(f"Training Stocks:    {len(train_stocks)}")
        cli.renderer.info(f"Total Samples:      {len(combined_train)}")

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            raw_importance = best_model.feature_importances_
            total = sum(raw_importance) if sum(raw_importance) > 0 else 1
            importance = {feat: (imp / total) * 100 for feat, imp in zip(feature_cols, raw_importance)}
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]

            cli.renderer.info("")
            cli.renderer.info("Top 15 Cross-Stock Features:")
            cli.renderer.info("┌────────────────────────────────┬────────────┐")
            cli.renderer.info("│           Feature              │ Importance │")
            cli.renderer.info("├────────────────────────────────┼────────────┤")
            for feat, imp in sorted_imp:
                feat_display = feat[:30] if len(feat) > 30 else feat
                cli.renderer.info(f"│ {feat_display:<30} │ {imp:>8.2f} % │")
            cli.renderer.info("└────────────────────────────────┴────────────┘")

        # Log to MLflow
        if use_mlflow:
            from core.skills.ml import MLflowTrackerSkill
            from .ml import MLCommand

            tracker = MLflowTrackerSkill()
            await tracker.init(experiment_name=experiment_name)
            await tracker.start_run(run_name=f"unified_{stocks_input}_{datetime.now().strftime('%Y%m%d_%H%M')}")

            await tracker.log_params({
                'stock_set': stocks_input,
                'n_train_stocks': len(train_stocks),
                'n_holdout_stocks': len(holdout_stocks),
                'target': target_type,
                'timeframe': timeframe,
                'years': years,
                'total_samples': len(combined_train),
            })

            metrics = {'validation_score': best_score, 'n_samples': len(combined_train)}
            if holdout_results and is_classification:
                metrics['holdout_avg_auc'] = avg_holdout_auc
                metrics['generalization_gap'] = best_score - avg_holdout_auc

            await tracker.log_metrics(metrics)

            model_uri = await tracker.log_model(best_model, f"unified_{stocks_input}_model")
            if model_uri:
                cli.renderer.info(f"Model logged to MLflow: {model_uri}")

            run_info = await tracker.end_run()
            if run_info:
                MLCommand.save_mlflow_state(experiment_name, run_info['run_id'])

        return CommandResult.ok(data={
            'stock_set': stocks_input,
            'train_stocks': train_stocks,
            'holdout_stocks': holdout_stocks,
            'best_model': best_name,
            'validation_score': best_score,
            'holdout_results': holdout_results,
            'total_samples': len(combined_train),
        })

    async def _run_cross_stock_normalized(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """
        Run cross-stock normalized training to avoid single-stock overfitting.

        Key techniques:
        1. Z-score normalization per stock (removes stock-specific scale)
        2. Sector-relative features (performance vs sector peers)
        3. Market regime indicators (shared across stocks)
        4. Leave-one-out cross-validation across stocks
        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from datetime import datetime

        cli.renderer.header("Cross-Stock Normalized Training")

        # Parse arguments
        stocks_input = args.flags.get("stocks", "nifty_bank")
        stocks = self._get_stocks_for_sweep(stocks_input)
        target_type = args.flags.get("target", "next_5d_up")
        timeframe = args.flags.get("timeframe", "60minute")
        years = int(args.flags.get("years", "3"))

        cli.renderer.info(f"Stocks: {len(stocks)} from {stocks_input}")
        cli.renderer.info(f"Target: {target_type}, Timeframe: {timeframe}, Years: {years}")
        cli.renderer.info("")

        target_config = self._parse_target(target_type)

        # ============ Phase 1: Load All Data ============
        cli.renderer.info("Phase 1: Loading all stock data...")

        all_data = {}
        for symbol in stocks:
            df = await self._load_stock_data(symbol, timeframe, years, cli)
            if df is not None and len(df) >= 100:
                all_data[symbol] = df
                cli.renderer.info(f"  {symbol}: {len(df)} samples")

        if len(all_data) < 3:
            cli.renderer.error("Need at least 3 stocks for cross-stock training")
            return CommandResult.fail("Insufficient stocks")

        # ============ Phase 2: Create Normalized Features ============
        cli.renderer.info("")
        cli.renderer.info("Phase 2: Creating normalized features per stock...")

        normalized_data = {}
        feature_names = None

        for symbol, df in all_data.items():
            X, y, feat_names = self._create_features_and_target(df.copy(), target_config)
            if X is None or len(X) < 50:
                continue

            # Z-score normalize features per stock
            scaler = StandardScaler()
            X_normalized = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

            # Add stock identifier and sector
            X_normalized['_symbol'] = symbol
            X_normalized['_sector'] = self._infer_sector(symbol)

            normalized_data[symbol] = {
                'X': X_normalized,
                'y': y,
                'scaler': scaler,
            }

            if feature_names is None:
                feature_names = feat_names

        cli.renderer.info(f"  Normalized {len(normalized_data)} stocks")

        # ============ Phase 3: Add Cross-Stock Features ============
        cli.renderer.info("")
        cli.renderer.info("Phase 3: Adding cross-stock features...")

        # Compute sector averages for relative features
        sector_stats = {}
        for symbol, data in normalized_data.items():
            sector = data['X']['_sector'].iloc[0]
            if sector not in sector_stats:
                sector_stats[sector] = []
            sector_stats[sector].append(data['X'][feature_names].mean())

        # Add sector-relative features
        for symbol, data in normalized_data.items():
            sector = data['X']['_sector'].iloc[0]
            if sector in sector_stats and len(sector_stats[sector]) > 1:
                sector_mean = pd.concat(sector_stats[sector], axis=1).mean(axis=1)
                for feat in feature_names[:10]:  # Top 10 features
                    data['X'][f'{feat}_vs_sector'] = data['X'][feat] - sector_mean.get(feat, 0)

        cli.renderer.info(f"  Added sector-relative features")

        # ============ Phase 4: Leave-One-Out Training ============
        cli.renderer.info("")
        cli.renderer.info("Phase 4: Leave-One-Out Cross-Validation...")

        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        results = []
        symbols = list(normalized_data.keys())

        for holdout_symbol in symbols[:min(5, len(symbols))]:  # Test on first 5
            # Train on all except holdout
            train_X_list = []
            train_y_list = []

            for symbol, data in normalized_data.items():
                if symbol != holdout_symbol:
                    # Drop metadata columns for training
                    X_train = data['X'].drop(columns=['_symbol', '_sector'], errors='ignore')
                    train_X_list.append(X_train)
                    train_y_list.append(data['y'])

            X_train_combined = pd.concat(train_X_list, ignore_index=True)
            y_train_combined = pd.concat(train_y_list, ignore_index=True)

            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

            # Align features
            train_features = [c for c in X_train_combined.columns if not c.startswith('_')]
            model.fit(X_train_combined[train_features].fillna(0), y_train_combined)

            # Test on holdout
            holdout_data = normalized_data[holdout_symbol]
            X_holdout = holdout_data['X'].drop(columns=['_symbol', '_sector'], errors='ignore')
            y_holdout = holdout_data['y']

            # Align holdout features
            for col in train_features:
                if col not in X_holdout.columns:
                    X_holdout[col] = 0
            X_holdout = X_holdout[train_features]

            y_pred_proba = model.predict_proba(X_holdout.fillna(0))[:, 1]
            auc = roc_auc_score(y_holdout, y_pred_proba)

            results.append({
                'holdout': holdout_symbol,
                'auc': auc,
                'n_train': len(y_train_combined),
                'n_test': len(y_holdout),
            })

            cli.renderer.info(f"  {holdout_symbol}: AUC={auc:.4f} (trained on {len(y_train_combined)} samples)")

        # ============ Results Summary ============
        cli.renderer.info("")
        cli.renderer.header("Cross-Stock Results")

        avg_auc = np.mean([r['auc'] for r in results])
        std_auc = np.std([r['auc'] for r in results])

        cli.renderer.info(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        cli.renderer.info("")

        cli.renderer.info("Per-Stock Performance:")
        for r in sorted(results, key=lambda x: -x['auc']):
            cli.renderer.info(f"  {r['holdout']:<12}: {r['auc']:.4f}")

        # Key insight
        cli.renderer.info("")
        if std_auc < 0.05:
            cli.renderer.info("LOW VARIANCE - Model generalizes well across stocks!")
        elif std_auc < 0.10:
            cli.renderer.info("MODERATE VARIANCE - Some stock-specific patterns")
        else:
            cli.renderer.info("HIGH VARIANCE - Consider more normalization or sector-specific models")

        return CommandResult.ok(data={
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'results': results,
            'stocks_used': list(normalized_data.keys()),
        })

    async def _run_with_comprehensive_features(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """
        Run ML with comprehensive features from /ml skills.

        Uses:
        - FeatureEngineeringSkill (Kaggle-style)
        - LLMFeatureReasonerSkill (multi-perspective)
        - Additional stock-specific features
        """
        import pandas as pd
        import numpy as np

        cli.renderer.header("Comprehensive Features ML")

        # Parse arguments
        symbol = args.positional[0] if args.positional else "SBIN"
        target_type = args.flags.get("target", "next_5d_up")
        timeframe = args.flags.get("timeframe", "60minute")
        years = int(args.flags.get("years", "3"))

        cli.renderer.info(f"Symbol: {symbol}")
        cli.renderer.info(f"Target: {target_type}, Timeframe: {timeframe}")
        cli.renderer.info("")

        # Load data
        df = await self._load_stock_data(symbol, timeframe, years, cli)
        if df is None or len(df) < 100:
            return CommandResult.fail("Insufficient data")

        target_config = self._parse_target(target_type)

        # Create base features
        cli.renderer.info("Creating base features...")
        X_base, y, base_features = self._create_features_and_target(df.copy(), target_config)

        if X_base is None:
            return CommandResult.fail("Feature creation failed")

        cli.renderer.info(f"  Base features: {len(base_features)}")

        # ============ Add Comprehensive Features ============
        cli.renderer.info("")
        cli.renderer.info("Adding comprehensive features...")

        X_enhanced = X_base.copy()

        # 1. Advanced momentum features
        cli.renderer.info("  [1/5] Advanced momentum features...")
        X_enhanced = self._add_advanced_momentum(X_enhanced, df)

        # 2. Volatility regime features
        cli.renderer.info("  [2/5] Volatility regime features...")
        X_enhanced = self._add_volatility_regime(X_enhanced, df)

        # 3. Volume profile features
        cli.renderer.info("  [3/5] Volume profile features...")
        X_enhanced = self._add_volume_profile(X_enhanced, df)

        # 4. Pattern recognition features
        cli.renderer.info("  [4/5] Pattern recognition features...")
        X_enhanced = self._add_pattern_features(X_enhanced, df)

        # 5. Cross-feature interactions
        cli.renderer.info("  [5/5] Cross-feature interactions...")
        X_enhanced = self._add_feature_interactions(X_enhanced)

        cli.renderer.info(f"  Total features: {len(X_enhanced.columns)}")

        # Clean up
        X_enhanced = X_enhanced.fillna(0).replace([np.inf, -np.inf], 0)

        # Align with target
        common_idx = X_enhanced.index.intersection(y.index)
        X_enhanced = X_enhanced.loc[common_idx]
        y = y.loc[common_idx]

        # ============ Train and Evaluate ============
        cli.renderer.info("")
        cli.renderer.info("Training with comprehensive features...")

        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_enhanced)):
            X_train = X_enhanced.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X_enhanced.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            scores.append(auc)

        avg_auc = np.mean(scores)
        cli.renderer.info(f"Average AUC: {avg_auc:.4f}")

        # Feature importance
        cli.renderer.info("")
        cli.renderer.info("Top 15 Features:")

        # Train final model
        model.fit(X_enhanced, y)
        importance = pd.Series(model.feature_importances_, index=X_enhanced.columns)
        importance = importance.sort_values(ascending=False)

        total_imp = importance.sum()
        for feat, imp in importance.head(15).items():
            pct = (imp / total_imp) * 100
            cli.renderer.info(f"  {feat:<35}: {pct:>5.1f}%")

        return CommandResult.ok(data={
            'symbol': symbol,
            'auc': avg_auc,
            'n_features': len(X_enhanced.columns),
            'top_features': importance.head(20).to_dict(),
        })

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
            marker = "★" if i == 0 else " "
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

    async def _run_swarm_learning(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """
        Run SwarmML auto-learning pipeline.

        Analyzes sweep results and feature importance patterns to:
        1. Learn which configs work best for which stock characteristics
        2. Identify underperforming stocks and suggest improvements
        3. Build transferable knowledge across stocks
        """
        import numpy as np
        from datetime import datetime

        cli.renderer.header("SwarmML Auto-Learning")
        cli.renderer.info("")

        # Load existing sweep results
        if not self.SWEEP_RESULTS_FILE.exists():
            cli.renderer.error("No sweep results found. Run: /stock-ml --sweep --stocks top10")
            return CommandResult.fail("No sweep results")

        with open(self.SWEEP_RESULTS_FILE) as f:
            sweep_results = json.load(f)

        if not sweep_results:
            cli.renderer.error("Empty sweep results")
            return CommandResult.fail("No results")

        cli.renderer.info(f"Analyzing {len(sweep_results)} sweep results...")

        # Load or initialize swarm learning state
        swarm_state = self._load_swarm_ml_state()

        # Initialize Q-Learner for real RL
        q_learner = StockMLQLearner()
        cli.renderer.info(f"Q-Learner: {q_learner.get_stats()['entries']} entries, ε={q_learner.epsilon:.3f}")

        # ============ Phase 1: Learn Stock Characteristics ============
        cli.renderer.info("")
        cli.renderer.info("Phase 1: Learning Stock Characteristics")

        stock_profiles = self._analyze_stock_profiles(sweep_results)
        swarm_state['stock_profiles'] = stock_profiles

        cli.renderer.info(f"  Analyzed {len(stock_profiles)} stock profiles")

        # ============ Phase 2: Learn Config Patterns ============
        cli.renderer.info("")
        cli.renderer.info("Phase 2: Learning Config Patterns")

        config_patterns = self._analyze_config_patterns(sweep_results)
        swarm_state['config_patterns'] = config_patterns

        cli.renderer.info(f"  Found {len(config_patterns['best_by_sector'])} sector patterns")
        cli.renderer.info(f"  Target rankings: {list(config_patterns['target_rankings'].keys())}")

        # ============ Phase 2b: Q-Learning Updates ============
        cli.renderer.info("")
        cli.renderer.info("Phase 2b: Q-Learning from Results")

        updates_count = 0
        for r in sweep_results:
            symbol = r.get('symbol', 'UNKNOWN')
            target = r.get('target', 'next_5d_up')
            timeframe = r.get('timeframe', 'day')
            auc = r.get('auc', 0.5)

            # Get state from profile
            profile = stock_profiles.get(symbol, {})
            if not profile:
                continue

            state = q_learner.get_state(profile)
            action = q_learner.get_action_key(target, timeframe)
            reward = q_learner.compute_reward({'auc': auc})

            # Q-learning update
            q_learner.update(state, action, reward)
            updates_count += 1

        cli.renderer.info(f"  Q-learning updates: {updates_count}")
        cli.renderer.info(f"  Q-table size: {q_learner.get_stats()['entries']}")
        cli.renderer.info(f"  Avg Q-value: {q_learner.get_stats()['avg_q']:.4f}")

        # Store Q-learner reference in state
        swarm_state['q_learner_stats'] = q_learner.get_stats()

        # ============ Phase 3: Identify Underperformers ============
        cli.renderer.info("")
        cli.renderer.info("Phase 3: Identifying Underperformers")

        underperformers = self._identify_underperformers(sweep_results, stock_profiles)
        swarm_state['underperformers'] = underperformers

        cli.renderer.info(f"  Found {len(underperformers)} underperforming configs")

        # ============ Phase 4: Generate Recommendations ============
        cli.renderer.info("")
        cli.renderer.info("Phase 4: Generating Recommendations")

        recommendations = self._generate_recommendations(
            sweep_results, stock_profiles, config_patterns, underperformers
        )
        swarm_state['recommendations'] = recommendations
        swarm_state['last_updated'] = datetime.now().isoformat()

        # Save learning state
        self._save_swarm_ml_state(swarm_state)
        cli.renderer.info("")
        cli.renderer.success(f"Learnings saved to {self.SWARM_ML_STATE_FILE}")

        # ============ Display Results ============
        cli.renderer.info("")
        cli.renderer.header("SwarmML Insights")

        # Top patterns
        cli.renderer.info("")
        cli.renderer.info("Top Performing Patterns:")
        for pattern in config_patterns.get('top_patterns', [])[:5]:
            cli.renderer.info(f"  {pattern['pattern']}: AUC {pattern['avg_auc']:.4f} ({pattern['count']} samples)")

        # Sector insights
        cli.renderer.info("")
        cli.renderer.info("Best Target by Sector:")
        for sector, best in list(config_patterns['best_by_sector'].items())[:8]:
            cli.renderer.info(f"  {sector:<15}: {best['target']:<12} (AUC {best['auc']:.4f})")

        # Recommendations
        cli.renderer.info("")
        cli.renderer.info("Top Recommendations:")
        for rec in recommendations[:5]:
            cli.renderer.info(f"  {rec['symbol']}: {rec['recommendation']}")
            if 'suggested_config' in rec:
                cli.renderer.info(f"    -> Try: {rec['suggested_config']}")

        # Cross-stock learnings
        cli.renderer.info("")
        cli.renderer.info("Cross-Stock Learnings:")
        cli.renderer.info(f"  Best overall timeframe: {config_patterns.get('best_timeframe', 'day')}")
        cli.renderer.info(f"  Best overall target: {config_patterns.get('best_target', 'next_1d_up')}")
        cli.renderer.info(f"  Optimal feature correlation: {config_patterns.get('feature_insight', 'momentum features dominate')}")

        # Q-Learning Insights
        cli.renderer.info("")
        cli.renderer.header("Q-Learning Strategy Recommendations")

        # Show Q-recommendations by sector
        for sector in ['banking', 'it', 'fmcg', 'auto', 'other']:
            # Create mock profile for sector query
            mock_profile = {'sector': sector, 'predictability': 'medium', 'auc_std': 0.05, 'target_sensitivity': {}}
            state = q_learner.get_state(mock_profile)
            recs = q_learner.get_recommendations(state, top_k=2)

            if recs and recs[0]['confidence'] > 0:
                cli.renderer.info(f"  {sector.upper()}:")
                for rec in recs[:2]:
                    conf_str = f"({rec['confidence']*100:.0f}% confident)" if rec['confidence'] > 0.3 else "(exploring)"
                    cli.renderer.info(f"    {rec['target']} + {rec['timeframe']}: Q={rec['q_value']:.3f} {conf_str}")

        # Transfer learning context
        cli.renderer.info("")
        cli.renderer.info("Transfer Learning:")
        context = q_learner.get_transfer_learning_context("banking|medium|high|medium")
        if context:
            for line in context.split('\n')[:4]:
                cli.renderer.info(f"  {line}")
        else:
            cli.renderer.info("  (Not enough data yet - run more sweeps)")

        return CommandResult.ok(data={
            'stock_profiles': stock_profiles,
            'config_patterns': config_patterns,
            'underperformers': underperformers,
            'recommendations': recommendations,
        })

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

    def _analyze_stock_profiles(self, sweep_results: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze each stock's performance characteristics.

        Groups stocks by their optimal configs and performance patterns.
        """
        import numpy as np

        profiles = {}

        # Group results by stock
        by_stock = {}
        for r in sweep_results:
            symbol = r.get('symbol', 'UNKNOWN')
            if symbol not in by_stock:
                by_stock[symbol] = []
            by_stock[symbol].append(r)

        for symbol, results in by_stock.items():
            if not results:
                continue

            aucs = [r.get('auc', 0) for r in results]
            accuracies = [r.get('accuracy', 0) for r in results]

            # Find best config for this stock
            best_result = max(results, key=lambda x: x.get('auc', 0))

            # Analyze sensitivity to different parameters
            by_target = {}
            by_timeframe = {}

            for r in results:
                target = r.get('target', 'unknown')
                timeframe = r.get('timeframe', 'day')

                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(r.get('auc', 0))

                if timeframe not in by_timeframe:
                    by_timeframe[timeframe] = []
                by_timeframe[timeframe].append(r.get('auc', 0))

            # Compute averages
            target_avg = {t: np.mean(aucs) for t, aucs in by_target.items()}
            timeframe_avg = {t: np.mean(aucs) for t, aucs in by_timeframe.items()}

            # Determine stock characteristics
            best_target = max(target_avg.items(), key=lambda x: x[1])[0] if target_avg else 'next_1d_up'
            best_timeframe = max(timeframe_avg.items(), key=lambda x: x[1])[0] if timeframe_avg else 'day'

            # Classify stock predictability
            max_auc = max(aucs) if aucs else 0
            if max_auc >= 0.70:
                predictability = 'high'
            elif max_auc >= 0.60:
                predictability = 'medium'
            else:
                predictability = 'low'

            # Determine sector (heuristic based on stock name)
            sector = self._infer_sector(symbol)

            profiles[symbol] = {
                'best_auc': max(aucs) if aucs else 0,
                'avg_auc': np.mean(aucs) if aucs else 0,
                'auc_std': np.std(aucs) if aucs else 0,
                'best_target': best_target,
                'best_timeframe': best_timeframe,
                'predictability': predictability,
                'sector': sector,
                'n_configs_tested': len(results),
                'target_sensitivity': target_avg,
                'timeframe_sensitivity': timeframe_avg,
                'best_config': {
                    'target': best_result.get('target'),
                    'timeframe': best_result.get('timeframe'),
                    'years': best_result.get('years'),
                    'auc': best_result.get('auc'),
                },
            }

        return profiles

    def _infer_sector(self, symbol: str) -> str:
        """Infer sector from stock symbol."""
        # Check against known sets
        for sector, stocks in self.STOCK_SETS.items():
            if symbol in stocks:
                if 'bank' in sector.lower() or sector in ['banks']:
                    return 'banking'
                if 'it' in sector.lower():
                    return 'it'
                if 'pharma' in sector.lower():
                    return 'pharma'
                if 'auto' in sector.lower():
                    return 'auto'
                if 'fmcg' in sector.lower():
                    return 'fmcg'
                if 'metal' in sector.lower():
                    return 'metal'
                if 'energy' in sector.lower():
                    return 'energy'
                if 'infra' in sector.lower() or 'realty' in sector.lower():
                    return 'infra'

        # Keyword matching
        symbol_upper = symbol.upper()
        if any(x in symbol_upper for x in ['BANK', 'FIN']):
            return 'banking'
        if any(x in symbol_upper for x in ['TECH', 'INFO', 'SOFT', 'WIPRO', 'INFY', 'TCS']):
            return 'it'
        if any(x in symbol_upper for x in ['PHARMA', 'SUN', 'CIPLA', 'LUPIN']):
            return 'pharma'
        if any(x in symbol_upper for x in ['AUTO', 'MOTOR', 'MARUTI', 'TATA']):
            return 'auto'

        return 'other'

    def _analyze_config_patterns(self, sweep_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns across all configs to find what works best.
        """
        import numpy as np

        patterns = {
            'target_rankings': {},
            'timeframe_rankings': {},
            'best_by_sector': {},
            'top_patterns': [],
            'best_timeframe': 'day',
            'best_target': 'next_1d_up',
            'feature_insight': '',
        }

        # Group by target
        by_target = {}
        for r in sweep_results:
            target = r.get('target', 'unknown')
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(r.get('auc', 0))

        patterns['target_rankings'] = {
            t: {'avg_auc': np.mean(aucs), 'count': len(aucs)}
            for t, aucs in by_target.items()
        }

        if patterns['target_rankings']:
            patterns['best_target'] = max(
                patterns['target_rankings'].items(),
                key=lambda x: x[1]['avg_auc']
            )[0]

        # Group by timeframe
        by_timeframe = {}
        for r in sweep_results:
            tf = r.get('timeframe', 'day')
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(r.get('auc', 0))

        patterns['timeframe_rankings'] = {
            t: {'avg_auc': np.mean(aucs), 'count': len(aucs)}
            for t, aucs in by_timeframe.items()
        }

        if patterns['timeframe_rankings']:
            patterns['best_timeframe'] = max(
                patterns['timeframe_rankings'].items(),
                key=lambda x: x[1]['avg_auc']
            )[0]

        # Group by sector (using inferred sectors)
        by_sector = {}
        for r in sweep_results:
            sector = self._infer_sector(r.get('symbol', ''))
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(r)

        for sector, results in by_sector.items():
            if not results:
                continue
            best = max(results, key=lambda x: x.get('auc', 0))
            patterns['best_by_sector'][sector] = {
                'target': best.get('target'),
                'timeframe': best.get('timeframe'),
                'auc': best.get('auc', 0),
                'symbol': best.get('symbol'),
            }

        # Find top patterns (target + timeframe combos)
        combo_results = {}
        for r in sweep_results:
            combo = f"{r.get('target', 'unknown')}_{r.get('timeframe', 'day')}"
            if combo not in combo_results:
                combo_results[combo] = []
            combo_results[combo].append(r.get('auc', 0))

        patterns['top_patterns'] = sorted([
            {'pattern': k, 'avg_auc': np.mean(v), 'count': len(v)}
            for k, v in combo_results.items()
        ], key=lambda x: -x['avg_auc'])

        # Feature insight (based on patterns)
        best_tf = patterns['best_timeframe']
        best_tgt = patterns['best_target']
        if '60minute' in best_tf or 'hour' in best_tf:
            patterns['feature_insight'] = 'Intraday patterns are predictive; focus on short-term momentum'
        elif '1d' in best_tgt:
            patterns['feature_insight'] = 'Short-term momentum features dominate; RSI, MACD effective'
        elif '20d' in best_tgt or '30d' in best_tgt:
            patterns['feature_insight'] = 'Trend-following features important; moving averages, ADX'
        else:
            patterns['feature_insight'] = 'Mixed signals; consider ensemble of timeframes'

        return patterns

    def _identify_underperformers(
        self, sweep_results: List[Dict], stock_profiles: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Identify underperforming stocks and configs.
        """
        import numpy as np

        underperformers = []

        # Calculate overall baseline
        all_aucs = [r.get('auc', 0) for r in sweep_results]
        baseline_auc = np.median(all_aucs) if all_aucs else 0.5
        threshold = baseline_auc - 0.05  # Underperform if 5% below median

        # Find stocks consistently below threshold
        for symbol, profile in stock_profiles.items():
            if profile['best_auc'] < threshold:
                underperformers.append({
                    'symbol': symbol,
                    'best_auc': profile['best_auc'],
                    'gap': baseline_auc - profile['best_auc'],
                    'sector': profile['sector'],
                    'reason': self._diagnose_underperformance(profile),
                })

        # Sort by gap (worst first)
        underperformers.sort(key=lambda x: -x['gap'])

        return underperformers

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

    def _generate_recommendations(
        self,
        sweep_results: List[Dict],
        stock_profiles: Dict[str, Dict],
        config_patterns: Dict[str, Any],
        underperformers: List[Dict],
    ) -> List[Dict]:
        """
        Generate actionable recommendations based on learnings.
        """
        recommendations = []

        # 1. Recommendations for underperformers
        for up in underperformers[:10]:
            symbol = up['symbol']
            profile = stock_profiles.get(symbol, {})
            sector = profile.get('sector', 'other')

            # Find what works for similar stocks in same sector
            sector_best = config_patterns['best_by_sector'].get(sector, {})

            rec = {
                'symbol': symbol,
                'current_best_auc': up['best_auc'],
                'recommendation': up['reason'],
                'priority': 'high' if up['gap'] > 0.1 else 'medium',
            }

            # Suggest alternative config based on sector patterns
            if sector_best and sector_best.get('auc', 0) > up['best_auc']:
                rec['suggested_config'] = (
                    f"target={sector_best['target']}, "
                    f"timeframe={sector_best['timeframe']}"
                )

            recommendations.append(rec)

        # 2. General recommendations
        best_tf = config_patterns.get('best_timeframe', 'day')
        best_tgt = config_patterns.get('best_target', 'next_1d_up')

        if best_tf == '60minute':
            recommendations.append({
                'symbol': 'ALL',
                'recommendation': f'60-minute data consistently outperforms daily; migrate all stocks to {best_tf}',
                'priority': 'high',
            })

        if best_tgt == 'next_1d_up':
            recommendations.append({
                'symbol': 'ALL',
                'recommendation': 'Short-term (1-day) predictions most reliable; avoid long-horizon targets',
                'priority': 'medium',
            })

        # 3. Sector-specific recommendations
        for sector, best in config_patterns.get('best_by_sector', {}).items():
            if best.get('auc', 0) >= 0.65:
                recommendations.append({
                    'symbol': f'{sector.upper()} SECTOR',
                    'recommendation': f"Focus on {best['target']} with {best['timeframe']} data (AUC {best['auc']:.4f})",
                    'priority': 'medium',
                })

        return recommendations

    def _show_swarm_insights(self, cli: "JottyCLI") -> CommandResult:
        """Display current swarm ML insights."""
        if not self.SWARM_ML_STATE_FILE.exists():
            cli.renderer.info("No swarm learnings yet. Run: /stock-ml --swarm-learn")
            return CommandResult.ok(data={})

        state = self._load_swarm_ml_state()

        cli.renderer.header("SwarmML Insights Dashboard")
        cli.renderer.info(f"Last updated: {state.get('last_updated', 'Never')}")
        cli.renderer.info("")

        # Stock profiles summary
        profiles = state.get('stock_profiles', {})
        if profiles:
            cli.renderer.info("Stock Predictability Tiers")

            high = [s for s, p in profiles.items() if p.get('predictability') == 'high']
            medium = [s for s, p in profiles.items() if p.get('predictability') == 'medium']
            low = [s for s, p in profiles.items() if p.get('predictability') == 'low']

            cli.renderer.info(f"  High ({len(high)}):   {', '.join(high[:10])}")
            cli.renderer.info(f"  Medium ({len(medium)}): {', '.join(medium[:10])}")
            cli.renderer.info(f"  Low ({len(low)}):    {', '.join(low[:10])}")

        # Config patterns
        patterns = state.get('config_patterns', {})
        if patterns:
            cli.renderer.info("")
            cli.renderer.info("Optimal Configurations")
            cli.renderer.info(f"  Best timeframe: {patterns.get('best_timeframe', 'N/A')}")
            cli.renderer.info(f"  Best target: {patterns.get('best_target', 'N/A')}")
            cli.renderer.info(f"  Insight: {patterns.get('feature_insight', 'N/A')}")

            cli.renderer.info("")
            cli.renderer.info("Target Rankings:")
            for target, stats in sorted(
                patterns.get('target_rankings', {}).items(),
                key=lambda x: -x[1].get('avg_auc', 0)
            )[:5]:
                cli.renderer.info(f"    {target:<15}: AUC {stats['avg_auc']:.4f} ({stats['count']} tests)")

        # Recommendations
        recs = state.get('recommendations', [])
        if recs:
            cli.renderer.info("")
            cli.renderer.info("Active Recommendations")
            for rec in recs[:8]:
                priority = rec.get('priority', 'low')
                marker = "!" if priority == 'high' else "-"
                cli.renderer.info(f"  {marker} {rec['symbol']}: {rec['recommendation']}")

        return CommandResult.ok(data=state)

    async def _run_swarm_refinement(
        self,
        symbol: str,
        args: ParsedArgs,
        cli: "JottyCLI"
    ) -> CommandResult:
        """
        Run swarm-guided refinement for a specific stock.

        Uses learnings to try alternative configs automatically.
        """
        cli.renderer.header(f"SwarmML Refinement: {symbol}")

        # Load learnings
        state = self._load_swarm_ml_state()
        profiles = state.get('stock_profiles', {})
        patterns = state.get('config_patterns', {})

        profile = profiles.get(symbol, {})
        if not profile:
            cli.renderer.info(f"No profile for {symbol}. Running initial sweep...")
            # Run quick sweep for this stock
            args.flags['stocks'] = symbol
            args.flags['sweep-targets'] = 'next_1d_up,next_5d_up,next_10d_up'
            args.flags['sweep-timeframes'] = 'day,60minute'
            return await self._run_sweep(args, cli)

        # Get recommendations for this stock's sector
        sector = profile.get('sector', 'other')
        sector_best = patterns.get('best_by_sector', {}).get(sector, {})

        cli.renderer.info(f"Current best: AUC {profile['best_auc']:.4f}")
        cli.renderer.info(f"  Target: {profile['best_config'].get('target')}")
        cli.renderer.info(f"  Timeframe: {profile['best_config'].get('timeframe')}")
        cli.renderer.info("")

        # Try sector-recommended config if different
        if sector_best:
            suggested_target = sector_best.get('target')
            suggested_tf = sector_best.get('timeframe')

            if (suggested_target != profile['best_config'].get('target') or
                suggested_tf != profile['best_config'].get('timeframe')):

                cli.renderer.info(f"Trying sector-recommended config:")
                cli.renderer.info(f"  Target: {suggested_target}")
                cli.renderer.info(f"  Timeframe: {suggested_tf}")
                cli.renderer.info("")

                # Load data and test
                years = int(args.flags.get("years", "3"))
                df = await self._load_stock_data(symbol, suggested_tf, years, cli)

                if df is not None and len(df) >= 100:
                    target_config = self._parse_target(suggested_target)
                    X, y, feature_names = self._create_features_and_target(df.copy(), target_config)

                    if X is not None and len(X) >= 100:
                        result = await self._quick_train(X, y, feature_names, target_config, cli)

                        new_auc = result.get('auc', 0)
                        improvement = new_auc - profile['best_auc']

                        cli.renderer.info(f"Result: AUC {new_auc:.4f}")

                        if improvement > 0:
                            cli.renderer.info(f"IMPROVEMENT: +{improvement:.4f}")
                            cli.renderer.info("Updating profile...")

                            # Update profile
                            profile['best_auc'] = new_auc
                            profile['best_config'] = {
                                'target': suggested_target,
                                'timeframe': suggested_tf,
                                'years': years,
                                'auc': new_auc,
                            }
                            profiles[symbol] = profile
                            state['stock_profiles'] = profiles
                            self._save_swarm_ml_state(state)
                        else:
                            cli.renderer.info(f"No improvement ({improvement:.4f})")

        return CommandResult.ok(data={'symbol': symbol, 'profile': profile})


# =============================================================================
# REAL Q-LEARNING INTEGRATION FOR STOCK ML
# =============================================================================

class StockMLQLearner:
    """
    Q-Learning for Stock ML Strategy Selection.

    Uses Jotty's Q-learning infrastructure to learn:
    - Which configs (target, timeframe) work for which stock characteristics
    - How to adapt strategies based on market conditions
    - Transfer learning across similar stocks

    State = (sector, volatility_regime, trend_strength, predictability_tier)
    Action = (target, timeframe, feature_set)
    Reward = AUC + Sharpe_bonus - drawdown_penalty
    """

    # Q-table persistence path
    Q_TABLE_PATH = Path.home() / ".jotty" / "stock_ml_qtable.json"

    def __init__(self):
        self.Q = {}  # State-Action -> (value, count, last_updated)
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor (lower for stock ML - focus on immediate)
        self.epsilon = 0.15  # Exploration rate

        # Load existing Q-table
        self._load_q_table()

    def _load_q_table(self):
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

    def _save_q_table(self):
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

    def update(self, state: str, action: str, reward: float, next_state: str = None):
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
