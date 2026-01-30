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

from typing import TYPE_CHECKING, Dict, Any, Optional, List
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
        "nifty50": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
                    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK",
                    "ASIANPAINT", "MARUTI", "TITAN", "SUNPHARMA", "BAJFINANCE",
                    "WIPRO", "ULTRACEMCO", "NESTLEIND"],
        "banks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                  "BANKBARODA", "INDUSINDBK", "PNB", "FEDERALBNK", "IDFCFIRSTB"],
        "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE"],
        "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "BIOCON"],
        "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
        "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
        "top10": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                  "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK"],
    }

    # Sweep results file
    SWEEP_RESULTS_FILE = Path.home() / ".jotty" / "stock_ml_sweep_results.json"

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
            cli.renderer.info("  /stock-ml --sweep --stocks RELIANCE,TCS   # Sweep specific stocks")
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
                experiment_name=experiment_name
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

    async def _run_stock_ml(self, X, y, feature_names, target_config, symbol,
                            max_iterations, cli, use_mlflow=False, experiment_name="stock"):
        """Run ML pipeline for stock prediction."""
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

        return {
            'symbol': symbol,
            'target': target_config,
            'best_model': best_name,
            'best_score': best_score,
            'results': results,
            'feature_importance': sorted_imp[:15] if hasattr(best_model, 'feature_importances_') else [],
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

        # Parse stocks
        if stocks_input in self.STOCK_SETS:
            stocks = self.STOCK_SETS[stocks_input]
        else:
            stocks = [s.strip().upper() for s in stocks_input.split(",")]

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

        cli.renderer.info("Usage:")
        cli.renderer.info("  /stock-ml --sweep --stocks top10")
        cli.renderer.info("  /stock-ml --sweep --stocks banks")
        cli.renderer.info("  /stock-ml --sweep --stocks RELIANCE,TCS,INFY")

        return CommandResult.ok(data=self.STOCK_SETS)

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        targets = list(self.TARGET_TYPES.keys())
        timeframes = list(self.TIMEFRAMES.keys())
        stock_sets = list(self.STOCK_SETS.keys())
        flags = ["--target", "--timeframe", "--years", "--iterations", "--mlflow",
                 "--experiment", "--list", "--targets", "--compare", "--benchmark",
                 "--compare-targets", "--compare-timeframes", "--sweep", "--grid",
                 "--stocks", "--sweep-targets", "--sweep-timeframes", "--sweep-periods",
                 "--leaderboard", "--lb", "--sets"]
        popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

        all_completions = targets + timeframes + stock_sets + flags + popular_stocks
        return [s for s in all_completions if s.lower().startswith(partial.lower())]
