"""
Stock ML - Training Mode Methods
==================================

Extended training modes for StockMLCommand: comparison, world-class backtest,
sweep, unified training, cross-stock normalization, comprehensive features.
"""

import warnings
import json
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
from pathlib import Path

warnings.filterwarnings('ignore')

from .base import CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class StockMLTrainingMixin:
    """Mixin providing extended training modes for StockMLCommand.

    These methods implement various ML training strategies and are called
    by StockMLCommand.execute() based on subcommand parsing.
    """

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
            from Jotty.core.skills.ml import MLflowTrackerSkill
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

    async def _run_world_class_backtest(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Run world-class comprehensive backtest with institutional-grade analysis."""
        import pandas as pd
        import numpy as np
        from datetime import datetime
        from pathlib import Path

        # Parse arguments
        symbol = args.positional[0].upper() if args.positional else None
        timeframe = args.flags.get("timeframe", args.flags.get("tf", "day"))
        target_type = args.flags.get("target", args.flags.get("t", "next_5d_up"))
        years = int(args.flags.get("years", args.flags.get("y", "2")))

        if not symbol:
            cli.renderer.error("Stock symbol required for world-class backtest.")
            cli.renderer.info("")
            cli.renderer.info("Usage: /stock-ml <SYMBOL> --wc [options]")
            cli.renderer.info("")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /stock-ml RELIANCE --wc")
            cli.renderer.info("  /stock-ml TCS --wc --target next_10d_up")
            cli.renderer.info("  /stock-ml HDFCBANK --wc --years 3")
            return CommandResult.fail("Symbol required")

        cli.renderer.header(f"World-Class ML Backtest: {symbol}")
        cli.renderer.info(f"Target: {target_type}")
        cli.renderer.info(f"Period: {years} years | Timeframe: {timeframe}")
        cli.renderer.info("")

        try:
            # Parse target config
            target_config = self._parse_target(target_type)
            is_classification = target_config['type'] == 'classification'

            # Load data
            cli.renderer.status(f"Loading {symbol} data...")
            df = await self._load_stock_data(symbol, timeframe, years, cli)
            if df is None or len(df) < 100:
                cli.renderer.error(f"Insufficient data for {symbol}")
                return CommandResult.fail("Insufficient data")

            cli.renderer.info(f"Loaded {len(df)} records ({df['date'].min().date()} to {df['date'].max().date()})")

            # Create features
            cli.renderer.status("Engineering features...")
            X, y, feature_names = self._create_features_and_target(df.copy(), target_config)
            if X is None or len(X) < 100:
                cli.renderer.error("Insufficient data after feature engineering")
                return CommandResult.fail("Insufficient data")

            cli.renderer.info(f"Features: {len(feature_names)}")

            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

            # Train-test split (temporal)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train models with timeframe-optimized hyperparameters
            cli.renderer.status("Training models with optimized hyperparameters...")
            import lightgbm as lgb
            import xgboost as xgb

            # Detect if intraday timeframe (use RoMaD-optimized params)
            is_intraday = timeframe.lower() in ['15min', '15minute', '30min', '30minute',
                                                  '60min', '60minute', 'hourly']

            if is_classification:
                if is_intraday:
                    # RoMaD-optimized params for intraday (shallow trees, slow learning)
                    models = {
                        'LightGBM': lgb.LGBMClassifier(
                            n_estimators=386, learning_rate=0.0067, max_depth=4,
                            num_leaves=32, min_child_samples=7, subsample=0.80,
                            colsample_bytree=0.50, reg_alpha=0.013, reg_lambda=0.017,
                            verbose=-1, random_state=42, n_jobs=-1,
                        ),
                        'XGBoost': xgb.XGBClassifier(
                            n_estimators=400, learning_rate=0.01, max_depth=4,
                            min_child_weight=5, subsample=0.75, colsample_bytree=0.5,
                            gamma=0.05, reg_alpha=0.01, reg_lambda=0.02,
                            verbosity=0, random_state=42, n_jobs=-1, tree_method='hist',
                        ),
                    }
                else:
                    # Daily timeframe params (deeper trees, faster learning)
                    models = {
                        'LightGBM': lgb.LGBMClassifier(
                            n_estimators=500, learning_rate=0.02, max_depth=8,
                            num_leaves=63, min_child_samples=20, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            verbose=-1, random_state=42, n_jobs=-1,
                        ),
                        'XGBoost': xgb.XGBClassifier(
                            n_estimators=500, learning_rate=0.02, max_depth=7,
                            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                            verbosity=0, random_state=42, n_jobs=-1, tree_method='hist',
                        ),
                    }
            else:
                if is_intraday:
                    # RoMaD-optimized params for intraday regression
                    models = {
                        'LightGBM': lgb.LGBMRegressor(
                            n_estimators=386, learning_rate=0.0067, max_depth=4,
                            num_leaves=32, min_child_samples=7, subsample=0.80,
                            colsample_bytree=0.50, reg_alpha=0.013, reg_lambda=0.017,
                            verbose=-1, random_state=42, n_jobs=-1,
                        ),
                        'XGBoost': xgb.XGBRegressor(
                            n_estimators=400, learning_rate=0.01, max_depth=4,
                            min_child_weight=5, subsample=0.75, colsample_bytree=0.5,
                            gamma=0.05, reg_alpha=0.01, reg_lambda=0.02,
                            verbosity=0, random_state=42, n_jobs=-1, tree_method='hist',
                        ),
                    }
                else:
                    # Daily timeframe params
                    models = {
                        'LightGBM': lgb.LGBMRegressor(
                            n_estimators=500, learning_rate=0.02, max_depth=8,
                            num_leaves=63, min_child_samples=20, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                            verbose=-1, random_state=42, n_jobs=-1,
                        ),
                        'XGBoost': xgb.XGBRegressor(
                            n_estimators=500, learning_rate=0.02, max_depth=7,
                            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                            verbosity=0, random_state=42, n_jobs=-1, tree_method='hist',
                        ),
                    }

            # Find best model
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, r2_score
            best_model = None
            best_score = -np.inf
            best_name = None

            for name, model in models.items():
                model.fit(X_train.values, y_train.values)
                pred = model.predict(X_test.values)

                if is_classification:
                    try:
                        proba = model.predict_proba(X_test.values)
                        score = roc_auc_score(y_test, proba[:, 1])
                    except:
                        score = accuracy_score(y_test, pred)
                else:
                    score = r2_score(y_test, pred)

                cli.renderer.info(f"  {name}: Score={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name

            cli.renderer.info(f"Best: {best_name} (Score: {best_score:.4f})")

            # Generate predictions for full test set
            if is_classification:
                predictions = best_model.predict_proba(X_test.values)[:, 1]
            else:
                predictions = best_model.predict(X_test.values)

            # Get price data for backtesting
            test_dates = df.loc[X_test.index, 'date']
            test_prices = df.loc[X_test.index, 'close'].values

            # Run World-Class Backtest Engine
            cli.renderer.info("")
            cli.renderer.header("Running World-Class Backtest Engine")

            from Jotty.core.skills.ml import (
                WorldClassBacktestEngine,
                TransactionCosts,
                ComprehensiveBacktestReportGenerator,
            )

            # Initialize engine
            engine = WorldClassBacktestEngine(risk_free_rate=0.05)

            # Create transaction costs
            costs = TransactionCosts(
                commission_pct=0.001,
                slippage_pct=0.001,
                market_impact_pct=0.0005,
            )

            # Prepare signals
            if is_classification:
                # Binary signal from probability (1 = long, 0 = flat)
                signals = (predictions > 0.5).astype(int)
            else:
                # Signal from return prediction
                signals = (predictions > 0).astype(int)

            # Prepare price dataframe for engine
            test_df = df.loc[X_test.index].copy()
            test_df = test_df.reset_index(drop=True)

            # Run backtest
            cli.renderer.status("Running comprehensive analysis...")
            result = engine.run_backtest(
                prices=test_df,
                signals=signals,
                benchmark=None,  # Use buy-and-hold as default
                costs=costs,
                walk_forward_windows=5,
                monte_carlo_sims=1000,
                target_volatility=0.10,
            )

            # Generate comprehensive report
            cli.renderer.info("")
            cli.renderer.header("Generating Comprehensive Report")

            report_generator = ComprehensiveBacktestReportGenerator()

            # Generate both markdown and PDF reports
            cli.renderer.status("Generating reports...")
            try:
                md_path, pdf_path = await report_generator.generate_report(
                    result=result,
                    template_name="quantitative"
                )
                cli.renderer.info(f"Markdown: {md_path}")
                cli.renderer.info(f"PDF: {pdf_path}")
            except Exception as e:
                cli.renderer.warning(f"Report generation failed: {e}")
                md_path = None
                pdf_path = None

            # Send to Telegram
            cli.renderer.info("")
            cli.renderer.status("Sending to Telegram...")
            try:
                sent = await report_generator.send_to_telegram(pdf_path or md_path, result)
                if sent:
                    cli.renderer.info("✓ Report sent to Telegram")
                else:
                    cli.renderer.warning("Telegram send failed")
            except Exception as e:
                cli.renderer.warning(f"Telegram error: {e}")

            # Print summary
            cli.renderer.info("")
            cli.renderer.header("Performance Summary")

            stats = result.statistical_tests
            risk = result.risk_metrics
            mc = result.monte_carlo

            cli.renderer.info("")
            cli.renderer.info("┌─────────────────────────────────────────────────────────┐")
            cli.renderer.info(f"│  Total Return (Gross):      {result.total_return*100:>+8.2f}%                   │")
            cli.renderer.info(f"│  Total Return (Net):        {result.total_return_net*100:>+8.2f}%                   │")
            cli.renderer.info(f"│  Sharpe Ratio:              {result.sharpe_ratio:>+8.2f}                     │")
            cli.renderer.info(f"│  Sortino Ratio:             {result.sortino_ratio:>+8.2f}                     │")
            cli.renderer.info(f"│  Max Drawdown:              {risk.max_drawdown*100:>+8.2f}%                   │")
            cli.renderer.info(f"│  Win Rate:                  {result.win_rate*100:>8.1f}%                   │")
            cli.renderer.info("├─────────────────────────────────────────────────────────┤")
            is_significant = stats.p_value < 0.05
            cli.renderer.info(f"│  P-Value:                   {stats.p_value:>8.4f}                     │")
            cli.renderer.info(f"│  Statistically Significant: {'Yes' if is_significant else 'No':>8}                     │")
            cli.renderer.info(f"│  Monte Carlo P(Profit):     {mc.prob_positive*100:>8.1f}%                   │")
            cli.renderer.info("└─────────────────────────────────────────────────────────┘")

            return CommandResult.ok(data={
                'symbol': symbol,
                'total_return': result.total_return,
                'total_return_net': result.total_return_net,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': risk.max_drawdown,
                'p_value': stats.p_value,
                'is_significant': is_significant,
                'report_paths': {
                    'markdown': str(md_path) if md_path else None,
                    'pdf': str(pdf_path) if pdf_path else None,
                }
            })

        except Exception as e:
            cli.renderer.error(f"World-class backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return CommandResult.fail(str(e))

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
            from Jotty.core.skills.ml import MLflowTrackerSkill
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
            }
        else:
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
            from Jotty.core.skills.ml import MLflowTrackerSkill
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

